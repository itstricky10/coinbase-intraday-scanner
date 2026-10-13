"""Unattended Coinbase public-market research scanner. No orders or account access.

Qualified observations -> public.scanner_signal_journal.
Near misses (one or two technical checks failed) -> public.scanner_near_misses.
Resolves worker-generated paper signals when completed candles show a stop/target.
Both tables require id text primary key, payload jsonb.
"""
import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

BASE = "https://api.exchange.coinbase.com"
HEADERS = {"User-Agent": "TradingDeskResearchWorker/2.1", "Accept": "application/json"}
STABLE = {"USD", "GBP", "USDC", "USDT", "DAI", "EUR", "PYUSD", "TUSD", "FDUSD", "GUSD", "PAX"}


def get(path, params=None):
    for attempt in range(4):
        try:
            response = requests.get(BASE + path, params=params, headers=HEADERS, timeout=20)
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Coinbase request failed: {path}")


def ema(values, span):
    alpha = 2 / (span + 1)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1 - alpha) * result
    return result


def rsi(values, period=14):
    if len(values) < period + 2:
        return None
    changes = [values[i] - values[i - 1] for i in range(1, len(values))]
    gain = sum(max(v, 0) for v in changes[:period]) / period
    loss = sum(max(-v, 0) for v in changes[:period]) / period
    for change in changes[period:]:
        gain = (gain * (period - 1) + max(change, 0)) / period
        loss = (loss * (period - 1) + max(-change, 0)) / period
    if loss == 0:
        return 100.0 if gain > 0 else 50.0
    return 100 - 100 / (1 + gain / loss)


def candles(pair, granularity):
    raw = get(f"/products/{pair}/candles", {"granularity": granularity})
    now = time.time()
    return sorted(
        (v for v in raw if len(v) >= 6 and float(v[0]) + granularity <= now),
        key=lambda v: v[0],
    )


def indicators(rows, granularity):
    if len(rows) < 60:
        return None
    closes = [float(x[4]) for x in rows]
    volumes = [float(x[5]) for x in rows]
    highs = [float(x[2]) for x in rows]
    lows = [float(x[1]) for x in rows]
    price = closes[-1]
    e20, e50 = ema(closes, 20), ema(closes, 50)
    strength = rsi(closes)
    if price <= 0 or strength is None:
        return None
    trs = [
        max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        for i in range(1, len(rows))
    ]
    atr = sum(trs[-14:]) / 14
    previous_vol = sorted(volumes[-21:-1])
    median = (previous_vol[9] + previous_vol[10]) / 2
    rvol = volumes[-1] / median if median > 0 else 0
    return {
        "price": price, "ema20": e20, "ema50": e50, "rsi": strength,
        "atr": atr, "rvol": rvol, "uptrend": price > e20 > e50,
        "breakout": price > max(highs[-21:-1]) and rvol >= 1.5,
        "support": min(lows[-11:-1]),
        "candle_utc": datetime.fromtimestamp(float(rows[-1][0]), timezone.utc).isoformat(),
        "age_minutes": (time.time() - (float(rows[-1][0]) + granularity)) / 60,
    }


def inspect(pair, fee_rate, max_spread):
    hour = indicators(candles(pair, 3600), 3600)
    day = indicators(candles(pair, 86400), 86400)
    if not hour or not day or hour["age_minutes"] > 120 or day["age_minutes"] > 2880:
        return None

    # These are the original five technical requirements. Do not relax them.
    checks = {
        "1h uptrend": hour["uptrend"],
        "1d uptrend": day["uptrend"],
        "1h RSI 50-72": 50 <= hour["rsi"] < 72,
        "1d RSI 48-75": 48 <= day["rsi"] < 75,
        "1h breakout with RVOL >= 1.5": hour["breakout"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if len(failed) > 2:
        return None

    ticker = get(f"/products/{pair}/ticker")
    bid, ask = float(ticker.get("bid") or 0), float(ticker.get("ask") or 0)
    if bid <= 0 or ask < bid:
        return None
    spread = (ask - bid) / ((ask + bid) / 2)
    if spread > max_spread:
        return None

    entry = ask
    stop = min(entry - 1.5 * hour["atr"], hour["support"] - 0.1 * hour["atr"])
    if stop <= 0 or stop >= entry:
        return None
    target = entry + 3 * (entry - stop)
    net_risk = (entry - stop) / entry + fee_rate + fee_rate * stop / entry
    net_reward = (target - entry) / entry - fee_rate - fee_rate * target / entry
    rr = net_reward / net_risk if net_risk > 0 else -1
    if rr < 1.5 or net_reward <= 0:
        return None

    observed = datetime.now(timezone.utc).isoformat()
    if not failed:
        signal_id = hashlib.sha256(f"worker-v1|{pair}|{hour['candle_utc']}".encode()).hexdigest()[:32]
        signal = {
            "ID": signal_id, "Observed UTC": observed, "Pair": pair,
            "Strategy": "1h breakout + daily trend (worker v1)",
            "Signal candle UTC": hour["candle_utc"],
            "Entry": entry, "Stop": stop, "Target": target, "Size GBP": 0,
            "Entry fee rate": fee_rate, "Exit fee rate": fee_rate,
            "Outcome": "OPEN", "Outcome UTC": "", "Net GBP": None, "R multiple": None,
            "Net R:R estimate": round(rr, 3), "Spread %": round(100 * spread, 3),
            "1h RSI": round(hour["rsi"], 1), "1d RSI": round(day["rsi"], 1),
            "1h RVOL": round(hour["rvol"], 2), "Source": "Coinbase Exchange public API",
            "Note": "Paper observation only; no order placed. Outcome checked using completed 1h candles.",
        }
        return "signal", signal

    # Near misses are research observations only, never trade instructions.
    near_id = hashlib.sha256(f"near-v1|{pair}|{hour['candle_utc']}".encode()).hexdigest()[:32]
    near = {
        "ID": near_id, "Type": "NEAR_MISS", "Observed UTC": observed,
        "Pair": pair, "Strategy": "1h breakout + daily trend (research near miss)",
        "Signal candle UTC": hour["candle_utc"],
        "Passed checks": [name for name, passed in checks.items() if passed],
        "Failed checks": failed, "Checks passed": len(checks) - len(failed),
        "Checks total": len(checks), "1h RSI": round(hour["rsi"], 1),
        "1d RSI": round(day["rsi"], 1), "1h RVOL": round(hour["rvol"], 2),
        "Spread %": round(100 * spread, 3), "Entry fee rate": fee_rate,
        "Exit fee rate": fee_rate, "Net R:R estimate": round(rr, 3),
        "Indicative entry": entry, "Indicative stop": stop, "Indicative target": target,
        "Outcome": "NOT_A_TRADE", "Source": "Coinbase Exchange public API",
        "Note": "Research only. Technical entry criteria not met. No order placed.",
    }
    return "near", near


def persist(table, record):
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    if not url.startswith("https://") or "your-project" in url:
        raise RuntimeError("Invalid SUPABASE_URL")
    response = requests.post(
        f"{url}/rest/v1/{table}?on_conflict=id",
        headers={
            "apikey": key, "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Prefer": "resolution=ignore-duplicates,return=minimal",
        },
        json={"id": record["ID"], "payload": record}, timeout=20,
    )
    response.raise_for_status()



def resolve_worker_signals(max_rows=200):
    """Check recent OPEN worker signals; leave older unobservable outcomes OPEN.

    A stop and target inside the same completed bar are treated as stop-first.
    This is paper research, not an executable fill or a validated backtest.
    """
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    if not url.startswith("https://") or "your-project" in url:
        raise RuntimeError("Invalid SUPABASE_URL")
    headers = {"apikey": key, "Authorization": "Bearer " + key,
               "Content-Type": "application/json"}
    base = f"{url}/rest/v1/scanner_signal_journal"
    response = requests.get(base, headers=headers,
                            params={"select": "id,payload", "order": "created_at.desc",
                                    "limit": max_rows}, timeout=20)
    response.raise_for_status()
    resolved = 0
    unavailable = 0
    checked = 0
    for item in response.json():
        p = item.get("payload")
        if not isinstance(p, dict) or p.get("Outcome") != "OPEN":
            continue
        # Never modify dashboard-created observations or manually inserted tests.
        if p.get("Strategy") != "1h breakout + daily trend (worker v1)":
            continue
        checked += 1
        try:
            observed = datetime.fromisoformat(p["Observed UTC"].replace("Z", "+00:00")).timestamp()
            entry, stop, target = (float(p[k]) for k in ("Entry", "Stop", "Target"))
            if not (0 < stop < entry < target):
                continue
            bars = candles(p["Pair"], 3600)
            # Only complete bars whose opening time is after observation.
            future = [b for b in bars if float(b[0]) > observed]
            if not future:
                unavailable += 1
                continue
            for bar in future:
                stop_hit = float(bar[1]) <= stop
                target_hit = float(bar[2]) >= target
                if not (stop_hit or target_hit):
                    continue
                p = p.copy()
                p["Outcome"] = "STOP" if stop_hit else "TARGET"
                p["Outcome UTC"] = datetime.fromtimestamp(
                    float(bar[0]) + 3600, timezone.utc).isoformat()
                p["Exit price assumption"] = stop if stop_hit else target
                p["Outcome method"] = "Completed 1h bar; stop-first if ambiguous; no slippage"
                size = float(p.get("Size GBP") or 0)
                if size > 0:
                    er = float(p.get("Entry fee rate") or 0)
                    xr = float(p.get("Exit fee rate") or 0)
                    exit_price = float(p["Exit price assumption"])
                    net = size * ((exit_price / entry) * (1 - xr) - (1 + er))
                    risk = size * ((1 + er) - (stop / entry) * (1 - xr))
                    p["Net GBP"] = round(net, 4)
                    p["R multiple"] = round(net / risk, 3) if risk > 0 else None
                else:
                    p["Net GBP"] = None
                    p["R multiple"] = None
                updated = requests.patch(
                    base, headers={**headers, "Prefer": "return=representation"},
                    params={"id": "eq." + str(item["id"]),
                            "payload->>Outcome": "eq.OPEN", "select": "id"},
                    json={"payload": p}, timeout=20)
                updated.raise_for_status()
                if updated.json():
                    resolved += 1
                break
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            print(f"Outcome check failed {p.get('Pair')}: {type(exc).__name__}: {exc}", flush=True)
    print(f"Outcome checks: worker_open={checked} resolved={resolved} no_future_bars={unavailable}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Scan without writing")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    limit = max(1, min(60, args.limit))
    if not args.dry_run and not all(
        os.environ.get(k) for k in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY")
    ):
        raise RuntimeError("GitHub Actions secrets SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

    if not args.dry_run:
        resolve_worker_signals()

    products = get("/products")
    pairs = [
        p["id"] for p in products
        if p.get("quote_currency") == "USD"
        and p.get("base_currency") not in STABLE
        and p.get("status") == "online"
        and not any(p.get(flag, False) for flag in
                    ("trading_disabled", "cancel_only", "limit_only", "post_only"))
    ]

    def turnover(pair):
        data = get(f"/products/{pair}/stats")
        return pair, float(data.get("volume") or 0) * float(data.get("last") or 0)

    liquidity = []
    errors = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(turnover, pair) for pair in pairs]
        for future in as_completed(futures):
            try:
                pair, amount = future.result()
                if amount >= 100000:
                    liquidity.append((pair, amount))
            except Exception:
                errors += 1

    chosen = [pair for pair, _ in sorted(liquidity, key=lambda x: x[1], reverse=True)[:limit]]
    print(f"Universe={len(pairs)} liquid={len(liquidity)} selected={len(chosen)} liquidity_errors={errors}", flush=True)
    qualified, near_misses = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(inspect, pair, .005, .005): pair for pair in chosen}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    kind, record = result
                    (qualified if kind == "signal" else near_misses).append(record)
            except Exception as exc:
                errors += 1
                print(f"Market {futures[future]} failed: {type(exc).__name__}", flush=True)

    # Limit database growth: store at most 10 near misses per run, best technical matches first.
    near_misses.sort(key=lambda x: (-x["Checks passed"], -x["Net R:R estimate"], x["Pair"]))
    near_misses = near_misses[:10]
    print(f"Qualified={len(qualified)} near_misses={len(near_misses)} request_errors={errors}", flush=True)
    for signal in qualified:
        print(f"Signal {signal['Pair']} net_RR={signal['Net R:R estimate']}", flush=True)
        if not args.dry_run:
            persist("scanner_signal_journal", signal)
    for near in near_misses:
        print(f"Near miss {near['Pair']} failed={'; '.join(near['Failed checks'])}", flush=True)
        if not args.dry_run:
            persist("scanner_near_misses", near)

    print("DRY RUN complete" if args.dry_run else "Scan and persistence complete", flush=True)
    if not chosen or errors > max(10, len(pairs) // 3):
        sys.exit("Too many market errors or no markets selected")


if __name__ == "__main__":
    main()
