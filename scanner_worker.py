"""Unattended Coinbase public-market research scanner. No orders or account access.

Writes qualified observations into public.scanner_signal_journal (id text primary key,
payload jsonb). A separate dashboard must read these observations to display them.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

BASE = "https://api.exchange.coinbase.com"
HEADERS = {"User-Agent": "TradingDeskResearchWorker/1.0", "Accept": "application/json"}
SESSION = requests.Session()
STABLE = {"USD", "GBP", "USDC", "USDT", "DAI", "EUR", "PYUSD", "TUSD", "FDUSD", "GUSD", "PAX"}


def get(path, params=None):
    for attempt in range(4):
        try:
            r = SESSION.get(BASE + path, params=params, headers=HEADERS, timeout=20)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
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
    changes = [values[i] - values[i-1] for i in range(1, len(values))]
    avg_gain = sum(max(v, 0) for v in changes[:period]) / period
    avg_loss = sum(max(-v, 0) for v in changes[:period]) / period
    for change in changes[period:]:
        avg_gain = (avg_gain * (period - 1) + max(change, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-change, 0)) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100 - 100 / (1 + avg_gain / avg_loss)


def candles(pair, granularity):
    raw = get(f"/products/{pair}/candles", {"granularity": granularity})
    now = time.time()
    rows = sorted((v for v in raw if len(v) >= 6 and float(v[0]) + granularity <= now), key=lambda v: v[0])
    return rows


def indicators(rows):
    if len(rows) < 60:
        return None
    closes = [float(x[4]) for x in rows]
    volumes = [float(x[5]) for x in rows]
    highs = [float(x[2]) for x in rows]
    lows = [float(x[1]) for x in rows]
    price = closes[-1]
    e20, e50 = ema(closes, 20), ema(closes, 50)
    strength = rsi(closes)
    if not price > 0 or strength is None:
        return None
    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])) for i in range(1, len(rows))]
    atr = sum(trs[-14:]) / 14
    previous_vol = sorted(volumes[-21:-1])
    median = (previous_vol[9] + previous_vol[10]) / 2
    relative_volume = volumes[-1] / median if median > 0 else 0
    return {"price": price, "ema20": e20, "ema50": e50, "rsi": strength,
            "atr": atr, "rvol": relative_volume, "uptrend": price > e20 > e50,
            "breakout": price > max(highs[-21:-1]) and relative_volume >= 1.5,
            "support": min(lows[-11:-1]), "candle_utc": datetime.fromtimestamp(rows[-1][0], timezone.utc).isoformat(),
            "age_minutes": (time.time() - (float(rows[-1][0]) + (rows[-1][0] - rows[-2][0]))) / 60}


def inspect(pair, fee_rate, max_spread):
    hour = indicators(candles(pair, 3600))
    day = indicators(candles(pair, 86400))
    if not hour or not day or hour["age_minutes"] > 120 or day["age_minutes"] > 2880:
        return None
    if not (hour["uptrend"] and day["uptrend"] and 50 <= hour["rsi"] < 72 and 48 <= day["rsi"] < 75 and hour["breakout"]):
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
    # Indicative 3R gross target. Does not assert executable liquidity at target.
    target = entry + 3 * (entry - stop)
    net_risk = (entry - stop) / entry + fee_rate + fee_rate * stop / entry
    net_reward = (target - entry) / entry - fee_rate - fee_rate * target / entry
    rr = net_reward / net_risk if net_risk > 0 else -1
    if rr < 1.5 or net_reward <= 0:
        return None
    signal_id = hashlib.sha256(f"worker-v1|{pair}|{hour['candle_utc']}".encode()).hexdigest()[:32]
    return {"ID": signal_id, "Observed UTC": datetime.now(timezone.utc).isoformat(), "Pair": pair,
            "Strategy": "1h breakout + daily trend (worker v1)", "Signal candle UTC": hour["candle_utc"],
            "Entry": entry, "Stop": stop, "Target": target, "Size GBP": 0,
            "Entry fee rate": fee_rate, "Exit fee rate": fee_rate,
            "Outcome": "OPEN", "Outcome UTC": "", "Net GBP": "", "R multiple": "",
            "Net R:R estimate": round(rr, 3), "Spread %": round(100*spread, 3),
            "1h RSI": round(hour["rsi"], 1), "1d RSI": round(day["rsi"], 1),
            "1h RVOL": round(hour["rvol"], 2), "Source": "Coinbase Exchange public API",
            "Note": "Paper observation only; no order placed. Outcome tracking not yet implemented."}


def persist(signal):
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    if not url.startswith("https://"):
        raise RuntimeError("Invalid SUPABASE_URL")
    r = requests.post(url + "/rest/v1/scanner_signal_journal?on_conflict=id",
                      headers={"apikey": key, "Authorization": "Bearer " + key,
                               "Content-Type": "application/json", "Prefer": "resolution=ignore-duplicates,return=minimal"},
                      json={"id": signal["ID"], "payload": signal}, timeout=20)
    r.raise_for_status()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Scan without writing")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    limit = max(1, min(60, args.limit))
    if not args.dry_run and not all(os.environ.get(k) for k in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY")):
        raise RuntimeError("GitHub Actions secrets SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
    products = get("/products")
    pairs = [p["id"] for p in products if p.get("quote_currency") == "USD" and
             p.get("base_currency") not in STABLE and p.get("status") == "online" and
             not any(p.get(flag, False) for flag in ("trading_disabled", "cancel_only", "limit_only", "post_only"))]
    def turnover(pair):
        data = get(f"/products/{pair}/stats")
        return pair, float(data.get("volume") or 0) * float(data.get("last") or 0)
    liquidity = []
    errors = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(turnover, p) for p in pairs]
        for future in as_completed(futures):
            try:
                pair, amount = future.result()
                if amount >= 100000:
                    liquidity.append((pair, amount))
            except Exception:
                errors += 1
    chosen = [p for p, _ in sorted(liquidity, key=lambda x: x[1], reverse=True)[:limit]]
    print(f"Universe={len(pairs)} liquid={len(liquidity)} selected={len(chosen)} liquidity_errors={errors}", flush=True)
    found = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(inspect, pair, .005, .005): pair for pair in chosen}
        for future in as_completed(futures):
            try:
                signal = future.result()
                if signal:
                    found.append(signal)
            except Exception as exc:
                errors += 1
                print(f"Market {futures[future]} failed: {type(exc).__name__}")
    print(f"Qualified={len(found)} request_errors={errors}")
    for signal in found:
        print(f"Signal {signal['Pair']} {signal['Strategy']} net_RR={signal['Net R:R estimate']}")
        if not args.dry_run:
            persist(signal)
    print("DRY RUN complete" if args.dry_run else "Scan and persistence complete")
    if not chosen or errors > max(10, len(pairs) // 3):
        sys.exit("Too many market errors or no markets selected")


if __name__ == "__main__":
    main()
