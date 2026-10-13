"""Coinbase spot-market research scanner. No authenticated endpoints or order placement."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st

BASE = 'https://api.exchange.coinbase.com'
HEADERS = {'User-Agent': 'SpotResearchScanner/4.0', 'Accept': 'application/json'}

st.set_page_config(page_title='Spot Scanner V4', layout='wide', initial_sidebar_state='collapsed')
st.markdown('''<style>.block-container{padding-top:1rem;padding-bottom:3rem;max-width:1050px}h1{font-size:1.7rem!important}@media(max-width:650px){.block-container{padding-left:.65rem;padding-right:.65rem}}</style>''', unsafe_allow_html=True)
st.title('⚡ Spot Scanner V4 — Broad Market')
st.caption('Coinbase Exchange public endpoints · research only · no trade execution · exchange listings can differ from Coinbase Advanced')

def fetch(path, params=None):
    for attempt in range(3):
        try:
            r = requests.get(BASE + path, params=params, headers=HEADERS, timeout=14)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(0.6 * (attempt + 1))

@st.cache_data(ttl=900, show_spinner=False)
def products():
    raw = fetch('/products')
    if not isinstance(raw, list):
        raise ValueError('Unexpected Coinbase products response')
    output = []
    for p in raw:
        pid = str(p.get('id', ''))
        quote = p.get('quote_currency')
        base = p.get('base_currency')
        if (quote in ('USD', 'GBP') and pid == f'{base}-{quote}'
                and p.get('status', 'online') == 'online'
                and not p.get('trading_disabled', False)
                and not p.get('cancel_only', False)
                and not p.get('limit_only', False)
                and base not in ('USD','GBP','USDC','USDT','DAI','EUR','PYUSD','TUSD','FDUSD','GUSD','PAX')):
            output.append(pid)
    return sorted(set(output))

@st.cache_data(ttl=600, show_spinner=False)
def stats(pair):
    x = fetch(f'/products/{pair}/stats')
    volume, last = float(x.get('volume', 0) or 0), float(x.get('last', 0) or 0)
    return {'Pair': pair, '24h turnover (quote)': volume * last, '24h volume': volume}

@st.cache_data(ttl=60, show_spinner=False)
def candles(pair, seconds):
    raw = fetch(f'/products/{pair}/candles', {'granularity': seconds})
    if not isinstance(raw, list) or not raw:
        raise ValueError('No candle data')
    d = pd.DataFrame(raw, columns=['time','low','high','open','close','volume'])
    for c in ('time','low','high','open','close','volume'):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna().sort_values('time').drop_duplicates('time')
    now = datetime.now(timezone.utc).timestamp()
    d = d.loc[d.time + seconds <= now].copy().reset_index(drop=True)
    d['utc'] = pd.to_datetime(d.time, unit='s', utc=True)
    return d

@st.cache_data(ttl=45, show_spinner=False)
def ticker(pair):
    return fetch(f'/products/{pair}/ticker')

def analyse(d):
    if len(d) < 60:
        raise ValueError('Fewer than 60 completed candles')
    close = d.close
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)
    previous = close.shift()
    tr = pd.concat([d.high-d.low, (d.high-previous).abs(), (d.low-previous).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    day = d.utc.dt.floor('D')
    typical = (d.high+d.low+d.close)/3
    vwap = (typical*d.volume).groupby(day).cumsum() / d.volume.groupby(day).cumsum().replace(0, np.nan)
    last, prior = d.iloc[-1], d.iloc[-21:-1]
    price = float(last.close)
    rv = float(last.volume / prior.volume.median()) if prior.volume.median() > 0 else 0.0
    high, low = float(prior.high.max()), float(prior.low.min())
    breakout = bool(price > high and rv >= 1.5)
    sweep = bool(last.low < low and price > low and price > last.open and rv >= 1.2)
    trend = bool(price > ema20.iloc[-1] > ema50.iloc[-1])
    momentum = bool(pd.notna(vwap.iloc[-1]) and price > vwap.iloc[-1] and 50 < rsi.iloc[-1] < 75)
    score = (30 if breakout else 0) + (25 if rv >= 1.5 else 12 if rv >= 1.1 else 0) + (20 if sweep else 0) + (15 if trend else 0) + (10 if momentum else 0)
    signal = 'SETUP' if (breakout or sweep) and trend and momentum and score >= 65 else ('AVOID' if price < ema50.iloc[-1] and rsi.iloc[-1] < 45 else 'WAIT')
    a = float(atr.iloc[-1]); support = float(d.low.iloc[-11:-1].min())
    stop = min(price - 1.5*a, support - 0.1*a)
    target = price + 2*(price-stop)
    if not np.isfinite([price, a, stop, target]).all() or stop <= 0 or a <= 0:
        raise ValueError('Invalid volatility or trade levels')
    return {'Signal': signal, 'Score': score, 'Last close': price, 'RVOL': round(rv, 2), 'RSI': round(float(rsi.iloc[-1]), 1),
            'VWAP': float(vwap.iloc[-1]) if pd.notna(vwap.iloc[-1]) else np.nan,
            'ATR': a, 'Breakout': breakout, 'Sweep': sweep, 'Uptrend': trend,
            'Entry': price, 'Stop': stop, 'Target': target, 'Candle UTC': last.utc.isoformat()}

with st.expander('⚙️ Scan settings', expanded=False):
    quotes = st.multiselect('Quote markets', ['USD','GBP'], default=['USD','GBP'])
    timeframe = st.selectbox('Signal timeframe', ['5 minutes','15 minutes','1 hour'], index=0)
    seconds = {'5 minutes':300,'15 minutes':900,'1 hour':3600}[timeframe]
    scan_limit = st.slider('Top liquid markets to analyse', 10, 100, 50, step=10)
    min_turnover = st.number_input('Minimum 24h turnover in quote currency', min_value=0.0, value=100000.0, step=50000.0)
    max_spread = st.number_input('Maximum spread (%)', min_value=0.0, max_value=10.0, value=0.50, step=0.05)
    maker = st.number_input('Maker fee per side (%) — verify with Coinbase', 0.0, 5.0, 0.25, 0.05)
    taker = st.number_input('Taker fee per side (%) — verify with Coinbase', 0.0, 5.0, 0.50, 0.05)
    entry_type = st.selectbox('Entry fee type', ['Taker','Maker'])
    exit_type = st.selectbox('Exit fee type', ['Taker','Maker'])
    capital = st.number_input('Capital (GBP)', min_value=1.0, value=100.0, step=10.0)
    max_risk = st.number_input('Maximum planned loss (GBP)', min_value=0.1, value=2.0, step=0.5)
    st.caption('USD and GBP turnover figures are NOT directly comparable; the scanner ranks within each quote market before combining candidates. Fee defaults are provisional.')

refresh = st.button('🔄 Refresh broad market scan', type='primary', use_container_width=True)
key = (tuple(sorted(quotes)), seconds, scan_limit, min_turnover, max_spread, maker, taker, entry_type, exit_type)

if refresh or 'scan' not in st.session_state or st.session_state.get('settings_key') != key:
    errors, rows = [], []
    with st.spinner('Discovering available spot pairs…'):
        try:
            universe = [p for p in products() if p.rsplit('-', 1)[-1] in quotes]
        except Exception as exc:
            st.error(f'Could not discover markets: {exc}')
            st.stop()
    st.caption(f'Discovered {len(universe)} eligible spot pairs. Checking 24-hour activity; this can take a little while on the first scan.')
    # Fetch activity across the entire discovered universe, not an alphabetically truncated subset.
    activity = []
    with st.spinner('Ranking markets by 24-hour turnover…'):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(stats, p): p for p in universe}
            for future in as_completed(futures):
                try:
                    s = future.result()
                    if s['24h turnover (quote)'] >= min_turnover:
                        activity.append(s)
                except Exception as exc:
                    errors.append(f"Activity {futures[future]}: {exc}")
    if not activity:
        st.error('No markets passed the turnover filter. Lower the minimum or try again.')
        st.stop()
    activity_df = pd.DataFrame(activity)
    # Allocate half of slots to each selected quote currency, then fill from remaining turnover leaders.
    chosen = []
    if len(quotes) == 2:
        for q in quotes:
            candidates = activity_df[activity_df.Pair.str.endswith('-'+q)].nlargest(scan_limit//2, '24h turnover (quote)')
            chosen.extend(candidates.Pair.tolist())
    else:
        chosen = activity_df.nlargest(scan_limit, '24h turnover (quote)').Pair.tolist()
    # Fill unused slots (quote turnovers are not FX-normalised; only used for overflow).
    for p in activity_df.sort_values('24h turnover (quote)', ascending=False).Pair:
        if len(chosen) >= scan_limit:
            break
        if p not in chosen:
            chosen.append(p)
    entry_rate = (maker if entry_type == 'Maker' else taker)/100
    exit_rate = (maker if exit_type == 'Maker' else taker)/100
    progress = st.progress(0, text='Analysing completed candles…')
    def inspect(pair):
        result = analyse(candles(pair, seconds))
        t = ticker(pair)
        bid, ask = float(t.get('bid') or 0), float(t.get('ask') or 0)
        spread = (ask-bid)/((ask+bid)/2)*100 if bid>0 and ask>0 and ask>=bid else np.nan
        result['Pair'] = pair
        result['Spread %'] = round(spread, 3) if np.isfinite(spread) else np.nan
        risk_pct = (result['Entry']-result['Stop'])/result['Entry'] + entry_rate + exit_rate*result['Stop']/result['Entry']
        profit_pct = (result['Target']-result['Entry'])/result['Entry'] - entry_rate - exit_rate*result['Target']/result['Entry']
        notional = min(capital/(1+entry_rate), max_risk/risk_pct) if risk_pct > 0 else 0
        result['Position GBP equivalent'] = round(notional, 2)
        result['Planned loss GBP'] = round(notional*risk_pct, 2)
        result['Target profit GBP'] = round(notional*profit_pct, 2)
        result['Net R:R'] = round(profit_pct/risk_pct, 2) if risk_pct > 0 else np.nan
        result['Trade filter'] = ('REVIEW' if result['Signal']=='SETUP' and np.isfinite(spread)
                                  and spread <= max_spread and profit_pct > 0 and result['Net R:R'] >= 1.5 else 'NO TRADE')
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(inspect, p): p for p in chosen}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append(f"Candles {futures[future]}: {exc}")
            progress.progress(i/len(futures), text=f'Analysed {i}/{len(futures)} markets')
    progress.empty()
    st.session_state.scan = pd.DataFrame(rows).sort_values(['Trade filter','Score','RVOL'], ascending=[False,False,False]) if rows else pd.DataFrame()
    st.session_state.errors = errors
    st.session_state.settings_key = key
    st.session_state.scanned_at = datetime.now(timezone.utc).isoformat()
    st.session_state.discovered = len(universe)
    st.session_state.activity_count = len(activity)

scan = st.session_state.scan
if scan.empty:
    st.error('No usable markets. Try refreshing or reducing the filters.')
    st.stop()
if st.session_state.errors:
    st.warning(f'{len(st.session_state.errors)} requests failed. Results may be incomplete.')
    with st.expander('Show request errors'):
        st.write(st.session_state.errors)
st.caption(f"Fetched {st.session_state.scanned_at} · {timeframe} · {len(scan)} analysed / {st.session_state.discovered} discovered · {st.session_state.activity_count} above volume filter")

setup_count = int((scan.Signal == 'SETUP').sum())
review_count = int((scan['Trade filter'] == 'REVIEW').sum())
up_count = int(scan.Uptrend.sum())
a,b,c,d = st.columns(4)
a.metric('Analysed', len(scan))
b.metric('Uptrends', up_count)
c.metric('Technical setups', setup_count)
d.metric('Review candidates', review_count)
st.caption('Breadth is based on analysed liquid markets, not the entire crypto market. REVIEW is a screening result, not an instruction to trade.')

st.subheader('📊 Ranked opportunities')
for _, r in scan.sort_values(['Trade filter','Score','RVOL'], ascending=[False,False,False]).head(12).iterrows():
    with st.container(border=True):
        left,right = st.columns([3,1])
        with left:
            st.markdown(f"**{r['Pair']}** · score {int(r['Score'])}/100")
            st.caption(f"RVOL {r['RVOL']}× · RSI {r['RSI']} · spread {r['Spread %']}% · net R:R {r['Net R:R']}:1")
        with right:
            st.markdown('🟢 **REVIEW**' if r['Trade filter']=='REVIEW' else '🟡 **WAIT**' if r['Signal']=='WAIT' else '🔴 **NO TRADE**')
with st.expander('Full market table', expanded=False):
    st.dataframe(scan, use_container_width=True, hide_index=True)

st.subheader('📥 Export analysis')
report = scan.copy()
report.insert(0, 'Scan fetched UTC', st.session_state.scanned_at)
report.insert(1, 'Timeframe', timeframe)
report.insert(2, 'Maker fee assumption %', maker)
report.insert(3, 'Taker fee assumption %', taker)
report.insert(4, 'Account capital GBP', capital)
report.insert(5, 'Risk limit GBP', max_risk)
st.download_button('Download V4 market analysis CSV', report.to_csv(index=False).encode('utf-8'),
                   file_name='coinbase_scanner_v4_report.csv', mime='text/csv', use_container_width=True)

pair = st.selectbox('Inspect market', scan.Pair.tolist())
try:
    detail = candles(pair, seconds).copy()
    detail['EMA20'] = detail.close.ewm(span=20, adjust=False).mean()
    detail['EMA50'] = detail.close.ewm(span=50, adjust=False).mean()
    st.subheader(f'{pair} — completed candles')
    st.line_chart(detail.set_index('utc')[['close','EMA20','EMA50']], height=290)
    st.bar_chart(detail.set_index('utc')[['volume']], height=130)
    item = scan.loc[scan.Pair==pair].iloc[0]
    st.write(f"**Indicative long levels:** entry {item['Entry']:.8g} · stop {item['Stop']:.8g} · target {item['Target']:.8g} (quote currency)")
    st.write(f"**At current fee assumptions:** notional £{item['Position GBP equivalent']:.2f} · planned loss £{item['Planned loss GBP']:.2f} · target profit £{item['Target profit GBP']:.2f} · net R:R {item['Net R:R']}:1")
except Exception as exc:
    st.warning(f'Could not render chart: {exc}')

st.info('No trades are executed. The report uses historical candles and public ticker snapshots, not executable Coinbase Advanced quotes. Fee tiers, conversion charges, order minimums, slippage and spread may change the result. A stop order does not guarantee the planned exit price. USD and GBP figures use GBP-equivalent position budgets, not an actual FX conversion.')
