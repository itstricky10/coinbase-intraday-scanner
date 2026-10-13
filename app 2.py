import io
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st

BASE = 'https://api.exchange.coinbase.com'
HEADERS = {'User-Agent': 'IntradayResearchScanner/3.0', 'Accept': 'application/json'}
DEFAULT = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD', 'DOGE-USD', 'ADA-USD', 'AVAX-USD', 'LINK-USD']

st.set_page_config(page_title='Spot Scanner V3', layout='wide', initial_sidebar_state='collapsed')
st.markdown('''<style>.block-container{padding-top:1rem;padding-bottom:3rem;max-width:1050px}h1{font-size:1.7rem!important}.stButton>button{min-height:44px;border-radius:12px}@media(max-width:650px){.block-container{padding-left:.65rem;padding-right:.65rem}}</style>''', unsafe_allow_html=True)
st.title('⚡ Spot Scanner V3')
st.caption('Coinbase Exchange public data • educational research • spot long setups only • no orders placed')

@st.cache_data(ttl=45, show_spinner=False)
def candles(pair, seconds):
    response = requests.get(f'{BASE}/products/{pair}/candles', params={'granularity': seconds}, headers=HEADERS, timeout=15)
    response.raise_for_status()
    raw = response.json()
    if not isinstance(raw, list) or not raw:
        raise ValueError(f'No candles returned: {raw}')
    frame = pd.DataFrame(raw, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    for field in ['low', 'high', 'open', 'close', 'volume']:
        frame[field] = pd.to_numeric(frame[field], errors='coerce')
    frame = frame.dropna().sort_values('time').drop_duplicates('time').reset_index(drop=True)
    frame['utc'] = pd.to_datetime(frame['time'], unit='s', utc=True)
    # Remove the active candle based on its actual timestamp, not merely its position.
    now = datetime.now(timezone.utc).timestamp()
    return frame.loc[frame['time'] + seconds <= now].copy().reset_index(drop=True)

@st.cache_data(ttl=30, show_spinner=False)
def ticker(pair):
    response = requests.get(f'{BASE}/products/{pair}/ticker', headers=HEADERS, timeout=12)
    response.raise_for_status()
    return response.json()


def indicators(frame):
    d = frame.copy()
    if len(d) < 60:
        raise ValueError('Need at least 60 completed candles')
    close = d['close']
    d['ema20'] = close.ewm(span=20, adjust=False).mean()
    d['ema50'] = close.ewm(span=50, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    ratio = gain / loss.replace(0, np.nan)
    d['rsi'] = 100 - 100 / (1 + ratio)
    d.loc[(loss == 0) & (gain > 0), 'rsi'] = 100
    d.loc[(loss == 0) & (gain == 0), 'rsi'] = 50
    previous_close = close.shift()
    tr = pd.concat([d['high'] - d['low'], (d['high'] - previous_close).abs(), (d['low'] - previous_close).abs()], axis=1).max(axis=1)
    d['atr'] = tr.rolling(14).mean()
    # Crypto trades 24/7: anchor VWAP at 00:00 UTC, not the arbitrary fetch boundary.
    day = d['utc'].dt.floor('D')
    typical = (d['high'] + d['low'] + d['close']) / 3
    numerator = (typical * d['volume']).groupby(day).cumsum()
    denominator = d['volume'].groupby(day).cumsum().replace(0, np.nan)
    d['vwap'] = numerator / denominator
    last = d.iloc[-1]
    previous20 = d.iloc[-21:-1]
    median_volume = float(previous20['volume'].median())
    rvol = float(last['volume'] / median_volume) if median_volume > 0 else 0.0
    prior_high = float(previous20['high'].max())
    prior_low = float(previous20['low'].min())
    breakout = bool(last['close'] > prior_high and rvol >= 1.5)
    sweep = bool(last['low'] < prior_low and last['close'] > prior_low and last['close'] > last['open'] and rvol >= 1.2)
    trend = bool(last['close'] > last['ema20'] > last['ema50'])
    momentum = bool(pd.notna(last['vwap']) and last['close'] > last['vwap'] and 50 < last['rsi'] < 75)
    score = (30 if breakout else 0) + (25 if rvol >= 1.5 else 12 if rvol >= 1.1 else 0) + (20 if sweep else 0) + (15 if trend else 0) + (10 if momentum else 0)
    # A setup requires price confirmation AND trend/momentum context.
    confirmed = (breakout or sweep) and trend and momentum
    signal = 'SETUP' if confirmed and score >= 65 else ('AVOID' if last['close'] < last['ema50'] and last['rsi'] < 45 else 'WAIT')
    atr = float(last['atr'])
    entry = float(last['close'])
    # Indicative, not executable orders. Prefer structural support when it is closer.
    support = float(d['low'].iloc[-11:-1].min())
    stop = min(entry - 1.5 * atr, support - 0.1 * atr)
    target = entry + 2 * (entry - stop)
    return d, dict(last=entry, rvol=rvol, rsi=float(last['rsi']), atr=atr, vwap=float(last['vwap']) if pd.notna(last['vwap']) else np.nan, trend=trend, breakout=breakout, sweep=sweep, score=score, signal=signal, entry=entry, stop=stop, target=target, candle_utc=last['utc'].isoformat())

with st.expander('⚙️ Markets, fees & risk settings'):
    raw = st.text_area('Coinbase Exchange pairs (one per line)', '\n'.join(DEFAULT), height=170)
    pairs = list(dict.fromkeys(x.strip().upper() for x in raw.splitlines() if x.strip()))[:20]
    interval = st.selectbox('Signal timeframe', ['5 minutes', '15 minutes', '1 hour'], index=0)
    seconds = {'5 minutes': 300, '15 minutes': 900, '1 hour': 3600}[interval]
    maker = st.number_input('Maker fee per side (%) — verify your actual tier', 0.0, 5.0, 0.25, 0.05)
    taker = st.number_input('Taker fee per side (%) — verify your actual tier', 0.0, 5.0, 0.50, 0.05)
    entry_type = st.selectbox('Entry order type', ['Taker', 'Maker'])
    exit_type = st.selectbox('Exit order type', ['Taker', 'Maker'])
    capital = st.number_input('Account capital (GBP)', min_value=1.0, value=100.0, step=10.0)
    max_risk = st.number_input('Maximum planned loss per trade (GBP)', min_value=0.1, value=2.0, step=0.5)
    st.caption('Fees are unverified assumptions. GBP-equivalent estimates exclude FX costs and slippage.')

refresh = st.button('🔄 Refresh market scan', type='primary', use_container_width=True)
settings_key = (tuple(pairs), seconds)
if 'scan' not in st.session_state or refresh or st.session_state.get('scan_settings') != settings_key:
    results, errors = [], []
    with st.spinner('Scanning completed Coinbase candles…'):
        for pair in pairs:
            try:
                _, m = indicators(candles(pair, seconds))
                t = ticker(pair)
                bid, ask = float(t.get('bid') or 0), float(t.get('ask') or 0)
                spread = (ask - bid) / ((ask + bid) / 2) * 100 if bid > 0 and ask > 0 else np.nan
                results.append({'Pair': pair, 'Signal': m['signal'], 'Score /100': m['score'], 'Last close': m['last'], 'RVOL': round(m['rvol'], 2), 'RSI': round(m['rsi'], 1), 'Spread %': round(spread, 3), 'VWAP': m['vwap'], 'ATR': m['atr'], 'Breakout': m['breakout'], 'Sweep': m['sweep'], 'Uptrend': m['trend'], 'Indicative entry': m['entry'], 'Indicative stop': m['stop'], 'Indicative target': m['target'], 'Candle UTC': m['candle_utc']})
            except Exception as exc:
                errors.append(f'{pair}: {exc}')
    st.session_state.scan = pd.DataFrame(results).sort_values('Score /100', ascending=False) if results else pd.DataFrame()
    st.session_state.errors = errors
    st.session_state.scan_settings = settings_key
    st.session_state.scanned_at = datetime.now(timezone.utc).isoformat()

if st.session_state.errors:
    st.warning('Some markets failed: ' + ' | '.join(st.session_state.errors))
scan = st.session_state.scan
if scan.empty:
    st.error('No usable data returned. Check pair symbols and Coinbase availability.')
    st.stop()
st.caption(f"Scan fetched: {st.session_state.scanned_at} · {interval} candles · refresh manually")
st.subheader('📊 Ranked opportunities')
for _, row in scan.head(8).iterrows():
    with st.container(border=True):
        left, right = st.columns([3, 1])
        with left:
            st.markdown(f"**{row['Pair']}** · {int(row['Score /100'])}/100")
            st.caption(f"RVOL {row['RVOL']}× · RSI {row['RSI']} · Spread {row['Spread %']}%")
        with right:
            st.markdown({'SETUP': '🟢 **SETUP**', 'WAIT': '🟡 **WAIT**', 'AVOID': '🔴 **AVOID**'}[row['Signal']])
st.caption('SETUP means rules passed, NOT a recommendation to buy. WAIT and AVOID are filters, not sell signals.')
with st.expander('Full sortable data'):
    st.dataframe(scan, use_container_width=True, hide_index=True)

st.subheader('📥 Share an analysis report')
report = scan.copy()
report.insert(0, 'Scan fetched UTC', st.session_state.scanned_at)
report.insert(1, 'Timeframe', interval)
report.insert(2, 'Maker fee assumption %', maker)
report.insert(3, 'Taker fee assumption %', taker)
st.download_button('Download market analysis CSV', data=report.to_csv(index=False).encode('utf-8'), file_name='coinbase_scanner_v3_report.csv', mime='text/csv', use_container_width=True)
st.caption('Upload this CSV in our ChatGPT conversation for a second-pass review. It contains public market data only.')

pair = st.selectbox('Inspect market', scan['Pair'].tolist())
d, m = indicators(candles(pair, seconds))
st.subheader(f'{pair} — completed candles')
st.line_chart(d.set_index('utc')[['close', 'ema20', 'ema50', 'vwap']], height=310)
st.bar_chart(d.set_index('utc')[['volume']], height=140)
st.caption(f"Indicative long levels: entry {m['entry']:.8g} · stop {m['stop']:.8g} · target {m['target']:.8g} (quote currency). Not live executable quotes.")

st.subheader('🎯 Fee-adjusted trade calculator')
entry = st.number_input('Entry price (quote currency)', min_value=0.00000001, value=float(m['entry']), format='%.8f')
stop = st.number_input('Stop price', min_value=0.00000001, value=max(0.00000001, float(m['stop'])), format='%.8f')
target = st.number_input('Target price', min_value=0.00000001, value=float(m['target']), format='%.8f')
entry_rate = (maker if entry_type == 'Maker' else taker) / 100
exit_rate = (maker if exit_type == 'Maker' else taker) / 100
if not (stop < entry < target):
    st.error('Spot long setup requires stop < entry < target.')
else:
    risk_pct = (entry - stop) / entry + entry_rate + exit_rate * stop / entry
    profit_pct = (target - entry) / entry - entry_rate - exit_rate * target / entry
    # Reserve entry fee from capital; risk includes exit fee at the stop.
    notional = min(capital / (1 + entry_rate), max_risk / risk_pct)
    risk, reward = notional * risk_pct, notional * profit_pct
    rr = reward / risk if risk > 0 else 0
    a, b = st.columns(2)
    a.metric('Position notional (GBP equivalent)', f'£{notional:.2f}')
    b.metric('Cash needed incl. entry fee', f'£{notional * (1 + entry_rate):.2f}')
    c, e, f = st.columns(3)
    c.metric('Planned loss incl. fees', f'£{risk:.2f}')
    e.metric('Target gain incl. fees', f'£{reward:.2f}')
    f.metric('Net reward/risk', f'{rr:.2f}:1')
    if profit_pct <= 0:
        st.error('Target does not cover estimated fees.')
    elif rr < 1.5:
        st.warning('Net reward/risk below 1.5:1. Consider rejecting the setup.')
    else:
        st.info('Reward/risk threshold passed; signal and execution still require independent confirmation.')
    st.caption('Assumes full fills at entered prices. Excludes spread beyond selected entry/exit, slippage, FX conversion, tax and minimum order sizes. Stops can execute worse than planned.')

st.subheader('🗒️ Manual trading journal')
template = pd.DataFrame(columns=['entry_time_utc', 'exit_time_utc', 'pair', 'side', 'quantity', 'entry_price', 'exit_price', 'entry_fee_gbp', 'exit_fee_gbp', 'fx_cost_gbp', 'realised_pnl_gbp', 'setup', 'notes'])
st.download_button('Download journal CSV template', template.to_csv(index=False).encode(), file_name='trade_journal.csv', mime='text/csv')
st.info('Public Coinbase Exchange snapshots are not Coinbase Advanced live quotes. Verify market availability, spreads and your own fee tier before any order. No automatic trading.')
