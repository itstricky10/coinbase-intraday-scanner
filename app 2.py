"""Coinbase Spot Scanner V6. Public-data research tool; does not execute trades."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st

BASE = 'https://api.exchange.coinbase.com'
HEADERS = {'User-Agent': 'SpotResearchScanner/6.0', 'Accept': 'application/json'}
st.set_page_config(page_title='Spot Scanner V6 | Trading Desk', page_icon='⚡', layout='wide', initial_sidebar_state='collapsed')
st.markdown('''<style>
.stApp{background:linear-gradient(160deg,#09131e 0%,#0b1522 55%,#101c2c 100%);color:#eaf2fa}
.block-container{max-width:1220px;padding-top:2.2rem;padding-bottom:3rem}
h1,h2,h3{letter-spacing:-.035em} h1{font-size:2rem!important}
[data-testid="stMetric"]{background:#132437;border:1px solid #274158;border-radius:15px;padding:12px}
[data-testid="stMetricLabel"]{color:#a5bfd0}[data-testid="stMetricValue"]{font-size:1.3rem}
[data-testid="stVerticalBlockBorderWrapper"]{border-color:#294156!important;border-radius:16px!important}
.stButton>button,.stDownloadButton>button{border-radius:12px;min-height:45px;font-weight:650}
.stTabs [data-baseweb="tab-list"]{gap:4px;overflow-x:auto}.stTabs [data-baseweb="tab"]{border-radius:10px;padding:8px 12px}
@media(max-width:650px){
.block-container{padding:3.5rem .7rem 2rem!important}
h1{font-size:1.5rem!important;line-height:1.25!important}h2{font-size:1.22rem!important}
[data-testid="stMetric"]{padding:7px 9px!important;border-radius:10px!important;min-height:0!important}
[data-testid="stMetricLabel"]{font-size:.75rem!important}
[data-testid="stMetricValue"]{font-size:1.12rem!important;line-height:1.15!important}
[data-testid="stHorizontalBlock"]{gap:.45rem!important}
.stTabs [data-baseweb="tab"]{padding:7px 8px!important;font-size:.85rem!important}
}
</style>''', unsafe_allow_html=True)
st.title('⚡ TRADING DESK V6')
st.caption('COINBASE SPOT · EARLY + CONFIRMED SETUPS · RESEARCH ONLY')


def fetch(path, params=None):
    for attempt in range(3):
        try:
            response = requests.get(BASE + path, params=params, headers=HEADERS, timeout=15)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(1.3 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(.6 * (attempt + 1))


@st.cache_data(ttl=900, show_spinner=False)
def products():
    data = fetch('/products')
    if not isinstance(data, list):
        raise ValueError('Unexpected product response')
    stable = {'USD', 'GBP', 'USDC', 'USDT', 'DAI', 'EUR', 'PYUSD', 'TUSD', 'FDUSD', 'GUSD', 'PAX'}
    pairs = []
    for p in data:
        base, quote = p.get('base_currency'), p.get('quote_currency')
        if (quote in ('USD', 'GBP') and base not in stable and p.get('id') == f'{base}-{quote}'
            and p.get('status') == 'online' and not p.get('trading_disabled', False)
            and not p.get('cancel_only', False) and not p.get('limit_only', False)
            and not p.get('post_only', False)):
            pairs.append(p['id'])
    return sorted(set(pairs))


@st.cache_data(ttl=600, show_spinner=False)
def stats(pair):
    s = fetch(f'/products/{pair}/stats')
    return {'Pair': pair, 'Turnover': float(s.get('volume') or 0) * float(s.get('last') or 0)}


@st.cache_data(ttl=60, show_spinner=False)
def candles(pair, seconds):
    raw = fetch(f'/products/{pair}/candles', {'granularity': seconds})
    if not isinstance(raw, list) or not raw:
        raise ValueError('No candles')
    d = pd.DataFrame(raw, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    for c in ['time', 'low', 'high', 'open', 'close', 'volume']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna().sort_values('time').drop_duplicates('time').reset_index(drop=True)
    d = d.loc[d.time + seconds <= datetime.now(timezone.utc).timestamp()].copy().reset_index(drop=True)
    d['utc'] = pd.to_datetime(d.time, unit='s', utc=True)
    return d


@st.cache_data(ttl=45, show_spinner=False)
def ticker(pair):
    return fetch(f'/products/{pair}/ticker')


def analyse(d, seconds):
    if len(d) < 60:
        raise ValueError('Not enough completed candles')
    d = d.copy()
    close = d.close
    d['EMA20'] = close.ewm(span=20, adjust=False).mean()
    d['EMA50'] = close.ewm(span=50, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    d['RSI'] = rsi.mask((loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)
    prev = close.shift()
    tr = pd.concat([d.high-d.low, (d.high-prev).abs(), (d.low-prev).abs()], axis=1).max(axis=1)
    d['ATR'] = tr.rolling(14).mean()
    day = d.utc.dt.floor('D')
    typical = (d.high + d.low + d.close) / 3
    d['VWAP'] = (typical * d.volume).groupby(day).cumsum() / d.volume.groupby(day).cumsum().replace(0, np.nan)
    last, prior = d.iloc[-1], d.iloc[-21:-1]
    price = float(last.close)
    median = float(prior.volume.median())
    rvol = float(last.volume / median) if median > 0 else 0.
    breakout = bool(price > prior.high.max() and rvol >= 1.5)
    sweep = bool(last.low < prior.low.min() and price > prior.low.min() and price > last.open and rvol >= 1.2)
    trend = bool(price > last.EMA20 > last.EMA50)
    momentum = bool(pd.notna(last.VWAP) and price > last.VWAP and 50 < last.RSI < 75)
    score = (30 if breakout else 0) + (25 if rvol >= 1.5 else 12 if rvol >= 1.1 else 0) + (20 if sweep else 0) + (15 if trend else 0) + (10 if momentum else 0)
    atr = float(last.ATR)
    support = float(d.low.iloc[-11:-1].min())
    stop = min(price - 1.5 * atr, support - .1 * atr)
    target = price + 2 * (price - stop)
    age = (datetime.now(timezone.utc).timestamp() - (float(last.time) + seconds)) / 60
    if not np.isfinite([price, atr, stop, target]).all() or price <= 0 or atr <= 0 or stop <= 0:
        raise ValueError('Invalid price/volatility data')
    return d, dict(price=price, rvol=rvol, rsi=float(last.RSI), atr=atr, vwap=float(last.VWAP),
                   breakout=breakout, sweep=sweep, trend=trend, momentum=momentum, score=score,
                   entry=price, stop=stop, target=target, age=age, candle=last.utc.isoformat(),
                   bearish=bool(price < last.EMA50 and last.RSI < 45))


with st.expander('⚙️ SCANNER SETTINGS · markets, fees and risk'):
    q1, q2 = st.columns(2)
    with q1:
        quotes = st.multiselect('Quote currencies', ['USD', 'GBP'], default=['USD', 'GBP'])
        scan_limit = st.slider('Markets to scan', 10, 100, 50, step=10)
        min_turnover = st.number_input('Minimum 24h turnover (quote currency)', min_value=0., value=100000., step=50000.)
        max_spread = st.number_input('Maximum spread (%)', min_value=0., max_value=10., value=.5, step=.05)
        use_higher = st.toggle('Include 1h / 4h / daily context', value=True, help='Extra API requests; scans take longer.')
    with q2:
        maker = st.number_input('Maker fee per side (%) — verify account', 0., 5., .25, .05)
        taker = st.number_input('Taker fee per side (%) — verify account', 0., 5., .50, .05)
        entry_type = st.selectbox('Entry fee', ['Taker', 'Maker'])
        exit_type = st.selectbox('Exit fee', ['Taker', 'Maker'])
        capital = st.number_input('Account capital (£)', min_value=1., value=100., step=10.)
        max_risk = st.number_input('Max planned loss (£)', min_value=.1, value=2., step=.5)
    st.caption('USD and GBP turnover are ranked separately; fee values are assumptions. Spread, slippage, FX and tax can change actual results.')

refresh = st.button('🔄  SCAN NOW', type='primary', use_container_width=True)

@st.fragment(run_every='5min')
def live_dashboard():
    key = (tuple(sorted(quotes)), scan_limit, min_turnover, max_spread, use_higher, maker, taker, entry_type, exit_type, capital, max_risk)
    if refresh or 'scan' not in st.session_state or st.session_state.get('key') != key or time.time() - st.session_state.get('scan_epoch', 0) >= 300:
        if not quotes:
            st.error('Select at least one quote currency.')
            st.stop()
        errors, rows = [], []
        try:
            with st.spinner('Discovering Coinbase spot markets…'):
                universe = [p for p in products() if p.rsplit('-', 1)[-1] in quotes]
        except Exception as exc:
            st.error(f'Market discovery failed: {exc}')
            st.stop()
        activity = []
        with st.spinner(f'Checking liquidity across {len(universe)} listed pairs…'):
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(stats, p): p for p in universe}
                for future in as_completed(futures):
                    try:
                        s = future.result()
                        if s['Turnover'] >= min_turnover:
                            activity.append(s)
                    except Exception as exc:
                        errors.append(f"Activity {futures[future]}: {exc}")
        if not activity:
            st.error('No markets passed the liquidity threshold. Lower the threshold or retry.')
            st.stop()
        activity_df = pd.DataFrame(activity)
        chosen = []
        if len(quotes) == 2:
            for quote in quotes:
                chosen.extend(activity_df[activity_df.Pair.str.endswith('-' + quote)].nlargest(scan_limit // 2, 'Turnover').Pair.tolist())
        for pair in activity_df.nlargest(len(activity_df), 'Turnover').Pair:
            if len(chosen) >= scan_limit:
                break
            if pair not in chosen:
                chosen.append(pair)
        er = (maker if entry_type == 'Maker' else taker) / 100
        xr = (maker if exit_type == 'Maker' else taker) / 100

        def higher_context(pair):
            if not use_higher:
                return {'1h trend': None, '4h trend': None, '1d trend': None, 'Higher TF': 'OFF'}
            hourly = candles(pair, 3600)
            _, h1 = analyse(hourly, 3600)
            # Coinbase Exchange does not support native 4h granularity. Aggregate completed 1h bars.
            four = (hourly.set_index('utc').resample('4h', origin='epoch')
                    .agg({'time':'first', 'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'})
                    .dropna(subset=['open', 'high', 'low', 'close']).reset_index())
            # The last group may contain fewer than four hourly candles; only use complete 4h groups.
            counts = hourly.set_index('utc').resample('4h', origin='epoch').size().to_numpy()
            four = four.loc[counts == 4].reset_index(drop=True)
            _, h4 = analyse(four, 14400)
            _, day = analyse(candles(pair, 86400), 86400)
            flags = [h1['trend'], h4['trend'], day['trend']]
            return {'1h trend': flags[0], '4h trend': flags[1], '1d trend': flags[2],
                    'Higher TF': f"1h {'UP' if flags[0] else 'DOWN'} · 4h {'UP' if flags[1] else 'DOWN'} · 1d {'UP' if flags[2] else 'DOWN'}"}

        def inspect(pair):
            _, fast = analyse(candles(pair, 300), 300)
            _, slow = analyse(candles(pair, 900), 900)
            t = ticker(pair)
            bid, ask = float(t.get('bid') or 0), float(t.get('ask') or 0)
            spread = 100 * (ask - bid) / ((ask + bid) / 2) if bid > 0 and ask >= bid else np.nan
            fresh = fast['age'] <= 10 and slow['age'] <= 30
            aligned = slow['trend'] and slow['momentum'] and 45 <= slow['rsi'] < 75
            higher = higher_context(pair)
            triggered = (fast['breakout'] or fast['sweep']) and fast['trend'] and fast['momentum']
            risk_pct = (fast['entry'] - fast['stop']) / fast['entry'] + er + xr * fast['stop'] / fast['entry']
            profit_pct = (fast['target'] - fast['entry']) / fast['entry'] - er - xr * fast['target'] / fast['entry']
            notional = min(capital / (1 + er), max_risk / risk_pct) if risk_pct > 0 else 0
            rr = profit_pct / risk_pct if risk_pct > 0 else np.nan
            eligible = (fresh and aligned and triggered and np.isfinite(spread) and spread <= max_spread
                        and profit_pct > 0 and rr >= 1.5 and fast['rsi'] < 75)
            # Early entries deliberately do NOT require the 15m trend to be fully aligned.
            # They still require a fresh candle, a volume-backed trigger, acceptable spread,
            # a positive net payoff and a controlled stop. They are not buy instructions.
            early = (fresh and not eligible and (fast['breakout'] or fast['sweep'])
                     and fast['rvol'] >= 1.2 and 40 <= fast['rsi'] < 75
                     and np.isfinite(spread) and spread <= max_spread
                     and profit_pct > 0 and np.isfinite(rr) and rr >= 1.2)
            status = ('CONFIRMED' if eligible else 'AGGRESSIVE' if early else 'STALE' if not fresh
                      else 'AVOID' if fast['bearish'] or slow['bearish'] else 'WATCH')
            reasons = []
            if not fresh: reasons.append('Candle data stale')
            if not triggered: reasons.append('No confirmed 5m trigger')
            if not aligned: reasons.append('15m trend/momentum not aligned')
            if not np.isfinite(spread) or spread > max_spread: reasons.append('Spread unavailable/too wide')
            if not np.isfinite(rr) or rr < 1.5 or profit_pct <= 0: reasons.append('Net reward/risk insufficient')
            if fast['rsi'] >= 75: reasons.append('5m RSI overextended')
            if eligible: reasons = ['Confirmed 5m trigger + 15m alignment; verify live execution']
            elif early: reasons = ['Early volume-backed 5m trigger; 15m confirmation NOT required; higher risk']
            if use_higher and higher['Higher TF'] != 'OFF':
                reasons.append(higher['Higher TF'])
            return {'Pair': pair, 'Status': status, **higher, 'Score 5m': fast['score'], '5m RSI': round(fast['rsi'], 1),
                    '15m RSI': round(slow['rsi'], 1), 'RVOL 5m': round(fast['rvol'], 2),
                    '5m uptrend': fast['trend'], '15m uptrend': slow['trend'], '15m aligned': aligned,
                    'Breakout': fast['breakout'], 'Sweep': fast['sweep'], 'Spread %': round(spread, 3),
                    'Net R:R': round(rr, 2), 'Entry': fast['entry'], 'Stop': fast['stop'],
                    'Target': fast['target'], 'Position GBP': round(notional, 2),
                    'Planned loss GBP': round(notional * risk_pct, 2),
                    'Target gain GBP': round(notional * profit_pct, 2),
                    '5m age min': round(fast['age'], 1), '15m age min': round(slow['age'], 1),
                    '5m candle UTC': fast['candle'], '15m candle UTC': slow['candle'],
                    'Why': '; '.join(reasons)}

        progress = st.progress(0, text='Scanning 5m / 15m and higher-timeframe context…')
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(inspect, p): p for p in chosen}
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    rows.append(future.result())
                except Exception as exc:
                    errors.append(f"Analysis {futures[future]}: {exc}")
                progress.progress(i / len(futures), text=f'Analysed {i}/{len(futures)} pairs')
        progress.empty()
        st.session_state.scan = pd.DataFrame(rows)
        st.session_state.errors = errors
        st.session_state.key = key
        st.session_state.scanned_at = datetime.now(timezone.utc).isoformat()
        st.session_state.scan_epoch = time.time()
        st.session_state.universe_count = len(universe)
        st.session_state.liquid_count = len(activity)

    scan = st.session_state.scan
    if scan.empty:
        st.error('No markets returned. Check request errors or try refreshing.')
        st.stop()
    rank = {'CONFIRMED': 0, 'AGGRESSIVE': 1, 'WATCH': 2, 'AVOID': 3, 'STALE': 4}
    scan = scan.assign(_rank=scan.Status.map(rank)).sort_values(['_rank', 'Score 5m', 'RVOL 5m'], ascending=[True, False, False]).drop(columns='_rank').reset_index(drop=True)
    if st.session_state.errors:
        st.warning(f"{len(st.session_state.errors)} market requests failed. Results may be incomplete.")
        with st.expander('View request errors'):
            st.write(st.session_state.errors)
    st.caption(f"Last scan: {st.session_state.scanned_at} · {len(scan)} analysed / {st.session_state.universe_count} discovered · {st.session_state.liquid_count} passed liquidity filter")

    confirmed = int((scan.Status == 'CONFIRMED').sum())
    aggressive = int((scan.Status == 'AGGRESSIVE').sum())
    watch = int((scan.Status == 'WATCH').sum())
    stale = int((scan.Status == 'STALE').sum())
    up = int(scan['15m uptrend'].sum())
    st.markdown('### 🚦 LIVE TRADE RADAR')
    if confirmed:
        st.success(f'🟢 {confirmed} CONFIRMED SCREENING SETUP(S) — verify current quotes and execution costs')
    elif aggressive:
        st.warning(f'🟠 {aggressive} EARLY SETUP(S) — higher risk, 15m confirmation incomplete')
    else:
        st.info('⚪ NO BUY SETUP — no action indicated by current screening rules.')
    a, b, c = st.columns(3)
    a.metric('🟢 Confirmed', confirmed)
    b.metric('🟠 Early', aggressive)
    c.metric('👀 Watch', watch)
    st.caption(f"{len(scan)} analysed · {up}/{len(scan)} in 15m uptrends · {stale} stale · Checked {st.session_state.scanned_at} UTC")
    st.markdown('#### 🎯 BEST OPPORTUNITIES')
    shortlist = scan.loc[scan.Status.isin(['CONFIRMED', 'AGGRESSIVE', 'WATCH'])].head(5)
    if shortlist.empty:
        st.caption('No qualifying or watchlist candidates. Next scan runs while the page remains active.')
    for _, candidate in shortlist.iterrows():
        with st.container(border=True):
            st.markdown(f"**{candidate['Pair']}** · **{candidate.Status}** · score {int(candidate['Score 5m'])}/100")
            st.caption(f"Spread {candidate['Spread %']}% · Net R:R {candidate['Net R:R']}:1 · {candidate['Higher TF']}")
            st.caption(candidate['Why'])
            if candidate.Status in ('CONFIRMED', 'AGGRESSIVE'):
                st.markdown(f"Entry **{candidate.Entry:.8g}** · Stop **{candidate.Stop:.8g}** · Target **{candidate.Target:.8g}**")
                st.caption(f"Position £{candidate['Position GBP']:.2f} · Planned loss £{candidate['Planned loss GBP']:.2f} · Target gain £{candidate['Target gain GBP']:.2f} (assumptions; excludes slippage/FX)")
    st.caption('Automatic rescan every 5 minutes while active. Signals are research prompts, not orders.')

    tab1, tab2, tab3 = st.tabs(['🏆 RANKINGS', '🔎 CHART LAB', '📥 REPORT'])
    with tab1:
        st.subheader('Opportunity leaderboard')
        filter_status = st.segmented_control('Show markets', ['All', 'Confirmed', 'Aggressive', 'Watch', 'Avoid', 'Stale'], default='All')
        visible = scan if filter_status == 'All' else scan.loc[scan.Status == filter_status.upper()]
        if visible.empty:
            st.info('No markets in this category on the last scan.')
        for _, r in visible.head(15).iterrows():
            with st.container(border=True):
                left, right = st.columns([3, 1])
                with left:
                    st.markdown(f"**{r['Pair']}**  ·  {int(r['Score 5m'])}/100")
                    st.caption(f"5m RSI {r['5m RSI']} · 15m RSI {r['15m RSI']} · RVOL {r['RVOL 5m']}×")
                    st.caption(f"Close {r.Entry:.8g} · Spread {r['Spread %']}% · Net R:R {r['Net R:R']}:1 · 15m {'↑' if r['15m uptrend'] else '↓'}")
                with right:
                    st.markdown({'CONFIRMED':'🟢 **CONFIRMED**','AGGRESSIVE':'🟠 **EARLY**','WATCH':'🟡 **WATCH**','AVOID':'🔴 **AVOID**','STALE':'⚪ **STALE**'}[r.Status])
        with st.expander('All markets · sortable detailed table'):
            st.dataframe(scan, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader('Chart lab')
        pair = st.selectbox('Choose a market', scan.Pair.tolist())
        r = scan.loc[scan.Pair == pair].iloc[0]
        chart_window = st.segmented_control('Chart window', ['6h', '12h', '24h', 'All'], default='6h', key='chart_window')
        overlays = st.toggle('Show entry / stop / target lines', value=False)
        st.caption(f"Signal: {r.Status} · 5m score: {r['Score 5m']}/100 · 15m RSI: {r['15m RSI']}")
        with st.container(border=True):
            st.markdown(f"### {pair}  ·  {r.Status}")
            st.write('**Why it passed or failed:** ' + r.Why)
            st.caption(f"Last completed candles: 5m {r['5m candle UTC']} · 15m {r['15m candle UTC']}")
        for tf, secs in [('5-minute chart', 300), ('15-minute confirmation', 900)]:
            try:
                d, _ = analyse(candles(pair, secs), secs)
                st.markdown(f'**{tf}**')
                if chart_window != 'All':
                    hours = {'6h': 6, '12h': 12, '24h': 24}[chart_window]
                    cutoff = d.utc.max() - pd.Timedelta(hours=hours)
                    d = d.loc[d.utc >= cutoff].copy()
                if len(d) < 2:
                    st.info('Not enough candles in the selected chart window.')
                    continue
                # Plotly is bundled with Streamlit's standard installation, but listed explicitly in requirements for portability.
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.77, .23], vertical_spacing=.035)
                fig.add_trace(go.Candlestick(x=d.utc, open=d.open, high=d.high, low=d.low, close=d.close,
                                             name='Candles', increasing_line_color='#19c5aa', decreasing_line_color='#ff6481'), row=1, col=1)
                for name, color in [('EMA20', '#4ba8ff'), ('EMA50', '#ffc66d')]:
                    fig.add_trace(go.Scatter(x=d.utc, y=d[name], mode='lines', name=name, line=dict(color=color, width=1.5)), row=1, col=1)
                fig.add_trace(go.Bar(x=d.utc, y=d.volume, name='Volume', marker_color='#436d85'), row=2, col=1)
                # Levels are derived from the 5m setup; on 15m they remain reference levels only.
                if overlays and np.isfinite([r.Entry, r.Stop, r.Target]).all():
                    for label, level, color in [('Entry', r.Entry, '#54c5ff'), ('Stop', r.Stop, '#ff6481'), ('Target', r.Target, '#3be0ac')]:
                        fig.add_hline(y=float(level), line_color=color, line_dash='dot', line_width=1,
                                      annotation_text=label, annotation_position='top left', row=1, col=1)
                fig.update_layout(height=340, margin=dict(l=4, r=4, t=15, b=5), paper_bgcolor='#101d2b',
                                  plot_bgcolor='#101d2b', font=dict(color='#dce8f2'),
                                  legend=dict(orientation='h', y=1.08, x=0, font=dict(size=10)), xaxis_rangeslider_visible=False)
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(gridcolor='#203447')
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            except Exception as exc:
                st.warning(f'Chart unavailable: {exc}')
        st.subheader('Indicative trade plan · long only')
        st.markdown(f"**Entry:** `{r.Entry:.8g}`　 **Stop:** `{r.Stop:.8g}`　 **Target:** `{r.Target:.8g}`")
        x, y = st.columns(2)
        x.metric('Position (£ equivalent)', f"£{r['Position GBP']:.2f}")
        y.metric('Net reward/risk', f"{r['Net R:R']}:1")
        st.caption(f"Estimated loss **£{r['Planned loss GBP']:.2f}** · Estimated target gain **£{r['Target gain GBP']:.2f}** (after assumed trading fees, before FX/slippage)")
        if r.Status not in ('CONFIRMED', 'AGGRESSIVE'):
            st.warning('Not a confirmed setup. ' + str(r.Why))
        st.caption('Levels are based on completed 5m candles, not current executable quotes. GBP-equivalent budget assumes FX conversion without costs; estimates exclude additional slippage, FX fees, taxes and order minimums. Stops may fill worse.')

    with tab3:
        st.subheader('Optional detailed report')
        report = scan.copy()
        report.insert(0, 'Scan fetched UTC', st.session_state.scanned_at)
        report.insert(1, 'Maker fee assumption %', maker)
        report.insert(2, 'Taker fee assumption %', taker)
        report.insert(3, 'Capital GBP', capital)
        report.insert(4, 'Risk budget GBP', max_risk)
        st.download_button('⬇️ DOWNLOAD V6 MARKET REPORT', report.to_csv(index=False).encode('utf-8'),
                           file_name='coinbase_scanner_v6_report.csv', mime='text/csv', use_container_width=True)
        st.caption('Optional deeper analysis: download and share this report. No account keys or personal data are exported.')
        with st.expander('What do the labels mean?'):
            st.markdown('**CONFIRMED:** a fresh 5m trigger plus 15m confirmation and acceptable costs. **AGGRESSIVE:** a volume-backed early 5m trigger without full 15m confirmation; higher risk. **WATCH:** incomplete confirmation. **AVOID:** bearish conditions. **STALE:** candle data is too old. None of these labels guarantees a trade outcome.')

    st.divider()
    st.caption('Higher timeframes are context, not proof of a trade; 4h bars are aggregated from 1h. Auto-check runs only while this browser session is active; Streamlit may sleep when inactive. Research only. Coinbase Exchange public candles and tickers are snapshots and may differ from Coinbase Advanced prices, listings, fees and execution. No leverage, account access or order placement. Never treat a screening signal as a guaranteed trade.')


live_dashboard()
