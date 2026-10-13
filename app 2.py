import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st

BASE='https://api.exchange.coinbase.com'
HEADERS={'User-Agent':'IntradayResearchScanner/1.0','Accept':'application/json'}
DEFAULT=['BTC-USD','ETH-USD','XRP-USD','SOL-USD','DOGE-USD','ADA-USD','AVAX-USD','LINK-USD']

st.set_page_config(page_title='Spot Scanner Mobile',layout='wide',initial_sidebar_state='collapsed')
st.markdown("""<meta name="viewport" content="width=device-width, initial-scale=1"/><style>
.block-container {padding-top:1rem;padding-bottom:4rem;max-width:1050px}
h1 {font-size:1.7rem!important;letter-spacing:-.04em}
[data-testid="stMetric"] {background:#14202b;border:1px solid #273849;border-radius:14px;padding:12px}
[data-testid="stMetricValue"] {font-size:1.25rem}
.stButton>button {min-height:44px;border-radius:12px}
@media(max-width:650px){.block-container{padding-left:.7rem;padding-right:.7rem}h2{font-size:1.25rem!important}
[data-testid="stHorizontalBlock"]{gap:.4rem}.stDataFrame{font-size:12px}}
</style>""",unsafe_allow_html=True)
st.title('⚡ Spot Scanner')
st.caption('Coinbase Exchange public market data • research only • no order execution • prices may be delayed')

@st.cache_data(ttl=45,show_spinner=False)
def candles(pair,seconds):
    r=requests.get(f'{BASE}/products/{pair}/candles',params={'granularity':seconds},headers=HEADERS,timeout=12)
    r.raise_for_status()
    data=r.json()
    if not isinstance(data,list) or not data: raise ValueError(f'No candles for {pair}: {data}')
    df=pd.DataFrame(data,columns=['time','low','high','open','close','volume'])
    for c in ['low','high','open','close','volume']:df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.sort_values('time').drop_duplicates('time').reset_index(drop=True)
    df['utc']=pd.to_datetime(df.time,unit='s',utc=True)
    return df

@st.cache_data(ttl=30,show_spinner=False)
def ticker(pair):
    r=requests.get(f'{BASE}/products/{pair}/ticker',headers=HEADERS,timeout=10)
    r.raise_for_status()
    return r.json()

def metrics(df):
    # Drop in-progress candle to avoid misleading breakout/volume signals.
    d=df.iloc[:-1].copy()
    if len(d)<40: raise ValueError('Insufficient completed candles')
    close=d.close
    d['ema20']=close.ewm(span=20,adjust=False).mean()
    d['ema50']=close.ewm(span=50,adjust=False).mean()
    typical=(d.high+d.low+d.close)/3
    d['vwap']=(typical*d.volume).cumsum()/d.volume.cumsum().replace(0,np.nan)
    tr=pd.concat([(d.high-d.low),(d.high-d.close.shift()).abs(),(d.low-d.close.shift()).abs()],axis=1).max(axis=1)
    d['atr']=tr.rolling(14).mean()
    delta=close.diff();up=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean();down=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    d['rsi']=100-100/(1+up/down.replace(0,np.nan))
    last=d.iloc[-1];prev=d.iloc[-2];window=d.iloc[-21:-1]
    rv=last.volume/max(1e-12,float(d.volume.iloc[-21:-1].median()))
    breakout=bool(last.close>window.high.max() and rv>=1.5)
    sweep=bool(last.low<window.low.min() and last.close>window.low.min() and last.close>last.open and rv>=1.2)
    trend=bool(last.close>last.ema20>last.ema50)
    momentum=bool(last.close>last.vwap and last.rsi>50 and last.rsi<78)
    score=(30 if breakout else 0)+(25 if rv>=1.5 else 12 if rv>=1.1 else 0)+(20 if sweep else 0)+(15 if trend else 0)+(10 if momentum else 0)
    return d,{'last':float(last.close),'rvol':rv,'rsi':float(last.rsi),'atr':float(last.atr),'vwap':float(last.vwap),'trend':trend,'breakout':breakout,'sweep':sweep,'score':score,'candle_utc':str(last.utc)}

with st.expander('⚙️ Markets, fees & risk settings', expanded=False):
    st.header('Scanner settings')
    raw=st.text_area('Coinbase Exchange pairs (one per line)', '\n'.join(DEFAULT),height=180)
    pairs=list(dict.fromkeys(x.strip().upper() for x in raw.splitlines() if x.strip()))[:20]
    interval=st.selectbox('Signal timeframe',['5 minutes','15 minutes','1 hour'],index=0)
    secs={'5 minutes':300,'15 minutes':900,'1 hour':3600}[interval]
    maker=st.number_input('Maker fee per side (%)',0.0,5.0,0.60,0.05)
    taker=st.number_input('Taker fee per side (%)',0.0,5.0,1.20,0.05)
    entry_fee=st.selectbox('Entry order type',['Taker','Maker'])
    exit_fee=st.selectbox('Exit order type',['Taker','Maker'])
    capital=st.number_input('Account capital (GBP)',min_value=1.0,value=100.0,step=10.0)
    max_risk=st.number_input('Maximum risk per trade (GBP)',min_value=0.1,value=2.0,step=0.5)
    st.caption('Fees are illustrative defaults, NOT your verified Coinbase tier. Enter your actual fee rates. USD/USDC quote markets add GBP conversion considerations.')
    run=st.button('Refresh market scan',type='primary')

if 'scan' not in st.session_state or run:
    results=[]; errors=[]
    with st.spinner('Fetching public candles…'):
        for pair in pairs:
            try:
                d,m=metrics(candles(pair,secs)); t=ticker(pair)
                bid=float(t.get('bid',0) or 0);ask=float(t.get('ask',0) or 0)
                spread=(ask-bid)/((ask+bid)/2)*100 if bid>0 and ask>0 else np.nan
                results.append({'Pair':pair,'Last completed close':m['last'],'Score /100':m['score'],'Relative volume':round(m['rvol'],2),'RSI 14':round(m['rsi'],1),'VWAP':m['vwap'],'ATR 14':m['atr'],'Spread %':round(spread,3),'Breakout':m['breakout'],'Sweep':m['sweep'],'Uptrend':m['trend'],'Candle UTC':m['candle_utc']})
            except Exception as e:errors.append(f'{pair}: {e}')
    st.session_state.scan=pd.DataFrame(results).sort_values('Score /100',ascending=False) if results else pd.DataFrame()
    st.session_state.errors=errors

if st.session_state.errors: st.warning('Some pairs could not be loaded: '+' | '.join(st.session_state.errors))
if st.session_state.scan.empty: st.error('No usable data. Check network, exchange availability and pair symbols.');st.stop()
st.subheader('📊 Opportunity ranking')
for _,r in st.session_state.scan.head(8).iterrows():
    with st.container(border=True):
        left,right=st.columns([3,1])
        with left:
            st.markdown(f"**{r['Pair']}** · Score {int(r['Score /100'])}/100")
            st.caption(f"RVOL {r['Relative volume']}× · RSI {r['RSI 14']} · Spread {r['Spread %']}%")
        with right:
            st.markdown('🟢 **Breakout**' if r['Breakout'] else '🟡 **Sweep**' if r['Sweep'] else '⚪ Watch')
with st.expander('Full sortable data table'):
    st.dataframe(st.session_state.scan,use_container_width=True,hide_index=True)
st.caption('Scores are heuristic filters, not predictive probabilities. Breakout: close above previous 20-bar high with ≥1.5× median volume. Sweep: undercut and reclaim previous 20-bar low with ≥1.2× volume. Intraday VWAP here starts at fetched history boundary, not a true exchange-session VWAP.')

pair=st.selectbox('Inspect market',st.session_state.scan.Pair.tolist())
d,m=metrics(candles(pair,secs))
st.subheader(f'{pair} — completed candles')
st.line_chart(d.set_index('utc')[['close','ema20','ema50','vwap']],height=320)
st.bar_chart(d.set_index('utc')[['volume']],height=150)

st.subheader('🎯 Fee-adjusted trade calculator')
entry=st.number_input('Entry price (quote currency)',min_value=0.00000001,value=float(m['last']),format='%.8f')
stop=st.number_input('Stop price',min_value=0.00000001,value=float(m['last'])*0.975,format='%.8f')
target=st.number_input('Target price',min_value=0.00000001,value=float(m['last'])*1.05,format='%.8f')
entry_rate=(maker if entry_fee=='Maker' else taker)/100
exit_rate=(maker if exit_fee=='Maker' else taker)/100
# Risk per unit includes entry fee and stop-side exit fee; assumes full fill and no slippage.
risk_pct=(entry-stop)/entry+entry_rate+exit_rate*stop/entry
profit_pct=(target-entry)/entry-entry_rate-exit_rate*target/entry
if not(stop<entry<target):st.error('For a spot long: stop < entry < target.')
elif risk_pct<=0:st.error('Invalid risk assumptions')
else:
    notional=min(capital,max_risk/risk_pct)
    risk=notional*risk_pct
    reward=notional*profit_pct
    rr=reward/risk
    a,b,c,dcol=st.columns(4)
    a.metric('Position notional (GBP equivalent)',f'£{notional:.2f}')
    b.metric('Planned loss incl. fees',f'£{risk:.2f}')
    c.metric('Target gain incl. fees',f'£{reward:.2f}')
    dcol.metric('Net reward / risk',f'{rr:.2f}:1')
    if profit_pct<=0:st.error('Target does not cover estimated trading fees.')
    elif rr<1.5:st.warning('Net reward/risk below 1.5:1; consider rejecting this setup.')
    else:st.success('Calculator threshold passed; this does NOT validate the chart signal or execution.')
    st.caption('Notional is a GBP-equivalent risk budget; convert at the actual GBP/quote rate before ordering. Excludes slippage, FX conversion, minimum orders and tax. Stop orders may fill worse than the stop price.')

st.subheader('Manual trading journal')
st.caption('Download the template, log each actual fill and retain your records locally. This app does not submit trades.')
template=pd.DataFrame(columns=['entry_time_utc','exit_time_utc','pair','side','quantity','entry_price','exit_price','entry_fee_gbp','exit_fee_gbp','fx_cost_gbp','realised_pnl_gbp','setup','notes'])
st.download_button('Download journal CSV template',template.to_csv(index=False).encode(),file_name='trade_journal.csv',mime='text/csv')
st.info('Data source: Coinbase Exchange public endpoints. Coinbase Advanced availability, pair listings, fees and pricing may differ. Public REST requests are snapshots, not streaming quotes. Do not trade solely on this dashboard.')
