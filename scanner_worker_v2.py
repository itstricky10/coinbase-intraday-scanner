"""Unattended Coinbase spot research: observe signals and resolve paper outcomes.
No trading credentials, orders, or account access. Uses Supabase server-side secrets.
"""
import argparse
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

BASE = 'https://api.exchange.coinbase.com'
HTTP = requests.Session()
HTTP.headers.update({'User-Agent': 'TradingDeskResearchWorker/2.0', 'Accept': 'application/json'})
STABLE = {'USD','GBP','USDC','USDT','DAI','EUR','PYUSD','TUSD','FDUSD','GUSD','PAX'}
COLS = ['ID','Observed UTC','Pair','Strategy','Signal candle UTC','Entry','Stop','Target',
        'Size GBP','Entry fee rate','Exit fee rate','Outcome','Outcome UTC','Net GBP','R multiple']


def request(method, url, **kwargs):
    for n in range(4):
        try:
            r = HTTP.request(method, url, timeout=20, **kwargs)
            if r.status_code in (429,500,502,503,504):
                if n < 3:
                    time.sleep(2**n)
                    continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if n == 3:
                raise
            time.sleep(2**n)


def get(path, params=None):
    return request('GET', BASE + path, params=params).json()


def db(method, params=None, body=None, prefer=None):
    url = os.environ['SUPABASE_URL'].rstrip('/')
    key = os.environ['SUPABASE_SERVICE_ROLE_KEY']
    if not url.startswith('https://') or 'your-project' in url:
        raise ValueError('Configure the actual SUPABASE_URL')
    headers = {'apikey': key, 'Authorization': 'Bearer ' + key, 'Content-Type':'application/json'}
    if prefer:
        headers['Prefer'] = prefer
    r = request(method, url + '/rest/v1/scanner_signal_journal', params=params, json=body, headers=headers)
    return r.json() if r.content else []


def ema(v, span):
    out = v[0]
    alpha = 2/(span+1)
    for x in v[1:]:
        out = alpha*x+(1-alpha)*out
    return out


def rsi(v, period=14):
    if len(v) < period+2:
        return None
    changes = [v[i]-v[i-1] for i in range(1,len(v))]
    gain = sum(max(x,0) for x in changes[:period])/period
    loss = sum(max(-x,0) for x in changes[:period])/period
    for x in changes[period:]:
        gain = (gain*(period-1)+max(x,0))/period
        loss = (loss*(period-1)+max(-x,0))/period
    return 100 if loss == 0 and gain > 0 else 50 if loss == 0 else 100-100/(1+gain/loss)


def candles(pair, granularity):
    data = get(f'/products/{pair}/candles', {'granularity':granularity})
    now = time.time()
    return sorted([v for v in data if len(v)>=6 and float(v[0])+granularity<=now],key=lambda x:x[0])


def indicators(rows):
    if len(rows)<60:
        return None
    close = [float(x[4]) for x in rows]
    high = [float(x[2]) for x in rows]
    low = [float(x[1]) for x in rows]
    vol = [float(x[5]) for x in rows]
    price = close[-1]
    e20,e50 = ema(close,20),ema(close,50)
    strength = rsi(close)
    tr = [max(high[i]-low[i],abs(high[i]-close[i-1]),abs(low[i]-close[i-1])) for i in range(1,len(rows))]
    prior_vol = sorted(vol[-21:-1]); median = (prior_vol[9]+prior_vol[10])/2
    rv = vol[-1]/median if median>0 else 0
    return {'price':price,'rsi':strength,'atr':sum(tr[-14:])/14,
            'uptrend':price>e20>e50,'breakout':price>max(high[-21:-1]) and rv>=1.5,
            'support':min(low[-11:-1]),'rvol':rv,
            'candle':datetime.fromtimestamp(rows[-1][0],timezone.utc).isoformat(),
            'age':(time.time()-(float(rows[-1][0])+float(rows[-1][0]-rows[-2][0])))/60}


def inspect(pair, fee, spread_max):
    hour = indicators(candles(pair,3600)); day = indicators(candles(pair,86400))
    if not hour or not day or hour['age']>120 or day['age']>2880:
        return None
    if not (hour['uptrend'] and day['uptrend'] and 50<=hour['rsi']<72
            and 48<=day['rsi']<75 and hour['breakout']):
        return None
    t = get(f'/products/{pair}/ticker')
    bid,ask = float(t.get('bid') or 0),float(t.get('ask') or 0)
    if bid<=0 or ask<bid or (ask-bid)/((ask+bid)/2)>spread_max:
        return None
    entry=ask
    stop=min(entry-1.5*hour['atr'],hour['support']-.1*hour['atr'])
    if stop<=0 or stop>=entry:
        return None
    target=entry+3*(entry-stop)
    risk=(entry-stop)/entry+fee+fee*stop/entry
    reward=(target-entry)/entry-fee-fee*target/entry
    if risk<=0 or reward<=0 or reward/risk<1.5:
        return None
    signal_id=hashlib.sha256(f"worker-v2|{pair}|{hour['candle']}".encode()).hexdigest()[:32]
    return {'ID':signal_id,'Observed UTC':datetime.now(timezone.utc).isoformat(),
            'Pair':pair,'Strategy':'WORKER_1H_BREAKOUT','Signal candle UTC':hour['candle'],
            'Entry':entry,'Stop':stop,'Target':target,'Size GBP':0.,
            'Entry fee rate':fee,'Exit fee rate':fee,'Outcome':'OPEN','Outcome UTC':'',
            'Net GBP':None,'R multiple':None,'Net R:R estimate':round(reward/risk,3),
            'Source':'Coinbase Exchange public API',
            'Note':'Unfilled paper observation; £0 position until explicit sizing. Not an executed trade.'}


def save_new(payload):
    # Ignore duplicates so a rerun cannot overwrite a resolved observation.
    db('POST',params={'on_conflict':'id'},body={'id':payload['ID'],'payload':payload},
       prefer='resolution=ignore-duplicates,return=minimal')


def resolve_open(max_rows=200):
    """Resolve only candles that start AFTER observation. Unknown history remains OPEN."""
    rows=db('GET',params={'select':'id,payload','order':'created_at.desc','limit':max_rows})
    open_rows=[x for x in rows if isinstance(x.get('payload'),dict)
               and x['payload'].get('Outcome')=='OPEN'
               and x['payload'].get('Pair') and x['payload'].get('Strategy')]
    resolved=0; unavailable=0
    for item in open_rows:
        p=item['payload'].copy()
        # Skip manual test rows, and never interpret unrecognized strategies.
        strategy=str(p.get('Strategy',''))
        if strategy not in ('WORKER_1H_BREAKOUT','1h breakout + daily trend (worker v1)',
                            'CONFIRMED','AGGRESSIVE','SWING'):
            continue
        secs=3600 if strategy in ('WORKER_1H_BREAKOUT','1h breakout + daily trend (worker v1)','SWING') else 300
        try:
            observed=datetime.fromisoformat(str(p['Observed UTC']).replace('Z','+00:00')).timestamp()
            entry,stop,target=[float(p[k]) for k in ('Entry','Stop','Target')]
            if not 0<stop<entry<target:
                continue
            bars=candles(p['Pair'],secs)
            # Coinbase's candle endpoint is limited to a recent window; do not invent older outcomes.
            future=[x for x in bars if float(x[0])>observed]
            if not future:
                unavailable+=1
                continue
            for bar in future:
                stop_hit=float(bar[1])<=stop
                target_hit=float(bar[2])>=target
                if not (stop_hit or target_hit):
                    continue
                exit_price=stop if stop_hit else target  # pessimistic if both within one candle
                p['Outcome']='STOP' if stop_hit else 'TARGET'
                p['Outcome UTC']=datetime.fromtimestamp(float(bar[0])+secs,timezone.utc).isoformat()
                p['Exit price assumption']=exit_price
                p['Outcome method']='Completed candle high/low; stop-first if ambiguous; no slippage modeled'
                size=float(p.get('Size GBP') or 0)
                if size>0:
                    er=float(p.get('Entry fee rate') or 0)
                    xr=float(p.get('Exit fee rate') or 0)
                    net=size*((exit_price/entry)*(1-xr)-(1+er))
                    risk=size*((1+er)-(stop/entry)*(1-xr))
                    p['Net GBP']=round(net,4)
                    p['R multiple']=round(net/risk,3) if risk>0 else None
                else:
                    # The original worker never sized a position: do not fabricate £ P/L.
                    p['Net GBP']=None
                    p['R multiple']=None
                # Conditional update: do not overwrite another writer's resolution.
                result=db('PATCH',params={'id':'eq.'+str(item['id']),
                                          'payload->>Outcome':'eq.OPEN','select':'id'},
                          body={'payload':p},prefer='return=representation')
                if result:
                    resolved+=1
                break
        except Exception as exc:
            print(f"Outcome check {p.get('Pair')}: {type(exc).__name__}: {exc}",flush=True)
    print(f'Outcome checks: open={len(open_rows)} resolved={resolved} no_new_candles={unavailable}',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--limit',type=int,default=30)
    args=parser.parse_args()
    if not args.dry_run and not all(os.getenv(x) for x in ('SUPABASE_URL','SUPABASE_SERVICE_ROLE_KEY')):
        raise RuntimeError('Missing GitHub Actions Supabase secrets')
    if not args.dry_run:
        resolve_open()
    products=get('/products')
    pairs=[p['id'] for p in products if p.get('quote_currency')=='USD'
           and p.get('base_currency') not in STABLE and p.get('status')=='online'
           and not any(p.get(flag,False) for flag in ('trading_disabled','cancel_only','limit_only','post_only'))]
    def turnover(pair):
        s=get(f'/products/{pair}/stats')
        return pair,float(s.get('volume') or 0)*float(s.get('last') or 0)
    liquid=[];errors=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(turnover,p) for p in pairs]):
            try:
                pair,amount=f.result()
                if amount>=100000:
                    liquid.append((pair,amount))
            except Exception:
                errors+=1
    chosen=[p for p,_ in sorted(liquid,key=lambda x:x[1],reverse=True)[:max(1,min(args.limit,60))]]
    print(f'Universe={len(pairs)} liquid={len(liquid)} selected={len(chosen)} liquidity_errors={errors}',flush=True)
    found=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(inspect,p,.005,.005):p for p in chosen}
        for f in as_completed(futures):
            try:
                result=f.result()
                if result:
                    found.append(result)
            except Exception as exc:
                errors+=1
                print(f'Market {futures[f]} failed: {type(exc).__name__}',flush=True)
    print(f'Qualified={len(found)} request_errors={errors}',flush=True)
    for p in found:
        print(f"Signal {p['Pair']} net_RR={p['Net R:R estimate']}",flush=True)
        if not args.dry_run:
            save_new(p)
    print('DRY RUN complete' if args.dry_run else 'Scan and persistence complete',flush=True)
    if not chosen or errors>max(10,len(pairs)//3):
        raise RuntimeError('Too many market errors or no markets selected')

if __name__=='__main__':
    main()
