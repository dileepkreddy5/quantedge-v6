"""Alt-Data Intelligence — alternative signals from news flow, volume microstructure,
price-action anomalies, attention proxies, insider velocity, liquidity/flow, peer-relative flow.
Honest: real options-flow/short-interest/satellite/card data unavailable on our tier (needs_source).
All computed signals derive from Polygon news + OHLCV bars (v/vw/n) + Form 4."""
import math, statistics as st
from datetime import datetime, timezone, timedelta

def _f(v):
    try: v=float(v); return v if math.isfinite(v) else None
    except: return None

def compute_altdata_features(bars, news, insider=None, peer_momentum=None):
    """bars: list of {v,vw,o,c,h,l,t,n} daily. news: list of Polygon news items. insider: Form4 agg."""
    f={}
    insider=insider or {}
    if not bars or len(bars)<30: return {"available":False}
    closes=[_f(b.get("c")) for b in bars if _f(b.get("c"))]
    vols=[_f(b.get("v")) for b in bars if _f(b.get("v"))]
    vwaps=[_f(b.get("vw")) for b in bars]
    ntxns=[_f(b.get("n")) for b in bars if _f(b.get("n"))]
    highs=[_f(b.get("h")) for b in bars]; lows=[_f(b.get("l")) for b in bars]
    opens=[_f(b.get("o")) for b in bars]

    # ===== VOLUME MICROSTRUCTURE =====
    if len(vols)>=40:
        recent_vol=st.mean(vols[-5:]); base_vol=st.mean(vols[-40:-5])
        f["unusual_volume"]=recent_vol/base_vol-1 if base_vol>0 else None  # volume surge
        f["volume_trend"]=st.mean(vols[-10:])/st.mean(vols[-40:-10])-1 if st.mean(vols[-40:-10])>0 else None
        # up-day vs down-day volume (accumulation)
        upv=sum(vols[i] for i in range(1,len(vols)) if i<len(closes) and closes[i]>closes[i-1])
        dnv=sum(vols[i] for i in range(1,len(vols)) if i<len(closes) and closes[i]<closes[i-1])
        if upv+dnv>0: f["accumulation_ratio"]=upv/(upv+dnv)  # >0.5 = accumulation
    # On-Balance-Volume slope
    if len(closes)>=30 and len(vols)>=30:
        obv=[0]
        for i in range(1,min(len(closes),len(vols))):
            obv.append(obv[-1]+(vols[i] if closes[i]>closes[i-1] else -vols[i] if closes[i]<closes[i-1] else 0))
        if len(obv)>=20:
            n=len(obv[-20:]); xs=list(range(n)); mx=st.mean(xs); my=st.mean(obv[-20:])
            d=sum((x-mx)**2 for x in xs)
            sl=sum((xs[i]-mx)*(obv[-20:][i]-my) for i in range(n))/d if d>0 else 0
            mv=st.mean([abs(v) for v in vols[-20:]])
            f["obv_slope"]=sl/mv if mv>0 else None  # normalized OBV trend

    # ===== TRANSACTION-COUNT (retail attention proxy) =====
    if len(ntxns)>=40:
        f["txn_count_surge"]=st.mean(ntxns[-5:])/st.mean(ntxns[-40:-5])-1 if st.mean(ntxns[-40:-5])>0 else None
        # avg trade size trend (institutional vs retail: rising size = institutional)
        if len(vols)>=40:
            sizes=[vols[i]/ntxns[i] for i in range(len(ntxns)) if ntxns[i] and ntxns[i]>0 and i<len(vols)]
            if len(sizes)>=40:
                f["avg_trade_size_trend"]=st.mean(sizes[-5:])/st.mean(sizes[-40:-5])-1 if st.mean(sizes[-40:-5])>0 else None

    # ===== PRICE-ACTION ANOMALIES =====
    if len(closes)>=30:
        rets=[closes[i]/closes[i-1]-1 for i in range(1,len(closes))]
        vol_ret=st.pstdev(rets[-20:]) if len(rets)>=20 else None
        # gap frequency (overnight gaps)
        if len(opens)>=20 and len(closes)>=20:
            gaps=[abs(opens[i]/closes[i-1]-1) for i in range(1,min(len(opens),len(closes))) if closes[i-1] and closes[i-1]>0]
            if gaps: f["gap_frequency"]=st.mean(gaps[-20:])
        # intraday range expansion (volatility regime)
        if len(highs)>=20 and len(lows)>=20:
            ranges=[(highs[i]-lows[i])/closes[i] for i in range(len(closes)) if closes[i] and closes[i]>0 and i<len(highs) and i<len(lows) and highs[i] and lows[i]]
            if len(ranges)>=20:
                f["range_expansion"]=st.mean(ranges[-5:])/st.mean(ranges[-20:-5])-1 if st.mean(ranges[-20:-5])>0 else None
        # volume-price divergence (price up but volume down = weak)
        if len(vols)>=10 and len(closes)>=10:
            price_chg=closes[-1]/closes[-10]-1; vol_chg=st.mean(vols[-5:])/st.mean(vols[-10:-5])-1 if st.mean(vols[-10:-5])>0 else 0
            f["volume_price_divergence"]=1.0 if (price_chg>0 and vol_chg<-0.1) else 0.0  # bearish divergence flag
        # volatility-as-attention
        if vol_ret is not None: f["volatility_attention"]=vol_ret

    # ===== NEWS FLOW =====
    if news:
        now=datetime.now(timezone.utc)
        def age_days(n):
            try: return (now-datetime.fromisoformat(n.get("published_utc","").replace("Z","+00:00"))).days
            except: return 999
        ages=[age_days(n) for n in news]
        last7=sum(1 for a in ages if a<=7); prior=sum(1 for a in ages if 7<a<=30)
        f["news_volume_7d"]=float(last7)
        f["news_velocity"]=(last7/(prior/3.0)-1) if prior>0 else (1.0 if last7>3 else 0.0)  # 7d vs weekly-avg of prior 3wk
        # sentiment dispersion from insights
        sents=[]
        for n in news[:40]:
            ins=n.get("insights",[])
            for i in ins:
                s=i.get("sentiment")
                if s=="positive": sents.append(1)
                elif s=="negative": sents.append(-1)
                elif s=="neutral": sents.append(0)
        if sents:
            f["news_sentiment_mean"]=st.mean(sents)
            f["news_sentiment_dispersion"]=st.pstdev(sents) if len(sents)>1 else 0.0  # high = controversial
            f["news_coverage_breadth"]=float(len(sents))

    # ===== INSIDER/FILING VELOCITY =====
    if insider.get("available"):
        f["insider_filing_velocity"]=float(insider.get("filings_parsed") or 0)
        f["insider_buy_signal"]=insider.get("buy_value_ratio")

    # ===== LIQUIDITY & FLOW =====
    if len(vols)>=20 and len(closes)>=20:
        dollar_vol=[vols[i]*closes[i] for i in range(min(len(vols),len(closes)))]
        if len(dollar_vol)>=20:
            f["dollar_volume_trend"]=st.mean(dollar_vol[-5:])/st.mean(dollar_vol[-20:-5])-1 if st.mean(dollar_vol[-20:-5])>0 else None
        # Amihud illiquidity: |return| / dollar volume
        rets=[abs(closes[i]/closes[i-1]-1) for i in range(1,len(closes))]
        amihud=[rets[i]/dollar_vol[i+1]*1e9 for i in range(min(len(rets),len(dollar_vol)-1)) if dollar_vol[i+1]>0]
        if amihud: f["amihud_illiquidity"]=st.mean(amihud[-20:]) if len(amihud)>=20 else st.mean(amihud)

    # ===== PEER-RELATIVE FLOW =====
    if peer_momentum is not None:
        f["relative_flow_vs_peers"]=peer_momentum

    # ===== HONEST NEEDS_SOURCE (unavailable on tier) =====
    # options_flow, short_interest, satellite, credit_card -> marked in catalog as needs_source

    return {k:v for k,v in f.items() if v is not None}

if __name__=="__main__":
    import random; random.seed(5)
    bars=[]; p=100
    for i in range(61):
        p*=(1+random.gauss(0.001,0.015)); v=random.uniform(2e7,4e7); n=random.uniform(2e5,4e5)
        bars.append({"c":p,"o":p*0.998,"h":p*1.01,"l":p*0.99,"v":v,"vw":p,"n":n,"t":i})
    news=[{"published_utc":f"2026-07-{19-min(18,i//3):02d}T10:00:00Z","insights":[{"sentiment":random.choice(["positive","negative","neutral"])}]} for i in range(30)]
    ins={"available":True,"filings_parsed":12,"buy_value_ratio":0.08}
    f=compute_altdata_features(bars,news,ins,peer_momentum=0.03)
    print(f"{sum(1 for k in f if not k.startswith('_'))} signals")
    for k,v in sorted(f.items()): print(f"  {k:28s} {round(v,4) if isinstance(v,float) else v}")
