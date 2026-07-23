"""Institutional Flow — flow/momentum of institutional activity (vs Ownership static holdings).
Money-flow indicators (MFI, Chaikin Money Flow, ADL), block-trade proxy, 13G filing flow,
insider velocity, smart-money footprint. From OHLCV bars + Form 4 + 13G."""
import math, statistics as st
from datetime import datetime, timezone
def _f(v):
    try: v=float(v); return v if math.isfinite(v) else None
    except: return None

def compute_iflow_features(bars, insider=None, ownership=None):
    f={}
    insider=insider or {}; ownership=ownership or {}
    if not bars or len(bars)<30: return {"available":False}
    closes=[_f(b.get("c")) for b in bars]; highs=[_f(b.get("h")) for b in bars]
    lows=[_f(b.get("l")) for b in bars]; vols=[_f(b.get("v")) for b in bars]
    ntxns=[_f(b.get("n")) for b in bars]

    if len(closes)>=21:
        mfvs=[]
        for i in range(len(closes)):
            h,l,c,v=highs[i],lows[i],closes[i],vols[i]
            if None in (h,l,c,v) or h==l: mfvs.append(0.0); continue
            mfm=((c-l)-(h-c))/(h-l)
            mfvs.append(mfm*v)
        vsum=sum(v for v in vols[-21:] if v)
        if vsum>0: f["chaikin_money_flow"]=sum(mfvs[-21:])/vsum
        adl=[0]
        for mfv in mfvs: adl.append(adl[-1]+mfv)
        if len(adl)>=20:
            n=20; xs=list(range(n)); seg=adl[-20:]; mx=st.mean(xs); my=st.mean(seg)
            d=sum((x-mx)**2 for x in xs)
            sl=sum((xs[i]-mx)*(seg[i]-my) for i in range(n))/d if d>0 else 0
            mv=st.mean([abs(v) for v in vols[-20:] if v]) or 1
            f["adl_slope"]=sl/mv

    if len(closes)>=15:
        tps=[(highs[i]+lows[i]+closes[i])/3 if None not in (highs[i],lows[i],closes[i]) else None for i in range(len(closes))]
        pos_mf=0.0; neg_mf=0.0
        for i in range(len(tps)-14,len(tps)):
            if i<1 or tps[i] is None or tps[i-1] is None or vols[i] is None: continue
            rmf=tps[i]*vols[i]
            if tps[i]>tps[i-1]: pos_mf+=rmf
            elif tps[i]<tps[i-1]: neg_mf+=rmf
        if pos_mf+neg_mf>0: f["money_flow_index"]=pos_mf/(pos_mf+neg_mf)

    if len(vols)>=40 and len(ntxns)>=40:
        sizes=[vols[i]/ntxns[i] for i in range(len(vols)) if vols[i] and ntxns[i] and ntxns[i]>0]
        if len(sizes)>=40:
            base=st.mean(sizes[-40:-5])
            f["avg_trade_size_trend"]=st.mean(sizes[-5:])/base-1 if base>0 else None
            f["block_trade_frequency"]=sum(1 for s in sizes[-10:] if s>1.5*base)/10 if base>0 else None
            # institutional_footprint repeated avg_trade_size_trend exactly.

    if len(closes)>=40 and len(vols)>=40:
        upv=sum(vols[i] for i in range(len(closes)-20,len(closes)) if i>0 and closes[i] and closes[i-1] and closes[i]>closes[i-1] and vols[i])
        dnv=sum(vols[i] for i in range(len(closes)-20,len(closes)) if i>0 and closes[i] and closes[i-1] and closes[i]<closes[i-1] and vols[i])
        if upv+dnv>0: f["accumulation_20d"]=upv/(upv+dnv)
        recent_dv=sum(vols[i]*closes[i] for i in range(len(closes)-10,len(closes)) if vols[i] and closes[i])
        prior_dv=sum(vols[i]*closes[i] for i in range(len(closes)-20,len(closes)-10) if vols[i] and closes[i])
        if prior_dv>0: f["dollar_flow_momentum"]=recent_dv/prior_dv-1

    if ownership.get("available"):
        holders=ownership.get("holders",[]); now=datetime.now(timezone.utc); recent=0
        for h in holders:
            try:
                dt=datetime.fromisoformat(h.get("date","").replace("Z","+00:00")) if h.get("date") else None
                if dt and (now-dt).days<=180: recent+=1
            except: pass
        f["recent_13g_filings"]=float(recent)
        f["institutional_holder_count"]=float(ownership.get("major_holders") or 0)

    if insider.get("available"):
        f["insider_flow_velocity"]=float(insider.get("filings_parsed") or 0)
        f["insider_net_flow"]=insider.get("buy_value_ratio")
        f["insider_cluster_flow"]=insider.get("cluster_buying")

    return {k:v for k,v in f.items() if v is not None}
