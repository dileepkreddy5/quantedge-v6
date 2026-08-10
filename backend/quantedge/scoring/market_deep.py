"""Market deepening — volatility suite, trading risk, volume/liquidity, short interest.
All real from price/volume history + Polygon short-interest. Missing->None."""
import math
from typing import Optional, Dict, List, Any

def _f(v):
    try: v=float(v); return v if math.isfinite(v) else None
    except: return None

def volatility_suite(closes, spy_closes=None):
    out={}
    if len(closes)<60: return out
    rets=[(closes[i]/closes[i-1]-1) for i in range(1,len(closes))]
    import statistics as st
    hv=st.pstdev(rets)*math.sqrt(252)
    rv=st.pstdev(rets[-21:])*math.sqrt(252) if len(rets)>=21 else None
    out["historical_vol"]=round(hv,4); out["realized_vol_1m"]=round(rv,4) if rv else None
    if len(closes)>=15:
        moves=[abs(closes[i]/closes[i-1]-1) for i in range(len(closes)-14,len(closes))]
        out["atr_pct"]=round(sum(moves)/len(moves),4)
    if len(rets)>=90:
        vs=[st.pstdev(rets[i-21:i])*math.sqrt(252) for i in range(21,len(rets))]
        cur=vs[-1]; below=sum(1 for v in vs if v<cur)
        out["vol_percentile"]=round(100*below/len(vs),1)
        if len(vs)>=42 and vs[-21]>0: out["vol_trend"]=round(vs[-1]/vs[-21]-1,4)
    if spy_closes and len(spy_closes)>=len(rets)+1:
        srets=[(spy_closes[i]/spy_closes[i-1]-1) for i in range(1,len(spy_closes))]
        n=min(len(rets),len(srets)); a=rets[-n:]; b=srets[-n:]
        mb=sum(b)/n; ma=sum(a)/n
        cov=sum((a[i]-ma)*(b[i]-mb) for i in range(n))/n; var=sum((x-mb)**2 for x in b)/n
        if var>0: out["beta"]=round(cov/var,3)
    return out

def trading_risk(closes):
    out={}
    if len(closes)<60: return out
    rets=[(closes[i]/closes[i-1]-1) for i in range(1,len(closes))]
    import statistics as st
    srt=sorted(rets); k=max(1,int(0.05*len(srt)))
    out["var_5pct"]=round(srt[k-1],4); out["cvar_5pct"]=round(sum(srt[:k])/k,4)
    peak=closes[0]; mdd=0
    for p in closes:
        if p>peak: peak=p
        dd=(p-peak)/peak
        if dd<mdd: mdd=dd
    out["max_drawdown"]=round(mdd,4)
    big=sum(1 for r in rets if abs(r)>0.02); out["gap_risk_freq"]=round(big/len(rets),4)
    downs=[r for r in rets if r<0]
    out["downside_dev"]=round(st.pstdev(downs)*math.sqrt(252),4) if len(downs)>2 else None
    return out

def volume_liquidity(closes, volumes):
    out={}
    if len(volumes)<21 or len(closes)!=len(volumes): return out
    adv=sum(volumes[-21:])/21
    out["avg_daily_volume"]=round(adv,0)
    out["rvol"]=round(volumes[-1]/adv,2) if adv>0 else None
    out["dollar_volume"]=round(closes[-1]*volumes[-1],0)
    tp=[closes[i] for i in range(len(closes)-20,len(closes))]
    vv=[volumes[i] for i in range(len(volumes)-20,len(volumes))]
    if sum(vv)>0:
        vwap=sum(tp[i]*vv[i] for i in range(20))/sum(vv)
        out["vwap_distance"]=round((closes[-1]-vwap)/vwap,4)
    import statistics as _st
    obv=0.0; obvs=[]
    for i in range(1,len(closes)):
        sign=1 if closes[i]>closes[i-1] else -1 if closes[i]<closes[i-1] else 0
        obv+=volumes[i]*sign; obvs.append(obv)
    rec=obvs[-63:]
    if len(rec)>=20 and _st.pstdev(rec)>0:
        xs=list(range(len(rec))); mx=sum(xs)/len(xs); my=sum(rec)/len(rec)
        num=sum((xs[i]-mx)*(rec[i]-my) for i in range(len(rec))); den=sum((x-mx)**2 for x in xs)
        if den>0:
            out["obv_slope"]=round((num/den)/(abs(my)+1e-9),6)
    return out

def short_interest_signals(records, avg_vol=None):
    out={}
    if not records: return out
    recs=sorted(records, key=lambda r: r.get("settlement_date",""), reverse=True)
    latest=recs[0]
    si=_f(latest.get("short_interest")); dtc=_f(latest.get("days_to_cover"))
    out["short_interest_shares"]=si
    out["days_to_cover"]=round(dtc,2) if dtc else None
    out["short_interest_date"]=latest.get("settlement_date")
    if len(recs)>=2:
        prev_si=_f(recs[1].get("short_interest"))
        if si and prev_si and prev_si>0: out["short_interest_trend"]=round(si/prev_si-1,4)
    tr=out.get("short_interest_trend",0) or 0
    if dtc and dtc>5 and tr>0.1: out["squeeze_risk"]="elevated"
    elif dtc and dtc>3: out["squeeze_risk"]="moderate"
    else: out["squeeze_risk"]="low"
    return out

# ===== DEEPENING: advanced risk, volatility detail, volume detail =====
def volatility_detail(closes):
    import statistics as st
    out={}
    if len(closes)<60: return out
    rets=[(closes[i]/closes[i-1]-1) for i in range(1,len(closes))]
    downs=[r for r in rets if r<0]; ups=[r for r in rets if r>0]
    out["downside_vol"]=round(st.pstdev(downs)*math.sqrt(252),4) if len(downs)>2 else None
    up_vol=st.pstdev(ups)*math.sqrt(252) if len(ups)>2 else None
    dv=out["downside_vol"]
    out["up_down_vol_ratio"]=round(up_vol/dv,3) if (up_vol and dv) else None
    if len(rets)>=90:
        vs=[st.pstdev(rets[i-21:i])*math.sqrt(252) for i in range(21,len(rets))]
        out["vol_of_vol"]=round(st.pstdev(vs)/st.mean(vs),3) if st.mean(vs)>0 else None
    return out

def advanced_risk(closes):
    out={}
    if len(closes)<60: return out
    peak=closes[0]; dds=[]
    for p in closes:
        if p>peak: peak=p
        dds.append((p-peak)/peak)
    out["ulcer_index"]=round(math.sqrt(sum(d*d for d in dds)/len(dds)),4)
    out["pain_index"]=round(abs(sum(dds)/len(dds)),4)
    rets=[(closes[i]/closes[i-1]-1) for i in range(1,len(closes))]
    total_ret=closes[-1]/closes[0]-1; max_dd=abs(min(dds))
    out["recovery_factor"]=round(total_ret/max_dd,3) if max_dd>0 else None
    srt=sorted(rets); n=len(srt)
    p95=srt[int(0.95*n)]; p5=srt[int(0.05*n)]
    out["tail_ratio"]=round(abs(p95/p5),3) if p5!=0 else None
    return out

def volume_detail(closes, volumes):
    out={}
    if len(volumes)<40 or len(closes)!=len(volumes): return out
    up_v=sum(volumes[i] for i in range(len(closes)-20,len(closes)) if closes[i]>closes[i-1])
    tot_v=sum(volumes[-20:])
    out["up_volume_ratio"]=round(up_v/tot_v,3) if tot_v>0 else None
    r10=sum(volumes[-10:])/10; p30=sum(volumes[-40:-10])/30
    out["volume_trend"]=round(r10/p30-1,3) if p30>0 else None
    out["accumulation_days"]=sum(1 for i in range(len(closes)-20,len(closes)) if closes[i]>closes[i-1] and volumes[i]>volumes[i-1])
    pos_mf=0; neg_mf=0
    for i in range(len(closes)-14,len(closes)):
        mf=closes[i]*volumes[i]
        if closes[i]>closes[i-1]: pos_mf+=mf
        else: neg_mf+=mf
    out["mfi_proxy"]=round(100-100/(1+pos_mf/neg_mf),1) if neg_mf>0 else None
    return out


def momentum_trend_suite(closes, volumes=None, spy_closes=None):
    """Compute the momentum/trend/risk-adjusted/liquidity signals that otherwise
    live only in the peer_stats bucket — so megacaps not in the last scan (whose
    me_factors are empty) still get real, price-derived values instead of blanks.
    All real math on the price series; missing -> None."""
    import math, statistics as st
    out = {}
    n = len(closes)
    if n < 60:
        return out
    rets = [(closes[i]/closes[i-1]-1) for i in range(1, n)]

    # --- momentum ladder (period returns) ---
    def _ret(days):
        if n > days: return round((closes[-1]/closes[-1-days]-1)*100, 2)
        return None
    out["mom_1m"]  = _ret(21)
    out["mom_3m"]  = _ret(63)
    out["mom_6m"]  = _ret(126)
    # classic 12-1 momentum: 12-month return excluding the most recent month
    if n > 273:
        r12 = closes[-22]/closes[-273]-1
        out["mom_12_1"] = round(r12*100, 2)

    # --- moving-average structure ---
    def _sma(k): return sum(closes[-k:])/k if n >= k else None
    sma50, sma200 = _sma(50), _sma(200)
    cur = closes[-1]
    if sma50 is not None:
        out["pct_above_ma50"] = round((cur/sma50-1)*100, 2)
    if sma200 is not None:
        out["pct_above_ma200"] = round((cur/sma200-1)*100, 2)
    # MA alignment: +1 price>50, +1 50>200, +1 price>200 -> 0..1
    if sma50 is not None and sma200 is not None:
        al = (cur > sma50) + (sma50 > sma200) + (cur > sma200)
        out["ma_alignment"] = round(al/3.0, 3)

    # --- Hurst exponent (rescaled-range, trend persistence) ---
    try:
        lags = range(2, 20)
        tau = [st.pstdev([closes[i]-closes[i-lag] for i in range(lag, n)]) for lag in lags]
        if all(t > 0 for t in tau):
            lx = [math.log(l) for l in lags]; ly = [math.log(t) for t in tau]
            mx, my = sum(lx)/len(lx), sum(ly)/len(ly)
            num = sum((lx[i]-mx)*(ly[i]-my) for i in range(len(lx)))
            den = sum((x-mx)**2 for x in lx)
            if den > 0:
                out["hurst"] = round(num/den, 3)
    except Exception:
        pass

    # --- risk-adjusted: match factor_engine scanner exactly ---
    # sharpe_3m = 3-month return (fraction) / (3-month annualised vol)
    if len(rets) >= 63 and n > 63:
        mom_3m_frac = closes[-1]/closes[-64]-1
        vol_3m = st.pstdev(rets[-63:])*math.sqrt(252)
        if vol_3m > 0:
            out["sharpe_3m"] = round(mom_3m_frac/vol_3m, 3)
        # vol-adjusted return: total return over full window / full annualised vol
        ann_vol = st.pstdev(rets)*math.sqrt(252)
        tot = closes[-1]/closes[0]-1
        if ann_vol > 0:
            out["vol_adj_return"] = round(tot/ann_vol, 3)

    # --- Amihud illiquidity — match scanner: (|ret|/(dollar_vol+1)).mean()*1e9 over 60d ---
    if volumes is not None and len(volumes) == n:
        dvol = [closes[i]*volumes[i] for i in range(n)][-60:]
        aret = [abs(r) for r in rets][-60:]
        m = min(len(dvol), len(aret))
        if m > 0 and sum(dvol[-m:])/m > 0:
            il = [aret[-m+j]/(dvol[-m+j]+1) for j in range(m)]
            out["amihud"] = round(sum(il)/len(il)*1e9, 3)
        else:
            out["amihud"] = 100.0
        # volume surge — scanner uses vol20/vol90 (not vol21)
        if n >= 90:
            v20 = sum(volumes[-20:])/20; v90 = sum(volumes[-90:])/90
            out["volume_surge"] = round(v20/v90, 3) if v90 > 0 else 1.0
        # OBV slope — match scanner exactly: cumulative OBV, last 63, polyfit slope
        # divided by (abs(mean)+1e-9). Uses sign of daily return like the scanner.
        obv = 0.0; obvs = []
        for i in range(1, n):
            sign = 1 if closes[i] > closes[i-1] else -1 if closes[i] < closes[i-1] else 0
            obv += volumes[i]*sign
            obvs.append(obv)
        rec = obvs[-63:]
        if len(rec) >= 20:
            mean_abs = abs(sum(rec)/len(rec))
            sd = st.pstdev(rec)
            if sd > 0:
                xs = list(range(len(rec))); mx = sum(xs)/len(xs); my = sum(rec)/len(rec)
                num = sum((xs[i]-mx)*(rec[i]-my) for i in range(len(rec)))
                den = sum((x-mx)**2 for x in xs)
                if den > 0:
                    slope = num/den
                    out["obv_slope_norm"] = round(slope/(mean_abs+1e-9), 6)

    return out
