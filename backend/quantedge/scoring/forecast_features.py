"""Forecast Intelligence — deep forward-looking signals WITHOUT analyst estimates.
Professional forward-analysis: earnings/revenue trajectory & quality, operating leverage,
ROIIC/intrinsic compounding, Rule of 40, cash-flow forecast, momentum quality, mean-reversion,
leading indicators, 2yr-stacked growth (base-effect normalized), forecast confidence.
Model-based projections from historical patterns, not analyst consensus."""
import math, statistics as st
def _f(v):
    try: v=float(v); return v if math.isfinite(v) else None
    except: return None
def _slope(ys):
    n=len(ys)
    if n<3: return None
    xs=list(range(n)); mx=st.mean(xs); my=st.mean(ys)
    d=sum((x-mx)**2 for x in xs)
    return sum((xs[i]-mx)*(ys[i]-my) for i in range(n))/d if d>0 else None
def _cagr_slope(ys):
    """normalized slope as fraction of mean level"""
    s=_slope(ys); m=st.mean(ys) if ys else None
    return (s/abs(m)) if (s is not None and m and m!=0) else None

def compute_forecast_features(merged, closes, fin_features):
    f={}
    if not merged or len(merged)<6: return {"available":False}
    def S(k,n=12): return [_f(x.get(k)) for x in merged[-n:] if _f(x.get(k)) is not None]
    def qoq(vals):
        return [vals[i]/vals[i-1]-1 for i in range(1,len(vals)) if vals[i-1] and vals[i-1]>0]

    rev=S("revenue",12); ni=S("net_income",12); oi=S("operating_income",12)
    gp=S("gross_profit",12); ocf=S("operating_cash_flow",12); eps=S("eps_diluted",12)

    # ========== EARNINGS TRAJECTORY ==========
    if len(ni)>=6:
        g=qoq(ni)
        if len(g)>=4:
            f["earnings_accel"]=st.mean(g[-2:])-st.mean(g[:2])  # 2nd derivative
            f["earnings_growth_recent"]=st.mean(g[-2:])
            f["earnings_growth_stability"]=1.0-min(1.0,st.pstdev(g)/(abs(st.mean(g))+0.02)) if len(g)>1 else None
        f["earnings_trend_slope"]=_cagr_slope(ni[-6:])
        f["earnings_positivity"]=sum(1 for x in ni[-8:] if x>0)/len(ni[-8:])
    if len(eps)>=6:
        eg=qoq(eps)
        if len(eg)>=4:
            f["eps_growth_recent"]=st.mean(eg[-2:])
            f["eps_accel"]=st.mean(eg[-2:])-st.mean(eg[:2])
        f["eps_trend_slope"]=_cagr_slope(eps[-6:])

    # ========== REVENUE TRAJECTORY & 2YR-STACKED ==========
    if len(rev)>=6:
        rg=qoq(rev)
        if len(rg)>=4:
            f["revenue_growth_persistence"]=1.0-min(1.0,st.pstdev(rg)/(abs(st.mean(rg))+0.02))
            f["revenue_accel"]=st.mean(rg[-2:])-st.mean(rg[:2])
            f["revenue_growth_recent"]=st.mean(rg[-2:])
        f["growth_consistency"]=sum(1 for x in rg if x>0)/len(rg) if rg else None
        f["revenue_trend_slope"]=_cagr_slope(rev[-8:]) if len(rev)>=8 else None
    # YoY + 2-year stacked (base-effect normalized)
    if len(rev)>=8:
        yoy=[rev[i]/rev[i-4]-1 for i in range(4,len(rev)) if rev[i-4] and rev[i-4]>0]
        if yoy:
            f["yoy_growth_recent"]=yoy[-1]
            f["yoy_growth_stability"]=1.0-min(1.0,st.pstdev(yoy)/(abs(st.mean(yoy))+0.02)) if len(yoy)>1 else None
            f["yoy_accel"]=yoy[-1]-yoy[-2] if len(yoy)>=2 else None
    if len(rev)>=9:  # 2yr stacked: sum of 2 consecutive YoY (smooths base effects)
        stacked=[(rev[i]/rev[i-4]-1)+(rev[i-4]/rev[i-8]-1) for i in range(8,len(rev)) if rev[i-4] and rev[i-8] and rev[i-4]>0 and rev[i-8]>0]
        if stacked: f["two_year_stacked_growth"]=stacked[-1]

    # ========== OPERATING LEVERAGE (incremental margin) ==========
    if len(rev)>=5 and len(oi)>=5:
        d_rev=rev[-1]-rev[-4] if len(rev)>=4 else None
        d_oi=oi[-1]-oi[-4] if len(oi)>=4 else None
        if d_rev and d_rev!=0 and d_oi is not None:
            f["incremental_op_margin"]=d_oi/d_rev  # each new $ revenue -> $ operating income
        # operating leverage ratio: %ΔOI / %Δrev
        if len(oi)>=4 and oi[-4] and oi[-4]>0 and rev[-4] and rev[-4]>0:
            pct_oi=oi[-1]/oi[-4]-1; pct_rev=rev[-1]/rev[-4]-1
            if pct_rev!=0: f["operating_leverage"]=pct_oi/pct_rev

    # ========== MARGIN TRAJECTORY ==========
    op_margins=[oi[i]/rev[i] for i in range(min(len(oi),len(rev))) if rev[i] and rev[i]>0]
    if len(op_margins)>=5:
        f["op_margin_trajectory"]=_slope(op_margins[-8:])
        f["op_margin_recent"]=op_margins[-1]
    gm=[gp[i]/rev[i] for i in range(min(len(gp),len(rev))) if rev[i] and rev[i]>0]
    if len(gm)>=5:
        f["gross_margin_trajectory"]=_slope(gm[-8:])

    # ========== RULE OF 40 (growth + margin) ==========
    rg_ttm=fin_features.get("revenue_growth")
    if rg_ttm is not None and op_margins:
        f["rule_of_40"]=rg_ttm+op_margins[-1]  # >0.40 = healthy for compounders
        f["rule_of_40_pass"]=1.0 if (rg_ttm+op_margins[-1])>=0.40 else 0.0

    # ========== CASH FLOW FORECAST ==========
    if len(ocf)>=6:
        ocfg=qoq(ocf)
        if len(ocfg)>=3:
            f["ocf_growth_recent"]=st.mean(ocfg[-2:])
            f["ocf_growth_persistence"]=1.0-min(1.0,st.pstdev(ocfg)/(abs(st.mean(ocfg))+0.05)) if len(ocfg)>1 else None
        f["ocf_trend_slope"]=_cagr_slope(ocf[-6:])
    # cash-backed earnings quality trend (OCF/NI rising = higher quality forward)
    if len(ocf)>=4 and len(ni)>=4:
        ratios=[ocf[i]/ni[i] for i in range(min(len(ocf),len(ni))) if ni[i] and ni[i]>0]
        if len(ratios)>=4:
            f["cash_conversion_trend"]=_slope(ratios[-6:]) if len(ratios)>=6 else None
            f["cash_conversion_level"]=ratios[-1]
    # FCF proxy = OCF - capex(≈ Δfixed_assets)
    fa=S("fixed_assets",12)
    if len(ocf)>=1 and len(fa)>=2:
        capex_proxy=max(0.0,fa[-1]-fa[-2])
        fcf_proxy=ocf[-1]-capex_proxy
        if rev and rev[-1]>0: f["fcf_margin_proxy"]=fcf_proxy/rev[-1]

    # ========== ROIIC / INTRINSIC COMPOUNDING ==========
    eq=S("equity",12); ltd=S("long_term_debt",12); tax=S("tax_expense",12); pre=S("pretax_income",12)
    if len(oi)>=5 and len(eq)>=5:
        # invested capital = equity + long-term debt
        ic_new=(eq[-1] or 0)+(ltd[-1] if ltd else 0 or 0)
        ic_old=(eq[-4] or 0)+(ltd[-4] if len(ltd)>=4 else 0 or 0) if len(eq)>=4 else None
        d_oi=oi[-1]-oi[-4] if len(oi)>=4 else None
        # effective tax rate
        etr=0.21
        if tax and pre and pre[-1] and pre[-1]>0: etr=min(0.5,max(0.0,tax[-1]/pre[-1]))
        if ic_old is not None and (ic_new-ic_old)!=0 and d_oi is not None:
            f["roiic"]=d_oi*(1-etr)/(ic_new-ic_old)  # return on incremental invested capital
    # intrinsic growth = ROIC * reinvestment rate (approx via retained/equity growth)
    roic=fin_features.get("roic")
    if roic is not None and len(eq)>=5:
        eqg=eq[-1]/eq[-4]-1 if len(eq)>=4 and eq[-4] and eq[-4]>0 else None
        if eqg is not None: f["intrinsic_growth_proxy"]=roic*max(0.0,min(1.0,eqg*4))

    # ========== WORKING CAPITAL / LEADING INDICATORS ==========
    ca=S("current_assets",12); cl=S("current_liabilities",12)
    if len(ca)>=5 and len(cl)>=5:
        wc=[ca[i]-cl[i] for i in range(min(len(ca),len(cl)))]
        if len(wc)>=5: f["working_capital_trend"]=_cagr_slope(wc[-6:])
    # cash trajectory
    cash=S("cash",12)
    if len(cash)>=5:
        f["cash_trajectory"]=_cagr_slope(cash[-6:])
    # tax-rate normalization flag (one-time tax boost inflating recent earnings)
    if tax and pre and len(tax)>=4 and len(pre)>=4:
        etrs=[tax[i]/pre[i] for i in range(min(len(tax),len(pre))) if pre[i] and pre[i]>0]
        if len(etrs)>=4:
            f["tax_rate_stability"]=1.0-min(1.0,st.pstdev(etrs[-4:])*5)

    # ========== MOMENTUM QUALITY ==========
    if closes and len(closes)>=126:
        rets=[closes[i]/closes[i-1]-1 for i in range(1,len(closes))]
        mom3=closes[-1]/closes[-63]-1; mom6=closes[-1]/closes[-126]-1
        mom12=closes[-1]/closes[-252]-1 if len(closes)>=252 else None
        f["momentum_3m"]=mom3; f["momentum_6m"]=mom6
        if mom12 is not None: f["momentum_12m"]=mom12
        f["momentum_alignment"]=1.0 if (mom3>0)==(mom6>0) else 0.0
        # momentum QUALITY: return / volatility (Sharpe-like, low-vol momentum is durable)
        vol=st.pstdev(rets[-126:])
        if vol>0: f["momentum_quality"]=mom6/(vol*math.sqrt(126))
        f["up_day_ratio"]=sum(1 for r in rets[-63:] if r>0)/63
        # trend acceleration: recent slope vs older slope
        if len(closes)>=120:
            s1=_slope(closes[-30:]); s2=_slope(closes[-60:-30])
            if s1 is not None and s2 is not None and closes[-1]>0:
                f["trend_acceleration"]=(s1-s2)/closes[-1]

    # ========== TREND / MEAN REVERSION ==========
    if closes and len(closes)>=60:
        sl=_slope(closes[-60:])
        if sl is not None and closes[-1]>0: f["price_trend_slope"]=sl*60/closes[-1]
        ma50=st.mean(closes[-50:])
        f["price_vs_ma50"]=closes[-1]/ma50-1 if ma50>0 else None
        if len(closes)>=200:
            ma200=st.mean(closes[-200:])
            f["ma_golden_cross"]=ma50/ma200-1 if ma200>0 else None
            dev=closes[-1]/ma200-1 if ma200>0 else None
            if dev is not None:
                f["mean_reversion_pull"]=-dev
                f["extension_from_mean"]=abs(dev)
        # 52-week high proximity
        if len(closes)>=252:
            hi=max(closes[-252:]); f["dist_from_52w_high"]=closes[-1]/hi-1
    if closes and len(closes)>=15:
        r14=[closes[i]/closes[i-1]-1 for i in range(len(closes)-14,len(closes))]
        gains=sum(r for r in r14 if r>0); losses=-sum(r for r in r14 if r<0)
        if gains+losses>0: f["rsi_signal"]=gains/(gains+losses)

    # ========== FORECAST CONFIDENCE ==========
    if closes and len(closes)>=120:
        rets=[closes[i]/closes[i-1]-1 for i in range(1,len(closes))]
        vol=st.pstdev(rets[-60:])
        f["forecast_confidence"]=1.0-min(1.0,vol*15)
    mom6=f.get("momentum_6m")
    if rg_ttm is not None and mom6 is not None:
        f["fundamental_technical_agree"]=1.0 if (rg_ttm>0)==(mom6>0) else 0.0
    if roic is not None: f["quality_anchor"]=roic
    # composite forward conviction: growth + momentum + quality aligned
    parts=[f.get("revenue_growth_recent"),f.get("momentum_6m"),f.get("op_margin_trajectory")]
    valid=[p for p in parts if p is not None]
    if len(valid)>=2: f["forward_composite"]=sum(1 for p in valid if p>0)/len(valid)

    return {k:v for k,v in f.items() if v is not None}

if __name__=="__main__":
    import random; random.seed(4)
    merged=[]
    for i in range(12):
        rev=50e9*(1+i*0.02); merged.append({"revenue":rev,"net_income":rev*0.30,"operating_income":rev*0.40,
            "gross_profit":rev*0.68,"operating_cash_flow":rev*0.42,"eps_diluted":2.5*(1+i*0.03),
            "equity":180e9+i*3e9,"long_term_debt":50e9,"tax_expense":rev*0.06,"pretax_income":rev*0.36,
            "current_assets":150e9+i*2e9,"current_liabilities":90e9+i*1e9,"cash":80e9+i*1e9,"fixed_assets":100e9+i*2e9})
    closes=[100]
    for _ in range(260): closes.append(closes[-1]*(1+random.gauss(0.0006,0.013)))
    fin={"revenue_growth":0.14,"roic":0.28}
    f=compute_forecast_features(merged,closes,fin)
    n=sum(1 for k in f if not k.startswith('_'))
    print(f"FORECAST: {n} signals")
    for k,v in sorted(f.items()):
        if not k.startswith('_'): print(f"  {k:28s} {round(v,4) if isinstance(v,float) else v}")
