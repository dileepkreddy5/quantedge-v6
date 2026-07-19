"""Business Intelligence — moat & durability engine. Scores business-model quality
from REAL quantitative proxies (persistent ROIC/excess returns, margin power &
stability, recurring revenue, operating leverage, reinvestment economics). Distinct
from Financial: owns SECOND-ORDER durability patterns, not raw levels. No fakes.
"""
import math
from typing import Optional, Dict, List, Any

def _f(v):
    try: v=float(v); return v if math.isfinite(v) else None
    except: return None
def _mean(xs):
    xs=[x for x in xs if x is not None]; return sum(xs)/len(xs) if xs else None
def _stdev(xs):
    xs=[x for x in xs if x is not None]
    if len(xs)<2: return None
    m=sum(xs)/len(xs); return (sum((x-m)**2 for x in xs)/len(xs))**0.5
def _slope(xs):
    xs=[x for x in xs if x is not None]
    n=len(xs)
    if n<3: return None
    mx=(n-1)/2; my=sum(xs)/n
    num=sum((i-mx)*(xs[i]-my) for i in range(n)); den=sum((i-mx)**2 for i in range(n))
    return num/den if den else None

def compute_business_features(merged, fin_features, wacc=None):
    f={}
    if not merged or len(merged)<8: return f
    def series(k, n=20): return [q.get(k) for q in merged[-n:]]
    def ttm_series(k, steps=12):
        out=[]
        for end in range(len(merged), 3, -1):
            window=[merged[j].get(k) for j in range(end-4,end)]
            if all(v is not None for v in window): out.append(sum(window))
        return list(reversed(out))[-steps:] if out else []

    roic=fin_features.get("roic"); roic_exgw=fin_features.get("roic_ex_goodwill")
    ni=series("net_income",20); eq=series("equity",20)
    roe_hist=[(ni[i]/eq[i]) if (ni[i] is not None and eq[i] not in (None,0)) else None for i in range(len(ni))]
    f["roe_mean_5y"]=_mean(roe_hist)
    md=_mean(roe_hist); sd=_stdev(roe_hist)
    f["roe_stability"]=(1-min(1,sd/abs(md))) if (md not in (None,0) and sd is not None) else None
    f["roic_current"]=roic_exgw or roic
    if (roic_exgw or roic) is not None and wacc:
        f["excess_return_spread"]=(roic_exgw or roic)-wacc

    rev_t=ttm_series("revenue"); cogs_t=ttm_series("cost_of_revenue")
    gm_hist=[(1-cogs_t[i]/rev_t[i]) if (i<len(cogs_t) and rev_t[i] not in (None,0) and cogs_t[i] is not None) else None for i in range(len(rev_t))]
    f["gross_margin_level"]=_mean(gm_hist)
    mg=_mean(gm_hist); sg=_stdev(gm_hist)
    f["gross_margin_stability"]=(1-min(1,sg/abs(mg))) if (mg not in (None,0) and sg is not None) else None
    f["gross_margin_trend"]=_slope(gm_hist)

    f["recurring_revenue_ratio"]=fin_features.get("deferred_rev_to_revenue")
    f["recurring_revenue_growth"]=fin_features.get("deferred_rev_growth")

    op_t=ttm_series("operating_income")
    if len(rev_t)>=8 and len(op_t)>=8:
        rev_growth=_slope(rev_t); op_growth=_slope(op_t)
        if rev_growth and op_growth and rev_growth>0:
            f["operating_leverage"]=op_growth/rev_growth
        om_hist=[(op_t[i]/rev_t[i]) if rev_t[i] not in (None,0) else None for i in range(min(len(op_t),len(rev_t)))]
        f["operating_margin_trend"]=_slope(om_hist)

    reinvest=fin_features.get("reinvestment_rate")
    if reinvest is not None and (roic_exgw or roic) is not None:
        f["reinvestment_quality"]=reinvest*(roic_exgw or roic)
    f["capital_intensity"]=fin_features.get("capex_intensity")

    if len(rev_t)>=8:
        growths=[(rev_t[i]/rev_t[i-1]-1) for i in range(1,len(rev_t)) if rev_t[i-1] not in (None,0)]
        mgr=_mean(growths); sgr=_stdev(growths)
        f["revenue_consistency"]=(1-min(1,sgr/abs(mgr))) if (mgr not in (None,0) and sgr is not None) else None
        f["revenue_growth_durability"]=mgr
    return f
