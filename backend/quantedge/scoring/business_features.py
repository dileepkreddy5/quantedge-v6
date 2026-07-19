"""Business Intelligence — FULL DEPTH moat/durability engine (~76 signals).
Every signal a real quantitative proxy from financial history, or None if absent.
Single ownership: durability/moat PATTERNS, not raw levels (Financial owns those).
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
    xs=[x for x in xs if x is not None]; n=len(xs)
    if n<3: return None
    mx=(n-1)/2; my=sum(xs)/n
    num=sum((i-mx)*(xs[i]-my) for i in range(n)); den=sum((i-mx)**2 for i in range(n))
    return num/den if den else None
def _stability(xs):
    m=_mean(xs); s=_stdev(xs)
    return (1-min(1,s/abs(m))) if (m not in (None,0) and s is not None) else None
def _cagr(series):
    s=[x for x in series if x is not None and x>0]
    if len(s)<2: return None
    yrs=len(s)/4.0
    return (s[-1]/s[0])**(1/yrs)-1 if yrs>0 else None

def compute_business_features(merged, fin_features, wacc=None, peer_data=None):
    f={}
    if not merged or len(merged)<8: return f
    wacc=wacc or 0.09
    # derive fields not directly present
    for q in merged:
        oi=q.get("operating_income"); da=q.get("depreciation_amortization")
        q["_ebitda_derived"]=(oi+da) if (oi is not None and da is not None) else None
        ltd=q.get("long_term_debt") or 0; std=q.get("short_term_debt") or q.get("short_term_debt2") or 0
        q["_total_debt"]=(ltd+std) if (q.get("long_term_debt") is not None or q.get("short_term_debt") is not None) else None
        oe=q.get("operating_expenses"); rd=q.get("rd")
        q["_sga_derived"]=(oe-rd) if (oe is not None and rd is not None) else oe
    def S(k,n=24): return [q.get(k) for q in merged[-n:]]
    def ttm(k,steps=16):
        out=[]
        for end in range(len(merged),3,-1):
            w=[merged[j].get(k) for j in range(end-4,end)]
            if all(v is not None for v in w): out.append(sum(w))
        return list(reversed(out))[-steps:] if out else []

    rev_t=ttm("revenue"); ni_t=ttm("net_income"); op_t=ttm("operating_income")
    cogs_t=ttm("cost_of_revenue"); gp_t=[rev_t[i]-cogs_t[i] for i in range(min(len(rev_t),len(cogs_t)))]
    ocf_t=ttm("operating_cash_flow"); capex_t=ttm("capex"); ebitda_t=ttm("_ebitda_derived")
    eq=S("equity"); ni=S("net_income"); assets=S("assets")
    roic=fin_features.get("roic_ex_goodwill") or fin_features.get("roic")

    # 1. MOAT STRENGTH
    f["roic_level"]=roic
    f["roic_wacc_spread"]=(roic-wacc) if roic is not None else None
    debt_s=S("_total_debt")
    roic_hist=[]
    for i in range(len(op_t)):
        idx=min(len(eq)-1, i+3)
        if 0<=idx<len(eq) and eq[idx] is not None:
            d=debt_s[idx] if idx<len(debt_s) and debt_s[idx] is not None else 0
            ic=(eq[idx] or 0)+d
            if ic and ic>0: roic_hist.append(op_t[i]*0.79/ic)
    f["roic_5y_avg"]=_mean(roic_hist)
    f["roic_trend"]=_slope(roic_hist)
    f["spread_persistence_yrs"]=(sum(1 for r in roic_hist if r>wacc)/4.0) if roic_hist else None
    f["excess_return_consistency"]=_stability(roic_hist)
    if ocf_t and capex_t and eq:
        fcf_t=[ocf_t[i]-abs(capex_t[i]) for i in range(min(len(ocf_t),len(capex_t)))]
        ic_now=(eq[-1] or 0)+((debt_s[-1] or 0) if debt_s else 0)
        f["croic"]=(fcf_t[-1]/ic_now) if (fcf_t and ic_now>0) else None
    if len(op_t)>=5 and len(eq)>=8:
        d_nopat=(op_t[-1]-op_t[-5])*0.79
        ic_new=(eq[-1] or 0)+((debt_s[-1] or 0) if debt_s else 0)
        ic_old=(eq[-5] or 0)+((debt_s[-5] or 0) if len(debt_s)>=5 and debt_s[-5] else 0)
        if ic_new-ic_old>0: f["incremental_roic"]=d_nopat/(ic_new-ic_old)
    if roic is not None and eq:
        ic_now=(eq[-1] or 0)+((debt_s[-1] or 0) if debt_s else 0)
        f["economic_profit"]=(roic-wacc)*ic_now
    rt=f.get("roic_trend")
    f["moat_direction"]=(1.0 if rt and rt>0.001 else -1.0 if rt and rt<-0.001 else 0.0) if rt is not None else None

    # 2. PRICING POWER
    gm_hist=[(gp_t[i]/rev_t[i]) if rev_t[i] else None for i in range(len(gp_t))]
    f["gross_margin_level"]=_mean(gm_hist); f["gross_margin_stability"]=_stability(gm_hist)
    f["gross_margin_trend"]=_slope(gm_hist)
    em_hist=[(ebitda_t[i]/rev_t[i]) if (i<len(rev_t) and rev_t[i]) else None for i in range(len(ebitda_t))] if ebitda_t else []
    f["ebitda_margin_level"]=_mean(em_hist) if em_hist else None
    f["ebitda_margin_trend"]=_slope(em_hist) if em_hist else None
    gmv=[x for x in gm_hist if x is not None]
    f["margin_resilience"]=(min(gmv)/_mean(gmv)) if gmv and _mean(gmv) else None
    if len(gp_t)>=8 and len(rev_t)>=8:
        gpg=_cagr(gp_t); rvg=_cagr(rev_t)
        f["price_vs_volume_growth"]=(gpg-rvg) if (gpg is not None and rvg is not None) else None
    f["gross_margin_vs_peers"]=None

    # 3. REVENUE QUALITY
    f["recurring_revenue_ratio"]=fin_features.get("deferred_rev_to_revenue")
    f["deferred_rev_growth"]=fin_features.get("deferred_rev_growth")
    f["revenue_consistency"]=_stability([(rev_t[i]/rev_t[i-1]-1) for i in range(1,len(rev_t)) if rev_t[i-1]])
    f["revenue_cagr_5y"]=_cagr(rev_t)
    if len(rev_t)>=8:
        n=len(rev_t); mx=(n-1)/2; my=sum(rev_t)/n
        ss_tot=sum((y-my)**2 for y in rev_t); sl=_slope(rev_t)
        pred=[my+sl*(i-mx) for i in range(n)] if sl else None
        if pred and ss_tot>0:
            ss_res=sum((rev_t[i]-pred[i])**2 for i in range(n))
            f["revenue_predictability"]=max(0,1-ss_res/ss_tot)
    f["organic_growth_est"]=fin_features.get("revenue_cagr_5y")
    f["revenue_diversification"]=None; f["backlog_rpo"]=None; f["customer_concentration"]=None

    # 4. UNIT ECONOMICS
    if len(op_t)>=5 and len(rev_t)>=5 and (rev_t[-1]-rev_t[-5])!=0:
        f["incremental_margin"]=(op_t[-1]-op_t[-5])/(rev_t[-1]-rev_t[-5])
    f["contribution_margin_proxy"]=_mean(gm_hist)
    f["marginal_roic"]=f.get("incremental_roic")
    if rev_t and assets and assets[-1]: f["capital_turnover"]=rev_t[-1]/assets[-1]
    f["cash_conversion_cycle"]=fin_features.get("cash_conversion_cycle")
    if ocf_t and ni_t and ni_t[-1]: f["cash_conversion_ratio"]=ocf_t[-1]/ni_t[-1]
    if ocf_t and capex_t and rev_t:
        fcf=ocf_t[-1]-abs(capex_t[-1]); f["fcf_margin"]=fcf/rev_t[-1] if rev_t[-1] else None
        fcf_hist=[(ocf_t[i]-abs(capex_t[i]))/rev_t[i] for i in range(min(len(ocf_t),len(capex_t),len(rev_t))) if rev_t[i]]
        f["fcf_margin_trend"]=_slope(fcf_hist)

    # 5. COMPETITIVE POSITION
    for k in ["relative_growth_vs_industry","roic_percentile_vs_peers","margin_percentile_vs_peers",
              "growth_percentile_vs_peers","scale_rank","market_share","share_trend"]:
        f[k]=None
    f["relative_reinvestment"]=fin_features.get("reinvestment_rate")

    # 6. MANAGEMENT & CAPITAL ALLOCATION
    f["buyback_yield"]=fin_features.get("buyback_yield")
    sh=S("diluted_shares"); shv=[x for x in sh if x is not None]
    if len(shv)>=8:
        f["buyback_consistency"]=1.0 if shv[-1]<shv[0] else 0.0
        f["share_count_cagr"]=_cagr(shv)
    f["dividend_growth"]=fin_features.get("dividend_growth")
    f["reinvestment_rate"]=fin_features.get("reinvestment_rate")
    f["reinvestment_quality"]=(fin_features.get("reinvestment_rate")*roic) if (fin_features.get("reinvestment_rate") is not None and roic is not None) else None
    gw=S("goodwill"); gwv=[x for x in gw if x is not None]
    f["ma_intensity"]=_cagr(gwv) if len(gwv)>=8 else None
    sbc_t=ttm("sbc")
    if sbc_t and rev_t: f["sbc_intensity"]=sbc_t[-1]/rev_t[-1] if rev_t[-1] else None
    div_t=ttm("dividends_paid")
    if div_t and ni_t and ni_t[-1]: f["total_payout_ratio"]=abs(div_t[-1])/ni_t[-1]
    f["capital_allocation_balance"]=None

    # 7. SCALE & OPERATING LEVERAGE
    if len(rev_t)>=8 and len(op_t)>=8:
        rvg=_slope(rev_t); opg=_slope(op_t)
        if rvg and opg and rvg>0: f["operating_leverage"]=opg/rvg
        om_hist=[(op_t[i]/rev_t[i]) if rev_t[i] else None for i in range(min(len(op_t),len(rev_t)))]
        f["operating_margin_trend"]=_slope(om_hist); f["margin_expansion_rate"]=_slope(om_hist)
    opex_t=ttm("operating_expenses")
    if opex_t and rev_t:
        oxr=[(opex_t[i]/rev_t[i]) if rev_t[i] else None for i in range(min(len(opex_t),len(rev_t)))]
        sl=_slope(oxr); f["opex_efficiency"]=-sl if sl is not None else None
    f["fixed_cost_absorption"]=f.get("incremental_margin")
    f["revenue_per_employee"]=None

    # 8. GROWTH QUALITY & DURABILITY
    growths=[(rev_t[i]/rev_t[i-1]-1) for i in range(1,len(rev_t)) if rev_t[i-1]]
    f["revenue_growth_durability"]=_mean(growths); f["growth_consistency"]=_stability(growths)
    f["reinvestment_runway"]=fin_features.get("reinvestment_rate")
    ci=fin_features.get("capex_intensity")
    f["growth_efficiency"]=(f["revenue_growth_durability"]/ci) if (f.get("revenue_growth_durability") and ci) else None
    if f.get("revenue_growth_durability") is not None and f.get("fcf_margin") is not None:
        f["rule_of_40"]=f["revenue_growth_durability"]+f["fcf_margin"]
    if roic is not None and fin_features.get("reinvestment_rate") is not None:
        f["sustainable_growth_rate"]=roic*fin_features.get("reinvestment_rate")
    if len(growths)>=8:
        f["growth_deceleration_risk"]=_mean(growths[-4:])-_mean(growths[-8:-4])

    # 9. BUSINESS RISK & DURABILITY
    if len(ni_t)>=8:
        eg=[(ni_t[i]/ni_t[i-4]-1) for i in range(4,len(ni_t)) if ni_t[i-4] and ni_t[i-4]>0]
        f["earnings_cyclicality"]=_stdev(eg)
    f["revenue_volatility"]=_stdev(growths)
    f["asset_intensity"]=fin_features.get("capex_intensity")
    rd_t=ttm("rd")
    if rd_t and rev_t: f["rd_intensity"]=rd_t[-1]/rev_t[-1] if rev_t[-1] else None
    f["leverage_stability"]=_stability(S("_total_debt"))
    f["customer_concentration_risk"]=None

    # 10. INTANGIBLE MOAT
    f["rd_intensity_moat"]=f.get("rd_intensity")
    if rd_t and len(rev_t)>=5 and rd_t[-1]:
        f["rd_productivity"]=(rev_t[-1]-rev_t[-5])/rd_t[-1]
    sga_t=ttm("_sga_derived")
    if sga_t and rev_t: f["brand_proxy"]=sga_t[-1]/rev_t[-1] if rev_t[-1] else None
    f["switching_costs"]=None; f["network_effects"]=None
    return f
