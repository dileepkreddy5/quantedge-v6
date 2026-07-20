"""Management Intelligence — capital allocation, insider activity, management effectiveness,
shareholder alignment, execution quality, governance. From financials + Form 4 insider data."""
import math, statistics as st

def _f(v):
    try: v=float(v); return v if math.isfinite(v) else None
    except: return None
def _sd(a,b):
    a=_f(a); b=_f(b); return a/b if (a is not None and b not in (None,0)) else None

def compute_management_features(merged, fin_features, insider=None, market_cap=None):
    f={}
    if not merged or len(merged)<4: return {"available":False}
    insider=insider or {}
    def cur(k):
        for x in reversed(merged):
            v=_f(x.get(k))
            if v is not None: return v
        return None
    def ttm(k):
        vals=[_f(x.get(k)) for x in merged]; vals=[v for v in vals if v is not None]
        return sum(vals[-4:]) if len(vals)>=4 else (sum(vals) if vals else None)
    def series(k,n=8):
        return [_f(x.get(k)) for x in merged[-n:] if _f(x.get(k)) is not None]

    # ===== CAPITAL ALLOCATION =====
    ni=ttm("net_income"); ocf=ttm("operating_cash_flow"); fcf=ttm("free_cash_flow")
    capex=ttm("capex"); div=ttm("dividends_paid"); bb=ttm("buybacks"); rev=ttm("revenue")
    assets=cur("assets"); eq=cur("equity")
    f["fcf_generation"]=_sd(fcf,rev)
    f["reinvestment_rate"]=_sd(capex,ocf)
    # buyback yield + dividend yield
    if market_cap:
        f["buyback_yield"]=_sd(bb,market_cap)
        f["dividend_yield"]=_sd(div,market_cap)
        f["total_payout_yield"]=_sd((bb or 0)+(div or 0),market_cap)
    f["payout_ratio"]=_sd((div or 0)+(bb or 0),fcf) if fcf and fcf>0 else None
    # ROIC trend (improving capital allocation)
    roics=[]
    for i in range(len(merged)-4,len(merged)):
        if i<0: continue
    roic_now=fin_features.get("roic")
    if roic_now is not None: f["roic_level"]=roic_now
    # dividend consistency (paid every quarter)
    divs=series("dividends_paid",8)
    if divs: f["dividend_consistency"]=sum(1 for d in divs if d and d>0)/len(divs)
    # dividend growth
    if len(divs)>=8:
        old=sum(divs[:4]); new=sum(divs[4:])
        f["dividend_growth"]=(new/old-1) if old>0 else None

    # ===== INSIDER ACTIVITY =====
    if insider.get("available"):
        f["insider_buy_sell_ratio"]=insider.get("buy_sell_txn_ratio")
        f["insider_buy_value_ratio"]=insider.get("buy_value_ratio")
        f["insider_net_value_norm"]=_sd(insider.get("net_insider_value"),market_cap) if market_cap else None
        f["insider_cluster_buying"]=insider.get("cluster_buying")
        f["insider_officer_net"]=float(insider.get("officer_net") or 0)
        f["insider_director_net"]=float(insider.get("director_net") or 0)
        f["insider_any_buying"]=insider.get("any_insider_buying")
        f["insider_unique_buyers"]=float(insider.get("unique_buyers") or 0)
        # sell pressure (heavy selling = negative). normalize sells vs mcap
        if market_cap and insider.get("sell_value"):
            f["insider_sell_pressure"]=insider["sell_value"]/market_cap

    # ===== MANAGEMENT EFFECTIVENESS =====
    # margin trend (expanding under current mgmt)
    margins=[]
    for x in merged[-8:]:
        r=_f(x.get("revenue")); oi=_f(x.get("operating_income"))
        if r and oi is not None and r>0: margins.append(oi/r)
    if len(margins)>=6:
        f["margin_trend"]=st.mean(margins[-3:])-st.mean(margins[:3])
        f["margin_current"]=margins[-1]
    # asset turnover trend
    ats=[]
    for x in merged[-8:]:
        r=_f(x.get("revenue")); a=_f(x.get("assets"))
        if r and a and a>0: ats.append(r*4/a)  # annualized
    if len(ats)>=6: f["asset_turnover_trend"]=st.mean(ats[-3:])-st.mean(ats[:3])
    # revenue growth consistency
    revs=series("revenue",8)
    if len(revs)>=6:
        growths=[revs[i]/revs[i-1]-1 for i in range(1,len(revs)) if revs[i-1]>0]
        if growths:
            f["revenue_consistency"]=1.0-min(1.0,st.pstdev(growths)/(abs(st.mean(growths))+0.01)) if len(growths)>1 else None
            f["revenue_growth_ttm"]=fin_features.get("revenue_growth")
    # ROE stability
    f["roe_level"]=fin_features.get("roe")

    # ===== SHAREHOLDER ALIGNMENT =====
    # share count trend (buyback = shrinking = aligned; dilution = misaligned)
    shares=series("diluted_shares",8)
    if len(shares)>=8:
        old=st.mean(shares[:4]); new=st.mean(shares[4:])
        f["share_count_change"]=(new/old-1) if old>0 else None  # negative=buyback=good
    # SBC discipline
    sbc=ttm("sbc")
    if sbc is not None and rev and rev>0: f["sbc_intensity"]=sbc/rev
    # buyback vs issuance (net)
    if bb is not None and sbc is not None:
        f["net_buyback_vs_sbc"]=_sd((bb or 0)-(sbc or 0),market_cap) if market_cap else None

    # ===== EXECUTION QUALITY =====
    # cash conversion (OCF/NI - management turning earnings to cash)
    f["cash_conversion"]=_sd(ocf,ni)
    # working capital efficiency
    ca=cur("current_assets"); cl=cur("current_liabilities")
    if ca and cl: f["working_capital_ratio"]=ca/cl
    # FCF margin stability
    fcfs=[]
    for x in merged[-8:]:
        r=_f(x.get("revenue")); fc=_f(x.get("free_cash_flow"))
        if r and fc is not None and r>0: fcfs.append(fc/r)
    if len(fcfs)>=6: f["fcf_margin_stability"]=1.0-min(1.0,st.pstdev(fcfs)/(abs(st.mean(fcfs))+0.01))

    # ===== GOVERNANCE =====
    # dilution control (already have share_count_change); insider ownership proxy via net buying
    if insider.get("available"):
        f["insider_alignment"]=insider.get("buy_value_ratio")
    # capital discipline: not over-issuing debt
    ltd_series=series("long_term_debt",8)
    if len(ltd_series)>=8 and assets:
        old=st.mean(ltd_series[:4]); new=st.mean(ltd_series[4:])
        f["debt_discipline"]=-(new/old-1) if old>0 else None  # negative debt growth = good (flip sign)

    return {k:v for k,v in f.items() if v is not None}
