"""Ownership Intelligence — ownership structure, insider ownership, institutional holders (13G),
share stability, smart-money signals, float/liquidity, conviction, concentration."""
import math, statistics as st
def _f(v):
    try: v=float(v); return v if math.isfinite(v) else None
    except: return None
def _sd(a,b):
    a=_f(a); b=_f(b); return a/b if (a is not None and b not in (None,0)) else None

def compute_ownership_features(merged, shares_out=None, market_cap=None, insider=None, ownership=None, avg_volume=None):
    f={}
    insider=insider or {}; ownership=ownership or {}
    if not merged or len(merged)<4: return {"available":False}
    def cur(k):
        for x in reversed(merged):
            v=_f(x.get(k))
            if v is not None: return v
        return None
    def series(k,n=8): return [_f(x.get(k)) for x in merged[-n:] if _f(x.get(k)) is not None]

    if shares_out: f["shares_outstanding_b"]=shares_out/1e9
    dil=cur("diluted_shares"); basic=cur("basic_shares")
    if dil and basic and basic>0: f["dilution_gap"]=dil/basic-1
    sh=series("diluted_shares",8)
    if len(sh)>=6:
        o=st.median(sh[:3]); n=st.median(sh[-3:])
        chg=(n/o-1) if o>0 else None
        if chg is not None and abs(chg)<0.4:
            f["share_count_trend"]=chg
            f["ownership_concentration_trend"]=-chg

    if insider.get("available"):
        f["insider_net_activity"]=float(insider.get("officer_net",0) or 0)+float(insider.get("director_net",0) or 0)
        f["insider_buy_ratio"]=insider.get("buy_value_ratio")
        f["insider_cluster"]=insider.get("cluster_buying")
        f["insider_unique_buyers"]=float(insider.get("unique_buyers") or 0)
        if market_cap and insider.get("net_insider_value") is not None:
            f["insider_net_conviction"]=insider["net_insider_value"]/market_cap

    if ownership.get("available"):
        f["major_holder_count"]=float(ownership.get("major_holders") or 0)
        f["top_holder_pct"]=ownership.get("top_holder_pct")
        f["institutional_concentration"]=ownership.get("concentration")
        f["avg_holder_stake"]=ownership.get("avg_holder_pct")

    if len(sh)>=8:
        vals=[sh[i]/sh[i-1]-1 for i in range(1,len(sh)) if sh[i-1]>0]
        if vals: f["share_count_stability"]=1.0-min(1.0,st.pstdev(vals)*20)
    sbc=cur("sbc"); rev=cur("revenue")
    if sbc is not None and rev and rev>0: f["dilution_pressure"]=sbc/rev

    if insider.get("available"):
        f["smart_money_buying"]=insider.get("any_insider_buying")
        f["officer_conviction"]=1.0 if float(insider.get("officer_net",0) or 0)>0 else 0.0
    if ownership.get("available") and ownership.get("major_holders"):
        f["institutional_interest"]=min(1.0,float(ownership["major_holders"])/5)

    if shares_out and market_cap: f["market_cap_b"]=market_cap/1e9
    if avg_volume and shares_out:
        f["turnover_ratio"]=avg_volume/shares_out
        f["float_liquidity"]=min(1.0,(avg_volume/shares_out)*1000)
    if dil and market_cap: f["implied_price"]=market_cap/dil

    if insider.get("available") and insider.get("buy_value_ratio") is not None:
        f["ownership_conviction"]=insider["buy_value_ratio"]
    bb=cur("buybacks")
    if bb and market_cap: f["buyback_intensity"]=bb/market_cap

    if f.get("ownership_concentration_trend") is not None:
        f["concentration_direction"]=f["ownership_concentration_trend"]
    if ownership.get("top_holder_pct") is not None:
        f["holder_concentration_risk"]=ownership["top_holder_pct"]

    return {k:v for k,v in f.items() if v is not None}
