"""Competitive Intelligence — head-to-head competitive dynamics vs sector peers.
Uses enriched peer fundamentals (fund_*) + own financials."""
import json, statistics as st

def _pf(obj):
    f=obj.get("factors") if isinstance(obj,dict) else None
    if isinstance(f,str):
        try: return json.loads(f)
        except: return {}
    return f if isinstance(f,dict) else {}

def _pctile(vals, mv, hib=True):
    if mv is None: return None
    vals=[v for v in vals if v is not None]
    if len(vals)<5: return None
    b=sum(1 for v in vals if v<mv)/len(vals)
    return b if hib else 1.0-b

def _med(vals):
    vals=[v for v in vals if v is not None]
    return st.median(vals) if len(vals)>=5 else None

def compute_competitive_features(own, peer_bucket):
    f={}
    if not peer_bucket or not peer_bucket.get("available"): return {"available":False}
    peers=peer_bucket.get("peers",[])
    if len(peers)<5: return {"available":False}
    f["_bucket"]=peer_bucket.get("bucket"); f["_peer_count"]=len(peers)

    def pv(key): return [_pf(p).get(key) for p in peers]
    def own_pctile(ok,pk,hib=True): return _pctile(pv(pk), own.get(ok), hib)
    def advantage(ok,pk):
        m=_med(pv(pk)); v=own.get(ok)
        return (v-m) if (v is not None and m is not None) else None

    mc=own.get("market_cap"); rev=own.get("revenue")
    f["scale_rank"]=_pctile([p.get("market_cap") for p in peers], mc, True)
    f["mcap_rank"]=f["scale_rank"]
    if rev and mc:
        peer_rev=[]
        for p in peers:
            pmc=p.get("market_cap"); pps=_pf(p).get("fund_ps")
            if pmc and pps and pps>0: peer_rev.append(pmc/pps)
        if peer_rev and rev:
            tot=sum(peer_rev)+rev
            f["market_share_proxy"]=rev/tot if tot>0 else None
    f["size_dominance"]=f.get("scale_rank")
    f["revenue_rank"]=f.get("scale_rank")

    f["net_margin_pctile"]=own_pctile("net_margin","fund_net_margin")
    f["gross_margin_pctile"]=own_pctile("gross_margin","fund_gross_margin")
    f["roic_pctile"]=own_pctile("roic","fund_roic_approx")
    f["roe_pctile"]=own_pctile("roe","fund_roe")
    f["margin_advantage"]=advantage("net_margin","fund_net_margin")
    f["roic_advantage"]=advantage("roic","fund_roic_approx")
    f["roe_advantage"]=advantage("roe","fund_roe")

    f["growth_pctile"]=own_pctile("revenue_growth","fund_revenue_growth")
    f["growth_advantage"]=advantage("revenue_growth","fund_revenue_growth")
    f["revenue_growth_vs_median"]=f.get("growth_advantage")
    if f.get("growth_advantage") is not None: f["share_gain_proxy"]=f["growth_advantage"]

    f["gross_margin_level"]=own.get("gross_margin")
    f["gross_margin_pctile_pp"]=own_pctile("gross_margin","fund_gross_margin")
    f["margin_advantage_pp"]=advantage("gross_margin","fund_gross_margin")
    f["price_premium_proxy"]=own_pctile("gross_margin","fund_gross_margin")

    f["asset_turnover_pctile"]=own_pctile("asset_turnover","fund_asset_turnover")
    f["ocf_margin_pctile"]=own_pctile("ocf_margin","fund_ocf_margin")
    f["efficiency_advantage"]=advantage("asset_turnover","fund_asset_turnover")
    f["capital_efficiency_vs_peers"]=own_pctile("roic","fund_roic_approx")

    f["excess_return_vs_peers"]=own_pctile("roic","fund_roic_approx")
    if own.get("roic") is not None and own.get("wacc") is not None:
        f["economic_moat_spread"]=own["roic"]-own["wacc"]
    f["margin_durability"]=own.get("gross_margin_stability")
    f["roic_persistence"]=own.get("roe_stability")

    f["pe_discount_vs_peers"]=own_pctile("pe","fund_pe",False)
    f["cheapness_rank"]=f.get("pe_discount_vs_peers")
    if own.get("pe") is not None:
        m=_med(pv("fund_pe"))
        if m and m>0: f["pe_relative_to_median"]=own["pe"]/m

    f["roe_strength_rank"]=own_pctile("roe","fund_roe")
    if own.get("current_ratio") is not None: f["liquidity_position"]=own["current_ratio"]
    f["profitability_rank"]=own_pctile("net_margin","fund_net_margin")
    f["profit_spread_vs_median"]=advantage("net_margin","fund_net_margin")

    f["growth_momentum_rank"]=own_pctile("revenue_growth","fund_revenue_growth")
    if own.get("earnings_growth") is not None: f["earnings_growth"]=own["earnings_growth"]

    ga=f.get("growth_advantage")
    if ga is not None:
        f["growth_decel_risk"]=ga
        f["share_loss_risk"]=1.0 if ga<-0.05 else 0.0
    ma=f.get("margin_advantage")
    if ma is not None: f["margin_compression_risk"]=1.0 if ma<-0.05 else 0.0

    if mc: f["absolute_scale"]=mc/1e9
    if own.get("employees"): f["employee_scale"]=own["employees"]
    f["scale_advantage"]=f.get("scale_rank")

    return {k:v for k,v in f.items() if (v is not None or k.startswith("_"))}
