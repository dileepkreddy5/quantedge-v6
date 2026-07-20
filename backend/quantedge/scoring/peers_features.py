"""Peers Intelligence — pure relative rank across the sector peer set on every enriched dimension."""
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

def compute_peers_features(own, peer_bucket):
    f={}
    if not peer_bucket or not peer_bucket.get("available"): return {"available":False}
    peers=peer_bucket.get("peers",[])
    if len(peers)<5: return {"available":False}
    f["_bucket"]=peer_bucket.get("bucket"); f["_peer_count"]=len(peers)
    def pv(k): return [_pf(p).get(k) for p in peers]
    def rank(own_key,pk,hib=True): return _pctile(pv(pk), own.get(own_key), hib)

    f["pe_rank"]=rank("pe","fund_pe",False)
    f["ps_rank"]=rank("ps","fund_ps",False)
    f["pb_rank"]=rank("pb","fund_pb",False)
    f["earnings_yield_rank"]=rank("earnings_yield","fund_earnings_yield",True)
    f["ocf_yield_rank"]=rank("ocf_yield","fund_ocf_yield",True)

    f["roic_rank"]=rank("roic","fund_roic_approx")
    f["roe_rank"]=rank("roe","fund_roe")
    f["roa_rank"]=rank("roa","fund_roa")
    qs=[x for x in [f.get("roic_rank"),f.get("roe_rank"),f.get("roa_rank")] if x is not None]
    if qs: f["quality_composite"]=st.mean(qs)

    f["revenue_growth_rank"]=rank("revenue_growth","fund_revenue_growth")
    f["earnings_growth_rank"]=rank("earnings_growth","fund_earnings_growth")

    f["gross_margin_rank"]=rank("gross_margin","fund_gross_margin")
    f["net_margin_rank"]=rank("net_margin","fund_net_margin")
    f["ocf_margin_rank"]=rank("ocf_margin","fund_ocf_margin")
    ps=[x for x in [f.get("gross_margin_rank"),f.get("net_margin_rank"),f.get("ocf_margin_rank")] if x is not None]
    if ps: f["profitability_composite"]=st.mean(ps)

    f["current_ratio_rank"]=rank("current_ratio","fund_current_ratio")
    f["asset_turnover_rank"]=rank("asset_turnover","fund_asset_turnover")

    all_ranks=[f.get(k) for k in ["roic_rank","roe_rank","net_margin_rank","gross_margin_rank",
               "revenue_growth_rank","earnings_yield_rank","current_ratio_rank"]]
    all_ranks=[x for x in all_ranks if x is not None]
    if all_ranks:
        f["overall_peer_rank"]=st.mean(all_ranks)
        f["peer_rank_consistency"]=1.0-min(1.0,st.pstdev(all_ranks)) if len(all_ranks)>1 else None
        f["top_quartile_count"]=sum(1 for x in all_ranks if x>=0.75)/len(all_ranks)
        f["bottom_quartile_count"]=sum(1 for x in all_ranks if x<=0.25)/len(all_ranks)
    f["peer_set_size"]=float(len(peers))
    return {k:v for k,v in f.items() if (v is not None or k.startswith("_"))}
