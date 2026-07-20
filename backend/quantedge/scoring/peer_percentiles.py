"""Shared peer-percentile helper. Given a ticker + the company's own metrics, fetch the
enriched peer bucket and return percentile ranks vs sector peers for fundamental factors.
Used by Business/Financial/Valuation/Industry to activate 'vs peers' competitive signals.
Reads the fund_* factors added by the peer-fundamental enrichment."""
import json

def _parse_factors(obj):
    fac=obj.get("factors") if isinstance(obj,dict) else None
    if isinstance(fac,str):
        try: return json.loads(fac)
        except: return {}
    return fac if isinstance(fac,dict) else {}

def _pctile(peer_vals, my_val, higher_better=True):
    """percentile of my_val among peer_vals (0-1). higher_better=False inverts (for P/E etc)."""
    if my_val is None: return None
    vals=[v for v in peer_vals if v is not None]
    if len(vals)<5: return None
    below=sum(1 for v in vals if v<my_val)/len(vals)
    return below if higher_better else (1.0-below)

async def compute_peer_percentiles(pool, ticker, own_metrics):
    """own_metrics: dict with keys like roic, gross_margin, net_margin, operating_margin,
    revenue_growth, roe, pe, market_cap. Returns percentile dict + peer context."""
    out={"_available":False}
    if pool is None: return out
    try:
        from services.peer_store import PeerStore
        pb=await PeerStore(pool).get_peers(ticker)
    except Exception:
        return out
    if not pb or not pb.get("available"): return out
    peers=pb.get("peers",[])
    if len(peers)<5: return out
    out["_available"]=True
    out["_bucket"]=pb.get("bucket")
    out["_peer_count"]=len(peers)

    # map own_metric -> (peer fund_ key, higher_is_better)
    M={
      "roic":("fund_roic_approx",True), "roe":("fund_roe",True),
      "gross_margin":("fund_gross_margin",True), "net_margin":("fund_net_margin",True),
      "operating_margin":("fund_net_margin",True),  # proxy (no separate op-margin in bulk)
      "revenue_growth":("fund_revenue_growth",True), "pe":("fund_pe",False),
      "ocf_margin":("fund_ocf_margin",True), "asset_turnover":("fund_asset_turnover",True),
    }
    for own_key,(pk,hib) in M.items():
        mv=own_metrics.get(own_key)
        if mv is None: continue
        peer_vals=[_parse_factors(p).get(pk) for p in peers]
        pc=_pctile(peer_vals, mv, hib)
        if pc is not None:
            out[f"{own_key}_percentile_vs_peers"]=pc

    # scale rank (by market_cap among peers)
    mc=own_metrics.get("market_cap")
    if mc is not None:
        peer_mc=[p.get("market_cap") for p in peers]
        sc=_pctile(peer_mc, mc, True)
        if sc is not None: out["scale_rank"]=sc

    # relative growth vs industry median
    rg=own_metrics.get("revenue_growth")
    if rg is not None:
        import statistics as st
        peer_g=[_parse_factors(p).get("fund_revenue_growth") for p in peers]
        peer_g=[v for v in peer_g if v is not None]
        if len(peer_g)>=5:
            out["relative_growth_vs_industry"]=rg-st.median(peer_g)
    return out
