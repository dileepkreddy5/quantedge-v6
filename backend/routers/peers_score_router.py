"""Peers Intelligence (scored) — GET /api/v6/peers_score/{ticker}. Relative rank across peer set."""
from __future__ import annotations
import math
from typing import Optional, Dict, Any
import httpx
from fastapi import APIRouter, HTTPException, Request, Depends
from ml.fundamentals.quality_engine import fetch_quarterly_financials
from quantedge.scoring.edgar_fetch import fetch_edgar_supplement
from quantedge.scoring.hybrid_merge import merge_quarters
from quantedge.fundamentals.peer_fundamentals import compute_peer_fundamentals
from quantedge.fundamentals.edgar_bulk import company_facts_from_bulk
from quantedge.fundamentals.universe_full import ticker_cik_map
from quantedge.scoring.peers_features import compute_peers_features
from quantedge.scoring.compute import score_signal
from quantedge.scoring.cat_peers_v6 import CATEGORIES, peers_rating
from services.peer_store import PeerStore
from auth.cognito_auth import get_optional_user, CognitoUser
from core.config import settings

router = APIRouter()
_CIK={}

def _san(o):
    if isinstance(o,float): return o if math.isfinite(o) else None
    if isinstance(o,dict): return {k:_san(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [_san(v) for v in o]
    return o

def _roll(children):
    sc=[c for c in children if c.get("score") is not None]
    tw=sum(c["weight"] for c in children); aw=sum(c["weight"] for c in sc)
    if not sc or tw==0: return None,0.0
    return round(sum(c["score"]*c["weight"] for c in sc)/aw,1), round(aw/tw,3)

def score_peers(features):
    cats=[]
    for cid,(label,wt,sigs) in CATEGORIES.items():
        scored=[]
        for spec in sigs:
            val=features.get(spec["field"])
            res=score_signal(val,spec,None)
            scored.append({"id":spec["id"],"label":spec["label"],"weight":spec["weight"],
                           "status":spec["status"],"evidence":spec["evidence"],"raw_value":val,**res})
        cs,cc=_roll(scored)
        cats.append({"id":cid,"label":label,"weight":wt,"score":cs,"confidence":cc,
                     "n_signals":len(sigs),"n_scored":sum(1 for s in scored if s["score"] is not None),"signals":scored})
    s,c=_roll(cats)
    return {"label":"Peers Intelligence","weight":2.0,"score":s,"confidence":c,"categories":cats}

async def compute_peers_intelligence(ticker: str, api_key: str, pool=None) -> Dict[str,Any]:
    ticker=ticker.upper().strip()
    peer_bucket=None
    if pool is not None:
        try: peer_bucket=await PeerStore(pool).get_peers(ticker)
        except Exception: pass
    if not peer_bucket or not peer_bucket.get("available"):
        return {"ticker":ticker,"available":False,"reason":"no peer set available"}
    # own fundamentals from bulk (consistent with peer factors)
    mcap=None
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r=await c.get(f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={api_key}")
            if r.status_code==200: mcap=((r.json() or {}).get("results",{}) or {}).get("market_cap")
    except Exception: pass
    global _CIK
    if not _CIK:
        try: _CIK=ticker_cik_map()
        except Exception: _CIK={}
    own={}
    cik=_CIK.get(ticker)
    if cik:
        try:
            facts=company_facts_from_bulk(cik)
            if facts:
                ff=compute_peer_fundamentals(facts, market_cap=mcap)
                own={"pe":ff.get("fund_pe"),"ps":ff.get("fund_ps"),"pb":ff.get("fund_pb"),
                     "roic":ff.get("fund_roic_approx"),"roe":ff.get("fund_roe"),"roa":ff.get("fund_roa"),
                     "net_margin":ff.get("fund_net_margin"),"gross_margin":ff.get("fund_gross_margin"),
                     "ocf_margin":ff.get("fund_ocf_margin"),"revenue_growth":ff.get("fund_revenue_growth"),
                     "earnings_growth":ff.get("fund_earnings_growth"),"current_ratio":ff.get("fund_current_ratio"),
                     "asset_turnover":ff.get("fund_asset_turnover"),"earnings_yield":ff.get("fund_earnings_yield"),
                     "ocf_yield":ff.get("fund_ocf_yield")}
        except Exception: pass
    if not own:
        # fallback: use the "me" record from peer bucket
        import json as _json
        me=peer_bucket.get("me",{})
        fac=me.get("factors")
        if isinstance(fac,str):
            try: fac=_json.loads(fac)
            except: fac={}
        if isinstance(fac,dict):
            own={"pe":fac.get("fund_pe"),"ps":fac.get("fund_ps"),"pb":fac.get("fund_pb"),
                 "roic":fac.get("fund_roic_approx"),"roe":fac.get("fund_roe"),"roa":fac.get("fund_roa"),
                 "net_margin":fac.get("fund_net_margin"),"gross_margin":fac.get("fund_gross_margin"),
                 "ocf_margin":fac.get("fund_ocf_margin"),"revenue_growth":fac.get("fund_revenue_growth"),
                 "earnings_growth":fac.get("fund_earnings_growth"),"current_ratio":fac.get("fund_current_ratio"),
                 "asset_turnover":fac.get("fund_asset_turnover"),"earnings_yield":fac.get("fund_earnings_yield"),
                 "ocf_yield":fac.get("fund_ocf_yield")}
    feats=compute_peers_features(own, peer_bucket)
    if feats.get("available") is False:
        return {"ticker":ticker,"available":False,"reason":"could not compute peer ranks"}
    tree=score_peers(feats)
    n_scored=sum(c["n_scored"] for c in tree["categories"]); n_total=sum(c["n_signals"] for c in tree["categories"])
    return {"ticker":ticker,"available":True,"intelligence":"peers",
            "score":tree["score"],"confidence":tree["confidence"],
            "peers_rating":peers_rating(tree["score"]),"weight_in_conviction":2.0,
            "bucket":feats.get("_bucket"),"peer_count":feats.get("_peer_count"),
            "coverage":{"scored":n_scored,"total":n_total},"tree":tree,
            "key_metrics":{k:feats.get(k) for k in
                ["overall_peer_rank","quality_composite","profitability_composite","roic_rank",
                 "net_margin_rank","revenue_growth_rank","pe_rank","top_quartile_count","peer_rank_consistency","earnings_yield_rank"]}}

@router.get("/peers_score/{ticker}")
async def get_peers_score(ticker: str, http_request: Request,
                          current_user: Optional[CognitoUser]=Depends(get_optional_user)):
    api_key=getattr(settings,"POLYGON_API_KEY","") or ""
    if not api_key: raise HTTPException(503,"data source unavailable")
    pool=getattr(http_request.app.state,"db",None)
    return {"data":_san(await compute_peers_intelligence(ticker, api_key, pool))}
