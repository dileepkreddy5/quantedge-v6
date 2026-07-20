"""Competitive Intelligence endpoint — GET /api/v6/competitive/{ticker}."""
from __future__ import annotations
import math
from typing import Optional, Dict, Any
import httpx
from fastapi import APIRouter, HTTPException, Request, Depends
from loguru import logger

from ml.fundamentals.quality_engine import fetch_quarterly_financials, estimate_wacc
from quantedge.scoring.edgar_fetch import fetch_edgar_supplement
from quantedge.scoring.hybrid_merge import merge_quarters
from quantedge.scoring.financial_features import compute_financial_features
from quantedge.scoring.business_features import compute_business_features
from quantedge.fundamentals.peer_fundamentals import compute_peer_fundamentals
from quantedge.fundamentals.edgar_bulk import company_facts_from_bulk
from quantedge.fundamentals.universe_full import ticker_cik_map
from quantedge.scoring.competitive_features import compute_competitive_features
from quantedge.scoring.compute import score_signal
from quantedge.scoring.cat_competitive_v6 import CATEGORIES, competitive_rating
from services.peer_store import PeerStore
from auth.cognito_auth import get_optional_user, CognitoUser
from core.config import settings

router = APIRouter()
_CIK_CACHE={}

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

def score_competitive(features):
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
    return {"label":"Competitive Intelligence","weight":8.0,"score":s,"confidence":c,"categories":cats}

async def compute_competitive_intelligence(ticker: str, api_key: str, pool=None) -> Dict[str,Any]:
    ticker=ticker.upper().strip()
    pq=await fetch_quarterly_financials(ticker, api_key, limit=12)
    if not pq: return {"ticker":ticker,"available":False,"reason":"no financial data"}
    ed=await fetch_edgar_supplement(ticker, years_back=4)
    merged=merge_quarters(pq, ed)
    if not merged or len(merged)<4: return {"ticker":ticker,"available":False,"reason":"insufficient history"}
    mcap=None; employees=None
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r=await c.get(f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={api_key}")
            if r.status_code==200:
                res=(r.json() or {}).get("results",{}); mcap=res.get("market_cap"); employees=res.get("total_employees")
    except Exception: pass
    wacc=estimate_wacc(beta=None)["mid"]
    fin=compute_financial_features(merged, market_cap=mcap, wacc=wacc)
    biz=compute_business_features(merged, fin, wacc=wacc) or {}
    # own fundamentals from bulk for consistency with peer factors
    own_fund={}
    global _CIK_CACHE
    if not _CIK_CACHE:
        try: _CIK_CACHE=ticker_cik_map()
        except Exception: _CIK_CACHE={}
    cik=_CIK_CACHE.get(ticker)
    if cik:
        try:
            facts=company_facts_from_bulk(cik)
            if facts: own_fund=compute_peer_fundamentals(facts, market_cap=mcap)
        except Exception: pass
    own={
      "market_cap":mcap,"employees":employees,"wacc":wacc,
      "revenue":fin.get("revenue"),
      "net_margin":own_fund.get("fund_net_margin") or fin.get("net_margin"),
      "gross_margin":own_fund.get("fund_gross_margin") or fin.get("gross_margin"),
      "roic":own_fund.get("fund_roic_approx") or fin.get("roic"),
      "roe":own_fund.get("fund_roe") or fin.get("roe"),
      "revenue_growth":own_fund.get("fund_revenue_growth") or fin.get("revenue_growth"),
      "pe":own_fund.get("fund_pe"),
      "asset_turnover":own_fund.get("fund_asset_turnover"),
      "ocf_margin":own_fund.get("fund_ocf_margin"),
      "earnings_growth":own_fund.get("fund_earnings_growth"),
      "current_ratio":own_fund.get("fund_current_ratio"),
      "gross_margin_stability":biz.get("gross_margin_stability"),
      "roe_stability":biz.get("roe_stability"),
    }
    peer_bucket=None
    if pool is not None:
        try: peer_bucket=await PeerStore(pool).get_peers(ticker)
        except Exception as e: logger.debug(f"peer bucket failed: {e}")
    feats=compute_competitive_features(own, peer_bucket)
    if feats.get("available") is False:
        return {"ticker":ticker,"available":False,"reason":"no peer data for competitive analysis"}
    tree=score_competitive(feats)
    n_scored=sum(c["n_scored"] for c in tree["categories"]); n_total=sum(c["n_signals"] for c in tree["categories"])
    return {"ticker":ticker,"available":True,"intelligence":"competitive",
            "score":tree["score"],"confidence":tree["confidence"],
            "competitive_rating":competitive_rating(tree["score"]),"weight_in_conviction":8.0,
            "bucket":feats.get("_bucket"),"peer_count":feats.get("_peer_count"),
            "coverage":{"scored":n_scored,"total":n_total},"tree":tree,
            "key_metrics":{k:feats.get(k) for k in
                ["scale_rank","market_share_proxy","net_margin_pctile","roic_pctile","growth_pctile",
                 "margin_advantage","economic_moat_spread","pe_discount_vs_peers","growth_advantage","gross_margin_level"]}}

@router.get("/competitive/{ticker}")
async def get_competitive(ticker: str, http_request: Request,
                          current_user: Optional[CognitoUser]=Depends(get_optional_user)):
    api_key=getattr(settings,"POLYGON_API_KEY","") or ""
    if not api_key: raise HTTPException(503,"data source unavailable")
    pool=getattr(http_request.app.state,"db",None)
    return {"data":_san(await compute_competitive_intelligence(ticker, api_key, pool))}
