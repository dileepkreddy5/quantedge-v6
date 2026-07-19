"""Business Intelligence endpoint — GET /api/v6/business/{ticker}.
Moat & durability scoring from real quantitative proxies. Reuses the Financial
data pipeline (merged + features) as single source. Qualitative signals honest
needs_source. Reusable for the conviction aggregator.
"""
from __future__ import annotations
import math
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request, Depends
from loguru import logger

from ml.fundamentals.quality_engine import fetch_quarterly_financials, estimate_wacc
from quantedge.scoring.edgar_fetch import fetch_edgar_supplement
from quantedge.scoring.hybrid_merge import merge_quarters
from quantedge.scoring.financial_features import compute_financial_features
from quantedge.scoring.business_features import compute_business_features
from quantedge.scoring.compute import score_signal
from quantedge.scoring.cat_business_v6 import CATEGORIES, moat_rating
from auth.cognito_auth import get_optional_user, CognitoUser
from core.config import settings

router = APIRouter()

def _san(o):
    if isinstance(o, float): return o if math.isfinite(o) else None
    if isinstance(o, dict): return {k: _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_san(v) for v in o]
    return o

def _roll(children):
    scored=[c for c in children if c.get("score") is not None]
    tw=sum(c["weight"] for c in children); aw=sum(c["weight"] for c in scored)
    if not scored or tw==0: return None,0.0
    return round(sum(c["score"]*c["weight"] for c in scored)/aw,1), round(aw/tw,3)

def score_business(features):
    cats=[]
    for cid,(label,wt,sigs) in CATEGORIES.items():
        scored=[]
        for spec in sigs:
            val=features.get(spec["field"])
            res=score_signal(val,spec,None)
            scored.append({"id":spec["id"],"label":spec["label"],"weight":spec["weight"],
                           "status":spec["status"],"evidence":spec["evidence"],
                           "raw_value":val,**res})
        cs,cc=_roll(scored)
        cats.append({"id":cid,"label":label,"weight":wt,"score":cs,"confidence":cc,
                     "n_signals":len(sigs),"n_scored":sum(1 for s in scored if s["score"] is not None),
                     "signals":scored})
    bs,bc=_roll(cats)
    return {"label":"Business Intelligence","weight":12.0,"score":bs,"confidence":bc,"categories":cats}

async def compute_business_intelligence(ticker: str, api_key: str) -> Dict[str, Any]:
    ticker=ticker.upper().strip()
    try:
        pq=await fetch_quarterly_financials(ticker,api_key,limit=24)
    except Exception:
        pq=[]
    if not pq:
        return {"ticker":ticker,"available":False,"reason":"no financial statements"}
    try:
        ed=await fetch_edgar_supplement(ticker,years_back=6)
    except Exception:
        ed={}
    merged=merge_quarters(pq,ed)
    wacc=estimate_wacc(beta=None)["mid"]
    # market cap for buyback-yield accuracy
    mcap=None
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as c:
            r=await c.get(f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={api_key}")
            if r.status_code==200:
                res=(r.json() or {}).get("results",{})
                mcap=res.get("market_cap")
    except Exception: pass
    fin_features=compute_financial_features(merged,market_cap=mcap,wacc=wacc)
    if mcap: fin_features["market_cap"]=mcap
    biz_features=compute_business_features(merged, fin_features, wacc=wacc)
    if not biz_features:
        return {"ticker":ticker,"available":False,"reason":"insufficient history for business analysis"}
    tree=score_business(biz_features)
    n_scored=sum(c["n_scored"] for c in tree["categories"])
    n_total=sum(c["n_signals"] for c in tree["categories"])
    return {"ticker":ticker,"available":True,"intelligence":"business",
            "score":tree["score"],"confidence":tree["confidence"],
            "moat_rating":moat_rating(tree["score"]),
            "weight_in_conviction":12.0,
            "coverage":{"scored":n_scored,"total":n_total},
            "tree":tree,
            "key_metrics":{k:biz_features.get(k) for k in
                ["excess_return_spread","roic_current","gross_margin_level","gross_margin_stability",
                 "recurring_revenue_ratio","operating_leverage","reinvestment_quality",
                 "revenue_consistency","capital_intensity","roe_stability"]}}

@router.get("/business/{ticker}")
async def get_business(ticker: str, http_request: Request,
                       current_user: Optional[CognitoUser]=Depends(get_optional_user)):
    api_key=getattr(settings,"POLYGON_API_KEY","") or ""
    if not api_key:
        raise HTTPException(503,"data source unavailable")
    result=await compute_business_intelligence(ticker, api_key)
    return {"data":_san(result)}
