"""Management Intelligence endpoint — GET /api/v6/management/{ticker}."""
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
from quantedge.scoring.insider_fetch import fetch_insider_activity
from quantedge.fundamentals.universe_full import ticker_cik_map
from quantedge.scoring.management_features import compute_management_features
from quantedge.scoring.compute import score_signal
from quantedge.scoring.cat_management_v6 import CATEGORIES, management_rating
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

def score_management(features):
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
    return {"label":"Management Intelligence","weight":6.0,"score":s,"confidence":c,"categories":cats}

async def compute_management_intelligence(ticker: str, api_key: str) -> Dict[str,Any]:
    ticker=ticker.upper().strip()
    pq=await fetch_quarterly_financials(ticker, api_key, limit=12)
    if not pq: return {"ticker":ticker,"available":False,"reason":"no financial data"}
    ed=await fetch_edgar_supplement(ticker, years_back=4)
    merged=merge_quarters(pq, ed)
    if not merged or len(merged)<4: return {"ticker":ticker,"available":False,"reason":"insufficient history"}
    mcap=None
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r=await c.get(f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={api_key}")
            if r.status_code==200: mcap=((r.json() or {}).get("results",{}) or {}).get("market_cap")
    except Exception: pass
    wacc=estimate_wacc(beta=None)["mid"]
    fin=compute_financial_features(merged, market_cap=mcap, wacc=wacc)
    # insider activity from Form 4
    global _CIK
    if not _CIK:
        try: _CIK=ticker_cik_map()
        except Exception: _CIK={}
    insider={}
    cik=_CIK.get(ticker)
    if cik:
        try: insider=await fetch_insider_activity(cik)
        except Exception: pass
    feats=compute_management_features(merged, fin, insider=insider, market_cap=mcap)
    if feats.get("available") is False:
        return {"ticker":ticker,"available":False,"reason":"insufficient data"}
    tree=score_management(feats)
    n_scored=sum(c["n_scored"] for c in tree["categories"]); n_total=sum(c["n_signals"] for c in tree["categories"])
    return {"ticker":ticker,"available":True,"intelligence":"management",
            "score":tree["score"],"confidence":tree["confidence"],
            "management_rating":management_rating(tree["score"]),"weight_in_conviction":6.0,
            "insider_available":insider.get("available",False),
            "coverage":{"scored":n_scored,"total":n_total},"tree":tree,
            "key_metrics":{k:feats.get(k) for k in
                ["roic_level","fcf_generation","total_payout_yield","insider_buy_value_ratio",
                 "insider_net_value_norm","margin_trend","share_count_change","cash_conversion","dividend_growth","insider_cluster_buying"]}}

@router.get("/management/{ticker}")
async def get_management(ticker: str, http_request: Request,
                         current_user: Optional[CognitoUser]=Depends(get_optional_user)):
    api_key=getattr(settings,"POLYGON_API_KEY","") or ""
    if not api_key: raise HTTPException(503,"data source unavailable")
    return {"data":_san(await compute_management_intelligence(ticker, api_key))}
