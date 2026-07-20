"""Ownership Intelligence endpoint — GET /api/v6/ownership/{ticker}."""
from __future__ import annotations
import math, datetime as dt
from typing import Optional, Dict, Any
import httpx
from fastapi import APIRouter, HTTPException, Request, Depends
from loguru import logger

from ml.fundamentals.quality_engine import fetch_quarterly_financials
from quantedge.scoring.edgar_fetch import fetch_edgar_supplement
from quantedge.scoring.hybrid_merge import merge_quarters
from quantedge.scoring.insider_fetch import fetch_insider_activity
from quantedge.scoring.ownership_fetch import fetch_ownership
from quantedge.fundamentals.universe_full import ticker_cik_map
from quantedge.scoring.ownership_features import compute_ownership_features
from quantedge.scoring.compute import score_signal
from quantedge.scoring.cat_ownership_v6 import CATEGORIES, ownership_rating
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

def score_ownership(features):
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
    return {"label":"Ownership Intelligence","weight":4.0,"score":s,"confidence":c,"categories":cats}

async def _avg_volume(ticker, api_key):
    try:
        end=dt.date.today(); start=end-dt.timedelta(days=40)
        async with httpx.AsyncClient(timeout=12) as c:
            u=f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}?adjusted=true&limit=50&apiKey={api_key}"
            r=await c.get(u)
            if r.status_code==200:
                vs=[b.get("v") for b in (r.json() or {}).get("results",[]) if b.get("v")]
                return sum(vs)/len(vs) if vs else None
    except Exception: pass
    return None

async def compute_ownership_intelligence(ticker: str, api_key: str) -> Dict[str,Any]:
    ticker=ticker.upper().strip()
    pq=await fetch_quarterly_financials(ticker, api_key, limit=12)
    if not pq: return {"ticker":ticker,"available":False,"reason":"no financial data"}
    ed=await fetch_edgar_supplement(ticker, years_back=4)
    merged=merge_quarters(pq, ed)
    if not merged or len(merged)<4: return {"ticker":ticker,"available":False,"reason":"insufficient history"}
    mcap=None; shares_out=None
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r=await c.get(f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={api_key}")
            if r.status_code==200:
                res=(r.json() or {}).get("results",{}); mcap=res.get("market_cap")
                shares_out=res.get("weighted_shares_outstanding") or res.get("share_class_shares_outstanding")
    except Exception: pass
    global _CIK
    if not _CIK:
        try: _CIK=ticker_cik_map()
        except Exception: _CIK={}
    cik=_CIK.get(ticker)
    insider={}; ownership={}
    if cik:
        try: insider=await fetch_insider_activity(cik)
        except Exception: pass
        try: ownership=await fetch_ownership(cik)
        except Exception: pass
    avgvol=await _avg_volume(ticker, api_key)
    feats=compute_ownership_features(merged, shares_out=shares_out, market_cap=mcap,
                                      insider=insider, ownership=ownership, avg_volume=avgvol)
    if feats.get("available") is False:
        return {"ticker":ticker,"available":False,"reason":"insufficient data"}
    tree=score_ownership(feats)
    n_scored=sum(c["n_scored"] for c in tree["categories"]); n_total=sum(c["n_signals"] for c in tree["categories"])
    return {"ticker":ticker,"available":True,"intelligence":"ownership",
            "score":tree["score"],"confidence":tree["confidence"],
            "ownership_rating":ownership_rating(tree["score"]),"weight_in_conviction":4.0,
            "insider_available":insider.get("available",False),"institutional_available":ownership.get("available",False),
            "top_holders":ownership.get("holders",[])[:5] if ownership.get("available") else [],
            "coverage":{"scored":n_scored,"total":n_total},"tree":tree,
            "key_metrics":{k:feats.get(k) for k in
                ["share_count_trend","insider_net_conviction","major_holder_count","top_holder_pct",
                 "dilution_pressure","buyback_intensity","ownership_conviction","insider_buy_ratio","float_liquidity","insider_cluster"]}}

@router.get("/ownership/{ticker}")
async def get_ownership(ticker: str, http_request: Request,
                        current_user: Optional[CognitoUser]=Depends(get_optional_user)):
    api_key=getattr(settings,"POLYGON_API_KEY","") or ""
    if not api_key: raise HTTPException(503,"data source unavailable")
    return {"data":_san(await compute_ownership_intelligence(ticker, api_key))}
