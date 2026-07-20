"""Institutional Flow endpoint — GET /api/v6/iflow/{ticker}."""
from __future__ import annotations
import math, datetime as dt
from typing import Optional, Dict, Any
import httpx
from fastapi import APIRouter, HTTPException, Request, Depends
from quantedge.scoring.insider_fetch import fetch_insider_activity
from quantedge.scoring.ownership_fetch import fetch_ownership
from quantedge.fundamentals.universe_full import ticker_cik_map
from quantedge.scoring.iflow_features import compute_iflow_features
from quantedge.scoring.compute import score_signal
from quantedge.scoring.cat_iflow_v6 import CATEGORIES, iflow_rating
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

def score_iflow(features):
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
    return {"label":"Institutional Flow","weight":2.0,"score":s,"confidence":c,"categories":cats}

async def _bars(ticker, api_key, days=90):
    try:
        end=dt.date.today(); start=end-dt.timedelta(days=days)
        async with httpx.AsyncClient(timeout=12) as c:
            u=f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}?adjusted=true&sort=asc&limit=120&apiKey={api_key}"
            r=await c.get(u)
            if r.status_code==200: return (r.json() or {}).get("results",[])
    except Exception: pass
    return []

async def compute_iflow_intelligence(ticker: str, api_key: str) -> Dict[str,Any]:
    ticker=ticker.upper().strip()
    bars=await _bars(ticker, api_key)
    if not bars or len(bars)<30:
        return {"ticker":ticker,"available":False,"reason":"insufficient price history"}
    global _CIK
    if not _CIK:
        try: _CIK=ticker_cik_map()
        except Exception: _CIK={}
    insider={}; ownership={}
    cik=_CIK.get(ticker)
    if cik:
        try: insider=await fetch_insider_activity(cik, days_back=180, max_filings=30)
        except Exception: pass
        try: ownership=await fetch_ownership(cik)
        except Exception: pass
    feats=compute_iflow_features(bars, insider=insider, ownership=ownership)
    if feats.get("available") is False:
        return {"ticker":ticker,"available":False,"reason":"could not compute flow signals"}
    tree=score_iflow(feats)
    n_scored=sum(c["n_scored"] for c in tree["categories"]); n_total=sum(c["n_signals"] for c in tree["categories"])
    return {"ticker":ticker,"available":True,"intelligence":"iflow",
            "score":tree["score"],"confidence":tree["confidence"],
            "iflow_rating":iflow_rating(tree["score"]),"weight_in_conviction":2.0,
            "coverage":{"scored":n_scored,"total":n_total},"tree":tree,
            "key_metrics":{k:feats.get(k) for k in
                ["money_flow_index","chaikin_money_flow","adl_slope","accumulation_20d","dollar_flow_momentum",
                 "avg_trade_size_trend","block_trade_frequency","recent_13g_filings","insider_net_flow","institutional_footprint"]}}

@router.get("/iflow/{ticker}")
async def get_iflow(ticker: str, http_request: Request,
                    current_user: Optional[CognitoUser]=Depends(get_optional_user)):
    api_key=getattr(settings,"POLYGON_API_KEY","") or ""
    if not api_key: raise HTTPException(503,"data source unavailable")
    return {"data":_san(await compute_iflow_intelligence(ticker, api_key))}
