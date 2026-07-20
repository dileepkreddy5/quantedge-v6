"""Macro Sensitivity endpoint — GET /api/v6/macro/{ticker}."""
from __future__ import annotations
import math, datetime as dt, asyncio
from typing import Optional, Dict, Any
import httpx
from fastapi import APIRouter, HTTPException, Request, Depends
from quantedge.scoring.macro_features import compute_macro_features
from quantedge.scoring.compute import score_signal
from quantedge.scoring.cat_macro_v6 import CATEGORIES, macro_rating
from auth.cognito_auth import get_optional_user, CognitoUser
from core.config import settings

router = APIRouter()
_PROXIES=["TLT","UUP","GLD","HYG","USO","SPY","IWM","VLUE","MTUM"]

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

def score_macro(features):
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
    return {"label":"Macro Sensitivity","weight":3.0,"score":s,"confidence":c,"categories":cats}

async def _closes(ticker, api_key, days=400):
    try:
        end=dt.date.today(); start=end-dt.timedelta(days=days)
        async with httpx.AsyncClient(timeout=12) as c:
            u=f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}?adjusted=true&sort=asc&limit=500&apiKey={api_key}"
            r=await c.get(u)
            if r.status_code==200: return [b["c"] for b in (r.json() or {}).get("results",[])]
    except Exception: pass
    return []

async def compute_macro_intelligence(ticker: str, api_key: str) -> Dict[str,Any]:
    ticker=ticker.upper().strip()
    stock=await _closes(ticker, api_key)
    if not stock or len(stock)<60:
        return {"ticker":ticker,"available":False,"reason":"insufficient price history"}
    proxy_closes=await asyncio.gather(*[_closes(p, api_key) for p in _PROXIES])
    proxies={name:cl for name,cl in zip(_PROXIES,proxy_closes)}
    feats=compute_macro_features(stock, proxies)
    if feats.get("available") is False:
        return {"ticker":ticker,"available":False,"reason":"could not compute macro exposures"}
    tree=score_macro(feats)
    n_scored=sum(c["n_scored"] for c in tree["categories"]); n_total=sum(c["n_signals"] for c in tree["categories"])
    return {"ticker":ticker,"available":True,"intelligence":"macro",
            "score":tree["score"],"confidence":tree["confidence"],
            "macro_rating":macro_rating(tree["score"]),"weight_in_conviction":3.0,
            "coverage":{"scored":n_scored,"total":n_total},"tree":tree,
            "key_metrics":{k:feats.get(k) for k in
                ["rate_beta","dollar_beta","inflation_hedge","market_beta","credit_beta",
                 "oil_beta","value_tilt","momentum_tilt","macro_resilience","defensiveness"]}}

@router.get("/macro/{ticker}")
async def get_macro(ticker: str, http_request: Request,
                    current_user: Optional[CognitoUser]=Depends(get_optional_user)):
    api_key=getattr(settings,"POLYGON_API_KEY","") or ""
    if not api_key: raise HTTPException(503,"data source unavailable")
    return {"data":_san(await compute_macro_intelligence(ticker, api_key))}
