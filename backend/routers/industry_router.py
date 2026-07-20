"""Industry Intelligence endpoint — GET /api/v6/industry/{ticker}.
Sector classification + sector-relative performance + industry position (peer-bucket
percentiles). From SIC + sector ETF + SPY prices + peer_stats. Reusable for conviction."""
from __future__ import annotations
import math, datetime as dt
from typing import Optional, Dict, Any
import httpx
from fastapi import APIRouter, HTTPException, Request, Depends
from loguru import logger

from quantedge.scoring.industry_features import compute_industry_features, sic_to_sector_etf
from quantedge.scoring.compute import score_signal
from quantedge.scoring.cat_industry_v6 import CATEGORIES, industry_rating
from services.peer_store import PeerStore
from auth.cognito_auth import get_optional_user, CognitoUser
from core.config import settings

router = APIRouter()
_POLY="https://api.polygon.io"

def _san(o):
    if isinstance(o,float): return o if math.isfinite(o) else None
    if isinstance(o,dict): return {k:_san(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [_san(v) for v in o]
    return o

def _roll(children):
    scored=[c for c in children if c.get("score") is not None]
    tw=sum(c["weight"] for c in children); aw=sum(c["weight"] for c in scored)
    if not scored or tw==0: return None,0.0
    return round(sum(c["score"]*c["weight"] for c in scored)/aw,1), round(aw/tw,3)

def score_industry(features):
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
                     "n_signals":len(sigs),"n_scored":sum(1 for s in scored if s["score"] is not None),
                     "signals":scored})
    isc,icc=_roll(cats)
    return {"label":"Industry Intelligence","weight":6.0,"score":isc,"confidence":icc,"categories":cats}

async def _closes(ticker, api_key, days=400):
    try:
        end=dt.date.today(); start=end-dt.timedelta(days=days)
        async with httpx.AsyncClient(timeout=15) as c:
            u=f"{_POLY}/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}?adjusted=true&sort=asc&limit=500&apiKey={api_key}"
            r=await c.get(u)
            if r.status_code==200: return [b["c"] for b in (r.json() or {}).get("results",[])]
    except Exception: pass
    return []

async def _details(ticker, api_key):
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r=await c.get(f"{_POLY}/v3/reference/tickers/{ticker}?apiKey={api_key}")
            if r.status_code==200: return (r.json() or {}).get("results",{}) or {}
    except Exception: pass
    return {}

async def compute_industry_intelligence(ticker: str, api_key: str, pool=None) -> Dict[str,Any]:
    ticker=ticker.upper().strip()
    det=await _details(ticker, api_key)
    sic=det.get("sic_code")
    etf,sector_name=sic_to_sector_etf(sic)
    stock=await _closes(ticker, api_key)
    sector=await _closes(etf, api_key) if etf else []
    spy=await _closes("SPY", api_key)
    if not stock or len(stock)<60:
        return {"ticker":ticker,"available":False,"reason":"insufficient price history"}
    peer_bucket=None
    if pool is not None:
        try: peer_bucket=await PeerStore(pool).get_peers(ticker)
        except Exception as e: logger.debug(f"peer bucket failed: {e}")
    feats=compute_industry_features(sic, stock, sector, spy,
        market_cap=det.get("market_cap"), employees=det.get("total_employees"),
        list_date=det.get("list_date"), peer_bucket=peer_bucket)
    tree=score_industry(feats)
    n_scored=sum(c["n_scored"] for c in tree["categories"])
    n_total=sum(c["n_signals"] for c in tree["categories"])
    return {"ticker":ticker,"available":True,"intelligence":"industry",
            "score":tree["score"],"confidence":tree["confidence"],
            "industry_rating":industry_rating(tree["score"]),"weight_in_conviction":6.0,
            "sector_name":feats.get("_sector_name"),"sector_etf":feats.get("_sector_etf"),
            "sic":sic,"sic_description":det.get("sic_description"),
            "coverage":{"scored":n_scored,"total":n_total},"tree":tree,
            "key_metrics":{k:feats.get(k) for k in
                ["rs_sector_3m","rs_sector_1y","composite_sector_rank","sector_trend_3m",
                 "sector_vs_spy_3m","beta_to_sector","sector_in_favor","years_public","market_cap_b","sector_peer_count"]}}

@router.get("/industry/{ticker}")
async def get_industry(ticker: str, http_request: Request,
                       current_user: Optional[CognitoUser]=Depends(get_optional_user)):
    api_key=getattr(settings,"POLYGON_API_KEY","") or ""
    if not api_key: raise HTTPException(503,"data source unavailable")
    pool=getattr(http_request.app.state,"db",None)
    result=await compute_industry_intelligence(ticker, api_key, pool)
    return {"data":_san(result)}
