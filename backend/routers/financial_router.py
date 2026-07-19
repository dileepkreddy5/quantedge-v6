"""Financial Intelligence endpoint — GET /api/v6/financial/{ticker}.
Polygon income+balance + EDGAR cashflow detail -> merge -> 66 real features ->
score 12-category catalog -> rolled-up tree. Peer percentiles when available.
"""
from __future__ import annotations
import math
from typing import Optional, Dict, List, Any
from fastapi import APIRouter, HTTPException, Request, Depends
from loguru import logger

from ml.fundamentals.quality_engine import fetch_quarterly_financials, estimate_wacc
from quantedge.scoring.edgar_fetch import fetch_edgar_supplement
from quantedge.scoring.hybrid_merge import merge_quarters
from quantedge.scoring.financial_features import compute_financial_features
from quantedge.scoring.compute import score_signal
from quantedge.scoring.cat_financial_v6 import CATEGORIES
from auth.cognito_auth import get_optional_user, CognitoUser
from core.config import settings

router = APIRouter()

import httpx as _httpx
_POLY = "https://api.polygon.io"

async def _get_market_cap(ticker: str, api_key: str):
    """Market cap: Polygon\'s field if present, else shares x latest price.
    Real primitives only — never faked."""
    try:
        async with _httpx.AsyncClient(timeout=12) as c:
            r = await c.get(f"{_POLY}/v3/reference/tickers/{ticker}?apiKey={api_key}")
            if r.status_code != 200:
                return None
            res = (r.json() or {}).get("results", {}) or {}
            mc = res.get("market_cap")
            if mc:
                return float(mc)
            shares = (res.get("weighted_shares_outstanding")
                      or res.get("share_class_shares_outstanding"))
            if not shares:
                return None
            pr = await c.get(f"{_POLY}/v2/aggs/ticker/{ticker}/prev?apiKey={api_key}")
            if pr.status_code != 200:
                return None
            results = (pr.json() or {}).get("results", [])
            if not results:
                return None
            close = results[0].get("c")
            return float(shares) * float(close) if close else None
    except Exception:
        return None

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

def score_financial(features, peers=None):
    peers=peers or {}
    cats=[]
    for cid,(label,wt,sigs) in CATEGORIES.items():
        scored=[]
        for spec in sigs:
            val=features.get(spec["field"])
            pv=peers.get(spec.get("peer_key")) if spec.get("peer_key") else None
            res=score_signal(val,spec,pv)
            scored.append({"id":spec["id"],"label":spec["label"],"weight":spec["weight"],
                           "status":spec["status"],"evidence":spec["evidence"],
                           "raw_value":val,**res})
        cs,cc=_roll(scored)
        cats.append({"id":cid,"label":label,"weight":wt,"score":cs,"confidence":cc,
                     "n_signals":len(sigs),"n_scored":sum(1 for s in scored if s["score"] is not None),
                     "signals":scored})
    fs,fc=_roll(cats)
    return {"label":"Financial Intelligence","weight":18.0,"score":fs,"confidence":fc,"categories":cats}

@router.get("/financial/{ticker}")
async def get_financial(ticker: str, http_request: Request,
                        current_user: Optional[CognitoUser]=Depends(get_optional_user)):
    ticker=ticker.upper().strip()
    api_key=getattr(settings,"POLYGON_API_KEY","") or ""
    if not api_key:
        raise HTTPException(503,"data source unavailable")
    try:
        pq=await fetch_quarterly_financials(ticker,api_key,limit=24)
    except Exception as e:
        logger.warning(f"financial: polygon fetch failed {ticker}: {e}"); pq=[]
    if not pq:
        return {"data":{"ticker":ticker,"available":False,"reason":"no financial statements"}}
    try:
        ed=await fetch_edgar_supplement(ticker,years_back=6)
    except Exception as e:
        logger.info(f"financial: edgar unavailable {ticker}: {e}"); ed={}
    merged=merge_quarters(pq,ed)

    market_cap = await _get_market_cap(ticker, api_key)
    wacc=estimate_wacc(beta=None)["mid"]
    feats=compute_financial_features(merged,market_cap=market_cap,wacc=wacc)

    peers={}
    pool=getattr(http_request.app.state,"db_pool",None)
    if pool is not None:
        try:
            from services.peer_store import PeerStore
            pdata=await PeerStore(pool).get_peers(ticker)
            if pdata.get("available"):
                fl={}
                for row in pdata.get("peers",[]):
                    for k,v in (row.get("factors") or {}).items():
                        fl.setdefault(k,[]).append(v)
                peers=fl
        except Exception as e:
            logger.info(f"financial: peers unavailable {ticker}: {e}")

    tree=score_financial(feats,peers)
    n_scored=sum(c["n_scored"] for c in tree["categories"])
    n_total=sum(c["n_signals"] for c in tree["categories"])
    result={"ticker":ticker,"available":True,"intelligence":"financial",
            "score":tree["score"],"confidence":tree["confidence"],
            "weight_in_conviction":18.0,
            "coverage":{"scored":n_scored,"total":n_total},
            "wacc_used":round(wacc,4),"market_cap":market_cap,
            "n_quarters":len(merged),"tree":tree,
            "key_metrics":{k:feats.get(k) for k in
                ["roic","roic_ex_goodwill","roic_wacc_spread","fcf_margin","owner_earnings",
                 "piotroski_f","altman_z","beneish_m","cash_conversion_cycle","shareholder_yield"]}}
    return {"data":_san(result)}
