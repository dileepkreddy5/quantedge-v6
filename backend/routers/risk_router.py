"""Risk Intelligence endpoint — GET /api/v6/risk/{ticker}.
Credit/forensic risk models + leverage/liquidity/tail/earnings-quality risk.
High score = Low Risk (signals oriented so safe=high). Reusable for conviction."""
from __future__ import annotations
import math, datetime as dt
from typing import Optional, Dict, Any, List
import httpx
from fastapi import APIRouter, HTTPException, Request, Depends
from loguru import logger

from ml.fundamentals.quality_engine import fetch_quarterly_financials, estimate_wacc
from quantedge.scoring.edgar_fetch import fetch_edgar_supplement
from quantedge.scoring.hybrid_merge import merge_quarters
from quantedge.scoring.risk_features import compute_risk_features
from quantedge.scoring.compute import score_signal
from quantedge.scoring.cat_risk_v6 import CATEGORIES, risk_rating
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

def score_risk(features):
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
    rs,rc=_roll(cats)
    return {"label":"Risk Intelligence","weight":6.0,"score":rs,"confidence":rc,"categories":cats}

async def _price_closes(ticker, api_key, days=400):
    try:
        end=dt.date.today(); start=end-dt.timedelta(days=days)
        async with httpx.AsyncClient(timeout=15) as c:
            u=f"{_POLY}/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}?adjusted=true&sort=asc&limit=500&apiKey={api_key}"
            r=await c.get(u)
            if r.status_code==200:
                return [b["c"] for b in (r.json() or {}).get("results",[])]
    except Exception: pass
    return []

async def _market_cap_beta(ticker, api_key):
    mcap=None; beta=None
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r=await c.get(f"{_POLY}/v3/reference/tickers/{ticker}?apiKey={api_key}")
            if r.status_code==200: mcap=((r.json() or {}).get("results",{}) or {}).get("market_cap")
    except Exception: pass
    return mcap, beta

async def compute_risk_intelligence(ticker: str, api_key: str) -> Dict[str,Any]:
    ticker=ticker.upper().strip()
    pq=await fetch_quarterly_financials(ticker, api_key, limit=12)
    if not pq: return {"ticker":ticker,"available":False,"reason":"no financial data"}
    ed=await fetch_edgar_supplement(ticker, years_back=4)
    merged=merge_quarters(pq, ed)
    if not merged or len(merged)<4:
        return {"ticker":ticker,"available":False,"reason":"insufficient financial history"}
    closes=await _price_closes(ticker, api_key)
    mcap, beta=await _market_cap_beta(ticker, api_key)
    if beta is None:
        try: beta=estimate_wacc(beta=None).get("beta_used")
        except Exception: beta=None
    feats=compute_risk_features(merged, price_closes=closes, market_cap=mcap, beta=beta)
    if feats.get("available") is False:
        return {"ticker":ticker,"available":False,"reason":"could not compute risk metrics"}
    tree=score_risk(feats)
    # extreme-valuation override: a debt-free co at 100x+ P/E still carries high investor risk
    pe=feats.get("pe_ratio")
    rating_score=tree["score"]
    if pe is not None and rating_score is not None:
        if pe>=100 and rating_score>72: rating_score=min(rating_score,70)   # cap below Low Risk
        elif pe>=50 and rating_score>78: rating_score=min(rating_score,76)
    n_scored=sum(c["n_scored"] for c in tree["categories"])
    n_total=sum(c["n_signals"] for c in tree["categories"])
    return {"ticker":ticker,"available":True,"intelligence":"risk",
            "score":tree["score"],"confidence":tree["confidence"],
            "risk_rating":risk_rating(rating_score),"weight_in_conviction":6.0,
            "coverage":{"scored":n_scored,"total":n_total},"tree":tree,
            "key_metrics":{k:feats.get(k) for k in
                ["altman_z","bankruptcy_prob","net_debt_to_ebitda","current_ratio",
                 "sloan_accruals","max_drawdown","annualized_vol","beta","pe_ratio","share_dilution"]}}

@router.get("/risk/{ticker}")
async def get_risk(ticker: str, http_request: Request,
                   current_user: Optional[CognitoUser]=Depends(get_optional_user)):
    api_key=getattr(settings,"POLYGON_API_KEY","") or ""
    if not api_key: raise HTTPException(503,"data source unavailable")
    result=await compute_risk_intelligence(ticker, api_key)
    return {"data":_san(result)}
