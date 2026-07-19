"""Market Intelligence endpoint — GET /api/v6/market/{ticker}.
Peer-relative scoring of momentum/trend/liquidity from the pre-computed peer_stats
bucketed factors, plus a fresh GARCH/HMM/Kalman regime read from price history.
Single ownership: price-based signals only. Reusable for the conviction aggregator.
"""
from __future__ import annotations
import math, datetime as dt
from typing import Optional, Dict, Any, List
import httpx
from fastapi import APIRouter, HTTPException, Request, Depends
from loguru import logger

from quantedge.scoring.market_features import score_market, market_rating
from auth.cognito_auth import get_optional_user, CognitoUser
from core.config import settings

router = APIRouter()
_POLY = "https://api.polygon.io"

def _san(o):
    if isinstance(o, float): return o if math.isfinite(o) else None
    if isinstance(o, dict): return {k: _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_san(v) for v in o]
    return o

async def _regime_read(ticker: str, api_key: str) -> Dict[str, Any]:
    """Fresh GARCH volatility + HMM regime from ~1yr daily returns (reuse real engines)."""
    out={}
    try:
        import pandas as pd, numpy as np
        end=dt.date.today(); start=end-dt.timedelta(days=420)
        async with httpx.AsyncClient(timeout=15) as c:
            u=f"{_POLY}/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}?adjusted=true&sort=asc&limit=500&apiKey={api_key}"
            r=await c.get(u)
            if r.status_code!=200: return out
            bars=(r.json() or {}).get("results",[])
        if len(bars)<60: return out
        closes=pd.Series([b["c"] for b in bars])
        volume=pd.Series([b.get("v",0) for b in bars])
        returns=closes.pct_change().dropna()
        try:
            from ml.models.regime_volatility import GJRGARCHModel, HMMRegimeClassifier, KalmanTrendFilter
            g=GJRGARCHModel(); gr=g.fit(returns)
            out["garch"]={"current_vol":gr.get("current_vol") or gr.get("annualized_vol"),
                          "vol_regime":gr.get("vol_regime") or gr.get("regime")}
            h=HMMRegimeClassifier(); h.fit(returns, volume)
            hr=h.predict_current_regime(returns, volume)
            out["regime"]={"current":hr.get("current_regime"),"confidence":hr.get("confidence") or hr.get("probability")}
            k=KalmanTrendFilter(); kr=k.fit(returns)
            out["kalman"]={"trend":kr.get("trend") or kr.get("trend_direction"),"state":kr.get("state")}
        except Exception as e:
            logger.info(f"market regime engines: {e}")
    except Exception as e:
        logger.info(f"market regime read failed {ticker}: {e}")
    return out

async def compute_market_intelligence(ticker: str, api_key: str, pool=None) -> Dict[str, Any]:
    ticker=ticker.upper().strip()
    me_factors=None; peer_list=[]; bucket=None
    if pool is not None:
        try:
            from services.peer_store import PeerStore
            pdata=await PeerStore(pool).get_peers(ticker)
            if pdata.get("available"):
                bucket=pdata.get("bucket")
                me_row=pdata.get("me",{})
                me_factors=me_row.get("factors") or {}
                if isinstance(me_factors,str):
                    import json; me_factors=json.loads(me_factors)
                for row in pdata.get("peers",[]):
                    pf=row.get("factors") or {}
                    if isinstance(pf,str):
                        import json; pf=json.loads(pf)
                    peer_list.append(pf)
        except Exception as e:
            logger.info(f"market: peers unavailable {ticker}: {e}")
    if not me_factors:
        return {"ticker":ticker,"available":False,"reason":"ticker not in peer universe (run the peer scan)"}
    tree=score_market(me_factors, peer_list)
    regime=await _regime_read(ticker, api_key)
    n_scored=sum(c["n_scored"] for c in tree["categories"])
    n_total=sum(c["n_signals"] for c in tree["categories"])
    # momentum ladder for the frontend
    ladder={k:me_factors.get(k) for k in ["mom_1m","mom_3m","mom_6m","mom_12_1"]}
    return {"ticker":ticker,"available":True,"intelligence":"market",
            "score":tree["score"],"confidence":tree["confidence"],
            "market_rating":market_rating(tree["score"]),
            "weight_in_conviction":5.0,"sector_bucket":bucket,
            "peer_count":len(peer_list),
            "coverage":{"scored":n_scored,"total":n_total},
            "tree":tree,"regime":regime,"momentum_ladder":ladder,
            "key_metrics":{"hurst":me_factors.get("hurst"),"sharpe_3m":me_factors.get("sharpe_3m"),
                "ma_alignment":me_factors.get("ma_alignment"),"amihud":me_factors.get("amihud"),
                "pct_above_ma50":me_factors.get("pct_above_ma50"),"pct_above_ma200":me_factors.get("pct_above_ma200")}}

@router.get("/market/{ticker}")
async def get_market(ticker: str, http_request: Request,
                     current_user: Optional[CognitoUser]=Depends(get_optional_user)):
    api_key=getattr(settings,"POLYGON_API_KEY","") or ""
    pool=getattr(http_request.app.state,"db_pool",None)
    result=await compute_market_intelligence(ticker, api_key, pool)
    return {"data":_san(result)}
