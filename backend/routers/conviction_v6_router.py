"""Conviction v6 endpoint — the MASTER aggregator. Unifies all live intelligence
modules into ONE consolidated conviction score via the extensible registry.
Currently live: Financial (18%). Each future tab plugs in by adding a scorer +
flipping its registry status. Deterministic; results briefly cached.
"""
from __future__ import annotations
import math, time
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request, Depends
from loguru import logger

from quantedge.scoring.conviction_agg import aggregate_conviction, MODULE_REGISTRY
from auth.cognito_auth import get_optional_user, CognitoUser
from core.config import settings

router = APIRouter()

_CACHE: Dict[str, Any] = {}
_TTL = 900  # 15 min, matching platform cache window

def _san(o):
    if isinstance(o, float): return o if math.isfinite(o) else None
    if isinstance(o, dict): return {k: _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_san(v) for v in o]
    return o

async def _financial_scorer_factory(http_request: Request, api_key: str, user):
    """Returns an async scorer(ticker) -> {score, confidence, coverage} that reuses
    the Financial Intelligence pipeline (one source of truth)."""
    from routers.financial_router import get_financial
    async def scorer(ticker: str):
        res = await get_financial(ticker, http_request, user)
        d = res.get("data", {})
        if not d.get("available"): return None
        return {"score": d.get("score"), "confidence": d.get("confidence"),
                "coverage": d.get("coverage")}
    return scorer

@router.get("/conviction/{ticker}")
async def get_conviction(ticker: str, http_request: Request,
                         current_user: Optional[CognitoUser] = Depends(get_optional_user)):
    ticker = ticker.upper().strip()
    now = time.time()
    if ticker in _CACHE and now - _CACHE[ticker]["t"] < _TTL:
        return {"data": _CACHE[ticker]["v"], "cached": True}

    api_key = getattr(settings, "POLYGON_API_KEY", "") or ""
    if not api_key:
        raise HTTPException(503, "data source unavailable")

    async def _valuation_scorer(ticker: str):
        from routers.valuation_router import compute_valuation_intelligence
        d = await compute_valuation_intelligence(ticker, api_key)
        if not d.get("available"): return None
        return {"score": d.get("score"), "confidence": d.get("confidence"), "coverage": d.get("coverage")}
    async def _market_scorer(ticker: str):
        from routers.market_router import compute_market_intelligence
        pool=getattr(http_request.app.state,"db",None)
        d = await compute_market_intelligence(ticker, api_key, pool)
        if not d.get("available"): return None
        return {"score": d.get("score"), "confidence": d.get("confidence"), "coverage": d.get("coverage")}
    async def _altdata_scorer(ticker: str):
        from routers.altdata_router import compute_altdata_intelligence
        d = await compute_altdata_intelligence(ticker, api_key)
        if not d.get("available"): return None
        return {"score": d.get("score"), "confidence": d.get("confidence"), "coverage": d.get("coverage")}
    async def _forecast_scorer(ticker: str):
        from routers.forecast_router import compute_forecast_intelligence
        d = await compute_forecast_intelligence(ticker, api_key)
        if not d.get("available"): return None
        return {"score": d.get("score"), "confidence": d.get("confidence"), "coverage": d.get("coverage")}
    async def _macro_scorer(ticker: str):
        from routers.macro_router import compute_macro_intelligence
        d = await compute_macro_intelligence(ticker, api_key)
        if not d.get("available"): return None
        return {"score": d.get("score"), "confidence": d.get("confidence"), "coverage": d.get("coverage")}
    async def _ownership_scorer(ticker: str):
        from routers.ownership_router import compute_ownership_intelligence
        d = await compute_ownership_intelligence(ticker, api_key)
        if not d.get("available"): return None
        return {"score": d.get("score"), "confidence": d.get("confidence"), "coverage": d.get("coverage")}
    async def _management_scorer(ticker: str):
        from routers.management_router import compute_management_intelligence
        d = await compute_management_intelligence(ticker, api_key)
        if not d.get("available"): return None
        return {"score": d.get("score"), "confidence": d.get("confidence"), "coverage": d.get("coverage")}
    async def _competitive_scorer(ticker: str):
        from routers.competitive_router import compute_competitive_intelligence
        pool = getattr(http_request.app.state, "db", None)
        d = await compute_competitive_intelligence(ticker, api_key, pool)
        if not d.get("available"): return None
        return {"score": d.get("score"), "confidence": d.get("confidence"), "coverage": d.get("coverage")}
    async def _industry_scorer(ticker: str):
        from routers.industry_router import compute_industry_intelligence
        pool = getattr(http_request.app.state, "db", None)
        d = await compute_industry_intelligence(ticker, api_key, pool)
        if not d.get("available"): return None
        return {"score": d.get("score"), "confidence": d.get("confidence"), "coverage": d.get("coverage")}
    async def _risk_scorer(ticker: str):
        from routers.risk_router import compute_risk_intelligence
        d = await compute_risk_intelligence(ticker, api_key)
        if not d.get("available"): return None
        return {"score": d.get("score"), "confidence": d.get("confidence"), "coverage": d.get("coverage")}
    async def _news_scorer(ticker: str):
        from routers.news_router import compute_news_intelligence
        d = await compute_news_intelligence(ticker, api_key)
        if not d.get("available"): return None
        return {"score": d.get("score"), "confidence": d.get("confidence"), "coverage": d.get("coverage")}
    async def _business_scorer(ticker: str):
        from routers.business_router import compute_business_intelligence
        d = await compute_business_intelligence(ticker, api_key)
        if not d.get("available"): return None
        return {"score": d.get("score"), "confidence": d.get("confidence"), "coverage": d.get("coverage")}
    scorers = {
        "financial": await _financial_scorer_factory(http_request, api_key, current_user),
        "valuation": _valuation_scorer,
        "market": _market_scorer,
        "business": _business_scorer,
        "news": _news_scorer,
        "risk": _risk_scorer,
        "industry": _industry_scorer,
        "competitive": _competitive_scorer,
        "management": _management_scorer,
        "ownership": _ownership_scorer,
        "macro": _macro_scorer,
        "forecast": _forecast_scorer,
        "alt_data": _altdata_scorer,
    }
    result = await aggregate_conviction(ticker, scorers)
    result = _san(result)
    _CACHE[ticker] = {"t": now, "v": result}
    return {"data": result, "cached": False}
