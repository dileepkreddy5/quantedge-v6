"""
QuantEdge v6.0 — Portfolio Sizer Router
========================================
Exposes /api/v6/portfolio/size — takes ranked tickers and returns
dollar-weighted position sizes.
"""

import json
import asyncio
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Request, Query, Body
from pydantic import BaseModel, Field
from loguru import logger

from research.position_sizer import PositionSizer
from research.regime_overlay import RegimeDetector
from auth.cognito_auth import get_optional_user, CognitoUser

router = APIRouter()


class SizeRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=1, max_length=30)
    total_capital: float = Field(..., gt=0, le=10_000_000)
    method: str = Field("hrp", pattern="^(equal|inverse_vol|ivp|erc|hrp)$")
    target_vol: float = Field(0.12, ge=0.05, le=0.40)
    max_position: float = Field(0.10, ge=0.02, le=0.50)
    min_position: float = Field(0.01, ge=0.005, le=0.05)
    apply_regime: bool = True


def _position_to_dict(p) -> dict:
    return {
        "ticker": p.ticker,
        "rank": p.rank,
        "weight": p.weight,
        "weight_pct": round(p.weight * 100, 2),
        "shares": p.shares,
        "dollars": p.dollars,
        "current_price": p.current_price,
        "volatility_annual": p.volatility_annual,
        "risk_contribution": p.risk_contribution,
        "risk_contribution_pct": round(p.risk_contribution * 100, 2),
    }


def _allocation_to_dict(a) -> dict:
    return {
        "total_capital": a.total_capital,
        "deployed_capital": a.deployed_capital,
        "cash_reserve": a.cash_reserve,
        "deployment_pct": round(a.deployed_capital / a.total_capital * 100, 1) if a.total_capital else 0,
        "method": a.method,
        "regime": a.regime,
        "regime_multiplier": a.regime_multiplier,
        "target_volatility": a.target_volatility,
        "estimated_portfolio_vol": a.estimated_portfolio_vol,
        "target_vol_pct": round(a.target_volatility * 100, 1),
        "estimated_vol_pct": round(a.estimated_portfolio_vol * 100, 1),
        "positions": [_position_to_dict(p) for p in a.positions],
        "n_positions": len(a.positions),
        "warnings": a.warnings,
    }


@router.post("/portfolio/size")
async def size_portfolio(
    http_request: Request,
    req: SizeRequest = Body(...),
    current_user: Optional[CognitoUser] = Depends(get_optional_user),
):
    """
    Size a portfolio given ranked tickers and total capital.

    Example request:
    {
      "tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
      "total_capital": 100000,
      "method": "hrp",
      "target_vol": 0.12,
      "max_position": 0.10,
      "apply_regime": true
    }
    """
    tickers = [t.upper().strip() for t in req.tickers]
    for t in tickers:
        if not t.replace("-", "").replace(".", "").isalnum() or len(t) > 10:
            raise HTTPException(422, f"Invalid ticker: {t}")

    # Fetch regime if requested
    regime_mult = 1.0
    regime_label = "normal"
    if req.apply_regime:
        try:
            detector = RegimeDetector()
            state = await asyncio.wait_for(detector.detect(), timeout=15.0)
            regime_mult = state.multiplier
            regime_label = state.regime
        except Exception as e:
            logger.warning(f"Regime detection failed, assuming normal: {e}")

    # Size portfolio
    try:
        sizer = PositionSizer()
        alloc = await asyncio.wait_for(
            sizer.size(
                ranked_tickers=tickers,
                total_capital=req.total_capital,
                method=req.method,
                target_vol=req.target_vol,
                max_position=req.max_position,
                min_position=req.min_position,
                regime_multiplier=regime_mult,
                regime_label=regime_label,
            ),
            timeout=90.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "Position sizing timeout")
    except Exception as e:
        logger.error(f"Portfolio sizing failed: {e}")
        raise HTTPException(500, f"Sizing failed: {e}")

    return {"data": _allocation_to_dict(alloc)}


@router.get("/portfolio/size_from_screener/{horizon}")
async def size_from_screener(
    horizon: str,
    http_request: Request,
    total_capital: float = Query(100000, gt=0, le=10_000_000),
    top_n: int = Query(15, ge=5, le=25),
    method: str = Query("hrp", pattern="^(equal|inverse_vol|ivp|erc|hrp)$"),
    target_vol: float = Query(0.12, ge=0.05, le=0.40),
    max_position: float = Query(0.10, ge=0.02, le=0.50),
    current_user: Optional[CognitoUser] = Depends(get_optional_user),
):
    """
    Convenience endpoint: fetch top_n from screener for given horizon,
    then size a portfolio.

    horizon: short_term / medium_term / long_term
    """
    VALID = ("short_term", "medium_term", "long_term")
    if horizon not in VALID:
        raise HTTPException(422, f"Invalid horizon. Must be one of {VALID}")

    # Re-use the screener's cached scan
    redis = http_request.app.state.redis
    scan_cache_key = "screener:scan:v1:n200"

    scan = None
    try:
        cached = await redis.get(scan_cache_key)
        if cached:
            scan = json.loads(cached)
    except Exception:
        pass

    if not scan or "rankings" not in scan:
        from research.universe_scanner import UniverseScanner
        from research.regime_overlay import (
            RegimeDetector, compute_breadth_from_scores, apply_regime_to_rankings
        )
        logger.info("Cache miss on scan — running fresh for size_from_screener")
        scanner = UniverseScanner(concurrency=8, skip_quality=False)
        scan_result = await asyncio.wait_for(scanner.scan(max_tickers=200), timeout=180.0)
        breadth = compute_breadth_from_scores(scan_result.get("raw_scores", {}))
        detector = RegimeDetector()
        regime_state = await detector.detect(breadth_hint=breadth)
        rankings_adjusted = apply_regime_to_rankings(scan_result["rankings"], regime_state)
        scan = {
            "rankings": rankings_adjusted,
            "regime": {
                "regime": regime_state.regime,
                "multiplier": regime_state.multiplier,
            },
        }
        try:
            await redis.setex(scan_cache_key, 4 * 3600, json.dumps(scan, default=str))
        except Exception:
            pass

    ranked = scan["rankings"].get(horizon, [])
    if not ranked:
        raise HTTPException(404, f"No rankings available for {horizon}")

    top_tickers = [r["ticker"] for r in ranked[:top_n]]
    regime_info = scan.get("regime", {"regime": "normal", "multiplier": 1.0})

    sizer = PositionSizer()
    alloc = await asyncio.wait_for(
        sizer.size(
            ranked_tickers=top_tickers,
            total_capital=total_capital,
            method=method,
            target_vol=target_vol,
            max_position=max_position,
            min_position=0.01,
            regime_multiplier=regime_info.get("multiplier", 1.0),
            regime_label=regime_info.get("regime", "normal"),
        ),
        timeout=90.0,
    )

    return {
        "data": {
            "horizon": horizon,
            "source_tickers": top_tickers,
            **_allocation_to_dict(alloc),
        }
    }
