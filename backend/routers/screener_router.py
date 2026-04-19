"""
QuantEdge v6.0 — Screener Router
=================================
Exposes /api/v6/screener/{horizon} — ranked universe scan with regime overlay.

Endpoints:
  GET  /screener/{horizon}           — ranked list for one horizon
  GET  /screener/all                 — all three horizons + regime state
  GET  /screener/regime              — current market regime only
  POST /screener/rescan              — force re-scan (bypasses cache, admin only)
  GET  /screener/ticker/{ticker}     — single-ticker factor breakdown

Caching strategy:
  - Full scan result cached 4 hours (expensive to recompute)
  - Regime cached 1 hour (cheap, but reduces Polygon load)
  - Single-ticker factor cached 6 hours
"""

import asyncio
import json
import time
from typing import Optional, List, Dict
from fastapi import APIRouter, HTTPException, Depends, Request, Query
from loguru import logger

from research.universe_scanner import UniverseScanner
from research.regime_overlay import (
    RegimeDetector, compute_breadth_from_scores, apply_regime_to_rankings
)
from research.factor_engine import FactorEngine
from auth.cognito_auth import get_optional_user, CognitoUser

router = APIRouter()


VALID_HORIZONS = ("short_term", "medium_term", "long_term")
SCAN_CACHE_TTL = 4 * 3600       # 4 hours
REGIME_CACHE_TTL = 3600          # 1 hour
TICKER_CACHE_TTL = 6 * 3600      # 6 hours


async def _get_or_build_scan(redis, max_tickers: int = 200) -> Dict:
    """
    Fetch cached scan or build a fresh one.
    Default 200 tickers for interactive responsiveness.
    Full 507-ticker scan runs as a scheduled job later.
    """
    cache_key = f"screener:scan:v1:n{max_tickers}"

    # Try cache
    try:
        cached = await redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            # Sanity check that cached data has the expected shape
            if "rankings" in data and "raw_scores" in data:
                return data
    except Exception:
        pass

    # Build fresh
    logger.info(f"Cache miss — building fresh scan for {max_tickers} tickers")
    scanner = UniverseScanner(concurrency=8, skip_quality=False)
    scan_result = await scanner.scan(max_tickers=max_tickers)

    # Detect regime using breadth from this scan
    breadth = compute_breadth_from_scores(scan_result.get("raw_scores", {}))
    detector = RegimeDetector()
    regime_state = await detector.detect(breadth_hint=breadth)

    # Apply regime multiplier
    rankings_adjusted = apply_regime_to_rankings(
        scan_result["rankings"], regime_state
    )

    result = {
        "scan_timestamp": scan_result["scan_timestamp"],
        "duration_seconds": scan_result["duration_seconds"],
        "universe_size": scan_result["universe_size"],
        "tickers_scored": scan_result["tickers_scored"],
        "tickers_failed": scan_result["tickers_failed"],
        "regime": {
            "regime": regime_state.regime,
            "multiplier": regime_state.multiplier,
            "vix_level": regime_state.vix_level,
            "spy_vol_20d": regime_state.spy_vol_20d,
            "breadth_pct_above_200ma": regime_state.breadth_pct_above_200ma,
            "reasoning": regime_state.reasoning,
            "timestamp": regime_state.timestamp,
        },
        "rankings": rankings_adjusted,
        "raw_scores": scan_result.get("raw_scores", {}),
    }

    try:
        await redis.setex(cache_key, SCAN_CACHE_TTL, json.dumps(result, default=str))
    except Exception as e:
        logger.warning(f"Failed to cache scan: {e}")

    return result


@router.get("/screener/regime")
async def get_regime(
    http_request: Request,
    current_user: Optional[CognitoUser] = Depends(get_optional_user),
):
    """Current market regime state (cheap — hit this often)."""
    redis = http_request.app.state.redis
    cache_key = "screener:regime:v1"

    try:
        cached = await redis.get(cache_key)
        if cached:
            return {"data": json.loads(cached), "cached": True}
    except Exception:
        pass

    try:
        detector = RegimeDetector()
        state = await asyncio.wait_for(detector.detect(), timeout=15.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Regime detection timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regime detection failed: {e}")

    result = {
        "regime": state.regime,
        "multiplier": state.multiplier,
        "vix_level": state.vix_level,
        "spy_vol_20d": state.spy_vol_20d,
        "breadth_pct_above_200ma": state.breadth_pct_above_200ma,
        "reasoning": state.reasoning,
        "timestamp": state.timestamp,
    }

    try:
        await redis.setex(cache_key, REGIME_CACHE_TTL, json.dumps(result, default=str))
    except Exception:
        pass

    return {"data": result, "cached": False}


@router.get("/screener/all")
async def get_all_rankings(
    http_request: Request,
    top_n: int = Query(25, ge=5, le=100),
    max_tickers: int = Query(200, ge=50, le=507),
    current_user: Optional[CognitoUser] = Depends(get_optional_user),
):
    """
    Ranked lists for all three horizons + regime state.
    top_n:       how many names to return per horizon (default 25)
    max_tickers: scan universe size (default 200 for responsiveness)
    """
    redis = http_request.app.state.redis

    try:
        scan = await asyncio.wait_for(
            _get_or_build_scan(redis, max_tickers=max_tickers),
            timeout=180.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Scan timeout — too many tickers")
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scan failed: {e}")

    # Trim to top_n per horizon
    trimmed_rankings = {
        h: scan["rankings"][h][:top_n] for h in VALID_HORIZONS
    }

    return {
        "data": {
            "scan_timestamp": scan["scan_timestamp"],
            "duration_seconds": scan["duration_seconds"],
            "universe_size": scan["universe_size"],
            "tickers_scored": scan["tickers_scored"],
            "tickers_failed": scan["tickers_failed"],
            "regime": scan["regime"],
            "rankings": trimmed_rankings,
        }
    }


@router.get("/screener/{horizon}")
async def get_horizon_ranking(
    horizon: str,
    http_request: Request,
    top_n: int = Query(25, ge=5, le=100),
    max_tickers: int = Query(200, ge=50, le=507),
    current_user: Optional[CognitoUser] = Depends(get_optional_user),
):
    """Ranked list for a single horizon."""
    if horizon not in VALID_HORIZONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid horizon '{horizon}'. Must be one of {VALID_HORIZONS}"
        )

    redis = http_request.app.state.redis

    try:
        scan = await asyncio.wait_for(
            _get_or_build_scan(redis, max_tickers=max_tickers),
            timeout=180.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Scan timeout")
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scan failed: {e}")

    return {
        "data": {
            "horizon": horizon,
            "scan_timestamp": scan["scan_timestamp"],
            "regime": scan["regime"],
            "rankings": scan["rankings"][horizon][:top_n],
        }
    }


@router.get("/screener/ticker/{ticker}")
async def get_ticker_factors(
    ticker: str,
    http_request: Request,
    current_user: Optional[CognitoUser] = Depends(get_optional_user),
):
    """4-factor breakdown for a single ticker."""
    ticker = ticker.upper().strip()
    if not ticker.replace("-", "").replace(".", "").isalnum() or len(ticker) > 10:
        raise HTTPException(status_code=422, detail="Invalid ticker symbol")

    redis = http_request.app.state.redis
    cache_key = f"screener:ticker:v1:{ticker}"

    try:
        cached = await redis.get(cache_key)
        if cached:
            return {"data": json.loads(cached), "cached": True}
    except Exception:
        pass

    try:
        engine = FactorEngine()
        fs = await asyncio.wait_for(
            engine.score(ticker, skip_quality=False),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Factor scoring timeout for {ticker}")
    except Exception as e:
        logger.error(f"Factor scoring failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Factor scoring failed: {e}")

    result = {
        "ticker": fs.ticker,
        "quality": fs.quality,
        "momentum": fs.momentum,
        "accumulation": fs.accumulation,
        "trend": fs.trend,
        "metrics": fs.metrics,
        "data_quality": fs.data_quality,
        "price_history_days": fs.price_history_days,
        "error": fs.error,
    }

    try:
        await redis.setex(cache_key, TICKER_CACHE_TTL, json.dumps(result, default=str))
    except Exception:
        pass

    return {"data": result, "cached": False}


@router.post("/screener/rescan")
async def force_rescan(
    http_request: Request,
    max_tickers: int = Query(200, ge=50, le=507),
    current_user: Optional[CognitoUser] = Depends(get_optional_user),
):
    """Force a fresh scan, bypassing cache. Expensive."""
    redis = http_request.app.state.redis

    # Clear the cache key
    try:
        await redis.delete(f"screener:scan:v1:n{max_tickers}")
        await redis.delete("screener:regime:v1")
    except Exception:
        pass

    try:
        scan = await asyncio.wait_for(
            _get_or_build_scan(redis, max_tickers=max_tickers),
            timeout=300.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Rescan timeout")
    except Exception as e:
        logger.error(f"Rescan failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rescan failed: {e}")

    return {
        "data": {
            "scan_timestamp": scan["scan_timestamp"],
            "duration_seconds": scan["duration_seconds"],
            "universe_size": scan["universe_size"],
            "tickers_scored": scan["tickers_scored"],
            "tickers_failed": scan["tickers_failed"],
            "regime": scan["regime"],
            "note": "Use /screener/all to retrieve rankings.",
        }
    }
