"""
QuantEdge v6.0 — Quality Score Router
======================================
Exposes /api/v6/quality/{ticker} endpoint that returns the Past Score.
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Optional
from loguru import logger
import json
import asyncio

from ml.fundamentals.quality_engine import QualityEngine, QualityScorecard
from auth.cognito_auth import get_optional_user, CognitoUser

router = APIRouter()


def _scorecard_to_dict(sc: QualityScorecard) -> dict:
    return {
        "ticker": sc.ticker,
        "past_score": sc.past_score,
        "sub_scores": sc.sub_scores,
        "metrics": sc.metrics,
        "piotroski_f_score": sc.piotroski_f_score,
        "altman_z_score": sc.altman_z_score,
        "strengths": sc.strengths,
        "weaknesses": sc.weaknesses,
        "n_quarters_used": sc.n_quarters_used,
        "data_quality": sc.data_quality,
    }


@router.get("/quality/{ticker}")
async def get_quality(
    ticker: str,
    http_request: Request,
    n_quarters: int = 40,
    current_user: Optional[CognitoUser] = Depends(get_optional_user),
):
    """
    Returns the Past Score (fundamental quality) for a ticker.
    Uses 10 years of SEC financials from Polygon.
    """
    ticker = ticker.upper().strip()
    if not ticker.replace("-", "").isalpha() or len(ticker) > 10:
        raise HTTPException(status_code=422, detail="Invalid ticker symbol")

    # Check cache first
    redis = http_request.app.state.redis
    cache_key = f"quality:v1:{ticker}:q{n_quarters}"
    try:
        cached = await redis.get(cache_key)
        if cached:
            return {"data": json.loads(cached), "cached": True}
    except Exception:
        pass

    # Compute
    try:
        engine = QualityEngine()  # picks up POLYGON_API_KEY from env
        scorecard = await asyncio.wait_for(
            engine.analyze(ticker, n_quarters=n_quarters),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Quality analysis timed out for {ticker}")
    except Exception as e:
        logger.error(f"Quality analysis failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Quality analysis error: {str(e)}")

    result = _scorecard_to_dict(scorecard)

    # Cache for 24 hours — fundamentals only change quarterly
    try:
        await redis.setex(cache_key, 86400, json.dumps(result, default=str))
    except Exception as e:
        logger.warning(f"Quality cache write error: {e}")

    return {"data": result, "cached": False}
