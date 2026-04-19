"""
QuantEdge v6.0 — News Momentum Router
======================================
Exposes /api/v6/news_momentum/{ticker} — analyst revision proxy signal
derived from Polygon news API.
"""

import json
import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from loguru import logger

from research.news_momentum import NewsMomentumEngine
from auth.cognito_auth import get_optional_user, CognitoUser

router = APIRouter()


@router.get("/news_momentum/{ticker}")
async def get_news_momentum(
    ticker: str,
    http_request: Request,
    current_user: Optional[CognitoUser] = Depends(get_optional_user),
):
    """
    News momentum signal (0-100) — analyst revision proxy.
    Based on: keyword extraction + sentiment drift + article volume trend.
    """
    ticker = ticker.upper().strip()
    if not ticker.replace("-", "").replace(".", "").isalnum() or len(ticker) > 10:
        raise HTTPException(422, "Invalid ticker")

    redis = http_request.app.state.redis
    cache_key = f"news_momentum:v1:{ticker}"

    try:
        cached = await redis.get(cache_key)
        if cached:
            return {"data": json.loads(cached), "cached": True}
    except Exception:
        pass

    try:
        engine = NewsMomentumEngine()
        signal = await asyncio.wait_for(engine.analyze(ticker), timeout=30.0)
    except asyncio.TimeoutError:
        raise HTTPException(504, f"News analysis timeout for {ticker}")
    except Exception as e:
        logger.error(f"News momentum failed for {ticker}: {e}")
        raise HTTPException(500, f"News momentum error: {e}")

    result = {
        "ticker": signal.ticker,
        "score": signal.score,
        "data_quality": signal.data_quality,
        "article_count_30d": signal.article_count_30d,
        "article_count_7d": signal.article_count_7d,
        "volume_trend": signal.volume_trend,
        "sentiment_30d_mean": signal.sentiment_30d_mean,
        "sentiment_7d_mean": signal.sentiment_7d_mean,
        "sentiment_drift": signal.sentiment_drift,
        "bullish_keyword_hits": signal.bullish_keyword_hits,
        "bearish_keyword_hits": signal.bearish_keyword_hits,
        "net_keyword_score": signal.net_keyword_score,
        "top_headlines": signal.top_headlines,
    }

    try:
        await redis.setex(cache_key, 3600, json.dumps(result, default=str))  # 1hr cache
    except Exception:
        pass

    return {"data": result, "cached": False}
