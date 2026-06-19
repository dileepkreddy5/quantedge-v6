"""
QuantEdge v6.0 — Per-Stock News Router
=======================================
Serves recent news articles for a ticker, normalized for the frontend News tab,
plus a lightweight sentiment summary. Reuses the existing Polygon news feed
(SentimentDataFeed.get_news), which is already Redis-cached.

Endpoint:
  GET /api/v6/news/{ticker}?limit=30  -> { data: {...}, cached: bool }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional, List, Dict
from fastapi import APIRouter, HTTPException, Request, Depends, Query
from loguru import logger

from data.feeds.market_data import SentimentDataFeed
from auth.cognito_auth import get_optional_user, CognitoUser

router = APIRouter()

NEWS_CACHE_TTL = 900  # 15 min


def _relative_time(iso: str) -> str:
    if not iso:
        return ""
    try:
        ts = iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        secs = delta.total_seconds()
        if secs < 0:
            return "just now"
        if secs < 3600:
            return f"{int(secs // 60)}m ago"
        if secs < 86400:
            return f"{int(secs // 3600)}h ago"
        days = int(secs // 86400)
        return f"{days}d ago" if days < 30 else f"{days // 30}mo ago"
    except Exception:
        return ""


def _sentiment_label(score: Optional[float]) -> str:
    if score is None:
        return "neutral"
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"


def _normalize(articles: List[Dict]) -> List[Dict]:
    out = []
    for a in articles:
        iso = a.get("published_utc", "") or a.get("published", "")
        sent = a.get("polygon_sentiment", None)
        if sent is None:
            sent = a.get("sentiment_score", None)
        out.append({
            "title": a.get("title", "") or "",
            "snippet": (a.get("description", "") or "")[:280],
            "source": a.get("publisher", "") or a.get("source", "") or "",
            "author": a.get("author", "") or "",
            "url": a.get("article_url", "") or a.get("url", "") or "",
            "published": iso,
            "relative_time": _relative_time(iso),
            "sentiment": _sentiment_label(sent),
            "sentiment_score": sent,
        })
    return out


def _summary(articles: List[Dict]) -> Dict:
    n = len(articles)
    if n == 0:
        return {"count": 0, "label": "no recent news", "pos": 0, "neg": 0, "neu": 0}
    pos = sum(1 for a in articles if a["sentiment"] == "positive")
    neg = sum(1 for a in articles if a["sentiment"] == "negative")
    neu = n - pos - neg
    if pos > neg and pos >= n * 0.4:
        label = "mostly positive"
    elif neg > pos and neg >= n * 0.4:
        label = "mostly negative"
    else:
        label = "mixed"
    return {"count": n, "label": label, "pos": pos, "neg": neg, "neu": neu}


@router.get("/news/{ticker}")
async def get_news(
    ticker: str,
    http_request: Request,
    limit: int = Query(30, ge=1, le=50),
    current_user: Optional[CognitoUser] = Depends(get_optional_user),
):
    ticker = ticker.upper().strip()
    if not ticker.replace("-", "").replace(".", "").isalnum() or len(ticker) > 10:
        raise HTTPException(422, "Invalid ticker")

    redis = http_request.app.state.redis
    cache_key = f"news:panel:v1:{ticker}:{limit}"

    try:
        cached = await redis.get(cache_key)
        if cached:
            return {"data": json.loads(cached), "cached": True}
    except Exception:
        pass

    try:
        feed = SentimentDataFeed()
        if hasattr(feed, "inject_redis"):
            feed.inject_redis(redis)
        raw = await feed.get_news(ticker=ticker, limit=limit)
    except Exception as e:
        logger.warning(f"News fetch error for {ticker}: {e}")
        raise HTTPException(502, "Could not fetch news")

    articles = _normalize(raw or [])
    payload = {
        "ticker": ticker,
        "summary": _summary(articles),
        "articles": articles,
    }

    try:
        await redis.setex(cache_key, NEWS_CACHE_TTL, json.dumps(payload))
    except Exception:
        pass

    return {"data": payload, "cached": False}
