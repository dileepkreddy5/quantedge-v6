"""
QuantEdge v6.0 — Per-Stock News Router (v2: Finnhub events + AI briefing)
"""

from __future__ import annotations

import json
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Dict

import httpx
from fastapi import APIRouter, HTTPException, Request, Depends, Query
from loguru import logger

from core.config import settings
from data.feeds.market_data import SentimentDataFeed
from data.feeds.finnhub_feed import FinnhubFeed
from auth.cognito_auth import get_optional_user, CognitoUser

router = APIRouter()

NEWS_CACHE_TTL = 1800  # 30 min


def _relative_time(iso: str) -> str:
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        secs = (datetime.now(timezone.utc) - dt).total_seconds()
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


def _normalize(articles: List[Dict]) -> List[Dict]:
    out = []
    for a in articles:
        iso = a.get("timestamp", "") or a.get("published_utc", "") or a.get("published", "")
        out.append({
            "title": a.get("title", "") or "",
            "one_liner": (a.get("description", "") or "")[:160],
            "source": a.get("publisher", "") or a.get("source", "") or "",
            "url": a.get("article_url", "") or a.get("url", "") or "",
            "published": iso,
            "relative_time": _relative_time(iso),
        })
    return out


async def _ai_key_points(ticker: str, facts: List[str], articles: List[Dict]) -> List[str]:
    key = getattr(settings, "ANTHROPIC_API_KEY", None)
    if not key:
        return []

    facts_block = "\n".join(f"- {f}" for f in facts) if facts else "- (no structured data available)"
    news_block = "\n".join(
        f"- {a['title']}: {a['one_liner']}" for a in articles[:15]
    ) if articles else "- (no recent articles)"

    prompt = (
        f"You are an equity research analyst preparing a morning briefing on {ticker}.\n\n"
        f"VERIFIED DATA (you may state these as fact):\n{facts_block}\n\n"
        f"RECENT NEWS HEADLINES (extract only what is stated; do NOT invent analyst "
        f"names, price targets, numbers, or dates not present here):\n{news_block}\n\n"
        f"Write up to 10 concise bullet points covering the most material developments "
        f"for this company right now: upcoming catalysts (earnings), analyst stance, and "
        f"key announcements (product, M&A, R&D, guidance) that appear above. Each bullet "
        f"one short sentence, under 20 words. Order by importance. Do not fabricate. "
        f"Return ONLY a JSON array of strings."
    )
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "content-type": "application/json",
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 700,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            data = resp.json()
            raw = "".join(b["text"] for b in data.get("content", []) if b.get("type") == "text")
            s, e = raw.find("["), raw.rfind("]")
            if s >= 0:
                pts = json.loads(raw[s:e + 1])
                return [str(p) for p in pts if isinstance(p, str)][:10]
    except Exception as ex:
        logger.warning(f"AI key points failed for {ticker}: {ex}")
    return []


async def build_briefing(ticker: str, redis, limit: int = 10) -> Dict:
    ticker = ticker.upper().strip()

    news_feed = SentimentDataFeed()
    if hasattr(news_feed, "inject_redis"):
        news_feed.inject_redis(redis)
    finnhub = FinnhubFeed(api_key=getattr(settings, "FINNHUB_API_KEY", ""), redis_client=redis)

    raw_news, events = await asyncio.gather(
        news_feed.get_news(ticker=ticker, limit=30),
        finnhub.get_events(ticker),
        return_exceptions=True,
    )
    if isinstance(raw_news, Exception):
        raw_news = []
    if isinstance(events, Exception):
        events = {}

    articles = _normalize(raw_news or [])[:limit]
    facts = FinnhubFeed.events_to_facts(events or {})
    key_points = await _ai_key_points(ticker, facts, articles)

    ai_available = bool(key_points)
    if not key_points:
        key_points = facts

    return {
        "ticker": ticker,
        "key_points": key_points,
        "ai_synthesized": ai_available,
        "events": events or {},
        "articles": articles,
        "summary": {"count": len(articles)},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/news/{ticker}")
async def get_news(
    ticker: str,
    http_request: Request,
    limit: int = Query(10, ge=1, le=20),
    current_user: Optional[CognitoUser] = Depends(get_optional_user),
):
    ticker = ticker.upper().strip()
    if not ticker.replace("-", "").replace(".", "").isalnum() or len(ticker) > 10:
        raise HTTPException(422, "Invalid ticker")

    redis = http_request.app.state.redis
    cache_key = f"news:briefing:v2:{ticker}:{limit}"

    try:
        cached = await redis.get(cache_key)
        if cached:
            return {"data": json.loads(cached), "cached": True}
    except Exception:
        pass

    try:
        payload = await build_briefing(ticker, redis, limit=limit)
    except Exception as e:
        logger.warning(f"Briefing build error for {ticker}: {e}")
        raise HTTPException(502, "Could not build news briefing")

    try:
        await redis.setex(cache_key, NEWS_CACHE_TTL, json.dumps(payload))
    except Exception:
        pass

    return {"data": payload, "cached": False}
