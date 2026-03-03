"""
QuantEdge v5.0 — Watchlist Router
Watchlist persisted in Redis (survives ECS restarts and container recycles).
Key: watchlist:{username}  Value: JSON list of items  TTL: 90 days
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, validator
from typing import List, Optional
from datetime import datetime
import json
from auth.cognito_auth import get_current_user, CognitoUser
from core.config import settings

router = APIRouter()

WATCHLIST_TTL = 60 * 60 * 24 * 90  # 90 days


class WatchlistItem(BaseModel):
    ticker: str
    notes: Optional[str] = None
    alert_above: Optional[float] = None
    alert_below: Optional[float] = None

    @validator("ticker")
    def validate_ticker(cls, v):
        v = v.upper().strip()
        if not v.isalpha() or len(v) > 10:
            raise ValueError("Invalid ticker")
        return v


async def _get_watchlist_from_redis(redis, username: str) -> List[dict]:
    raw = await redis.get(f"watchlist:{username}")
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []


async def _save_watchlist_to_redis(redis, username: str, items: List[dict]):
    await redis.setex(
        f"watchlist:{username}",
        WATCHLIST_TTL,
        json.dumps(items, default=str)
    )


@router.get("/watchlist")
async def get_watchlist(
    request: Request,
    current_user: CognitoUser = Depends(get_current_user),
):
    redis = request.app.state.redis
    items = await _get_watchlist_from_redis(redis, current_user.username)
    return {"watchlist": items, "count": len(items)}


@router.post("/watchlist")
async def add_to_watchlist(
    item: WatchlistItem,
    request: Request,
    current_user: CognitoUser = Depends(get_current_user),
):
    redis = request.app.state.redis
    items = await _get_watchlist_from_redis(redis, current_user.username)

    if any(w["ticker"] == item.ticker for w in items):
        raise HTTPException(status_code=409, detail=f"{item.ticker} already in watchlist")

    entry = {
        "ticker": item.ticker,
        "notes": item.notes,
        "alert_above": item.alert_above,
        "alert_below": item.alert_below,
        "added_at": datetime.utcnow().isoformat(),
    }
    items.append(entry)
    await _save_watchlist_to_redis(redis, current_user.username, items)
    return {"message": f"Added {item.ticker}", "item": entry}


@router.delete("/watchlist/{ticker}")
async def remove_from_watchlist(
    ticker: str,
    request: Request,
    current_user: CognitoUser = Depends(get_current_user),
):
    redis = request.app.state.redis
    ticker = ticker.upper()
    items = await _get_watchlist_from_redis(redis, current_user.username)
    before = len(items)
    items = [w for w in items if w["ticker"] != ticker]
    if len(items) == before:
        raise HTTPException(status_code=404, detail=f"{ticker} not in watchlist")
    await _save_watchlist_to_redis(redis, current_user.username, items)
    return {"message": f"Removed {ticker}"}
