"""
QuantEdge v6.0 — Ascent Radar Router
=====================================
Public, read-only board of US companies climbing toward larger-cap tiers.
Discovery tool, not advice.

Endpoints:
  GET  /ascent/board       — top-25 board + 3d/1w/1m deltas + first-seen
  GET  /ascent/board.csv   — same board as CSV download
  GET  /ascent/top/{n}     — top-N teaser (homepage uses n=5)
  POST /ascent/rescan      — force a fresh scan now (owner only)
"""

from __future__ import annotations

import csv
import io
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from loguru import logger

from auth.cognito_auth import get_optional_user, CognitoUser

router = APIRouter()


def _store(request: Request):
    store = getattr(request.app.state, "ascent_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Ascent Radar not initialized (DB required)")
    return store


@router.get("/ascent/board")
async def ascent_board(request: Request):
    store = _store(request)
    try:
        return await store.get_latest_board(top_n=25)
    except Exception as e:
        logger.warning(f"Ascent board error: {e}")
        raise HTTPException(status_code=500, detail="Failed to load ascent board")


@router.get("/ascent/top/{n}")
async def ascent_top(n: int, request: Request):
    store = _store(request)
    n = max(1, min(n, 25))
    return await store.get_latest_board(top_n=n)


@router.get("/ascent/board.csv")
async def ascent_board_csv(request: Request):
    store = _store(request)
    board = await store.get_latest_board(top_n=25)
    rows = board.get("rows", [])

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "rank", "ticker", "name", "sector", "ascent_score", "tier",
        "strength", "volume", "tier_score", "near_high",
        "rank_change_3d", "rank_change_1w", "rank_change_1m",
        "first_seen", "is_new", "flags",
    ])
    for r in rows:
        d3 = r.get("delta_3d") or {}
        w1 = r.get("delta_1w") or {}
        m1 = r.get("delta_1m") or {}
        w.writerow([
            r["rank"], r["ticker"], r.get("name", ""), r.get("sector", ""),
            r["ascent_score"], r.get("tier", ""),
            r.get("strength_score"), r.get("volume_score"),
            r.get("tier_score"), r.get("high_score"),
            d3.get("rank_change", ""), w1.get("rank_change", ""), m1.get("rank_change", ""),
            r.get("first_seen", ""), r.get("is_new", False),
            "; ".join(r.get("flags", [])),
        ])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ascent_radar.csv"},
    )


@router.post("/ascent/rescan")
async def ascent_rescan(request: Request,
                        current_user: CognitoUser = Depends(get_optional_user)):
    if current_user is None:
        raise HTTPException(status_code=401, detail="Login required to force a rescan")
    job = getattr(request.app.state, "ascent_job", None)
    if job is None:
        raise HTTPException(status_code=503, detail="Ascent job not initialized")
    result = await job.run()
    return {"status": "ok", **result}
