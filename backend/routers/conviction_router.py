"""Conviction endpoint — runs the signal-registry intelligence tree for a ticker.

/api/v6/conviction/{ticker} returns the full scored tree: intelligence score,
per-category scores, and every leaf signal with value, score, method, status
(live/defined), and evidence. Reuses the quality-engine data fetch (Polygon), so
no new data pipe. Peer distributions come from PeerStore when available; absent
peers fall back to absolute bands. Response is recursively NaN-sanitized.
"""
from __future__ import annotations
import math
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Depends
from loguru import logger

from ml.fundamentals.quality_engine import fetch_quarterly_financials, estimate_wacc
from quantedge.scoring.features import compute_features
from quantedge.scoring.rollup_full import run_financial
from auth.cognito_auth import get_optional_user, CognitoUser
from core.config import settings

router = APIRouter()


def _sanitize(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


@router.get("/conviction/{ticker}")
async def get_conviction(
    ticker: str,
    http_request: Request,
    n_quarters: int = 40,
    current_user: Optional[CognitoUser] = Depends(get_optional_user),
):
    ticker = ticker.upper().strip()
    api_key = getattr(settings, "POLYGON_API_KEY", "") or ""
    if not api_key:
        raise HTTPException(status_code=503, detail="Financial data source unavailable")

    try:
        quarters = await fetch_quarterly_financials(ticker, api_key, limit=n_quarters)
    except Exception as e:
        logger.warning(f"conviction fetch failed for {ticker}: {e}")
        quarters = []

    if not quarters:
        return {"data": {"ticker": ticker, "available": False,
                         "reason": "no financial statements available"}, "cached": False}

    wacc = estimate_wacc(beta=None)["high"]

    peers = {}
    pool = getattr(http_request.app.state, "db_pool", None)
    if pool is not None:
        try:
            from services.peer_store import PeerStore
            ps = PeerStore(pool)
            pdata = await ps.get_peers(ticker)
            if pdata.get("available"):
                factor_lists = {}
                for row in pdata.get("peers", []):
                    for k, v in (row.get("factors") or {}).items():
                        factor_lists.setdefault(k, []).append(v)
                peers = factor_lists
        except Exception as e:
            logger.info(f"conviction: peers unavailable for {ticker}: {e}")

    feats = compute_features(quarters, wacc=wacc)
    tree = run_financial(feats, peers)

    n_scored = sum(c["n_scored"] for c in tree["categories"])
    n_total = sum(c["n_signals"] for c in tree["categories"])
    n_live = sum(c["n_live"] for c in tree["categories"])

    result = {
        "ticker": ticker, "available": True,
        "conviction": tree["score"], "confidence": tree["confidence"],
        "coverage": {"scored": n_scored, "live": n_live, "total": n_total},
        "wacc_used": round(wacc, 4),
        "intelligences": [tree], "n_quarters": len(quarters),
    }
    return {"data": _sanitize(result), "cached": False}
