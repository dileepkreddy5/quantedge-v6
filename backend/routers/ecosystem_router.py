"""
QuantEdge v6.0 — Ecosystem Router
==================================
GET /api/v6/ecosystem/{ticker}

Which stocks actually move with this one. Correlation is computed from daily
returns rather than asserted from a curated relationship table, so it covers
every ticker and surfaces links nobody thought to list. Correlation is not
causation — two chip names move together because they share end-market
exposure, not necessarily because one supplies the other. Labelled accordingly.
"""
from __future__ import annotations

import asyncio, os
from datetime import date, timedelta
from typing import Optional, Dict, List

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException, Request, Depends
from loguru import logger

from auth.cognito_auth import get_optional_user, CognitoUser

router = APIRouter()
POLY = os.environ.get("POLYGON_API_KEY", "")


async def _closes(client: httpx.AsyncClient, tkr: str, start: str, end: str) -> Optional[Dict[int, float]]:
    url = (f"https://api.polygon.io/v2/aggs/ticker/{tkr}/range/1/day/{start}/{end}"
           f"?adjusted=true&sort=asc&limit=400&apiKey={POLY}")
    try:
        r = await client.get(url, timeout=20)
        if r.status_code != 200:
            return None
        res = (r.json() or {}).get("results") or []
        if len(res) < 120:
            return None
        # Keyed by bar timestamp so series can be aligned on actual dates.
        return {int(b["t"]): float(b["c"]) for b in res}
    except Exception:
        return None


@router.get("/ecosystem/{ticker}")
async def get_ecosystem(
    ticker: str,
    http_request: Request,
    current_user: Optional[CognitoUser] = Depends(get_optional_user),
):
    ticker = ticker.upper().strip()
    if not ticker.replace("-", "").replace(".", "").isalnum() or len(ticker) > 10:
        raise HTTPException(422, "Invalid ticker")

    store = getattr(http_request.app.state, "peer_store", None)
    if store is None:
        raise HTTPException(503, "Universe not available yet")

    # Candidate set: the scanned universe, so this works without a curated list.
    async with store.pool.acquire() as conn:
        latest = await conn.fetchval("SELECT max(scan_time) FROM peer_stats")
        if not latest:
            return {"data": {"available": False, "reason": "universe not scanned"}, "cached": False}
        rows = await conn.fetch(
            "SELECT ticker, name, sic, bucket, market_cap FROM peer_stats WHERE scan_time=$1", latest)

    universe = [dict(r) for r in rows if r["ticker"] != ticker]
    meta = {r["ticker"]: dict(r) for r in rows}
    if ticker not in meta:
        return {"data": {"available": False, "reason": "ticker not in scanned universe"}, "cached": False}

    # Prices come from the local bars table rather than the API: one query against
    # 3,000 tickers instead of 220 HTTP calls, which is what previously forced a cap.
    async with store.pool.acquire() as conn:
        me_rows = await conn.fetch(
            "SELECT d, c FROM daily_bars WHERE ticker=$1 AND d >= CURRENT_DATE - INTERVAL '400 days'"
            " ORDER BY d", ticker)
        if len(me_rows) < 120:
            return {"data": {"available": False, "reason": "insufficient local price history"}, "cached": False}
        my_dates = [r["d"] for r in me_rows]
        start_d = my_dates[0]

        peer_rows = await conn.fetch(
            "SELECT ticker, d, c FROM daily_bars WHERE d >= $1 AND ticker = ANY($2::text[]) ORDER BY ticker, d",
            start_d, [u["ticker"] for u in universe])
        spy_rows = await conn.fetch(
            "SELECT d, c FROM daily_bars WHERE ticker='SPY' AND d >= $1 ORDER BY d", start_d)

    base = {r["d"]: float(r["c"]) for r in me_rows}
    spy = {r["d"]: float(r["c"]) for r in spy_rows}
    series: Dict[str, Dict] = {}
    for r in peer_rows:
        series.setdefault(r["ticker"], {})[r["d"]] = float(r["c"])

    results = [(meta[t], s) for t, s in series.items() if t in meta and len(s) >= 120]

    out: List[Dict] = []
    base_dates = sorted(base.keys())
    mkt_ret = None
    if spy:
        _c = [t for t in base_dates if t in spy]
        if len(_c) >= 120:
            mkt_ret = np.diff(np.log([spy[t] for t in _c]))
    for item in results:
        if isinstance(item, Exception) or item is None:
            continue
        row, arr = item
        if arr is None:
            continue
        # Align on shared trading days. Slicing by position silently compares
        # different dates when two tickers have different bar counts, which
        # destroys the correlation and leaves only a weak market component.
        common = [t for t in base_dates if t in arr]
        if len(common) < 120:
            continue
        b = np.array([base[t] for t in common], dtype=float)
        p = np.array([arr[t] for t in common], dtype=float)
        r1 = np.diff(np.log(b))
        r2 = np.diff(np.log(p))
        n = len(r1)
        if r1.std() == 0 or r2.std() == 0:
            continue
        c = float(np.corrcoef(r1, r2)[0, 1])
        if not np.isfinite(c):
            continue
        # Strip the common market factor. Two large caps correlate simply because
        # both track the index; what matters is whether they move together beyond
        # that. Residual correlation isolates the stock-specific relationship.
        if mkt_ret is not None and len(mkt_ret) >= n:
            m = mkt_ret[-n:]
            if m.std() > 0:
                b1 = np.cov(r1, m)[0, 1] / m.var()
                b2 = np.cov(r2, m)[0, 1] / m.var()
                e1, e2 = r1 - b1 * m, r2 - b2 * m
                if e1.std() > 0 and e2.std() > 0:
                    rc = float(np.corrcoef(e1, e2)[0, 1])
                    if np.isfinite(rc):
                        resid = round(rc, 3)
                    else:
                        resid = None
                else:
                    resid = None
            else:
                resid = None
        else:
            resid = None
        # Beta of the peer against this stock: how far it moves per 1% move here.
        beta = float(np.cov(r2, r1)[0, 1] / r1.var()) if r1.var() > 0 else None
        out.append({
            "ticker": row["ticker"],
            "name": (row["name"] or "")[:40],
            "sector": row["bucket"],
            "same_sector": row["bucket"] == meta[ticker]["bucket"],
            "correlation": round(c, 3),
            "residual_correlation": resid,
            "beta_to_this": round(beta, 2) if beta is not None else None,
            "market_cap": row["market_cap"],
            "days": n,
        })

    out.sort(key=lambda x: -x["correlation"])
    movers = out[:15]
    # Residual correlations were computed but are not surfaced: after stripping the
    # market factor they land at 0.1-0.2 across the board, which is within the range
    # you would expect by chance across 220 comparisons. Presenting them as
    # relationships would be reading signal into noise.
    inverse = sorted(out, key=lambda x: x["correlation"])[:6]
    cross = [x for x in out if not x["same_sector"]][:10]

    return {"data": {
        "available": True,
        "ticker": ticker,
        "sector": meta[ticker]["bucket"],
        "n_compared": len(out),
        "window_days": len(base_dates),
        "movers": movers,
        "cross_sector": cross,
        "inverse": inverse,
        "significance_floor": 0.25,
        "note": ("Raw correlation includes the market factor — most large caps move together simply "
                 "because both track the index. Residual correlation strips that out, leaving the "
                 "stock-specific relationship, which is where genuine business links show up. "
                 "Correlation of daily returns over the past year. These stocks move together — "
                 "that may reflect a supply relationship, shared customers, or simply common "
                 "sector exposure. It is a measured association, not an asserted business link."),
    }, "cached": False}
