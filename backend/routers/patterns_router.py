"""Pattern Lab — GET /api/v6/patterns/analogs/{ticker}?window=60

Serves historical-analog distributions from the precomputed window library.
The heavy lifting (508k+ windows) is a nightly artifact; queries are one
vectorized Euclidean pass plus banded DTW on a 300-window shortlist —
sub-second. Distributions ship with library-wide base rates and the
one-macro-era caveat; below 15 episodes a cell is null, not a number.
"""
from __future__ import annotations
import numpy as np
from datetime import date
from fastapi import APIRouter, HTTPException, Query, Request
from quantedge.patterns.engine import PatternLibrary

router = APIRouter()
_LIB = PatternLibrary("/app/models/patterns")


@router.get("/patterns/analogs/{ticker}")
async def pattern_analogs(ticker: str, request: Request,
                          window: int = Query(60, description="20 or 60")):
    if window not in (20, 60):
        raise HTTPException(status_code=422, detail="window must be 20 or 60")
    tk = ticker.upper().strip()
    if not tk.replace('-', '').isalpha() or len(tk) > 10:
        raise HTTPException(status_code=422, detail="invalid ticker")

    pool = getattr(request.app.state, "db", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="database not connected")

    rows = await pool.fetch(
        "SELECT d, c FROM daily_bars WHERE ticker=$1 ORDER BY d DESC LIMIT $2",
        tk, window + 5)
    if len(rows) < window:
        raise HTTPException(status_code=404,
                            detail=f"{tk}: only {len(rows)} bars on record; "
                                   f"{window} needed")
    rows = rows[::-1]
    closes = np.array([r["c"] for r in rows], dtype=np.float64)
    q_end: date = rows[-1]["d"]

    res = _LIB.query(closes, tk, window, q_end)
    if res is None:
        raise HTTPException(status_code=503,
                            detail="pattern library not built yet")
    res["ticker"] = tk
    res["as_of"] = q_end.isoformat()
    return res
