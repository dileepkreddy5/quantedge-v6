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


@router.get("/patterns/formations")
async def formations_library():
    """Classical-formation scan artifact: occurrences, breakout stats and
    forward distributions per formation, measured — never asserted."""
    import json
    from core.artifact_paths import artifact_read_path
    p = artifact_read_path("formations_scan.json")
    if p is None:
        raise HTTPException(status_code=503, detail="formation scan not run yet")
    return json.loads(p.read_text())


@router.get("/patterns/conditions/{ticker}")
async def ticker_conditions(ticker: str, request: Request):
    """Where the ticker sits TODAY on each measured condition (multi-horizon
    momentum, 52w-high distance, volatility percentile), with the historical
    forward distribution of its current quintile vs the unconditional base."""
    import json
    import numpy as np
    from core.artifact_paths import artifact_read_path
    tk = ticker.upper().strip()
    if not tk.replace('-', '').isalpha() or len(tk) > 10:
        raise HTTPException(status_code=422, detail="invalid ticker")
    p = artifact_read_path("conditions_scan.json")
    if p is None:
        raise HTTPException(status_code=503, detail="condition scan not run yet")
    art = json.loads(p.read_text())

    pool = getattr(request.app.state, "db", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="database not connected")
    rows = await pool.fetch(
        "SELECT c FROM daily_bars WHERE ticker=$1 ORDER BY d DESC LIMIT 320", tk)
    if len(rows) < 260:
        raise HTTPException(status_code=404,
                            detail=f"{tk}: {len(rows)} bars on record; 260 needed")
    c = np.array([r["c"] for r in rows], np.float64)[::-1]
    lr = np.diff(np.log(c))
    vol21 = float(np.std(lr[-21:]))
    hist = [float(np.std(lr[j - 20:j + 1])) for j in range(20, len(lr) - 1, 10)]
    vals = {
        "mom_20d": float(c[-1] / c[-21] - 1),
        "mom_60d": float(c[-1] / c[-61] - 1),
        "mom_120d": float(c[-1] / c[-121] - 1),
        "mom_252d": float(c[-1] / c[-253] - 1),
        "dist_52w_high": float(c[-1] / c[-252:].max() - 1),
        "vol_21d_pctile": float((np.array(hist) < vol21).mean()) if hist else 0.5,
    }
    out = {"ticker": tk, "generated": art["generated"], "samples": art["samples"],
           "note": art["note"], "base": art["base"], "conditions": {}}
    for name, v in vals.items():
        spec = art["conditions"].get(name)
        if not spec:
            continue
        q = int(np.digitize([v], spec["quintile_edges"])[0])  # 0..4
        out["conditions"][name] = {
            "value": round(v, 4), "quintile": q + 1,
            "cell": spec["cells"].get(f"Q{q+1}"),
        }
    return out
