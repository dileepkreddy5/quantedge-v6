"""Pattern Lab Phase 2 — conditional forward-return scan.

For every deep-history ticker-date (stride 5): where did the stock sit on
each CONDITION — multi-horizon momentum, distance from its 52-week high,
realized-volatility percentile — and what happened over the next 5/20/60
sessions? Buckets are quintiles computed over the full scan, so "Q5
momentum" means top-fifth relative to everything measured, not an arbitrary
threshold. This turns three research modes (momentum/reversal, price
extremes, volatility) into measured distributions from one pass.

Look-ahead: conditions use data <= t only; outcomes start at t+1.
Overlap: stride-5 sampling; per-condition stats report raw n — these are
overlapping samples and the artifact says so rather than pretending
independence (quintile cells have tens of thousands of samples; the
purpose is distribution shape, not t-statistics).
"""
from __future__ import annotations
import numpy as np
from datetime import date
from loguru import logger

STRIDE = 5
FWD = (5, 20, 60)
CONDITIONS = ("mom_20d", "mom_60d", "mom_120d", "mom_252d",
              "dist_52w_high", "vol_21d_pctile")


def _stats(v: np.ndarray) -> dict | None:
    v = v[np.isfinite(v)]
    if len(v) < 100:
        return None
    return {"n": int(len(v)),
            "positive_pct": round(float((v > 0).mean()) * 100, 1),
            "median_pct": round(float(np.median(v)) * 100, 2),
            "p25_pct": round(float(np.percentile(v, 25)) * 100, 2),
            "p75_pct": round(float(np.percentile(v, 75)) * 100, 2)}


async def scan_conditions(pool, out_path: str) -> dict:
    import json
    from pathlib import Path
    tickers = [r["ticker"] for r in await pool.fetch(
        "SELECT ticker FROM daily_bars GROUP BY ticker HAVING count(*) >= 750")]
    cond_vals = {c: [] for c in CONDITIONS}
    fwd_vals = {h: [] for h in FWD}
    for tk in tickers:
        rows = await pool.fetch(
            "SELECT c, v FROM daily_bars WHERE ticker=$1 ORDER BY d", tk)
        c = np.array([r["c"] for r in rows], np.float64)
        if len(c) < 300 or c.min() < 3.0:
            continue
        lr = np.diff(np.log(c), prepend=np.log(c[0]))
        for i in range(260, len(c) - max(FWD), STRIDE):
            vol21 = float(np.std(lr[i - 20:i + 1]))
            hist_vols = [np.std(lr[j - 20:j + 1]) for j in range(i - 250, i, 10)]
            cond_vals["mom_20d"].append(c[i] / c[i - 20] - 1)
            cond_vals["mom_60d"].append(c[i] / c[i - 60] - 1)
            cond_vals["mom_120d"].append(c[i] / c[i - 120] - 1)
            cond_vals["mom_252d"].append(c[i] / c[i - 252] - 1)
            cond_vals["dist_52w_high"].append(c[i] / c[i - 251:i + 1].max() - 1)
            cond_vals["vol_21d_pctile"].append(
                float((np.array(hist_vols) < vol21).mean()))
            for h in FWD:
                fwd_vals[h].append(c[i + h] / c[i] - 1)
    n = len(fwd_vals[FWD[0]])
    logger.info(f"[conditions] {n} ticker-date samples from {len(tickers)} tickers")
    cond = {k: np.array(v) for k, v in cond_vals.items()}
    fwd = {h: np.array(v) for h, v in fwd_vals.items()}

    out = {}
    for cname, cv in cond.items():
        qs = np.nanpercentile(cv, [20, 40, 60, 80])
        bucket = np.digitize(cv, qs)          # 0..4 = Q1..Q5
        cells = {}
        for q in range(5):
            m = bucket == q
            cells[f"Q{q+1}"] = {
                "range": [round(float(cv[m].min()), 4) if m.any() else None,
                          round(float(cv[m].max()), 4) if m.any() else None],
                **{f"fwd_{h}d": _stats(fwd[h][m]) for h in FWD}}
        out[cname] = {"quintile_edges": [round(float(x), 4) for x in qs],
                      "cells": cells}
    art = {"generated": date.today().isoformat(), "samples": n,
           "universe": len(tickers), "stride": STRIDE,
           "note": ("Overlapping stride-5 samples; n is raw sample count, not "
                    "independent observations. Purpose is distribution shape "
                    "across quintiles, not significance claims."),
           "base": {f"fwd_{h}d": _stats(fwd[h]) for h in FWD},
           "conditions": out}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(art))
    return {"samples": n}
