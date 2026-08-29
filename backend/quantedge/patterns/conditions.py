"""Pattern Lab — conditional scan v2.

Adds: momentum acceleration (20d momentum now vs 20 sessions ago),
volatility trend (vol percentile change over 60 sessions), days since 52w
high — and PATTERN EVOLUTION: each ticker-date is assigned a discrete
state (trend x volatility tercile); transitions state_t -> state_{t+20}
are counted with the forward return that followed each transition. All
measured; overlapping stride-5 samples disclosed as such.
"""
from __future__ import annotations
import numpy as np
from datetime import date
from loguru import logger

STRIDE = 5
FWD = (5, 20, 60)
MIN_CELL = 100


def _stats(v):
    v = v[np.isfinite(v)]
    if len(v) < MIN_CELL: return None
    return {"n": int(len(v)), "positive_pct": round(float((v > 0).mean()) * 100, 1),
            "median_pct": round(float(np.median(v)) * 100, 2),
            "p25_pct": round(float(np.percentile(v, 25)) * 100, 2),
            "p75_pct": round(float(np.percentile(v, 75)) * 100, 2)}


def _state(mom60: float, volq: int) -> str:
    t = "UP" if mom60 > 0.03 else "DOWN" if mom60 < -0.03 else "FLAT"
    return f"{t}_{('LOWVOL','MIDVOL','HIGHVOL')[volq]}"


async def scan_conditions(pool, out_path: str) -> dict:
    import json
    from pathlib import Path
    tickers = [r["ticker"] for r in await pool.fetch(
        "SELECT ticker FROM daily_bars GROUP BY ticker HAVING count(*) >= 750")]
    C = {k: [] for k in ("mom_20d", "mom_60d", "mom_120d", "mom_252d", "mom_accel",
                         "dist_52w_high", "days_since_high", "vol_21d_pctile", "vol_trend")}
    F = {h: [] for h in FWD}
    trans: dict[str, dict[str, list]] = {}
    for tk in tickers:
        rows = await pool.fetch("SELECT c, v FROM daily_bars WHERE ticker=$1 ORDER BY d", tk)
        c = np.array([r["c"] for r in rows], np.float64)
        if len(c) < 340 or c.min() < 3.0: continue
        lr = np.diff(np.log(c), prepend=np.log(c[0]))
        v21 = np.array([np.std(lr[max(0, j - 20):j + 1]) for j in range(len(c))])
        for i in range(280, len(c) - max(FWD), STRIDE):
            vp = float((v21[i - 250:i] < v21[i]).mean())
            vp_old = float((v21[i - 310:i - 60] < v21[i - 60]).mean()) if i >= 310 else 0.5
            hi_idx = int(np.argmax(c[i - 251:i + 1]))
            C["mom_20d"].append(c[i] / c[i - 20] - 1)
            C["mom_60d"].append(c[i] / c[i - 60] - 1)
            C["mom_120d"].append(c[i] / c[i - 120] - 1)
            C["mom_252d"].append(c[i] / c[i - 252] - 1)
            C["mom_accel"].append((c[i] / c[i - 20] - 1) - (c[i - 20] / c[i - 40] - 1))
            C["dist_52w_high"].append(c[i] / c[i - 251:i + 1].max() - 1)
            C["days_since_high"].append(251 - hi_idx)
            C["vol_21d_pctile"].append(vp)
            C["vol_trend"].append(vp - vp_old)
            for h in FWD: F[h].append(c[i + h] / c[i] - 1)
            # evolution: state now -> state at +20, with the return over that leg
            volq_now = 0 if vp <= 0.33 else 2 if vp >= 0.67 else 1
            vp20 = float((v21[i - 230:i + 20] < v21[i + 20]).mean())
            volq_20 = 0 if vp20 <= 0.33 else 2 if vp20 >= 0.67 else 1
            s0 = _state(c[i] / c[i - 60] - 1, volq_now)
            s1 = _state(c[i + 20] / c[i - 40] - 1, volq_20)
            trans.setdefault(s0, {}).setdefault(s1, []).append(c[i + 20] / c[i] - 1)
    n = len(F[FWD[0]])
    logger.info(f"[conditions v2] {n} samples from {len(tickers)} tickers")
    Ca = {k: np.array(v) for k, v in C.items()}
    Fa = {h: np.array(v) for h, v in F.items()}
    out = {}
    for name, cv in Ca.items():
        qs = np.nanpercentile(cv, [20, 40, 60, 80])
        b = np.digitize(cv, qs)
        out[name] = {"quintile_edges": [round(float(x), 4) for x in qs],
                     "cells": {f"Q{q+1}": {"range": [round(float(cv[b == q].min()), 4) if (b == q).any() else None,
                                                     round(float(cv[b == q].max()), 4) if (b == q).any() else None],
                                           **{f"fwd_{h}d": _stats(Fa[h][b == q]) for h in FWD}}
                               for q in range(5)}}
    evolution = {}
    for s0, dests in trans.items():
        tot = sum(len(v) for v in dests.values())
        evolution[s0] = {"n": tot, "transitions": {
            s1: {"pct": round(100 * len(v) / tot, 1),
                 "fwd20": _stats(np.array(v))}
            for s1, v in sorted(dests.items(), key=lambda kv: -len(kv[1])) if len(v) >= MIN_CELL}}
    art = {"generated": date.today().isoformat(), "samples": n, "universe": len(tickers),
           "stride": STRIDE,
           "note": ("Overlapping stride-5 samples; n is raw sample count, not independent "
                    "observations — distribution shape, not significance claims."),
           "state_definition": ("trend from 60d momentum (+/-3% thresholds) x 21d-vol tercile; "
                                "transition measured over 20 sessions"),
           "base": {f"fwd_{h}d": _stats(Fa[h]) for h in FWD},
           "conditions": out, "evolution": evolution}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(art))
    return {"samples": n, "states": len(evolution)}
