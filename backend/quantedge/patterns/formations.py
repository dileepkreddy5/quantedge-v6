"""Pattern Lab — classical formation detection v2 (LMW-style).

v2 adds what a research report needs: each occurrence tagged with its
formation regime, volume slope and volatility percentile; follow-through
measured (did price continue in the breakout direction at +20d); summary
carries full distributions per horizon plus conditional splits by regime
and volume. Horizons NaN per-occurrence near the data end (bounding the
scan by the longest horizon is the truncation bug — twice caught today).
"""
from __future__ import annotations
import numpy as np
from datetime import date
from loguru import logger

TOL = 0.015
SMOOTH_SIGMA = 3.0
MIN_EXTREMA_GAP = 3
HORIZONS = (5, 20, 60, 120)
MIN_CELL = 15


def _smooth(c, sigma=SMOOTH_SIGMA):
    r = int(4 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    return np.convolve(np.pad(c, r, mode="edge"), k, mode="valid")


def _extrema(sm):
    ext = []
    for i in range(1, len(sm) - 1):
        if sm[i] > sm[i - 1] and sm[i] >= sm[i + 1]: t = 1
        elif sm[i] < sm[i - 1] and sm[i] <= sm[i + 1]: t = -1
        else: continue
        if ext and ext[-1][1] == t:
            if (t == 1 and sm[i] > sm[ext[-1][0]]) or (t == -1 and sm[i] < sm[ext[-1][0]]):
                ext[-1] = (i, t)
            continue
        if ext and i - ext[-1][0] < MIN_EXTREMA_GAP: continue
        ext.append((i, t))
    return ext


def _near(a, b, tol=TOL): return abs(a - b) / ((a + b) / 2) < tol
def _slope(idx, val): return float(np.polyfit(idx, val, 1)[0]) if len(idx) >= 2 else 0.0


def classify_last5(c, ext, at):
    if at < 4: return None
    e = ext[at - 4:at + 1]
    idx = np.array([i for i, _ in e]); typ = [t for _, t in e]; v = c[idx]
    if typ == [1, -1, 1, -1, 1]:
        if v[2] > v[0] and v[2] > v[4] and _near(v[0], v[4]) and _near(v[1], v[3]): return "head_shoulders"
        if _near(v[0], v[2]) and _near(v[2], v[4]): return "triple_top"
        if _near(v[0], v[2]) and v[4] < v[2] * (1 - TOL): return "double_top"
        ts = _slope(idx[[0, 2, 4]], v[[0, 2, 4]] / v[0]); bs = _slope(idx[[1, 3]], v[[1, 3]] / v[0])
        if abs(ts) < 5e-4 and bs > 5e-4: return "ascending_triangle"
        if ts < -5e-4 and abs(bs) < 5e-4: return "descending_triangle"
        if ts < -5e-4 and bs > 5e-4: return "symmetrical_triangle"
        if abs(ts) < 5e-4 and abs(bs) < 5e-4 and not _near(v[0], v[1]): return "rectangle"
        if ts > 5e-4 and bs > ts: return "rising_wedge"
        if ts < -5e-4 and bs < ts * 0.999: return "falling_wedge"
    if typ == [-1, 1, -1, 1, -1]:
        if v[2] < v[0] and v[2] < v[4] and _near(v[0], v[4]) and _near(v[1], v[3]): return "inv_head_shoulders"
        if _near(v[0], v[2]) and _near(v[2], v[4]): return "triple_bottom"
        if _near(v[0], v[2]) and v[4] > v[2] * (1 + TOL): return "double_bottom"
    return None


def _stats(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)])
    if len(v) < MIN_CELL: return None
    return {"n": int(len(v)), "positive_pct": round(float((v > 0).mean()) * 100, 1),
            "median_pct": round(float(np.median(v)), 2), "mean_pct": round(float(v.mean()), 2),
            "p25_pct": round(float(np.percentile(v, 25)), 2), "p75_pct": round(float(np.percentile(v, 75)), 2)}


async def scan_formations(pool, out_path: str) -> dict:
    import json
    from pathlib import Path
    tickers = [r["ticker"] for r in await pool.fetch(
        "SELECT ticker FROM daily_bars GROUP BY ticker HAVING count(*) >= 750")]

    spy = await pool.fetch("SELECT d, c FROM daily_bars WHERE ticker='SPY' ORDER BY d")
    regime_by_date = {}
    if len(spy) > 220:
        sc = np.array([r["c"] for r in spy]); sds = [r["d"] for r in spy]
        for i in range(220, len(sc)):
            trend = "BULL" if sc[i] / sc[i - 63] - 1 > 0 else "BEAR"
            vol = "HIGH_VOL" if np.std(np.diff(np.log(sc[i - 21:i + 1]))) * np.sqrt(252) > 0.18 else "LOW_VOL"
            regime_by_date[sds[i]] = f"{trend}_{vol}"

    occ = {}
    for tk in tickers:
        rows = await pool.fetch("SELECT d, c, v FROM daily_bars WHERE ticker=$1 ORDER BY d", tk)
        c = np.array([r["c"] for r in rows], np.float64)
        if c.min() < 3.0: continue
        vv = np.array([float(r["v"] or 0) for r in rows], np.float64)
        ds = [r["d"] for r in rows]
        lr = np.diff(np.log(np.maximum(c, 1e-9)), prepend=np.log(max(c[0], 1e-9)))
        v21 = np.array([np.std(lr[max(0, j - 20):j + 1]) for j in range(len(c))])
        sm = _smooth(c); ext = _extrema(sm)
        for at in range(4, len(ext)):
            name = classify_last5(c, ext, at)
            if name is None: continue
            end_i = ext[at][0]; conf = end_i + 3
            if conf + min(HORIZONS) >= len(c): continue
            entry = c[conf]; start_i = ext[at - 4][0]
            vseg = vv[start_i:end_i + 1]
            vz_s = vseg.std()
            vslope = float(np.polyfit(np.arange(len(vseg)), (vseg - vseg.mean()) / vz_s, 1)[0]) if vz_s > 0 else 0.0
            lo = max(0, end_i - 251)
            vol_pct = float((v21[lo:end_i] < v21[end_i]).mean()) if end_i > lo else 0.5
            fwd = {}
            for h in HORIZONS:
                fwd[h] = (round(float(c[conf + h] / entry - 1) * 100, 2)
                          if conf + h < len(c) and np.isfinite(c[conf + h]) else None)
            up = bool(c[conf] > sm[end_i])
            ft = (None if fwd[20] is None else bool((fwd[20] > 0) == up))
            occ.setdefault(name, []).append({
                "ticker": tk, "start": ds[start_i].isoformat(), "end": ds[end_i].isoformat(),
                "duration": int(end_i - start_i), "breakout_up": up, "follow_through": ft,
                "regime": regime_by_date.get(ds[end_i], "UNKNOWN"),
                "volume_slope": round(vslope, 3), "vol_pctile": round(vol_pct, 2),
                **{f"fwd_{h}d": fwd[h] for h in HORIZONS}})

    summary = {}
    for name, lst in occ.items():
        lst.sort(key=lambda o: (o["ticker"], o["end"]))
        kept, last = [], {}
        for o in lst:
            le = last.get(o["ticker"])
            if le and (date.fromisoformat(o["end"]) - le).days < o["duration"]: continue
            last[o["ticker"]] = date.fromisoformat(o["end"]); kept.append(o)
        f20 = [o["fwd_20d"] for o in kept]
        ftv = [o["follow_through"] for o in kept if o["follow_through"] is not None]
        by_regime = {}
        for rg in set(o["regime"] for o in kept):
            by_regime[rg] = _stats([o["fwd_20d"] for o in kept if o["regime"] == rg])
        summary[name] = {
            "occurrences": len(kept), "raw_detections": len(lst),
            "median_duration": int(np.median([o["duration"] for o in kept])) if kept else 0,
            "duration_p25_p75": ([int(np.percentile([o["duration"] for o in kept], 25)),
                                  int(np.percentile([o["duration"] for o in kept], 75))] if kept else None),
            "breakout_up_pct": round(100 * np.mean([o["breakout_up"] for o in kept]), 1) if kept else None,
            "follow_through_pct": round(100 * np.mean(ftv), 1) if len(ftv) >= MIN_CELL else None,
            "confirmation": "3 sessions past final extremum; direction = close vs smoothed level",
            "distributions": {f"{h}d": _stats([o[f"fwd_{h}d"] for o in kept]) for h in HORIZONS},
            "by_regime": by_regime,
            "by_volume": {"rising": _stats([o["fwd_20d"] for o in kept if o["volume_slope"] > 0]),
                          "falling": _stats([o["fwd_20d"] for o in kept if o["volume_slope"] < 0])},
            "by_volatility": {"high": _stats([o["fwd_20d"] for o in kept if o["vol_pctile"] >= 0.67]),
                              "low": _stats([o["fwd_20d"] for o in kept if o["vol_pctile"] <= 0.33])},
            "examples": sorted(kept, key=lambda o: o["end"], reverse=True)[:12]}
    art = {"generated": date.today().isoformat(),
           "method": ("Gaussian-smoothed closes (sigma=3), alternating extrema, LMW five-point "
                      "templates (1.5% tol); outcomes from a 3-session confirmation bar; "
                      "per-ticker non-overlapping; horizons NaN individually near data end; "
                      "cells under 15 occurrences report null"),
           "universe": len(tickers), "formations": summary}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(art))
    logger.info(f"[formations v2] {sum(v['occurrences'] for v in summary.values())} occurrences")
    return {k: v["occurrences"] for k, v in summary.items()}
