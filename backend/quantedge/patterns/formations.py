"""Pattern Lab Phase 2 — classical formation detection (Lo-Mamaysky-Wang style).

Method: Gaussian kernel smoothing of closes, local extrema of the smoothed
series, formation classification from consecutive extrema sequences, with
tolerances per LMW's formalization. Detection uses data <= T only; outcomes
are measured strictly after the formation completes. The nightly scan walks
every deep-history ticker and records completed formations with breakout
direction and forward returns — profitability is CALCULATED, never asserted.
"""
from __future__ import annotations
import numpy as np
from datetime import date
from loguru import logger

TOL = 0.015          # LMW peak-equality tolerance (1.5%)
SMOOTH_SIGMA = 3.0   # smoothing bandwidth in sessions
MIN_EXTREMA_GAP = 3  # sessions between extrema
HORIZONS = (5, 20, 60)


def _smooth(c: np.ndarray, sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    r = int(4 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    pad = np.pad(c, r, mode="edge")
    return np.convolve(pad, k, mode="valid")


def _extrema(sm: np.ndarray) -> list[tuple[int, int]]:
    """[(index, +1 max / -1 min)] alternating, min gap enforced."""
    ext = []
    for i in range(1, len(sm) - 1):
        if sm[i] > sm[i - 1] and sm[i] >= sm[i + 1]:
            t = 1
        elif sm[i] < sm[i - 1] and sm[i] <= sm[i + 1]:
            t = -1
        else:
            continue
        if ext and ext[-1][1] == t:            # same type: keep the more extreme
            if (t == 1 and sm[i] > sm[ext[-1][0]]) or (t == -1 and sm[i] < sm[ext[-1][0]]):
                ext[-1] = (i, t)
            continue
        if ext and i - ext[-1][0] < MIN_EXTREMA_GAP:
            continue
        ext.append((i, t))
    return ext


def _near(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) / ((a + b) / 2) < tol


def _slope(idx: np.ndarray, val: np.ndarray) -> float:
    return float(np.polyfit(idx, val, 1)[0]) if len(idx) >= 2 else 0.0


def classify_last5(c: np.ndarray, ext: list[tuple[int, int]], at: int) -> str | None:
    """Classify the formation ending at extrema position `at` (index into ext),
    using that extremum and the four before it — LMW's five-point templates,
    plus trendline formations from the same points."""
    if at < 4:
        return None
    e = ext[at - 4:at + 1]
    idx = np.array([i for i, _ in e]); typ = [t for _, t in e]
    v = c[idx]
    if typ == [1, -1, 1, -1, 1]:               # max-min-max-min-max
        if v[2] > v[0] and v[2] > v[4] and _near(v[0], v[4]) and _near(v[1], v[3]):
            return "head_shoulders"
        if _near(v[0], v[2]) and _near(v[2], v[4]):
            return "triple_top"
        if _near(v[0], v[2]) and v[4] < v[2] * (1 - TOL):
            return "double_top"
        top_s = _slope(idx[[0, 2, 4]], v[[0, 2, 4]] / v[0])
        bot_s = _slope(idx[[1, 3]], v[[1, 3]] / v[0])
        if abs(top_s) < 5e-4 and bot_s > 5e-4:
            return "ascending_triangle"
        if top_s < -5e-4 and abs(bot_s) < 5e-4:
            return "descending_triangle"
        if top_s < -5e-4 and bot_s > 5e-4:
            return "symmetrical_triangle"
        if abs(top_s) < 5e-4 and abs(bot_s) < 5e-4 and not _near(v[0], v[1]):
            return "rectangle"
        if top_s > 5e-4 and bot_s > top_s:
            return "rising_wedge"
        if top_s < -5e-4 and bot_s < top_s * 0.999:
            return "falling_wedge"
    if typ == [-1, 1, -1, 1, -1]:              # min-max-min-max-min
        if v[2] < v[0] and v[2] < v[4] and _near(v[0], v[4]) and _near(v[1], v[3]):
            return "inv_head_shoulders"
        if _near(v[0], v[2]) and _near(v[2], v[4]):
            return "triple_bottom"
        if _near(v[0], v[2]) and v[4] > v[2] * (1 + TOL):
            return "double_bottom"
    return None


async def scan_formations(pool, out_path: str) -> dict:
    """Walk every deep-history ticker, record completed formations + outcomes."""
    import json
    from pathlib import Path
    tickers = [r["ticker"] for r in await pool.fetch(
        "SELECT ticker FROM daily_bars GROUP BY ticker HAVING count(*) >= 750")]
    occ: dict[str, list] = {}
    for tk in tickers:
        rows = await pool.fetch(
            "SELECT d, c, v FROM daily_bars WHERE ticker=$1 ORDER BY d", tk)
        c = np.array([r["c"] for r in rows], np.float64)
        if c.min() < 3.0:
            continue
        ds = [r["d"] for r in rows]
        sm = _smooth(c)
        ext = _extrema(sm)
        for at in range(4, len(ext)):
            name = classify_last5(c, ext, at)
            if name is None:
                continue
            end_i = ext[at][0]
            # confirmation bar: smoothing peeks ~sigma ahead; outcomes start
            # after a 3-session confirmation to stay strictly post-formation.
            conf = end_i + 3
            if conf + max(HORIZONS) >= len(c):
                continue
            entry = c[conf]
            fwd = {h: float(c[conf + h] / entry - 1) for h in HORIZONS}
            start_i = ext[at - 4][0]
            breakout_up = c[conf] > sm[end_i]
            occ.setdefault(name, []).append({
                "ticker": tk, "start": ds[start_i].isoformat(),
                "end": ds[end_i].isoformat(),
                "duration": int(end_i - start_i),
                "breakout_up": bool(breakout_up),
                **{f"fwd_{h}d": round(fwd[h] * 100, 2) for h in HORIZONS},
            })
    summary = {}
    for name, lst in occ.items():
        # Non-overlap per ticker: greedy by end date.
        lst.sort(key=lambda o: (o["ticker"], o["end"]))
        kept, last = [], {}
        for o in lst:
            le = last.get(o["ticker"])
            if le and (date.fromisoformat(o["end"]) - le).days < o["duration"]:
                continue
            last[o["ticker"]] = date.fromisoformat(o["end"]); kept.append(o)
        arr20 = np.array([o["fwd_20d"] for o in kept])
        summary[name] = {
            "occurrences": len(kept), "raw_detections": len(lst),
            "median_duration": int(np.median([o["duration"] for o in kept])) if kept else 0,
            "breakout_up_pct": round(100 * np.mean([o["breakout_up"] for o in kept]), 1) if kept else None,
            "fwd20": {
                "positive_pct": round(float((arr20 > 0).mean()) * 100, 1),
                "median_pct": round(float(np.median(arr20)), 2),
                "p25_pct": round(float(np.percentile(arr20, 25)), 2),
                "p75_pct": round(float(np.percentile(arr20, 75)), 2),
            } if len(kept) >= 15 else None,
            "examples": sorted(kept, key=lambda o: o["end"], reverse=True)[:12],
        }
    art = {"generated": date.today().isoformat(),
           "method": ("Gaussian-smoothed closes (sigma=3), alternating local extrema, "
                      "LMW five-point templates with 1.5% tolerance; outcomes measured "
                      "from a 3-session confirmation bar after formation end; "
                      "per-ticker non-overlapping occurrences"),
           "universe": len(tickers), "formations": summary}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(art))
    logger.info(f"[formations] {sum(v['occurrences'] for v in summary.values())} occurrences "
                f"across {len(summary)} formation types")
    return {k: v["occurrences"] for k, v in summary.items()}
