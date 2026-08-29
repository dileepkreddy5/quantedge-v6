"""Pattern Lab — analog query engine.

Given a ticker's current W-day trajectory, finds its nearest historical
analogs and returns the empirical distribution of what followed.

Rigor decisions, each preventing a specific known failure:
- Two-stage matching: z-normalized Euclidean distance against the whole
  library as one vectorized pass, then Sakoe-Chiba-banded DTW (band 10% of W)
  to re-rank the top pool. Full DTW everywhere melts 3 vCPUs; Euclidean-only
  misses time-warped twins (UCR literature: banded DTW on a Euclidean
  shortlist recovers nearly all of full DTW's accuracy).
- Episode dedup: windows overlapping in (ticker, time) are the same episode
  photographed twice. Overlap inflation is precisely the panel's IC bug in
  new clothes — analogs are greedily deduped to non-overlapping episodes per
  ticker before any statistic is computed. Reported n = episodes.
- Self-match exclusion: the query ticker's own recent windows (those
  overlapping the query period) are excluded — matching yourself is not
  evidence.
- Base rate alongside every figure: 64% positive means nothing without the
  library-wide positive rate over the same horizon. Both always ship.
- NaN outcomes (window too close to data end) excluded from distributions,
  and the exclusion counted.
- Splits by volume-slope tercile and by formation regime (RQ5/RQ6) are only
  reported when a cell has >= 15 episodes; below that: null, not a number.
"""
from __future__ import annotations
import numpy as np
from datetime import date, timedelta
from pathlib import Path
from loguru import logger

HORIZONS = (1, 5, 20, 60, 120, 252)
TOP_EUCLID = 300      # pool re-ranked by DTW
TOP_RETURN = 120      # analogs kept after DTW, before dedup
MIN_CELL = 15         # minimum episodes for a split cell


def _znorm(x: np.ndarray) -> np.ndarray | None:
    s = x.std()
    if not np.isfinite(s) or s < 1e-9:
        return None
    return (x - x.mean()) / s


def _dtw_banded(a: np.ndarray, b: np.ndarray, band: int) -> float:
    n = len(a)
    D = np.full((n + 1, n + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(n, i + band)
        ai = a[i - 1]
        for j in range(lo, hi + 1):
            d = (ai - b[j - 1]) ** 2
            D[i, j] = d + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, n]))


def _dist_stats(vals: np.ndarray) -> dict | None:
    v = vals[np.isfinite(vals)]
    if len(v) == 0:
        return None
    return {
        "n": int(len(v)),
        "positive_pct": round(float((v > 0).mean()) * 100, 1),
        "median_pct": round(float(np.median(v)) * 100, 2),
        "mean_pct": round(float(v.mean()) * 100, 2),
        "p10_pct": round(float(np.percentile(v, 10)) * 100, 2),
        "p25_pct": round(float(np.percentile(v, 25)) * 100, 2),
        "p75_pct": round(float(np.percentile(v, 75)) * 100, 2),
        "p90_pct": round(float(np.percentile(v, 90)) * 100, 2),
        "negative_pct": round(float((v < 0).mean()) * 100, 1),
        "outcome_vol_pct": round(float(v.std()) * 100, 2),
    }


class PatternLibrary:
    def __init__(self, lib_dir: str):
        self.dir = Path(lib_dir)
        self._cache: dict[int, dict] = {}

    def load(self, W: int) -> dict | None:
        if W in self._cache:
            return self._cache[W]
        p = self.dir / f"library_{W}d.npz"
        if not p.exists():
            return None
        z = np.load(p, allow_pickle=False)
        lib = {k: z[k] for k in z.files}
        lib["trajs_f32"] = lib["trajs"].astype(np.float32)
        # Library-wide base rates, computed once per load.
        lib["base"] = {h: _dist_stats(lib[f"fwd_{h}d"]) for h in HORIZONS}
        self._cache[W] = lib
        logger.info(f"[patterns] loaded {W}d library: {len(lib['trajs'])} windows")
        return lib

    def query(self, closes: np.ndarray, ticker: str, W: int,
              query_end: date, filters: dict | None = None) -> dict | None:
        """filters: optional conditional matching over the retained episodes —
        {'volume': 'rising'|'falling', 'vola': 'high'|'low',
         'regime': <regime str>, 'extreme': 'near_high'|'near_low'}.
        Applied AFTER dedup, so n shrinks and the response says by how much."""
        lib = self.load(W)
        if lib is None or len(closes) < W:
            return None
        q = _znorm(closes[-W:].astype(np.float64))
        if q is None:
            return None

        T = lib["trajs_f32"]                                  # (N, W)
        # Stage 1: vectorized z-Euclidean over the entire library.
        d2 = ((T - q.astype(np.float32)) ** 2).sum(axis=1)

        # Self-match exclusion: same ticker, window overlapping the query span.
        q_start_ord = (query_end - timedelta(days=int(W * 1.6))).toordinal()
        self_mask = (lib["ticker"] == ticker) & (lib["start_ord"] >= q_start_ord)
        d2[self_mask] = np.inf

        idx = np.argpartition(d2, TOP_EUCLID)[:TOP_EUCLID]
        # Stage 2: banded DTW re-rank of the shortlist.
        band = max(2, W // 10)
        dtw = np.array([_dtw_banded(q, T[i].astype(np.float64), band)
                        for i in idx])
        order = idx[np.argsort(dtw)][:TOP_RETURN]
        dtw_sorted = np.sort(dtw)[:TOP_RETURN]

        # Episode dedup: greedy, best-first, per (ticker, time) overlap.
        kept, kept_d, seen = [], [], {}
        for pos, i in enumerate(order):
            tk = str(lib["ticker"][i]); s0 = int(lib["start_ord"][i])
            spans = seen.setdefault(tk, [])
            # calendar-day overlap test against episodes already kept
            if any(abs(s0 - s1) < int(W * 1.4) for s1 in spans):
                continue
            spans.append(s0)
            kept.append(int(i)); kept_d.append(float(dtw_sorted[pos]))
        kept = np.array(kept, dtype=int)
        if len(kept) == 0:
            return None

        pre_filter_n = len(kept)
        applied = {}
        if filters:
            m = np.ones(len(kept), dtype=bool)
            vs = lib["vslope"][kept]
            if filters.get("volume") == "rising":  m &= vs > 0; applied["volume"] = "rising"
            if filters.get("volume") == "falling": m &= vs < 0; applied["volume"] = "falling"
            vp = lib.get("vol_pct")
            if vp is not None and filters.get("vola") == "high":
                m &= vp[kept] >= 0.67; applied["vola"] = "high"
            if vp is not None and filters.get("vola") == "low":
                m &= vp[kept] <= 0.33; applied["vola"] = "low"
            if filters.get("regime"):
                m &= lib["regime"][kept] == filters["regime"]; applied["regime"] = filters["regime"]
            dh, dl = lib.get("d52h"), lib.get("d52l")
            if dh is not None and filters.get("extreme") == "near_high":
                m &= dh[kept] >= -0.05; applied["extreme"] = "near_high"
            if dl is not None and filters.get("extreme") == "near_low":
                m &= dl[kept] <= 0.10; applied["extreme"] = "near_low"
            kept = kept[m]
            kept_d = [d for d, keep in zip(kept_d, m) if keep]
            if len(kept) == 0:
                return {"window_days": W, "episodes": 0,
                        "pre_filter_episodes": pre_filter_n,
                        "filters_applied": applied,
                        "insufficient": "no episodes match these conditions"}

        # Similarity as a percentage users can read: normalized against the
        # distance of a completely dissimilar pair (~2*sqrt(W) for z-series).
        worst = 2.0 * np.sqrt(W)
        sims = np.clip(1.0 - np.array(kept_d) / worst, 0, 1) * 100

        out = {
            "window_days": W,
            "episodes": int(len(kept)),
            "pre_filter_episodes": int(pre_filter_n),
            "filters_applied": applied,
            "search_scope": {"windows_searched": int(len(T)),
                             "tickers_in_library": int(len(np.unique(lib["ticker"])))},
            "windows_matched_before_dedup": int(TOP_RETURN),
            "distributions": {}, "base_rates": lib["base"],
            "splits": {"volume_slope": {}, "regime": {}},
            "analogs": [],
        }
        for h in HORIZONS:
            out["distributions"][f"{h}d"] = _dist_stats(lib[f"fwd_{h}d"][kept])
            # Excess vs SPY over each episode's own dates: the question is not
            # "did it go up" but "did it beat what the market did anyway".
            sk = lib.get(f"spy_fwd_{h}d")
            if sk is not None:
                ex = lib[f"fwd_{h}d"][kept] - sk[kept]
                out.setdefault("excess_vs_spy", {})[f"{h}d"] = _dist_stats(ex)

        # RQ5: volume-slope terciles of the matched episodes.
        vs = lib["vslope"][kept]
        if len(kept) >= MIN_CELL * 2:
            lo_t, hi_t = np.percentile(vs, 33), np.percentile(vs, 67)
            for name, m in (("rising", vs >= hi_t), ("flat", (vs > lo_t) & (vs < hi_t)),
                            ("falling", vs <= lo_t)):
                cell = lib["fwd_20d"][kept[m]]
                out["splits"]["volume_slope"][name] = (
                    _dist_stats(cell) if m.sum() >= MIN_CELL else None)

        # RQ6: formation-regime split.
        rg = lib["regime"][kept]
        for r in np.unique(rg):
            m = rg == r
            out["splits"]["regime"][str(r)] = (
                _dist_stats(lib["fwd_20d"][kept[m]]) if m.sum() >= MIN_CELL else None)

        # Top analogs for the overlay chart.
        for rank, (i, s) in enumerate(zip(kept[:12], sims[:12])):
            _end = (date.fromordinal(int(lib["end_ord"][i])).isoformat()
                    if "end_ord" in lib else None)
            out["analogs"].append({
                "vol_pctile": (round(float(lib["vol_pct"][i]), 2) if "vol_pct" in lib else None),
                "dist_52w_high": (round(float(lib["d52h"][i]) * 100, 1) if "d52h" in lib else None),
                "ticker": str(lib["ticker"][i]),
                "start": date.fromordinal(int(lib["start_ord"][i])).isoformat(),
                "end": _end,
                "duration_sessions": W,
                "regime": str(lib["regime"][i]),
                "volume_slope": round(float(lib["vslope"][i]), 3),
                "similarity_pct": round(float(s), 1),
                "trajectory": [round(float(x), 3) for x in lib["trajs"][i]],
                "fwd": {f"{h}d": (None if not np.isfinite(lib[f"fwd_{h}d"][i])
                                  else round(float(lib[f"fwd_{h}d"][i]) * 100, 2))
                        for h in HORIZONS},
            })
        out["query_trajectory"] = [round(float(x), 3) for x in q]
        out["episodes_for_paths"] = [
            {"ticker": str(lib["ticker"][i]),
             "end": date.fromordinal(int(lib["end_ord"][i])).isoformat()}
            for i in kept[:60] if "end_ord" in lib]
        out["method"] = {
            "normalization": "z-score per window (mean 0, std 1) — scale-free shape",
            "stage1": "z-normalized Euclidean distance, full library, vectorized",
            "stage2": f"Sakoe-Chiba banded DTW (band {max(2, W // 10)}) on top {TOP_EUCLID}",
            "dedup": "greedy non-overlapping episodes per ticker",
            "look_ahead": "outcomes baked at build time strictly after window end",
            "parameter_fitting": ("none — analog matching fits no parameters, so a "
                                  "discovery/validation split is not applicable; controls "
                                  "are dedup, base rates, and date-range disclosure"),
        }
        ords = lib["start_ord"][kept]
        out["episode_date_range"] = [date.fromordinal(int(ords.min())).isoformat(),
                                     date.fromordinal(int(ords.max())).isoformat()]
        out["caveat"] = (
            "Library spans 2021-2026 — one macro era. Distributions answer "
            "'what followed this shape in recent regimes', not a universal law. "
            "n counts non-overlapping episodes, not raw windows.")
        return out
