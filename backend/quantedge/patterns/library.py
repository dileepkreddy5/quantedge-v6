"""Pattern Lab — window library builder.

Slides fixed-length windows over every ticker with enough history, stores each
as a z-normalized close trajectory plus its context (volume slope, forward
returns, market regime), and writes one compact .npz per window length.

Design decisions, and why:
- Z-normalized closes (not returns): shape matching per Lo/Mamaysky and the
  PR-DTW literature — a $180 stock and a $900 stock with the same trajectory
  must be the same pattern. Mean 0, std 1 per window.
- Stride 5 (weekly), not 1: adjacent daily windows are near-duplicates that
  bloat the library 5x and then get deduped at query time anyway. Weekly
  stride keeps every distinct episode reachable within 2 DTW warping steps.
- Forward returns computed from the window's LAST bar, at +1/+5/+20/+60d,
  stored alongside — the outcome is baked at build time, never recomputed.
- Windows whose forward horizon runs past the data end carry NaN outcomes and
  are excluded from distributions (not from matching).
- Volume slope: OLS slope of z-normalized volume over the window — the
  stage-2 dimension, captured now so the library needn't rebuild.
- float16 trajectories: 60d x ~400k windows ~ 48MB. Precision loss is far
  below the noise floor of price data.
"""
from __future__ import annotations
import numpy as np
from datetime import date
from loguru import logger

WINDOWS = (20, 60)
STRIDE = 5
HORIZONS = (1, 5, 20, 60, 120, 252)  # long horizons NaN near data end; stats exclude them
MIN_BARS = 750          # ~3y minimum history to contribute
MIN_PRICE = 3.0         # sub-$3 names: spreads dominate shape
MIN_DOLLAR_VOL = 1e6    # thinly traded shapes are microstructure, not pattern


def _znorm(x: np.ndarray) -> np.ndarray | None:
    s = x.std()
    if not np.isfinite(s) or s < 1e-9:
        return None                      # flat window: no shape to match
    return (x - x.mean()) / s


async def build_library(pool, out_dir: str) -> dict:
    from pathlib import Path
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    tickers = [r["ticker"] for r in await pool.fetch(
        "SELECT ticker FROM daily_bars GROUP BY ticker HAVING count(*) >= $1", MIN_BARS)]
    logger.info(f"[patterns] building library from {len(tickers)} tickers")

    # SPY regime by date: tag every window with the market state it formed in.
    spy = await pool.fetch("SELECT d, c FROM daily_bars WHERE ticker='SPY' ORDER BY d")
    regime_by_date: dict[date, str] = {}
    if len(spy) > 220:
        closes = np.array([r["c"] for r in spy]); ds = [r["d"] for r in spy]
        for i in range(220, len(closes)):
            ret63 = closes[i] / closes[i - 63] - 1
            vol21 = np.std(np.diff(np.log(closes[i - 21:i + 1]))) * np.sqrt(252)
            trend = "BULL" if ret63 > 0 else "BEAR"
            vol = "HIGH_VOL" if vol21 > 0.18 else "LOW_VOL"
            regime_by_date[ds[i]] = f"{trend}_{vol}"

    # SPY forward returns by date-position: lets every window carry the
    # market's move over its own outcome horizon, so the engine can report
    # EXCESS return ("did this beat what the market did anyway?") instead of
    # raw return alone.
    spy_c = np.array([r["c"] for r in spy], dtype=np.float64) if len(spy) else np.zeros(0)
    spy_pos = {r["d"].toordinal(): i for i, r in enumerate(spy)}

    stats = {}
    for W in WINDOWS:
        trajs, vslopes, fwd = [], [], {h: [] for h in HORIZONS}
        spy_fwd = {h: [] for h in HORIZONS}
        meta_tk, meta_start, meta_end = [], [], []
        vol_pcts, d52hs, d52ls = [], [], []
        skipped = {"flat": 0, "price": 0, "liquidity": 0}
        regimes = []
        for tk in tickers:
            rows = await pool.fetch(
                "SELECT d, c, v FROM daily_bars WHERE ticker=$1 ORDER BY d", tk)
            if len(rows) < W + max(HORIZONS) + 1:
                continue
            c = np.array([r["c"] for r in rows], dtype=np.float64)
            v = np.array([float(r["v"] or 0) for r in rows], dtype=np.float64)
            ds = [r["d"] for r in rows]
            # Per-date context for conditional filtering: 21d realized vol and
            # its percentile vs the trailing year; distance from 52w high/low.
            lr = np.diff(np.log(np.maximum(c, 1e-9)), prepend=np.log(max(c[0], 1e-9)))
            v21 = np.array([np.std(lr[max(0, j - 20):j + 1]) for j in range(len(c))])
            # Bound by the SHORTEST horizon: longer horizons NaN out per-window.
            # Bounding by max(HORIZONS) silently dropped the most recent ~10
            # months of windows — the panel truncation bug, reintroduced here
            # when 120/252d were added. Caught by the window count falling
            # 509k -> 421k on rebuild.
            for i in range(0, len(c) - W - min(HORIZONS), STRIDE):
                seg = c[i:i + W]
                if seg.min() < MIN_PRICE:
                    skipped["price"] += 1; continue
                if np.median(seg * v[i:i + W]) < MIN_DOLLAR_VOL:
                    skipped["liquidity"] += 1; continue
                z = _znorm(seg)
                if z is None:
                    skipped["flat"] += 1; continue
                vz = _znorm(v[i:i + W])
                vs = 0.0 if vz is None else float(
                    np.polyfit(np.arange(W), vz, 1)[0])
                end = i + W - 1
                trajs.append(z.astype(np.float16))
                vslopes.append(vs)
                for h in HORIZONS:
                    fwd[h].append(c[end + h] / c[end] - 1
                                  if end + h < len(c) else np.nan)
                meta_tk.append(tk); meta_start.append(ds[i].toordinal())
                meta_end.append(ds[end].toordinal())
                _lo = max(0, end - 251)
                vol_pcts.append(float((v21[_lo:end] < v21[end]).mean()) if end > _lo else 0.5)
                d52hs.append(float(c[end] / c[_lo:end + 1].max() - 1))
                d52ls.append(float(c[end] / c[_lo:end + 1].min() - 1))
                sp = spy_pos.get(ds[end].toordinal())
                for h in HORIZONS:
                    spy_fwd[h].append(spy_c[sp + h] / spy_c[sp] - 1
                                      if sp is not None and sp + h < len(spy_c)
                                      else np.nan)
                regimes.append(regime_by_date.get(ds[end], "UNKNOWN"))
        np.savez_compressed(
            out / f"library_{W}d.npz",
            trajs=np.stack(trajs) if trajs else np.zeros((0, W), np.float16),
            vslope=np.array(vslopes, np.float32),
            **{f"fwd_{h}d": np.array(fwd[h], np.float32) for h in HORIZONS},
            **{f"spy_fwd_{h}d": np.array(spy_fwd[h], np.float32) for h in HORIZONS},
            ticker=np.array(meta_tk), start_ord=np.array(meta_start, np.int32),
            end_ord=np.array(meta_end, np.int32),
            vol_pct=np.array(vol_pcts, np.float32),
            d52h=np.array(d52hs, np.float32), d52l=np.array(d52ls, np.float32),
            regime=np.array(regimes),
        )
        stats[W] = {"windows": len(trajs), "skipped": skipped}
        logger.info(f"[patterns] {W}d: {len(trajs)} windows "
                    f"(skipped {skipped})")
    return stats
