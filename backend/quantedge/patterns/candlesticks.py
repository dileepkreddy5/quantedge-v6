"""Pattern Lab — Japanese candlestick patterns, detected and then MEASURED.

Ten patterns the academic tests actually examine (Marshall, Young & Rose 2006;
Lu et al. 2012). Detection is the textbook geometry; the verdict comes from
scanning the whole universe and recording what followed — a pattern with no
measurable edge in five years of data will say so on the chart.
"""
from __future__ import annotations
import numpy as np
from datetime import date
from loguru import logger

HORIZONS = (5, 21, 63, 126, 252)
MIN_OCC = 200

def detect(o, h, l, c) -> list[dict]:
    """Returns [{i, name, direction}] with i = index of the completing candle."""
    n = len(c); out = []
    body = np.abs(c - o); rng = np.maximum(h - l, 1e-9)
    upper = h - np.maximum(o, c); lower = np.minimum(o, c) - l
    avg_body = np.array([body[max(0, i - 10):i].mean() if i > 0 else body[i] for i in range(n)]) + 1e-9
    sma10 = np.array([c[max(0, i - 10):i].mean() if i > 0 else c[i] for i in range(n)])
    for i in range(2, n):
        down = c[i - 1] < sma10[i - 1]; up = c[i - 1] > sma10[i - 1]
        # doji: body < 10% of range
        if body[i] <= 0.1 * rng[i]:
            out.append({"i": i, "name": "doji", "direction": "neutral"})
        # hammer / hanging man: small body near top, lower shadow >= 2x body
        if body[i] <= 0.35 * rng[i] and lower[i] >= 2 * body[i] and upper[i] <= 0.15 * rng[i]:
            out.append({"i": i, "name": "hammer" if down else "hanging_man", "direction": "bullish" if down else "bearish"})
        # engulfing
        if c[i] > o[i] and c[i - 1] < o[i - 1] and o[i] <= c[i - 1] and c[i] >= o[i - 1] and body[i] > body[i - 1]:
            out.append({"i": i, "name": "bullish_engulfing", "direction": "bullish"})
        if c[i] < o[i] and c[i - 1] > o[i - 1] and o[i] >= c[i - 1] and c[i] <= o[i - 1] and body[i] > body[i - 1]:
            out.append({"i": i, "name": "bearish_engulfing", "direction": "bearish"})
        # harami: small body inside prior large body
        if body[i - 1] > 1.2 * avg_body[i - 1] and body[i] < 0.5 * body[i - 1] and \
           max(o[i], c[i]) <= max(o[i - 1], c[i - 1]) and min(o[i], c[i]) >= min(o[i - 1], c[i - 1]):
            out.append({"i": i, "name": "bullish_harami" if c[i - 1] < o[i - 1] else "bearish_harami",
                        "direction": "bullish" if c[i - 1] < o[i - 1] else "bearish"})
        # morning / evening star: big candle, small gap candle, big reversal candle closing past midpoint
        b2, b1, b0 = body[i - 2], body[i - 1], body[i]
        if b2 > 1.2 * avg_body[i - 2] and b1 < 0.3 * b2 and b0 > 0.6 * b2:
            mid2 = (o[i - 2] + c[i - 2]) / 2
            if c[i - 2] < o[i - 2] and c[i] > o[i] and c[i] > mid2:
                out.append({"i": i, "name": "morning_star", "direction": "bullish"})
            if c[i - 2] > o[i - 2] and c[i] < o[i] and c[i] < mid2:
                out.append({"i": i, "name": "evening_star", "direction": "bearish"})
        # three white soldiers / black crows: three consecutive strong same-direction bodies, each closing beyond the last
        if all(c[j] > o[j] and body[j] > 0.8 * avg_body[j] for j in (i - 2, i - 1, i)) and c[i] > c[i - 1] > c[i - 2]:
            out.append({"i": i, "name": "three_white_soldiers", "direction": "bullish"})
        if all(c[j] < o[j] and body[j] > 0.8 * avg_body[j] for j in (i - 2, i - 1, i)) and c[i] < c[i - 1] < c[i - 2]:
            out.append({"i": i, "name": "three_black_crows", "direction": "bearish"})
    return out


def _stats(v):
    v = np.array([x for x in v if x is not None and np.isfinite(x)])
    if len(v) < MIN_OCC: return None
    return {"n": int(len(v)), "positive_pct": round(float((v > 0).mean()) * 100, 1),
            "median_pct": round(float(np.median(v)) * 100, 2),
            "p25_pct": round(float(np.percentile(v, 25)) * 100, 2), "p75_pct": round(float(np.percentile(v, 75)) * 100, 2)}


async def scan_candlesticks(pool, out_path: str) -> dict:
    """Universe-wide measurement: for every occurrence, forward returns at each
    horizon from the next session's close; split by regime and by period
    (2021-2024 vs 2025+) so a faded edge is visible. Nightly, not on request."""
    import json
    from pathlib import Path
    tickers = [r["ticker"] for r in await pool.fetch(
        "SELECT ticker FROM daily_bars GROUP BY ticker HAVING count(*) >= 750")]
    spy = await pool.fetch("SELECT d, c FROM daily_bars WHERE ticker='SPY' ORDER BY d")
    regime = {}
    if len(spy) > 220:
        sc = np.array([r["c"] for r in spy]); sds = [r["d"] for r in spy]
        for i in range(220, len(sc)):
            regime[sds[i]] = ("BULL" if sc[i] / sc[i - 63] > 1 else "BEAR") + ("_HIGH_VOL" if np.std(np.diff(np.log(sc[i - 21:i + 1]))) * np.sqrt(252) > 0.18 else "_LOW_VOL")
    occ = {}
    base = {h: [] for h in HORIZONS}
    for tk in tickers:
        rows = await pool.fetch("SELECT d, o, h, l, c, v FROM daily_bars WHERE ticker=$1 ORDER BY d", tk)
        c = np.array([r["c"] for r in rows], np.float64)
        if c.min() < 3.0: continue
        vol = np.array([float(r["v"] or 0) for r in rows]); v20 = np.array([vol[max(0, i - 20):i].mean() if i else vol[i] for i in range(len(vol))]) + 1e-9
        o = np.array([r["o"] or r["c"] for r in rows], np.float64); h = np.array([r["h"] or r["c"] for r in rows], np.float64)
        l = np.array([r["l"] or r["c"] for r in rows], np.float64); ds = [r["d"] for r in rows]
        # base rate sample: every 5th session
        for i in range(60, len(c) - 5, 5):
            for hz in HORIZONS:
                base[hz].append(c[i + hz] / c[i] - 1 if i + hz < len(c) else None)
        for occ_ in detect(o, h, l, c):
            i = occ_["i"]
            if i + 1 >= len(c): continue
            entry = c[i + 1]   # next session's close: no look-ahead on the completing candle
            rec = {"regime": regime.get(ds[i], "UNKNOWN"), "period": "2021-2024" if ds[i].year <= 2024 else "2025+",
                   "vol_confirmed": bool(vol[i] >= 1.5 * v20[i])}   # Lee-Swaminathan: volume-confirmed vs not
            for hz in HORIZONS:
                rec[hz] = c[i + 1 + hz] / entry - 1 if i + 1 + hz < len(c) else None
            occ.setdefault(occ_["name"], []).append(rec)
    summary = {}
    for name, lst in occ.items():
        by_h = {}
        for hz in HORIZONS:
            by_h[f"{hz}d"] = {"all": _stats([r[hz] for r in lst]),
                              "by_regime": {rg: _stats([r[hz] for r in lst if r["regime"] == rg]) for rg in set(r["regime"] for r in lst)},
                              "by_period": {p: _stats([r[hz] for r in lst if r["period"] == p]) for p in ("2021-2024", "2025+")},
                              "by_volume": {"confirmed": _stats([r[hz] for r in lst if r["vol_confirmed"]]),
                                            "unconfirmed": _stats([r[hz] for r in lst if not r["vol_confirmed"]])}}
        summary[name] = {"occurrences": len(lst), "horizons": by_h}
    art = {"generated": date.today().isoformat(), "universe": len(tickers), "min_occurrences": MIN_OCC,
           "base": {f"{hz}d": _stats(base[hz]) for hz in HORIZONS},
           "method": ("textbook candlestick geometry; entry at the NEXT session's close after the completing "
                      "candle; forward returns at 5/21/63/126/252 sessions; split by SPY regime and by period; "
                      f"cells under {MIN_OCC} occurrences report null"),
           "patterns": summary}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True); Path(out_path).write_text(json.dumps(art))
    logger.info(f"[candlesticks] {sum(v['occurrences'] for v in summary.values())} occurrences, {len(summary)} patterns")
    return {k: v["occurrences"] for k, v in summary.items()}
