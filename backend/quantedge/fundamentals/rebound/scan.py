"""Rebound scan v1 — discounted quality that has stopped falling.

The original engine referenced by the job never existed in this repo (its
signature wanted sqlite paths and an insider-trading DB from an external
research environment). This is a from-scratch honest v1 on the data we have:
daily_bars (5y), universe market caps, and a quality join against the
nightly multibagger artifact. No insider dimension — that data source is
not on this plan, so it is absent rather than imitated.

Gates (all must pass; each one's purpose stated):
  1. price >= $3, median dollar-vol >= $1M      — microstructure floor
  2. drawdown <= -30% from the 2y prior high    — meaningfully discounted
  3. trough >= 20 sessions ago                   — a base has had time to form
  4. close >= trough * 1.10                      — bounced 10%: not a falling knife
  5. 20d log-price slope >= 0                    — decline has actually stopped

Recovery progress = (close - trough) / (prior_high - trough), staged:
  stabilizing < 25% <= recovering < 60% <= late_recovery.
"""
from __future__ import annotations
import json
import numpy as np
from datetime import date
from loguru import logger
from core.artifact_paths import artifact_read_path


async def run_scan(pool) -> dict:
    tickers = await pool.fetch("""
        SELECT b.ticker, u.name, u.market_cap
        FROM (SELECT ticker FROM daily_bars GROUP BY ticker HAVING count(*) >= 500) b
        LEFT JOIN universe u ON u.ticker = b.ticker""")
    # Quality join: multibagger artifact (score, piotroski) where available.
    quality = {}
    mp = artifact_read_path("scan_artifact.json")
    if mp:
        try:
            mb = json.loads(mp.read_text())
            for tier_rows in (mb.get("tiers") or {}).values():
                for r in tier_rows:
                    quality[r["ticker"]] = {"mb_score": r.get("score"),
                                            "piotroski": r.get("piotroski")}
        except Exception:
            pass

    rows, gate_counts = [], {"universe": len(tickers), "liquidity": 0,
                             "discounted": 0, "base_formed": 0,
                             "off_trough": 0, "stabilized": 0}
    for t in tickers:
        tk = t["ticker"]
        bars = await pool.fetch(
            "SELECT d, c, v FROM daily_bars WHERE ticker=$1 ORDER BY d", tk)
        c = np.array([r["c"] for r in bars], np.float64)
        v = np.array([float(r["v"] or 0) for r in bars], np.float64)
        ds = [r["d"] for r in bars]
        if c[-1] < 3.0 or np.median(c[-60:] * v[-60:]) < 1e6:
            continue
        gate_counts["liquidity"] += 1
        look = min(len(c), 504)                       # 2y prior-high window
        hi_i = len(c) - look + int(np.argmax(c[-look:]))
        prior_high = float(c[hi_i])
        dd = c[-1] / prior_high - 1
        if dd > -0.30:
            continue
        gate_counts["discounted"] += 1
        tr_i = hi_i + int(np.argmin(c[hi_i:]))
        trough = float(c[tr_i])
        if len(c) - 1 - tr_i < 20:
            continue
        gate_counts["base_formed"] += 1
        if c[-1] < trough * 1.10:
            continue
        gate_counts["off_trough"] += 1
        slope20 = float(np.polyfit(np.arange(20), np.log(c[-20:]), 1)[0])
        if slope20 < 0:
            continue
        gate_counts["stabilized"] += 1
        prog = (c[-1] - trough) / (prior_high - trough) * 100 if prior_high > trough else 0.0
        stage = ("stabilizing" if prog < 25 else
                 "recovering" if prog < 60 else "late_recovery")
        mc = float(t["market_cap"] or 0)
        tier = "small" if mc < 2e9 else "mid" if mc < 1e10 else "large"
        q = quality.get(tk, {})
        rows.append({
            # Schema the existing /rebound router's _shape() expects.
            "ticker": tk, "name": t["name"] or tk, "tier": tier,
            # score = recovery progress (0-100, measured) plus a quality nudge
            # where the multibagger scan knows the name. Documented, not tuned.
            "score": round(prog + (10 if q.get("mb_score") else 0), 1),
            "stage": stage,
            "drawdown": round(abs(dd), 4),          # positive fraction per router
            "price": round(float(c[-1]), 2),        # latest close at scan time
            "prior_high": round(prior_high, 2),
            "prior_high_date": ds[hi_i].isoformat(),
            "trough": round(trough, 2), "trough_date": ds[tr_i].isoformat(),
            "days_since_low": int(len(c) - 1 - tr_i),
            "off_trough_pct": round((c[-1] / trough - 1) * 100, 1),
            "slope_20d_ann_pct": round(slope20 * 252 * 100, 1),
            "thesis": (f"{abs(dd)*100:.0f}% off its {ds[hi_i].year} high, trough "
                       f"{int(len(c)-1-tr_i)} sessions back, {((c[-1]/trough-1)*100):.0f}% "
                       f"recovered, 20d slope positive"),
            "recovery": {"progress_pct": round(prog, 1),
                         "reached_high": bool(c[-1] >= prior_high),
                         "upside_to_high_pct": round((prior_high / c[-1] - 1) * 100, 1)},
            **q,
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    tiers = {tn: [r for r in rows if r["tier"] == tn][:50]
             for tn in ("small", "mid", "large")}
    art = {"generated": date.today().isoformat(),
           "as_of": date.today().isoformat(),
           "tiers": tiers,
           "method": ("v1: 2y prior high, >=30% drawdown, trough >=20 sessions old, "
                      ">=10% bounce off trough, non-negative 20d slope; quality "
                      "fields joined from the multibagger scan where present; no "
                      "insider dimension — that data source is not available"),
           "stage_counts": gate_counts, "n_universe": len(tickers),
           "n_passed_gates": len(rows), "rows": rows}
    logger.info(f"[rebound v1] {len(rows)} names passed of {len(tickers)} "
                f"(gates: {gate_counts})")
    return art
