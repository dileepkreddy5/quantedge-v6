"""REBOUND Layer 4 — confirmation: is money actually moving in? (steps 15-16)

Volume half (step-15) — HONEST metrics only. True buy/sell volume does not
exist in daily bars (every trade has both sides); platforms showing "68%
buying" invent it. What daily bars DO support:

  vol_1w_ratio / vol_1m_ratio   recent average volume vs a 60-day baseline
                                that ENDS BEFORE the recent window (so a
                                volume spike can't inflate its own baseline)
  up_day_share_1w / _1m         fraction of volume traded on days the stock
                                CLOSED UP — the accumulation footprint:
                                heavy volume on up days + thin on down days
  accum_streak_weeks            consecutive weeks with up-day share > 55%
                                (persistent accumulation during a base is
                                the pattern; one spike is noise)

Buyback half (step-16) — Ikenberry-Lakonishok-Vermaelen (1995): open-market
repurchases AFTER price declines were followed by years of outperformance.
Management is the ultimate insider; buying back discounted shares with the
company's own cash is the strongest corporate confirmation there is.

  buyback_ttm / buyback_to_mktcap / active_through_decline

This layer never gates a stock in/out — it CONFIRMS. Its components feed the
score (step 20) and the TURNING stage detector (step 19).
PIT: buybacks filtered by filed <= as_of; bars end at as_of by construction.
"""
from __future__ import annotations
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from quantedge.fundamentals.rebound.bulk_extra import (
    Series, knowable, ttm_points,
)

Bars = List[Tuple[date, float, float]]  # ascending (date, close, volume)


# ── volume signals (step-15) ──────────────────────────────────

def _window_stats(bars: Bars) -> Tuple[float, float]:
    """(avg_volume, up_day_volume_share) for a bar window. The first bar of
    the window has no prior close inside it, so it's skipped for up/down."""
    if len(bars) < 2:
        return 0.0, 0.0
    vols = [v for _, _, v in bars[1:]]
    avg = sum(vols) / len(vols) if vols else 0.0
    up_vol = sum(v for (_, c, v), (_, pc, _) in zip(bars[1:], bars[:-1]) if c > pc)
    tot = sum(vols)
    return avg, (up_vol / tot if tot > 0 else 0.0)


def volume_signals(bars: Bars, as_of: date, params: dict) -> Dict:
    p = params["rebound"]["confirm"]
    bars = [b for b in bars if b[0] <= as_of]
    week_n, month_n, base_n = (p["volume_recent_week_days"],
                               p["volume_recent_month_days"],
                               p["volume_baseline_days"])
    if len(bars) < base_n + month_n + 2:
        return {"ok": False, "reason": "insufficient_bars", "n_bars": len(bars)}

    recent_month = bars[-month_n:]
    recent_week = bars[-week_n:]
    baseline = bars[-(base_n + month_n):-month_n]   # ends BEFORE the recent month

    base_avg, _ = _window_stats(baseline)
    if base_avg <= 0:
        return {"ok": False, "reason": "zero_baseline_volume"}
    w_avg, w_up = _window_stats(recent_week)
    m_avg, m_up = _window_stats(recent_month)

    # accumulation streak: consecutive most-recent 5-bar weeks with
    # up-day volume share above the frozen bar
    streak = 0
    i = len(bars)
    while i - 5 >= 1:
        wk = bars[i - 6: i]     # include one prior bar for the first up/down
        _, up = _window_stats(wk)
        if up > p["accum_updayvol_min"]:
            streak += 1
            i -= 5
        else:
            break

    return {
        "ok": True,
        "vol_1w_ratio": round(w_avg / base_avg, 3),
        "vol_1m_ratio": round(m_avg / base_avg, 3),
        "up_day_share_1w": round(w_up, 3),
        "up_day_share_1m": round(m_up, 3),
        "accum_streak_weeks": streak,
    }


# ── buyback confirmation (step-16) ────────────────────────────

def buyback_confirm(buybacks: Series, market_cap: Optional[float],
                    as_of: date, params: dict) -> Dict:
    """TTM repurchase dollars from the cash-flow statement. `buybacks` may mix
    quarterly and annual windows (bulk_extra merges both); TTM is the rolling
    4-quarter sum where quarterly exists, else the latest annual figure."""
    p = params["rebound"]["confirm"]
    bb = knowable(buybacks, as_of)
    if not bb:
        return {"ok": False, "reason": "no_buyback_data"}
    lookback = as_of - timedelta(days=int(p["buyback_lookback_quarters"] * 95))
    ttm = knowable(ttm_points(bb), as_of)
    if ttm:
        bb_ttm = ttm[-1][1]
    else:
        recent = [v for e, v, _ in bb if e >= lookback]
        bb_ttm = recent[-1] if recent else 0.0
    active = bb_ttm > 0 and bb[-1][1] > 0 and bb[-1][0] >= lookback
    out = {
        "ok": True,
        "buyback_ttm": round(bb_ttm, 0),
        "active_through_decline": bool(active),
    }
    if market_cap and market_cap > 0:
        out["buyback_to_mktcap"] = round(bb_ttm / market_cap, 4)
    return out


# ── the layer summary ─────────────────────────────────────────

def compute_confirm(bars: Bars, buybacks: Series, market_cap: Optional[float],
                    as_of: date, params: dict) -> Dict:
    vol = volume_signals(bars, as_of, params)
    bb = buyback_confirm(buybacks, market_cap, as_of, params)
    result: Dict = {"volume": vol, "buyback": bb}

    reasons: List[str] = []
    if vol.get("ok"):
        if vol["vol_1m_ratio"] >= 1.3:
            reasons.append(f"volume {vol['vol_1m_ratio']}x baseline (1m)")
        if vol["up_day_share_1m"] >= 0.55:
            reasons.append(f"{vol['up_day_share_1m']:.0%} of volume on up-days")
        if vol["accum_streak_weeks"] >= 3:
            reasons.append(f"{vol['accum_streak_weeks']}-wk accumulation streak")
    if bb.get("ok") and bb["active_through_decline"]:
        frag = "company buying back stock"
        if bb.get("buyback_to_mktcap"):
            frag += f" ({bb['buyback_to_mktcap']:.1%} of mktcap TTM)"
        reasons.append(frag)
    result["reason"] = "; ".join(reasons) if reasons else "no confirmation signals yet"
    result["n_confirmations"] = len(reasons)
    return result
