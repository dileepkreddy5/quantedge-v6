"""REBOUND stage detector (step-19) — WHERE in the rebound cycle is it?

Born from a real design argument: waiting for full confirmation keeps the
safest entries but donates the fastest gains (rebounds are front-loaded off
the bottom); buying falling knives catches bottoms and catastrophes alike.
So the tracker does not pick a side — it LABELS the stage and shows all of
them, and the backtest (step 21) measures each stage's hit rate, return and
drawdown separately, so the risk/reward trade-off is answered with data.

Stages (evaluated top-down, frozen params from step-11):
  RECOVERING  price >= +25% off its 1y low — clearly off the bottom,
              least remaining risk, least remaining discount
  TURNING     base established (>=60d since the low) AND accumulation
              evidence: streak >= 3 wks OR 1m up-day volume share >= 55%
  BASING      base established, no accumulation evidence yet
  FALLING     still near/making lows — highest risk, fullest discount;
              only reaches the list at all by clearing the knife filter

Pure function: consumes the outputs of drawdown_structure (step-12) and
volume_signals (step-15) — no recomputation, no I/O.
"""
from __future__ import annotations
from typing import Dict


def classify_stage(drawdown: Dict, volume: Dict, params: dict) -> Dict:
    p = params["rebound"]["stages"]
    if not drawdown.get("ok"):
        return {"ok": False, "stage": "UNKNOWN", "reason": "no price structure"}

    off_low = drawdown["pct_off_low"]
    days_since_low = drawdown["days_since_1y_low"]
    based = days_since_low >= p["basing_min_days_since_low"]

    accum = False
    accum_why = ""
    if volume.get("ok"):
        if volume["accum_streak_weeks"] >= p["turning_min_accum_weeks"]:
            accum = True
            accum_why = f"{volume['accum_streak_weeks']}-wk accumulation streak"
        elif volume["up_day_share_1m"] >= p["turning_updayvol_min"]:
            accum = True
            accum_why = f"{volume['up_day_share_1m']:.0%} up-day volume (1m)"

    if off_low >= p["recovering_min_off_low"]:
        stage, why = "RECOVERING", f"+{off_low:.0%} off the low"
    elif based and accum:
        stage, why = "TURNING", f"based {days_since_low}d; {accum_why}"
    elif based:
        stage, why = "BASING", f"{days_since_low}d since the low, no accumulation yet"
    else:
        stage, why = "FALLING", f"low set {days_since_low}d ago"

    return {"ok": True, "stage": stage, "reason": why,
            "days_since_low": days_since_low, "pct_off_low": round(off_low, 4),
            "accumulating": accum}
