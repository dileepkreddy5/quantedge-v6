"""REBOUND score assembly (step-20) — six layers, one number, one thesis.

HARD GATES (in/out, no partial credit — a stock is on the list or it isn't):
  1. discount.qualifies            (step-12: deep AND cheap vs own history)
  2. health.qualifies              (step-13: growth streak AND F-score bars)
  3. NOT disqualifiers.disqualified (step-14: the knife filter)

SCORE (0-100, only for stocks that pass all gates) — weights are FROZEN in
params.yaml (scoring block, committed with this module BEFORE the backtest,
journal-noted). The backtest (step 21) judges whether THIS score, with THESE
weights, ranks forward returns. Tuning weights after seeing results requires
a journal entry and re-running the gate — the standing rule.

Component budget (sums to 100):
  discount   25   depth of drawdown (15) + cheapness percentile in own history (10)
  health     35   growth streak (15) + F-score (10) + margin (5) + accruals (3) + ROIC (2)
  confirm    25   accumulation streak (8) + up-day share (5) + volume ratio (4)
                  + active buyback (4) + insider cluster (4)
  rd_bonus    5   R&D intensity (3) + R&D growing (2)   [CLS 2001]
  data_trust 10   full valuation history (6) + all knife checks verifiable (4)

The thesis line is assembled from the layers' own reason fragments — every
claim on the page traces to a computed number.
"""
from __future__ import annotations
from typing import Dict, List, Optional


def _pts(x: float, lo: float, hi: float, budget: float) -> float:
    """Linear points: lo->0, hi->budget, clipped."""
    if hi == lo:
        return 0.0
    f = (x - lo) / (hi - lo)
    return budget * max(0.0, min(1.0, f))


def score_candidate(
    discount: Dict, health: Dict, disq: Dict, confirm: Dict,
    insider: Optional[Dict], stage: Dict, params: dict,
) -> Dict:
    # ── hard gates ────────────────────────────────────────────
    gates = {
        "discount": bool(discount.get("qualifies")),
        "health": bool(health.get("qualifies")),
        "knife_filter": not bool(disq.get("disqualified")),
    }
    if not all(gates.values()):
        return {"passes": False, "gates": gates, "score": None,
                "failed": [k for k, v in gates.items() if not v]}

    w = params["rebound"]["scoring"]
    dd = discount["drawdown"]
    val = discount.get("valuation", {})
    vol = confirm.get("volume", {})
    bb = confirm.get("buyback", {})
    gs = health.get("growth", {})
    mt = health.get("margin", {})
    rd = health.get("rd", {})
    ro = health.get("roic", {})

    comp: Dict[str, float] = {}

    # discount: 25
    comp["dd_depth"] = _pts(dd["drawdown_from_3y_high"], 0.35, 0.75, w["dd_depth"])
    pctl = val.get("ps_percentile_own")
    comp["cheap_vs_self"] = (_pts(100 - pctl, 50, 100, w["cheap_vs_self"])
                             if pctl is not None else 0.0)

    # health: 35
    comp["growth_streak"] = _pts(gs.get("streak", 0), 3, 8, w["growth_streak"])
    comp["fscore"] = _pts(health.get("piotroski", 0), 4, 9, w["fscore"])
    comp["margin"] = w["margin"] if mt.get("ok") and mt.get("expanding") else 0.0
    acc = health.get("accruals")
    comp["accruals"] = w["accruals"] if (acc is not None and acc < 0) else 0.0
    comp["roic"] = w["roic"] if ro.get("ok") and ro.get("improving") else 0.0

    # confirmation: 25
    if vol.get("ok"):
        comp["accum_streak"] = _pts(vol["accum_streak_weeks"], 0, 5, w["accum_streak"])
        comp["upday_share"] = _pts(vol["up_day_share_1m"], 0.50, 0.70, w["upday_share"])
        comp["vol_ratio"] = _pts(vol["vol_1m_ratio"], 1.0, 2.0, w["vol_ratio"])
    else:
        comp["accum_streak"] = comp["upday_share"] = comp["vol_ratio"] = 0.0
    comp["buyback"] = w["buyback"] if bb.get("ok") and bb.get("active_through_decline") else 0.0
    comp["insider"] = w["insider"] if insider and insider.get("cluster") else 0.0

    # R&D bonus: 5
    comp["rd_intensity"] = (w["rd_intensity"]
                            if rd.get("ok") and rd.get("rd_to_mktcap", 0) >= 0.05 else 0.0)
    comp["rd_growth"] = (w["rd_growth"]
                         if rd.get("ok") and rd.get("rd_yoy_growth", 0) > 0 else 0.0)

    # data trust: 10 — opacity is penalized, never rewarded
    comp["val_history"] = w["val_history"] if discount.get("valuation_history") else 0.0
    n_unverified = len(disq.get("unverified", []))
    comp["verifiable"] = max(0.0, w["verifiable"] * (1 - n_unverified / 4.0))

    total = round(sum(comp.values()), 1)

    # ── thesis line: every fragment comes from a layer's own reason ──
    frags: List[str] = []
    if discount.get("reason"):
        frags.append(discount["reason"])
    if health.get("reason"):
        frags.append(health["reason"])
    if confirm.get("reason") and confirm.get("n_confirmations", 0) > 0:
        frags.append(confirm["reason"])
    if insider and insider.get("cluster"):
        frags.append(insider["reason"])

    return {
        "passes": True,
        "gates": gates,
        "score": total,
        "components": {k: round(v, 2) for k, v in comp.items()},
        "stage": stage.get("stage", "UNKNOWN"),
        "stage_reason": stage.get("reason", ""),
        "thesis": " · ".join(frags),
    }
