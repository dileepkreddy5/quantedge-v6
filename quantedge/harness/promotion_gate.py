"""promotion_gate — the ONLY bridge from research to production, as code.

PROMOTION.md is the law; this is its enforcement. A unit reaches
status: production ONLY if check() returns PASS against the FROZEN
kill-threshold in params.yaml. The bars are read from params, never passed
in, so a caller cannot quietly soften them. Honesty is mechanical.

Manual §9, §15, §22.2, Table 20.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class GateDecision:
    passed: bool
    reasons: list[str] = field(default_factory=list)   # every failure, named

    def __bool__(self) -> bool:
        return self.passed


def check(validation: dict, params: dict) -> GateDecision:
    """Judge a unit's measured validation block against the frozen bar.

    validation expects keys: mechanism (str), lift, t_stat,
    calibration_max_dev, n_regimes_passed, cost_survived (bool).
    """
    kt = params["kill_threshold"]
    reasons: list[str] = []

    # 0. Mechanism is a hard prerequisite (§3) — no mechanism, no consideration.
    mech = (validation.get("mechanism") or "").strip()
    if not mech:
        reasons.append("No economic mechanism stated (§3) — cannot be promoted.")

    # 1. Cost survival (§13) — gross-only results are inadmissible.
    if not validation.get("cost_survived", False):
        reasons.append("Did not survive realistic costs (§13).")

    # 2. Lift over base rate, after costs (§15).
    lift = validation.get("lift")
    if lift is None or lift < kt["min_oos_lift_over_base_rate"]:
        reasons.append(
            f"OOS lift {lift} < required {kt['min_oos_lift_over_base_rate']}x base rate."
        )

    # 3. Significance, after multiple-testing correction (§15).
    t = validation.get("t_stat")
    if t is None or t < kt["min_t_stat"]:
        reasons.append(f"t-stat {t} < required {kt['min_t_stat']} (post-correction).")

    # 4. Calibration: predicted vs realized within tolerance across deciles.
    cal = validation.get("calibration_max_dev")
    tol = kt["calibration_tolerance_pct"] / 100.0
    if cal is None or cal > tol:
        reasons.append(f"Calibration deviation {cal} > tolerance {tol}.")

    # 5. Regime stability — must hold across >= N distinct regimes (§15).
    nreg = validation.get("n_regimes_passed", 0)
    if nreg < kt["min_distinct_regimes"]:
        reasons.append(
            f"Held in {nreg} regimes < required {kt['min_distinct_regimes']}."
        )

    return GateDecision(passed=(len(reasons) == 0), reasons=reasons)
