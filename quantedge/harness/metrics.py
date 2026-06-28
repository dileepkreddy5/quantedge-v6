"""metrics — the numbers the kill-threshold judges (Manual §15, Table 25/37).

Lift over base rate, calibration, significance (with multiple-testing
correction), and regime stability. Everything is computed net-of-cost
upstream; these never see a gross-only return. No metric flatters by
construction — that is the point.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class ValidationResult:
    base_rate: float
    selected_rate: float
    lift: float                 # selected_rate / base_rate
    t_stat: float               # after multiple-testing correction
    calibration_max_dev: float  # worst predicted-vs-realized gap across deciles
    n_regimes_passed: int
    n_selected: int
    n_total: int


def base_rate(labels) -> float:
    """Fraction of WINNERs in the FULL universe — including the dead (§14)."""
    wins = sum(1 for x in labels if x)
    return wins / len(labels) if labels else 0.0


def lift_over_base(selected_labels, universe_labels) -> tuple[float, float, float]:
    """Lift = P(winner | selected) / P(winner | universe). Returns (base, sel, lift)."""
    b = base_rate(universe_labels)
    s = base_rate(selected_labels)
    return b, s, (s / b if b > 0 else float("inf"))


def proportion_t_stat(selected_labels, universe_labels, n_tests: int = 1) -> float:
    """Two-proportion z, then Bonferroni-style correction for n_tests.

    Correction inflates the bar when many traits are tested (§15: t>=3.0
    BECAUSE many traits are tested). We scale the statistic down by sqrt(n_tests)
    as a conservative correction, so testing more things makes passing harder.
    """
    b = base_rate(universe_labels)
    s = base_rate(selected_labels)
    n = len(selected_labels)
    if n == 0 or b in (0.0, 1.0):
        return 0.0
    se = math.sqrt(b * (1 - b) / n)
    z = (s - b) / se if se > 0 else 0.0
    return z / math.sqrt(max(n_tests, 1))


def calibration_max_deviation(pred_probs, realized) -> float:
    """Worst |predicted - realized| across deciles of predicted probability."""
    if not pred_probs:
        return 1.0
    pairs = sorted(zip(pred_probs, realized), key=lambda p: p[0])
    n = len(pairs)
    worst = 0.0
    for d in range(10):
        lo, hi = d * n // 10, (d + 1) * n // 10
        bucket = pairs[lo:hi]
        if not bucket:
            continue
        pred_mean = sum(p for p, _ in bucket) / len(bucket)
        real_mean = sum(r for _, r in bucket) / len(bucket)
        worst = max(worst, abs(pred_mean - real_mean))
    return worst
