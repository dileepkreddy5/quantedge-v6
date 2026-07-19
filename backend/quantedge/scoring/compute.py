"""Signal scoring core — turns one raw value into a 0-100 score.

Cross-sectional percentile against sector peers is the PRIMARY score; absolute
floors/caps override at the extremes. Falls back to absolute bands when no peer
distribution is available, so every signal works from day one.
"""
from __future__ import annotations
import math
from typing import Optional, List, Dict, Any


def _finite(v) -> Optional[float]:
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def percentile_rank(value: float, distribution: List[float],
                    higher_is_better: bool = True) -> Optional[float]:
    vals = [x for x in (_finite(d) for d in distribution) if x is not None]
    v = _finite(value)
    if v is None or len(vals) < 5:
        return None
    below = sum(1 for x in vals if x < v)
    equal = sum(1 for x in vals if x == v)
    pct = 100.0 * (below + 0.5 * equal) / len(vals)
    return pct if higher_is_better else (100.0 - pct)


def absolute_band(value: float, good: float, great: float,
                  higher_is_better: bool = True) -> Optional[float]:
    v = _finite(value)
    if v is None:
        return None
    if not higher_is_better:
        value, good, great = -v, -good, -great
    else:
        value = v
    if value <= good:
        return max(0.0, 40.0 + (value - (good - (great - good))) / (great - good) * 25.0)
    if value <= great:
        return 65.0 + (value - good) / (great - good) * 25.0
    return min(100.0, 90.0 + (value - great) / (great - good) * 10.0)


def score_signal(value: Optional[float], spec: Dict[str, Any],
                 peer_values: Optional[List[float]] = None) -> Dict[str, Any]:
    v = _finite(value)
    hib = spec.get("higher_is_better", True)
    status = spec.get("status", "live")
    # 'reference' = computed & displayed but NOT scored (owned by another module,
    # e.g. EV multiples belong to Valuation) — prevents double-counting.
    if status == "reference":
        return {"value": v, "score": None, "method": "reference",
                "reason": "display-only; scored in its owning module"}
    # 'needs_source' = honest placeholder, no data yet
    if status == "needs_source":
        return {"value": None, "score": None, "method": "needs_source",
                "reason": "data source pending"}
    if v is None:
        return {"value": None, "score": None, "method": "missing",
                "reason": "input unavailable"}

    method = "absolute"
    score = None
    if peer_values:
        p = percentile_rank(v, peer_values, hib)
        if p is not None:
            score, method = p, "percentile"
    if score is None:
        score = absolute_band(v, spec.get("good", 0.0), spec.get("great", 1.0), hib)
        method = "absolute"
    if score is None:
        return {"value": v, "score": None, "method": "missing",
                "reason": "not rankable and no absolute band"}

    clamped = None
    floor, cap = spec.get("floor"), spec.get("cap")
    if floor is not None and ((hib and v < floor) or (not hib and v > floor)):
        score = min(score, spec.get("floor_score", 30.0)); clamped = "floor"
    if cap is not None and ((hib and v > cap) or (not hib and v < cap)):
        score = max(score, spec.get("cap_score", 85.0)); clamped = "cap"

    return {"value": round(v, 6), "score": round(float(score), 1),
            "method": method, "clamped": clamped}
