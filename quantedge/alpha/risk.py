"""Risk layer (Stage H) — turns a ranked book into sized positions.

Independent of which backtest variant wins: any tradeable version of the
model needs (1) volatility-target position sizing so the book runs at a
chosen risk level rather than whatever the raw names happen to carry, and
(2) a drawdown governor that cuts gross exposure after losses. Pure
functions, unit-tested; no lookahead (uses only trailing realized vol).
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple


def inverse_vol_weights(vols: Dict[str, float]) -> Dict[str, float]:
    """Weight names inversely to their trailing volatility (risk parity-ish),
    normalized to sum to 1. Zero/missing vols are dropped."""
    inv = {t: 1.0 / v for t, v in vols.items() if v and v > 0}
    s = sum(inv.values())
    if s <= 0:
        return {}
    return {t: w / s for t, w in inv.items()}


def vol_target_leverage(weights: Dict[str, float], vols: Dict[str, float],
                        target_ann_vol: float, avg_corr: float = 0.3,
                        max_leverage: float = 1.0) -> float:
    """Gross leverage so the weighted book hits target_ann_vol. Portfolio vol
    approximated with a flat average-correlation assumption (honest: we don't
    have a full covariance matrix, so we state the assumption rather than fake
    precision). Capped at max_leverage (long-only default = 1.0)."""
    names = [t for t in weights if t in vols and vols[t] > 0]
    if not names:
        return 0.0
    w = [weights[t] for t in names]
    v = [vols[t] for t in names]
    var = 0.0
    for i in range(len(names)):
        for j in range(len(names)):
            corr = 1.0 if i == j else avg_corr
            var += w[i] * w[j] * v[i] * v[j] * corr
    port_vol = math.sqrt(var) if var > 0 else 0.0
    if port_vol <= 0:
        return 0.0
    return max(0.0, min(max_leverage, target_ann_vol / port_vol))


def drawdown_governor(current_drawdown: float, cut_start: float = -0.08,
                      cut_floor: float = -0.20, min_scale: float = 0.3) -> float:
    """Scale gross exposure down as drawdown deepens. Flat 1.0 until cut_start,
    linearly down to min_scale at cut_floor, then holds. current_drawdown is
    negative (e.g. -0.12). Returns a multiplier in [min_scale, 1.0]."""
    if current_drawdown >= cut_start:
        return 1.0
    if current_drawdown <= cut_floor:
        return min_scale
    frac = (current_drawdown - cut_start) / (cut_floor - cut_start)
    return 1.0 - frac * (1.0 - min_scale)


def size_book(ranked_longs: List[str], vols: Dict[str, float],
              target_ann_vol: float, current_drawdown: float,
              max_leverage: float = 1.0) -> Dict[str, float]:
    """Full sizing: inverse-vol weights x vol-target leverage x drawdown
    governor. Returns {ticker: final_weight} summing to <= max_leverage."""
    base = inverse_vol_weights({t: vols[t] for t in ranked_longs if t in vols})
    if not base:
        return {}
    lev = vol_target_leverage(base, vols, target_ann_vol, max_leverage=max_leverage)
    gov = drawdown_governor(current_drawdown)
    scale = lev * gov
    return {t: round(w * scale, 5) for t, w in base.items()}
