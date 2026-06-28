"""costs — gross returns are never shown alone (Manual §13).

Every strategy result passes through costs before any metric is reported.
Costs are committed in params.yaml so they can't be quietly softened to
flatter a result.
"""
from __future__ import annotations


def round_trip_cost_bps(params: dict) -> float:
    c = params["costs"]
    per_side = c["commission_bps"] + c["slippage_bps"] + c["spread_bps"]
    return 2.0 * per_side


def apply_costs(gross_return: float, turnover: float, params: dict) -> float:
    """Net return after round-trip costs scaled by turnover (1.0 = full turnover)."""
    cost_frac = (round_trip_cost_bps(params) / 1e4) * turnover
    return gross_return - cost_frac
