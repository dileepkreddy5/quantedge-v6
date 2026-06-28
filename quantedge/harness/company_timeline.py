"""company_timeline — the direct test of the mission (Manual §16.3).

Every company carries a full point-in-time trajectory: each month, the score
the system WOULD have assigned using only data available then, alongside the
price. The mission claim is 'the score rises before the price and the
headlines.' This module measures whether that actually happened on real
history — it does not assert it.

If the score led the price: early detection is demonstrated.
If it did not: that is the most important negative result the project can
produce, and far better learned here than with capital (§16.3).
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date


@dataclass
class TimelinePoint:
    as_of: date
    score: float          # what the system would have said, PIT — no look-ahead
    price: float
    # optional: the evidence attached at that moment, for the audit trail
    drivers: dict | None = None


@dataclass
class LeadResult:
    score_inflection: date | None   # first month the score durably turned up
    price_inflection: date | None   # first month the price durably turned up
    lead_days: int | None           # price_inflection - score_inflection
    detected_early: bool            # score led price by a positive margin


def _first_durable_rise(points, key, min_consec: int):
    """First date after which `key` rises for >= min_consec consecutive months."""
    vals = [(p.as_of, getattr(p, key)) for p in points]
    run = 0
    start = None
    for i in range(1, len(vals)):
        if vals[i][1] > vals[i - 1][1]:
            if run == 0:
                start = vals[i - 1][0]
            run += 1
            if run >= min_consec:
                return start
        else:
            run = 0
            start = None
    return None


def measure_lead(timeline: list[TimelinePoint], min_consec: int = 2) -> LeadResult:
    """Did the score inflect before the price? The mission, measured."""
    pts = sorted(timeline, key=lambda p: p.as_of)
    s_inf = _first_durable_rise(pts, "score", min_consec)
    p_inf = _first_durable_rise(pts, "price", min_consec)
    lead = None
    early = False
    if s_inf and p_inf:
        lead = (p_inf - s_inf).days
        early = lead > 0     # score turned up strictly before price
    return LeadResult(s_inf, p_inf, lead, early)
