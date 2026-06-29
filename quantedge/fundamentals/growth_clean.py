"""Clean quarterly growth — reject bad-data quarters and tiny-base noise.

Two real distortions found in the raw data:
1. Parsing artifacts: a quarter wildly out of line with neighbors (PRG showing
   $3.7M between two $600M quarters) — a mis-parsed/partial filing. Drop it.
2. Tiny-base noise: growth off a near-zero base ($0.4M -> $10M = +2500%) is
   meaningless for a pre-revenue company. Require a minimum year-ago base.
"""
from __future__ import annotations
from datetime import date

MIN_BASE_REVENUE = 25_000_000   # year-ago quarter must exceed this for growth to count
NEIGHBOR_FLOOR = 0.20           # drop a quarter < 20% of its neighbors' median (artifact)


def _median(xs):
    s = sorted(xs); n = len(s)
    return s[n//2] if n % 2 else (s[n//2-1]+s[n//2])/2


def clean_quarters(q):
    """q = [(end_date, value, filed)]. Drop parsing-artifact quarters."""
    if len(q) < 3:
        return q
    vals = [v for _, v, _ in q]
    cleaned = []
    for i, (e, v, f) in enumerate(q):
        lo = max(0, i-2); hi = min(len(q), i+3)
        neigh = [vals[j] for j in range(lo, hi) if j != i and vals[j] > 0]
        if neigh:
            med = _median(neigh)
            # drop if this quarter is implausibly small vs neighbors (artifact)
            if med > 0 and v > 0 and v < NEIGHBOR_FLOOR * med:
                continue
        cleaned.append((e, v, f))
    return cleaned


def clean_growth_signal(q, as_of):
    """YoY growth from cleaned quarters, requiring a real revenue base.

    Returns (growth, base_ok). growth is None if not reliably computable.
    """
    q = clean_quarters(q)
    visible = [(e, v) for e, v, f in q if f <= as_of and v > 0]
    if len(visible) < 5:
        return None, False
    visible.sort()
    by_end = dict(visible)
    ends = sorted(by_end)
    latest = ends[-1]
    prior = date(latest.year-1, latest.month, min(latest.day, 28))
    base_end = min(by_end, key=lambda k: abs((k-prior).days))
    if abs((base_end-prior).days) > 45:
        return None, False
    base_val = by_end[base_end]
    if base_val < MIN_BASE_REVENUE:
        return None, False          # tiny base — growth % is noise
    return by_end[latest]/base_val - 1.0, True
