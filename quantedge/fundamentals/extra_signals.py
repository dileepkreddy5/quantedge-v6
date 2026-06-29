"""Additional research-paper signals from EDGAR data (no new fetches needed).

- gross_margin_trend: is gross margin expanding? (quality improvement)
- accruals (Sloan 1996): (net income - operating cash flow) / assets.
  High accruals = earnings driven by accounting, not cash = lower-quality,
  one of the most robust anomalies in the literature. Lower is better.
- debt_trend: is leverage rising? Rising debt funding growth is a red flag
  for small-caps (the manual's survival test).
All computed from facts we already fetch in edgar_pit / edgar_facts.
"""
from __future__ import annotations


def gross_margin_trend(f: dict):
    """Latest gross margin minus prior-year gross margin (annual)."""
    gp, rev = f.get("gross_profit", {}), f.get("revenue", {})
    yrs = sorted(set(gp) & set(rev))
    if len(yrs) < 2: return None
    y, p = yrs[-1], yrs[-2]
    if rev[y] and rev[p]:
        return round((gp[y]/rev[y]) - (gp[p]/rev[p]), 4)
    return None


def accruals(f: dict):
    """Sloan accruals: (net_income - op_cash_flow) / assets. Lower = better."""
    ni, ocf, a = f.get("net_income", {}), f.get("op_cash_flow", {}), f.get("assets", {})
    yrs = sorted(set(ni) & set(ocf) & set(a))
    if not yrs: return None
    y = yrs[-1]
    if a[y]:
        return round((ni[y] - ocf[y]) / a[y], 4)
    return None


def debt_trend(f: dict):
    """Change in liabilities/assets vs prior year. Positive = leverage rising."""
    li, a = f.get("liabilities", {}), f.get("assets", {})
    yrs = sorted(set(li) & set(a))
    if len(yrs) < 2: return None
    y, p = yrs[-1], yrs[-2]
    if a[y] and a[p]:
        return round((li[y]/a[y]) - (li[p]/a[p]), 4)
    return None
