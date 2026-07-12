"""REBOUND Layer 3 — disqualifiers: the falling-knife filter (step-14).

The Brandes Institute falling-knives finding: stocks down 60%+ beat the
market on average over 3 years — but ~13% go bankrupt, and they ruin the
average holder. This layer removes the knives BEFORE ranking. Four hard
checks against the FROZEN params; tripping ANY ONE disqualifies, regardless
of how good the discount and health layers look:

  revenue_shrinking   latest TTM revenue below the TTM one year earlier —
                      a "beaten-down bargain" with a shrinking business is
                      a value trap, not a rebound
  leverage_spiking    liabilities/assets up more than +0.10 YoY — decline
                      being financed with debt (the survival test)
  dilution            share count up >10% YoY — the rebound you're buying
                      is being sold out from under you (split-guarded)
  cash_runway         if operating cash flow is negative: months of cash
                      left at the current burn < 18 — companies that need
                      to raise money during THEIR crash raise it on terms
                      that bury the equity

PIT: every series filtered by filed <= as_of. Pure functions, no I/O.
"""
from __future__ import annotations
from datetime import date, timedelta
from typing import Dict, List, Optional

from quantedge.fundamentals.rebound.bulk_extra import (
    Series, knowable, latest_knowable, ttm_points,
)


def _ttm_year_ago(ttm: Series) -> Optional[float]:
    """The TTM value whose window ended ~1 year before the latest window."""
    if len(ttm) < 2:
        return None
    target = ttm[-1][0] - timedelta(days=365)
    candidates = [p for p in ttm[:-1] if abs((p[0] - target).days) <= 60]
    return candidates[-1][1] if candidates else None


def check_revenue_shrinking(q_revenue: Series, as_of: date) -> Dict:
    ttm = knowable(ttm_points(q_revenue), as_of)
    if len(ttm) < 5:
        return {"ok": False, "reason": "insufficient_ttm_history", "flag": False}
    prior = _ttm_year_ago(ttm)
    if prior is None or prior <= 0:
        return {"ok": False, "reason": "no_year_ago_ttm", "flag": False}
    chg = ttm[-1][1] / prior - 1.0
    return {"ok": True, "ttm_now": round(ttm[-1][1], 0),
            "ttm_year_ago": round(prior, 0),
            "ttm_change": round(chg, 4), "flag": chg < 0}


def check_leverage(known_annual: dict, max_increase: float) -> Dict:
    li, a = known_annual.get("liabilities", {}), known_annual.get("assets", {})
    yrs = sorted(set(li) & set(a))
    if len(yrs) < 2:
        return {"ok": False, "reason": "insufficient_annual_history", "flag": False}
    y, p = yrs[-1], yrs[-2]
    if not a[y] or not a[p]:
        return {"ok": False, "reason": "zero_assets", "flag": False}
    delta = (li[y] / a[y]) - (li[p] / a[p])
    return {"ok": True, "leverage_now": round(li[y] / a[y], 4),
            "leverage_delta_yoy": round(delta, 4), "flag": delta > max_increase}


def check_dilution(shares_pit: Series, as_of: date, max_dilution: float) -> Dict:
    """Share count YoY from the dei cover-page series. Split guard: a jump
    >1.5x (or shrink <0.6x) is a split/reverse-split, not dilution."""
    sh = knowable(shares_pit, as_of)
    if len(sh) < 2:
        return {"ok": False, "reason": "insufficient_share_history", "flag": False}
    now = sh[-1]
    target = now[0] - timedelta(days=365)
    prior = [p for p in sh[:-1] if abs((p[0] - target).days) <= 90]
    if not prior or prior[-1][1] <= 0:
        return {"ok": False, "reason": "no_year_ago_shares", "flag": False}
    ratio = now[1] / prior[-1][1]
    if ratio > 1.5 or ratio < 0.6:
        return {"ok": True, "share_ratio_yoy": round(ratio, 3), "flag": False,
                "note": "treated as split/reverse-split, not dilution"}
    return {"ok": True, "share_ratio_yoy": round(ratio, 3),
            "dilution_yoy": round(ratio - 1.0, 4),
            "flag": (ratio - 1.0) > max_dilution}


def check_cash_runway(cash_pit: Series, known_annual: dict, as_of: date,
                      min_months: float) -> Dict:
    """If the latest annual operating cash flow is negative, months of cash
    at that burn rate must exceed min_months. Positive OCF = no burn = pass.
    Burn basis is annual OCF (quarterly OCF is filed as YTD spans in 10-Qs
    and is not reliably extractable from the bulk file — honest limitation)."""
    ocf_by_fy = known_annual.get("op_cash_flow", {})
    if not ocf_by_fy:
        return {"ok": False, "reason": "no_ocf", "flag": False}
    ocf = ocf_by_fy[max(ocf_by_fy)]
    if ocf is None:
        return {"ok": False, "reason": "no_ocf", "flag": False}
    if ocf >= 0:
        return {"ok": True, "ocf_annual": round(ocf, 0), "burning": False,
                "flag": False}
    cash = latest_knowable(cash_pit, as_of)
    if not cash:
        # burning cash AND we can't see the cash balance — treat as
        # disqualified: unverifiable survival is not survivable-by-assumption
        return {"ok": True, "burning": True, "flag": True,
                "note": "negative OCF with no visible cash balance"}
    monthly_burn = abs(ocf) / 12.0
    runway = cash[1] / monthly_burn if monthly_burn > 0 else float("inf")
    return {"ok": True, "ocf_annual": round(ocf, 0), "burning": True,
            "cash": round(cash[1], 0), "runway_months": round(runway, 1),
            "flag": runway < min_months}


def compute_disqualifiers(
    q_revenue: Series,
    shares_pit: Series,
    cash_pit: Series,
    known_annual: dict,
    as_of: date,
    params: dict,
) -> Dict:
    """ANY tripped flag disqualifies. Checks that cannot run (missing data)
    do NOT trip — but are listed in `unverified` so scoring can penalize
    opacity instead of rewarding it."""
    p = params["rebound"]["disqualifiers"]
    checks = {
        "revenue_shrinking": check_revenue_shrinking(q_revenue, as_of),
        "leverage_spiking": check_leverage(known_annual, p["max_leverage_increase_yoy"]),
        "dilution": check_dilution(shares_pit, as_of, p["max_share_dilution_yoy"]),
        "cash_runway": check_cash_runway(cash_pit, known_annual, as_of,
                                         p["min_cash_runway_months"]),
    }
    flags: List[str] = [name for name, c in checks.items() if c.get("flag")]
    unverified: List[str] = [name for name, c in checks.items() if not c.get("ok")]
    return {
        "checks": checks,
        "flags": flags,
        "unverified": unverified,
        "disqualified": bool(flags),
        "reason": ("DISQUALIFIED: " + ", ".join(flags)) if flags
                  else "clears all knife filters"
                  + (f" ({len(unverified)} unverifiable)" if unverified else ""),
    }
