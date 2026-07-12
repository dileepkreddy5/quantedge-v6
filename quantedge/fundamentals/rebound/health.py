"""REBOUND Layer 2 — business health (step-13).

Answers: "while the price fell, was the BUSINESS getting better?" This is the
Piotroski (2000) core insight applied to the beaten-down set: the discount is
not the edge — discount x improving fundamentals is the edge.

Components (each with its paper):
  growth_streak     consecutive quarters of YoY revenue growth, from CLEANED
                    quarterly XBRL (artifact quarters dropped, tiny bases
                    rejected — reuses growth_clean rules)
  margin_trajectory quarterly gross-margin slope over the last 8 quarters —
                    the inflection detector, finer than annual deltas
  piotroski         the 9-check F-score (reused from multibagger_score)
  accruals          Sloan (1996): cash-backed earnings (reused)
  debt_trend        leverage direction (reused)
  rd_factor         Chan-Lakonishok-Sougiannis (2001): TTM R&D / market cap —
                    expensed R&D hides value from earnings-based screens
  roic_direction    proxy: net income / (assets - current liabilities),
                    latest FY vs prior — honestly labeled a PROXY (no NOPAT
                    tax adjustment from bulk annual tags)

PIT: every quarterly series is filtered by filed <= as_of before use; annual
facts arrive already filtered via edgar_pit.knowable_as_of.
Pure functions, no I/O — scan feeds today, backtest feeds history.
"""
from __future__ import annotations
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from quantedge.fundamentals.growth_clean import clean_quarters
from quantedge.fundamentals.multibagger_score import score as piotroski_score
from quantedge.fundamentals.extra_signals import accruals, debt_trend
from quantedge.fundamentals.rebound.bulk_extra import Series, knowable, ttm_points


# ── growth streak ─────────────────────────────────────────────

def growth_streak(q_revenue: Series, as_of: date, min_base: float) -> Dict:
    """Consecutive most-recent quarters with YoY revenue growth > 0.

    Each quarter is matched to its year-ago quarter (365d +/- 45d). The streak
    breaks on: negative YoY, missing year-ago match, or a base below min_base
    (tiny-base growth is noise, not health). Artifact quarters are dropped
    before any comparison (growth_clean rules)."""
    q = [x for x in clean_quarters(knowable(q_revenue, as_of)) if x[1] > 0]
    if len(q) < 6:
        return {"ok": False, "reason": "insufficient_quarters", "n_quarters": len(q)}
    by_end = {e: v for e, v, _ in q}
    ends = sorted(by_end)

    def year_ago(e: date) -> Optional[date]:
        target = e - timedelta(days=365)
        best = min(ends, key=lambda k: abs((k - target).days))
        return best if abs((best - target).days) <= 45 else None

    streak, yoys = 0, []
    for e in reversed(ends):
        base_end = year_ago(e)
        if base_end is None:
            break
        base = by_end[base_end]
        if base < min_base:
            break
        g = by_end[e] / base - 1.0
        if g <= 0:
            break
        streak += 1
        yoys.append(round(g, 4))
    return {
        "ok": True,
        "streak": streak,
        "latest_yoy": yoys[0] if yoys else None,
        "yoy_path": yoys[:8],           # newest first
        "n_quarters": len(q),
    }


# ── margin trajectory ─────────────────────────────────────────

def margin_trajectory(q_gross_profit: Series, q_revenue: Series, as_of: date,
                      window: int = 8) -> Dict:
    """Gross-margin slope over the last `window` quarters (least squares on
    margin vs quarter index) + latest-vs-year-ago delta."""
    gp = {e: v for e, v, _ in knowable(q_gross_profit, as_of)}
    rv = {e: v for e, v, _ in clean_quarters(knowable(q_revenue, as_of)) if v > 0}
    ends = sorted(set(gp) & set(rv))[-window:]
    margins = [(e, gp[e] / rv[e]) for e in ends if rv[e] > 0]
    if len(margins) < 5:
        return {"ok": False, "reason": "insufficient_margin_history",
                "n_points": len(margins)}
    ys = [m for _, m in margins]
    n = len(ys)
    xbar, ybar = (n - 1) / 2.0, sum(ys) / n
    num = sum((i - xbar) * (y - ybar) for i, y in enumerate(ys))
    den = sum((i - xbar) ** 2 for i in range(n))
    slope_per_q = num / den if den else 0.0
    return {
        "ok": True,
        "gross_margin_now": round(ys[-1], 4),
        "gross_margin_slope_per_q": round(slope_per_q, 5),
        "margin_delta_vs_year_ago": round(ys[-1] - ys[-5], 4) if n >= 5 else None,
        "expanding": slope_per_q > 0,
        "n_points": n,
    }


# ── R&D factor ────────────────────────────────────────────────

def rd_factor(rd_quarterly: Series, market_cap: Optional[float], as_of: date) -> Dict:
    """Chan-Lakonishok-Sougiannis: TTM R&D / market cap, plus R&D YoY growth.
    High R&D intensity = value hidden by expensing; growing R&D during a
    drawdown = management investing through the storm."""
    ttm = knowable(ttm_points(rd_quarterly), as_of)
    if not ttm or not market_cap or market_cap <= 0:
        return {"ok": False, "reason": "no_rd_or_mktcap"}
    rd_now = ttm[-1][1]
    out = {
        "ok": True,
        "rd_ttm": round(rd_now, 0),
        "rd_to_mktcap": round(rd_now / market_cap, 4),
    }
    year_ago = ttm[-1][0] - timedelta(days=365)
    prior = [p for p in ttm[:-1] if abs((p[0] - year_ago).days) <= 60]
    if prior and prior[-1][1] > 0:
        out["rd_yoy_growth"] = round(rd_now / prior[-1][1] - 1.0, 4)
    return out


# ── ROIC direction (proxy) ────────────────────────────────────

def roic_direction(known_annual: dict) -> Dict:
    """PROXY: net_income / (assets - current liabilities), latest vs prior FY.
    Not tax-adjusted NOPAT — labeled honestly and used directionally only."""
    ni = known_annual.get("net_income", {})
    a = known_annual.get("assets", {})
    cl = known_annual.get("cur_liab", {})
    yrs = sorted(set(ni) & set(a) & set(cl))
    if len(yrs) < 2:
        return {"ok": False, "reason": "insufficient_annual_history"}
    def roic(y):
        ic = a[y] - cl[y]
        return ni[y] / ic if ic and ic > 0 else None
    r_now, r_prev = roic(yrs[-1]), roic(yrs[-2])
    if r_now is None or r_prev is None:
        return {"ok": False, "reason": "invalid_invested_capital"}
    return {
        "ok": True,
        "roic_proxy_now": round(r_now, 4),
        "roic_proxy_prior": round(r_prev, 4),
        "improving": r_now > r_prev,
        "note": "proxy: NI/(assets-cur_liab), not tax-adjusted NOPAT",
    }


# ── the layer verdict ─────────────────────────────────────────

def compute_health(
    q_revenue: Series,
    q_gross_profit: Series,
    rd_quarterly: Series,
    known_annual: dict,          # {metric:{fy:val}} already PIT-filtered
    market_cap: Optional[float],
    ticker: str,
    as_of: date,
    params: dict,
) -> Dict:
    """Layer verdict against the FROZEN params. Gate = growth streak AND
    F-score bars; margin/accruals/R&D/ROIC feed the score + thesis line."""
    p = params["rebound"]["health"]

    gs = growth_streak(q_revenue, as_of, p["min_revenue_base_usd"])
    mt = margin_trajectory(q_gross_profit, q_revenue, as_of)
    rd = rd_factor(rd_quarterly, market_cap, as_of)
    ro = roic_direction(known_annual)
    pio = piotroski_score(ticker, known_annual)
    acc = accruals(known_annual)
    dbt = debt_trend(known_annual)

    result: Dict = {
        "growth": gs, "margin": mt, "rd": rd, "roic": ro,
        "piotroski": pio.piotroski,
        "piotroski_checks": pio.piotroski_checks,
        "accruals": acc,
        "debt_trend": dbt,
    }

    streak_ok = gs.get("ok") and gs["streak"] >= p["min_growth_quarters"]
    pio_ok = pio.piotroski >= p["min_piotroski"]
    result["qualifies"] = bool(streak_ok and pio_ok)

    reasons: List[str] = []
    if gs.get("ok") and gs["streak"] > 0:
        reasons.append(f"{gs['streak']} straight growth quarters"
                       + (f" (latest +{gs['latest_yoy']:.0%})" if gs.get("latest_yoy") else ""))
    reasons.append(f"F-score {pio.piotroski}/9")
    if mt.get("ok") and mt["expanding"]:
        reasons.append(f"gross margin expanding ({mt['gross_margin_now']:.0%})")
    if rd.get("ok") and rd["rd_to_mktcap"] >= 0.05:
        frag = f"R&D {rd['rd_to_mktcap']:.0%} of mktcap"
        if rd.get("rd_yoy_growth") and rd["rd_yoy_growth"] > 0:
            frag += f", up {rd['rd_yoy_growth']:.0%} YoY"
        reasons.append(frag)
    if acc is not None and acc < 0:
        reasons.append("cash-backed earnings")
    if ro.get("ok") and ro["improving"]:
        reasons.append("ROIC improving")
    result["reason"] = "; ".join(reasons)
    return result
