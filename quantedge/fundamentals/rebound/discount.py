"""REBOUND Layer 1 — the discount (step-12).

Answers, with point-in-time discipline: "is this stock ACTUALLY cheap —
versus its own price history AND its own valuation history?"

Two families of evidence, because each alone lies:
  PRICE  — drawdown says the stock fell, but a stock can fall AND stay
           expensive (still at 30x sales after -50%).
  VALUE  — P/S and EV/S versus the company's OWN 5-year record. Own-history
           comparison (not sector) avoids the classic screener failure where
           every bank looks "cheap" and every SaaS "expensive" forever.
           Mechanism: De Bondt-Thaler (1985) overreaction is about a stock
           being punished beyond its own norm.

Every valuation point is timestamped by when it became KNOWABLE:
  TTM revenue knowable at max(filed) of its 4 quarters; shares at their
  filing; the price used is the first close on/after that knowable date.
Pure functions — the nightly scan feeds today's data, the backtest (step 21)
feeds historical windows. No I/O here.
"""
from __future__ import annotations
from bisect import bisect_left
from datetime import date, timedelta
from statistics import median
from typing import Dict, List, Optional, Tuple

from quantedge.fundamentals.rebound.bulk_extra import (
    Series, knowable, latest_knowable, ttm_points,
)

Closes = List[Tuple[date, float]]  # ascending (date, close)


# ── price-series helpers ──────────────────────────────────────

def _truncate(closes: Closes, as_of: date) -> Closes:
    return [(d, c) for d, c in closes if d <= as_of]


def _close_on_or_after(closes: Closes, d: date, max_slip_days: int = 10) -> Optional[float]:
    dates = [x[0] for x in closes]
    i = bisect_left(dates, d)
    if i < len(closes) and (closes[i][0] - d).days <= max_slip_days:
        return closes[i][1]
    return None


def drawdown_structure(closes: Closes, as_of: date, lookback_years: int = 3) -> Dict:
    """The shape of the decline: how deep, how long, and where price sits now."""
    px = _truncate(closes, as_of)
    px = [(d, c) for d, c in px if d >= as_of - timedelta(days=int(lookback_years * 365.25))]
    if len(px) < 200:  # <~10 months of bars: too young to judge a 3y drawdown
        return {"ok": False, "reason": "insufficient_price_history", "n_bars": len(px)}

    price_now = px[-1][1]
    hi_date, hi = max(px, key=lambda x: x[1])
    dd = 1.0 - price_now / hi if hi > 0 else 0.0

    yr = [(d, c) for d, c in px if d >= as_of - timedelta(days=365)]
    lo_date, lo = min(yr, key=lambda x: x[1])
    off_low = price_now / lo - 1.0 if lo > 0 else 0.0

    # depth-in-own-terms: current drawdown's percentile among the stock's own
    # rolling drawdowns — is THIS decline unusual FOR THIS stock? A -40% dip
    # is routine for a triple-beta small cap and a five-alarm event for KO.
    dds = []
    running_hi = px[0][1]
    for _, c in px:
        running_hi = max(running_hi, c)
        dds.append(1.0 - c / running_hi if running_hi > 0 else 0.0)
    dd_percentile = 100.0 * sum(1 for x in dds if x <= dd) / len(dds)

    return {
        "ok": True,
        "price_now": round(price_now, 4),
        "high_3y": round(hi, 4),
        "high_3y_date": hi_date.isoformat(),
        "drawdown_from_3y_high": round(dd, 4),
        "days_underwater": (as_of - hi_date).days,
        "low_1y": round(lo, 4),
        "low_1y_date": lo_date.isoformat(),
        "days_since_1y_low": (as_of - lo_date).days,
        "pct_off_low": round(off_low, 4),
        "dd_percentile_own_history": round(dd_percentile, 1),
    }


# ── own-history valuation ─────────────────────────────────────

def valuation_vs_own_history(
    closes: Closes,
    q_revenue: Series,
    shares_pit: Series,
    cash_pit: Series,
    debt_pit: Series,
    as_of: date,
    history_years: int = 5,
    min_points: int = 6,
) -> Dict:
    """Current P/S (TTM) and EV/S (TTM) versus the company's own record.

    For each historical TTM point: the valuation THAT DAY, using only shares
    knowable that day and the first close on/after the knowable date. The
    current point uses everything knowable as of `as_of`. Output includes the
    percentile of today's multiple in its own history (0 = cheapest ever)."""
    px = _truncate(closes, as_of)
    ttm = knowable(ttm_points(q_revenue), as_of)
    ttm = [p for p in ttm if p[2] >= as_of - timedelta(days=int(history_years * 365.25))]
    if len(ttm) < min_points or len(px) < 200:
        return {"ok": False, "reason": "insufficient_valuation_history",
                "n_ttm_points": len(ttm)}

    def mult_at(knowable_date: date, ttm_rev: float) -> Optional[Tuple[float, Optional[float]]]:
        if ttm_rev <= 0:
            return None
        sh = latest_knowable(shares_pit, knowable_date)
        p = _close_on_or_after(px, knowable_date)
        if not sh or sh[1] <= 0 or p is None:
            return None
        mcap = p * sh[1]
        ps = mcap / ttm_rev
        ev_s = None
        cash = latest_knowable(cash_pit, knowable_date)
        debt = latest_knowable(debt_pit, knowable_date)
        if cash is not None and debt is not None:
            ev_s = (mcap + debt[1] - cash[1]) / ttm_rev
        return ps, ev_s

    hist_ps, hist_evs = [], []
    for _, rev, kdate in ttm[:-1]:
        m = mult_at(kdate, rev)
        if m:
            hist_ps.append(m[0])
            if m[1] is not None:
                hist_evs.append(m[1])

    now = mult_at(as_of, ttm[-1][1])
    if now is None or len(hist_ps) < min_points - 1:
        return {"ok": False, "reason": "could_not_price_history",
                "n_ps_points": len(hist_ps)}
    ps_now, evs_now = now

    med_ps = median(hist_ps)
    out = {
        "ok": True,
        "ps_ttm_now": round(ps_now, 3),
        "ps_5y_median": round(med_ps, 3),
        "ps_vs_5y_median": round(ps_now / med_ps, 3) if med_ps > 0 else None,
        "ps_percentile_own": round(
            100.0 * sum(1 for x in hist_ps if x <= ps_now) / len(hist_ps), 1),
        "n_valuation_points": len(hist_ps),
        "ttm_revenue": round(ttm[-1][1], 0),
    }
    if evs_now is not None and len(hist_evs) >= min_points - 1:
        med_ev = median(hist_evs)
        out["ev_s_ttm_now"] = round(evs_now, 3)
        out["ev_s_vs_5y_median"] = round(evs_now / med_ev, 3) if med_ev > 0 else None
    return out


# ── the layer verdict ─────────────────────────────────────────

def compute_discount(
    closes: Closes,
    q_revenue: Series,
    shares_pit: Series,
    cash_pit: Series,
    debt_pit: Series,
    as_of: date,
    params: dict,
) -> Dict:
    """Layer verdict against the FROZEN params (step-11). Returns all metrics
    + `qualifies` + human-readable `reason` fragments for the thesis line."""
    p = params["rebound"]
    dd = drawdown_structure(closes, as_of, p["discount"]["lookback_high_years"])
    result: Dict = {"drawdown": dd}
    if not dd["ok"]:
        result.update({"qualifies": False, "reason": dd["reason"]})
        return result

    deep_enough = dd["drawdown_from_3y_high"] >= p["universe"]["min_drawdown_from_3y_high"]
    priced_ok = dd["price_now"] >= p["universe"]["min_price_usd"]

    val = valuation_vs_own_history(
        closes, q_revenue, shares_pit, cash_pit, debt_pit, as_of,
        history_years=p["discount"]["valuation_history_years"],
    )
    result["valuation"] = val

    reasons = [f"-{dd['drawdown_from_3y_high']:.0%} from 3y high"]
    if val.get("ok"):
        cheap_vs_self = (val["ps_vs_5y_median"] is not None
                         and val["ps_vs_5y_median"] <= p["discount"]["ps_vs_5y_median_max"])
        if cheap_vs_self:
            reasons.append(
                f"P/S {val['ps_ttm_now']} = {val['ps_vs_5y_median']:.0%} of own 5y median")
        qualifies = deep_enough and priced_ok and cheap_vs_self
        result["valuation_history"] = True
    else:
        # young company: no 5y record to compare against. Price evidence alone
        # may pass the layer, but the flag is carried so scoring can discount it.
        qualifies = deep_enough and priced_ok
        result["valuation_history"] = False
        reasons.append("no 5y valuation history")

    result["qualifies"] = bool(qualifies)
    result["reason"] = "; ".join(reasons)
    return result
