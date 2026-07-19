"""Module A final piece: HYBRID MERGER — unifies Polygon (income+balance) with
EDGAR XBRL (capex, dividends, buybacks, receivables, goodwill, D&A, SBC, R&D,
inventory, payables, leases) into one clean per-fiscal-quarter dataset.

Alignment by (fiscal_year, fiscal_period) + period-end proximity, so non-calendar
fiscal years (MSFT=June) align correctly. D&A synthesized from depreciation +
amortization when the combined tag is absent. Rolling window relative to today.
"""
from __future__ import annotations
import datetime as dt
from typing import List, Dict, Optional, Any

def _valid_shares(primary, fallback):
    """Reject absurd share counts (negative, or implausibly small). Real large-cap
    share counts are in the hundreds of millions to billions; a value under 1e6 or
    negative is a bad Polygon field — use the fallback instead."""
    for v in (primary, fallback):
        try:
            fv = float(v)
            if fv > 1e6:  # at least 1M shares (any real public co); rejects -4M, 1M-ish glitches
                return fv
        except (TypeError, ValueError):
            continue
    return None

def _nearest_edgar_val(edgar_series: List[dict], period_end: str, tol_days: int = 45) -> Optional[float]:
    if not edgar_series or not period_end:
        return None
    try:
        target = dt.date.fromisoformat(period_end)
    except ValueError:
        return None
    best, best_gap = None, tol_days + 1
    for pt in edgar_series:
        try:
            e = dt.date.fromisoformat(pt["end"])
        except (KeyError, ValueError):
            continue
        gap = abs((e - target).days)
        if gap <= tol_days and gap < best_gap:
            best, best_gap = pt["val"], gap
    return best

def merge_quarters(polygon_quarters: List[Any], edgar: Dict[str, List[dict]]) -> List[Dict[str, Any]]:
    merged = []
    for q in polygon_quarters:
        pe = getattr(q, "period_end", None) or getattr(q, "filing_date", None)
        row: Dict[str, Any] = {
            "fiscal_year": getattr(q, "fiscal_year", None),
            "fiscal_period": getattr(q, "fiscal_period", None),
            "period_end": pe,
            "revenue": getattr(q, "revenue", None),
            "cost_of_revenue": getattr(q, "cost_of_revenue", None),
            "gross_profit": getattr(q, "gross_profit", None),
            "operating_income": getattr(q, "operating_income", None),
            "operating_expenses": getattr(q, "operating_expenses", None),
            "net_income": getattr(q, "net_income", None),
            "pretax_income": getattr(q, "pretax_income", None),
            "tax_expense": getattr(q, "tax_expense", None),
            "nonoperating_income": getattr(q, "nonoperating_income", None),
            "eps_diluted": getattr(q, "eps_diluted", None),
            "diluted_shares": _valid_shares(getattr(q, "diluted_shares", None), getattr(q, "basic_shares", None)),
            "basic_shares": _valid_shares(getattr(q, "basic_shares", None), getattr(q, "diluted_shares", None)),
            "assets": getattr(q, "total_assets", None),
            "current_assets": getattr(q, "current_assets", None),
            "current_liabilities": getattr(q, "current_liabilities", None),
            "liabilities": getattr(q, "total_liabilities", None),
            "equity": getattr(q, "total_equity", None),
            "long_term_debt": getattr(q, "long_term_debt", None),
            "cash": getattr(q, "cash", None),
            "fixed_assets": getattr(q, "fixed_assets", None),
            "operating_cash_flow": getattr(q, "operating_cash_flow", None),
        }
        for metric in ["capex","dividends_paid","buybacks","receivables","goodwill",
                       "intangibles","sbc","interest_expense","rd","inventory",
                       "accounts_payable","operating_lease_liab","deferred_revenue",
                       "retained_earnings","short_term_debt2","operating_lease_total"]:
            row[metric] = _nearest_edgar_val(edgar.get(metric, []), pe)
        # interest expense: prefer the fuller InterestExpense tag when present
        ie_full = _nearest_edgar_val(edgar.get("interest_expense_full", []), pe)
        if ie_full is not None:
            row["interest_expense"] = ie_full
        # short-term debt: use the supplemental tag if base is absent
        if row.get("short_term_debt") is None:
            row["short_term_debt"] = row.get("short_term_debt2")
        da = _nearest_edgar_val(edgar.get("depreciation_amortization", []), pe)
        if da is None:
            dep = _nearest_edgar_val(edgar.get("depreciation", []), pe)
            amo = _nearest_edgar_val(edgar.get("amortization", []), pe)
            if dep is not None or amo is not None:
                da = (dep or 0.0) + (amo or 0.0)
        row["depreciation_amortization"] = da
        if row["operating_cash_flow"] is not None and row["capex"] is not None:
            row["free_cash_flow"] = row["operating_cash_flow"] - abs(row["capex"])
        else:
            row["free_cash_flow"] = None
        merged.append(row)
    merged.sort(key=lambda r: (r.get("fiscal_year") or 0, r.get("fiscal_period") or ""))
    # Backfill invalid share counts from the most recent valid quarter (carry-forward),
    # so per-share metrics never divide by a broken share count.
    last_valid=None
    for r in merged:
        if r.get("diluted_shares"): last_valid=r["diluted_shares"]
        elif last_valid: r["diluted_shares"]=last_valid
    last_valid=None
    for r in reversed(merged):
        if r.get("diluted_shares"): last_valid=r["diluted_shares"]
        elif last_valid: r["diluted_shares"]=last_valid
    return merged
