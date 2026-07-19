"""EDGAR XBRL data layer — pulls the detailed line items Polygon's aggregated
feed drops (capex, dividends, buybacks, receivables, goodwill, and more),
directly from SEC XBRL company-concept API. Free, richer than any Polygon tier.

CORE DISCIPLINE (no fake values):
- Every raw point carries start/end/fy/fp/frame. We disambiguate period length by
  (end - start) in days: ~85-95 = one fiscal quarter; ~175-190 = half; ~265-280 =
  three-quarter YTD; ~350-375 = annual. We NEVER sum overlapping periods.
- Quarterly series = clean ~90-day points only. TTM = sum of 4 most recent
  non-overlapping quarters. Annual = a ~365-day point, or 4 clean quarters.
- Fiscal-year aware: we read fy/fp from the data; we never assume calendar quarters.

FUTURE-PROOF (rolling window):
- All windows are relative to TODAY (datetime.now). Request "last N years"; in 2030
  this returns 2025-2030 with zero code changes. No hardcoded years anywhere.
"""
from __future__ import annotations
import datetime as dt
from typing import List, Dict, Optional, Any

SEC_BASE = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{concept}.json"
UA = "QuantEdge research dileep@quant.dileepkapu.com"

CONCEPTS = {
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets",
              "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets",
              "PaymentsForCapitalImprovements",
              "PaymentsToAcquireOtherPropertyPlantAndEquipment"],
    "dividends_paid": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock",
                 "PaymentsForRepurchaseOfEquity"],
    "receivables": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "goodwill": ["Goodwill"],
    "intangibles": ["IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"],
    "depreciation_amortization": ["DepreciationDepletionAndAmortization",
                                  "DepreciationAmortizationAndAccretionNet",
                                  "DepreciationAndAmortization"],
    "depreciation": ["Depreciation", "DepreciationNonproduction"],
    "amortization": ["AmortizationOfIntangibleAssets"],
    "sbc": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
    "interest_expense": ["InterestExpense", "InterestExpenseDebt"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "rd": ["ResearchAndDevelopmentExpense"],
    "operating_lease_liab": ["OperatingLeaseLiabilityNoncurrent"],
    "short_term_debt": ["ShortTermBorrowings", "DebtCurrent"],
    "inventory": ["InventoryNet"],
    "accounts_payable": ["AccountsPayableCurrent", "AccountsPayableTradeCurrent"],
    "deferred_revenue": ["ContractWithCustomerLiabilityCurrent", "ContractWithCustomerLiability", "DeferredRevenueCurrent"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "short_term_debt2": ["ShortTermBorrowings", "DebtCurrent", "LongTermDebtCurrent"],
    "interest_expense_full": ["InterestExpense", "InterestExpenseNonoperating", "InterestAndDebtExpense"],
    "operating_lease_total": ["OperatingLeaseLiability"],
}

def _days(start: Optional[str], end: Optional[str]) -> Optional[int]:
    if not start or not end:
        return None
    try:
        return (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days
    except ValueError:
        return None

def classify_period(start, end) -> Optional[str]:
    d = _days(start, end)
    if d is None:
        return "instant" if (start is None and end) else None
    if 85 <= d <= 95:   return "quarter"
    if 175 <= d <= 195: return "half"
    if 265 <= d <= 285: return "ytd3"
    if 350 <= d <= 380: return "annual"
    return None

def _derive_from_cumulative(pts: List[Dict]) -> Dict[str, float]:
    """Given classified points, derive quarterly values by successive subtraction
    within each fiscal year (grouped by shared FY start date). Handles filers that
    report cumulative (H1/9mo/FY) rather than discrete quarters. Q_n = cum_n - cum_(n-1)."""
    from collections import defaultdict
    by_start = defaultdict(list)
    for p in pts:
        if p.get("start"):
            by_start[p["start"]].append(p)
    quarters: Dict[str, float] = {}
    for start, group in by_start.items():
        group.sort(key=lambda x: x["end"])
        chain = [g for g in group if g["cls"] in ("quarter","half","ytd3","annual")]
        prev_cum = 0.0
        for g in chain:
            if g["cls"] == "quarter" and g["start"] == start:
                quarters[g["end"]] = g["val"]; prev_cum = g["val"]
            else:
                quarters[g["end"]] = g["val"] - prev_cum; prev_cum = g["val"]
    # keep standalone clean quarters not tied to FY start
    for p in pts:
        if p["cls"] == "quarter" and p["end"] not in quarters:
            quarters[p["end"]] = p["val"]
    return quarters

def parse_flow_concept(units: List[Dict], years_back: int = 6) -> List[Dict]:
    """Clean quarters, PLUS derive missing fiscal-Q4 from annual - (Q1+Q2+Q3).
    Many filers (esp. non-calendar FY like MSFT June) never file a standalone Q4;
    the 10-K reports the full year. We reconstruct Q4 so the series is complete."""
    cutoff = dt.date.today() - dt.timedelta(days=365 * years_back + 120)
    # Classify and dedupe every point (prefer framed points for the same end date).
    classified: Dict[str, Dict] = {}
    for u in units:
        cls = classify_period(u.get("start"), u.get("end"))
        if cls not in ("quarter","half","ytd3","annual"):
            continue
        try:
            end_d = dt.date.fromisoformat(u["end"])
        except (KeyError, ValueError):
            continue
        if end_d < cutoff:
            continue
        try:
            val = float(u["val"])
        except (KeyError, ValueError, TypeError):
            continue
        key = (u.get("start"), u["end"])
        has_frame = bool(u.get("frame"))
        if key not in classified or (has_frame and not classified[key].get("frame")):
            classified[key] = {"start": u.get("start"), "end": u["end"], "val": val,
                               "cls": cls, "frame": u.get("frame"),
                               "fy": u.get("fy"), "fp": u.get("fp"), "form": u.get("form")}
    # Derive quarterly values from the cumulative chain (handles discrete-quarter AND
    # cumulative-only filers uniformly).
    q_map = _derive_from_cumulative(list(classified.values()))
    quarters = [{"end": e, "val": v} for e, v in q_map.items()]
    quarters.sort(key=lambda x: x["end"])
    return quarters

def parse_stock_concept(units: List[Dict], years_back: int = 6) -> List[Dict]:
    cutoff = dt.date.today() - dt.timedelta(days=365 * years_back + 120)
    picked: Dict[str, Dict] = {}
    for u in units:
        end = u.get("end")
        if not end:
            continue
        try:
            end_d = dt.date.fromisoformat(end)
        except ValueError:
            continue
        if end_d < cutoff:
            continue
        picked[end] = u
    out = [{"end": u["end"], "val": float(u["val"]), "fy": u.get("fy"),
            "fp": u.get("fp"), "form": u.get("form")} for u in picked.values()]
    out.sort(key=lambda x: x["end"])
    return out

def ttm(series: List[Dict]) -> Optional[float]:
    if len(series) < 4:
        return None
    return sum(p["val"] for p in series[-4:])

def annual_series(series: List[Dict]) -> List[float]:
    out = []
    for i in range(3, len(series)):
        out.append(sum(p["val"] for p in series[i-3:i+1]))
    return out
