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
              "PaymentsToAcquireProductiveAssets"],
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

def parse_flow_concept(units: List[Dict], years_back: int = 6) -> List[Dict]:
    """Clean quarters, PLUS derive missing fiscal-Q4 from annual - (Q1+Q2+Q3).
    Many filers (esp. non-calendar FY like MSFT June) never file a standalone Q4;
    the 10-K reports the full year. We reconstruct Q4 so the series is complete."""
    cutoff = dt.date.today() - dt.timedelta(days=365 * years_back + 120)
    q_picked: Dict[str, Dict] = {}
    a_picked: Dict[str, Dict] = {}
    for u in units:
        cls = classify_period(u.get("start"), u.get("end"))
        try:
            end_d = dt.date.fromisoformat(u["end"])
        except (KeyError, ValueError):
            continue
        if end_d < cutoff:
            continue
        if cls == "quarter":
            key = u["end"]; has_frame = bool(u.get("frame"))
            if key not in q_picked or (has_frame and not q_picked[key].get("frame")):
                q_picked[key] = u
        elif cls == "annual":
            a_picked[u["end"]] = u
    quarters = [{"end": u["end"], "start": u.get("start"), "val": float(u["val"]),
                 "fy": u.get("fy"), "fp": u.get("fp"), "form": u.get("form")}
                for u in q_picked.values()]
    # Derive Q4 = annual - sum(3 quarters inside the annual window)
    for ann in a_picked.values():
        a_s, a_e = ann.get("start"), ann.get("end")
        if not a_s or not a_e:
            continue
        inside = [q for q in quarters if q["start"] and a_s <= q["start"] and q["end"] <= a_e]
        if len(inside) == 3:
            try:
                q4_val = float(ann["val"]) - sum(q["val"] for q in inside)
                last_end = max(q["end"] for q in inside)
                q4_start = (dt.date.fromisoformat(last_end) + dt.timedelta(days=1)).isoformat()
                if not any(q["end"] == a_e for q in quarters):
                    quarters.append({"end": a_e, "start": q4_start, "val": q4_val,
                                     "fy": ann.get("fy"), "fp": "Q4",
                                     "form": ann.get("form"), "derived": True})
            except (ValueError, KeyError):
                continue
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
