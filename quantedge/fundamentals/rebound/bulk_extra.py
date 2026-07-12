"""Extra point-in-time extractions from a companyfacts bulk blob (step-12).

Everything REBOUND needs beyond what bulk_adapter already extracts, each
series carrying its FILED date so `knowable_as_of`-style filtering works:

  quarterly_shares  — dei EntityCommonStockSharesOutstanding (cover-page count,
                      updated every 10-Q/10-K: the honest share count over time)
  cash_series       — cash & equivalents (+ short-term investments if filed)
  debt_series       — best-effort total debt (long-term + current portions)
  rd_series         — quarterly R&D expense           (used by health, step 13)
  buyback_series    — payments for share repurchases  (used by confirm, step 16)
  ttm_points        — rolling 4-quarter revenue sums, each stamped with the
                      date the FULL window became knowable (max of the 4 filings)

All return [(period_end_or_asof, value, filed_date)] ascending. Filtering to
`filed <= as_of` is the caller's PIT contract — helper `knowable` provided.
"""
from __future__ import annotations
from datetime import date
from typing import List, Optional, Tuple

Series = List[Tuple[date, float, date]]  # (period_end, value, filed)


def _rows(section: dict, tag: str, unit_pref=("USD", "shares")) -> list:
    c = section.get(tag)
    if not c:
        return []
    units = c.get("units", {})
    for u in unit_pref:
        if u in units:
            return units[u]
    return units[list(units.keys())[0]] if units else []


def _instant_series(section: dict, tags: List[str]) -> Series:
    """Instant (balance-sheet) values: keyed by 'end', deduped preferring
    the EARLIEST filing of each period-end (first knowable)."""
    merged = {}
    for tag in tags:
        for r in _rows(section, tag):
            end, filed, val = r.get("end"), r.get("filed"), r.get("val")
            if end is None or filed is None or val is None:
                continue
            try:
                e, f = date.fromisoformat(end), date.fromisoformat(filed)
            except Exception:
                continue
            cur = merged.get(e)
            if cur is None or f < cur[1] or (cur[0] == 0 and val != 0):
                merged[e] = (float(val), f)
    return [(e, v, f) for e, (v, f) in sorted(merged.items())]


def _duration_series(section: dict, tags: List[str], min_days=80, max_days=100) -> Series:
    """Duration (income/cash-flow) values for ~quarterly windows."""
    merged = {}
    for tag in tags:
        for r in _rows(section, tag):
            s, e, filed, val = r.get("start"), r.get("end"), r.get("filed"), r.get("val")
            if not (s and e and filed) or val is None:
                continue
            try:
                sd, ed, fd = date.fromisoformat(s), date.fromisoformat(e), date.fromisoformat(filed)
            except Exception:
                continue
            if min_days <= (ed - sd).days <= max_days:
                cur = merged.get(ed)
                if cur is None or fd < cur[1] or (cur[0] == 0 and val != 0):
                    merged[ed] = (float(val), fd)
    return [(e, v, f) for e, (v, f) in sorted(merged.items())]


# ── public extractors ─────────────────────────────────────────

def quarterly_shares(facts: dict) -> Series:
    dei = facts.get("facts", {}).get("dei", {})
    out = _instant_series(dei, ["EntityCommonStockSharesOutstanding"])
    if out:
        return out
    gaap = facts.get("facts", {}).get("us-gaap", {})
    return _instant_series(gaap, ["CommonStockSharesOutstanding"])


def cash_series(facts: dict) -> Series:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    return _instant_series(gaap, [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ])


def debt_series(facts: dict) -> Series:
    """Best-effort total debt: prefer explicit totals, else LT + current."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    total = _instant_series(gaap, ["DebtLongtermAndShorttermCombinedAmount", "LongTermDebt"])
    if total:
        return total
    lt = dict((e, (v, f)) for e, v, f in _instant_series(gaap, ["LongTermDebtNoncurrent"]))
    st = dict((e, (v, f)) for e, v, f in _instant_series(gaap, ["LongTermDebtCurrent", "DebtCurrent"]))
    ends = sorted(set(lt) | set(st))
    out = []
    for e in ends:
        v = (lt.get(e, (0.0, None))[0] or 0.0) + (st.get(e, (0.0, None))[0] or 0.0)
        f = max([x[1] for x in (lt.get(e), st.get(e)) if x and x[1]], default=None)
        if f:
            out.append((e, v, f))
    return out


def rd_series(facts: dict) -> Series:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    return _duration_series(gaap, ["ResearchAndDevelopmentExpense"])


def buyback_series(facts: dict) -> Series:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    # accept quarterly AND annual windows (companies differ in how they file)
    q = _duration_series(gaap, ["PaymentsForRepurchaseOfCommonStock"], 80, 100)
    a = _duration_series(gaap, ["PaymentsForRepurchaseOfCommonStock"], 350, 380)
    return sorted(q + [x for x in a if x[0] not in {e for e, _, _ in q}])


# ── TTM builder + PIT helpers ─────────────────────────────────

def ttm_points(q_revenue: Series) -> Series:
    """Rolling 4-quarter revenue sums. Each point is stamped with the date the
    FULL window became knowable = the LATEST filing among its 4 quarters —
    the strict PIT stance: you cannot know a TTM before its last piece filed."""
    out = []
    for i in range(3, len(q_revenue)):
        window = q_revenue[i - 3: i + 1]
        # span between FIRST and LAST period-END of 4 consecutive quarters is
        # ~3 quarters (~273 days) — NOT 365. Guard rejects gapped windows
        # (a missing quarter stretches the span to ~365+).
        span = (window[-1][0] - window[0][0]).days
        if not (240 <= span <= 310):
            continue
        out.append((
            window[-1][0],
            sum(v for _, v, _ in window),
            max(f for _, _, f in window),
        ))
    return out


def knowable(series: Series, as_of: date) -> Series:
    return [(e, v, f) for e, v, f in series if f <= as_of]


def latest_knowable(series: Series, as_of: date) -> Optional[Tuple[date, float, date]]:
    k = knowable(series, as_of)
    return k[-1] if k else None


def quarterly_gross_profit(facts: dict) -> Series:
    """Quarterly gross profit; falls back to revenue - cost_of_revenue where
    GrossProfit isn't filed quarterly (common). Used by health (step-13)."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    gp = _duration_series(gaap, ["GrossProfit"])
    if gp:
        return gp
    rev = _duration_series(gaap, [
        "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
        "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"])
    cost = _duration_series(gaap, [
        "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"])
    cd = {e: (v, f) for e, v, f in cost}
    out = []
    for e, v, f in rev:
        if e in cd:
            out.append((e, v - cd[e][0], max(f, cd[e][1])))
    return out
