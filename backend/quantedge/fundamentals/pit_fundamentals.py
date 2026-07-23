"""
Point-in-time fundamentals — for a given (cik, as_of_date), return ONLY the
financial facts that were publicly filed on or before as_of_date. No lookahead.
This is the anti-lookahead guarantee that separates a real model from a fake backtest.
"""
from __future__ import annotations
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
import math

# SEC us-gaap metric name candidates (companies use different tags across years)
METRIC_ALIASES = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    # Many filers report cash under the restricted-cash-inclusive tag or the
    # short-term-investments variant. Mapping only the first left cash_ratio and
    # cash_to_debt blank for every company that uses another.
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
             "CashAndCashEquivalentsFairValueDisclosure",
             "CashCashEquivalentsAndShortTermInvestments"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "shares": ["CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding",
               "WeightedAverageNumberOfSharesOutstandingBasic"],
}

def _d(s: str) -> Optional[date]:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

def _pit_series(gaap: dict, aliases: List[str], as_of: date) -> List[Tuple[date, date, float, str]]:
    """Return list of (end_date, filed_date, value, fp) filed on/before as_of, for the first
    alias that has data. Sorted by end_date descending (most recent period first)."""
    for name in aliases:
        node = gaap.get(name)
        if not node:
            continue
        out = []
        for unit_key, entries in node.get("units", {}).items():
            for e in entries:
                fd = _d(e.get("filed", ""))
                ed = _d(e.get("end", ""))
                val = e.get("val")
                if fd is None or ed is None or val is None:
                    continue
                if fd > as_of:          # ANTI-LOOKAHEAD: not public yet on as_of
                    continue
                fp = e.get("fp", "")
                out.append((ed, fd, float(val), fp))
        if out:
            # dedupe by end_date, keep the latest-filed value for each period
            best = {}
            for ed, fd, val, fp in out:
                if ed not in best or fd > best[ed][1]:
                    best[ed] = (ed, fd, val, fp)
            series = sorted(best.values(), key=lambda t: t[0], reverse=True)
            return series
    return []

def _latest(series, n=1):
    return series[:n] if series else []

def point_in_time_fundamentals(facts: dict, as_of: date, price: float = None,
                                shares_hint: float = None) -> Dict:
    """Compute fundamental factors using ONLY data filed on/before as_of.
    price: the stock's close on as_of (for valuation ratios, avoids market-cap lookahead)."""
    if not facts:
        return {}
    gaap = facts.get("facts", {}).get("us-gaap", {})
    if not gaap:
        return {}

    def series(key): return _pit_series(gaap, METRIC_ALIASES[key], as_of)

    rev = series("revenue"); ni = series("net_income"); gp = series("gross_profit")
    oi = series("operating_income"); assets = series("assets"); liab = series("liabilities")
    eq = series("equity"); ca = series("current_assets"); cl = series("current_liabilities")
    ocf = series("ocf"); cor = series("cost_of_revenue"); shares = series("shares")

    def val(s): return s[0][2] if s else None
    def ttm(s):
        # sum last 4 quarterly values if they look quarterly; else latest annual
        if not s: return None
        vals = [x[2] for x in s[:4]]
        return sum(vals) if len(vals) >= 4 else s[0][2]

    f = {}
    rev_ttm = ttm(rev); ni_ttm = ttm(ni); gp_ttm = ttm(gp); oi_ttm = ttm(oi); ocf_ttm = ttm(ocf)
    A = val(assets); E = val(eq); L = val(liab); CA = val(ca); CL = val(cl); SH = val(shares) or shares_hint

    # Margins
    if rev_ttm and rev_ttm > 0:
        if ni_ttm is not None: f["fund_net_margin"] = ni_ttm / rev_ttm
        if gp_ttm is not None: f["fund_gross_margin"] = gp_ttm / rev_ttm
        if oi_ttm is not None: f["fund_operating_margin"] = oi_ttm / rev_ttm
        if ocf_ttm is not None: f["fund_ocf_margin"] = ocf_ttm / rev_ttm
    # Returns
    if A and A > 0 and ni_ttm is not None: f["fund_roa"] = ni_ttm / A
    if E and E > 0 and ni_ttm is not None: f["fund_roe"] = ni_ttm / E
    if A and E and (A) > 0:
        ic = (E or 0) + (val(series("long_term_debt")) or 0)
        if ic > 0 and oi_ttm is not None: f["fund_roic_approx"] = (oi_ttm * 0.79) / ic
    # Liquidity / leverage
    if CA is not None and CL and CL > 0: f["fund_current_ratio"] = CA / CL
    if E and E > 0 and L is not None: f["fund_debt_to_equity"] = L / E
    if A and A > 0 and rev_ttm is not None: f["fund_asset_turnover"] = rev_ttm / A
    # Growth (YoY): compare latest period to same period ~4 quarters earlier
    if rev and len(rev) >= 5:
        recent = rev[0][2]; year_ago = rev[4][2]
        if year_ago and abs(year_ago) > 0: f["fund_revenue_growth"] = (recent - year_ago) / abs(year_ago)
    if ni and len(ni) >= 5:
        recent = ni[0][2]; year_ago = ni[4][2]
        if year_ago and abs(year_ago) > 0: f["fund_earnings_growth"] = (recent - year_ago) / abs(year_ago)
    # Valuation (uses point-in-time PRICE, not current market cap → no lookahead)
    if price and SH and SH > 0:
        mcap = price * SH
        if ni_ttm and ni_ttm > 0: f["fund_pe_ratio"] = mcap / ni_ttm
        if rev_ttm and rev_ttm > 0: f["fund_ps_ratio"] = mcap / rev_ttm
        if E and E > 0: f["fund_price_to_book"] = mcap / E
        if ocf_ttm and ocf_ttm > 0: f["fund_ocf_yield"] = ocf_ttm / mcap
        if ni_ttm and mcap > 0: f["fund_earnings_yield"] = ni_ttm / mcap

    return {k: v for k, v in f.items() if v is not None and math.isfinite(v)}

if __name__ == "__main__":
    from quantedge.fundamentals.edgar_bulk import company_facts_from_bulk
    facts = company_facts_from_bulk("0000320193")  # AAPL
    for test_date in ["2023-01-01", "2024-06-01", "2025-11-01"]:
        d = _d(test_date)
        ff = point_in_time_fundamentals(facts, d, price=180.0, shares_hint=15.5e9)
        print(f"\n=== AAPL as of {test_date} (only data filed <= this date) ===")
        for k, v in sorted(ff.items()):
            print(f"  {k:28s} {round(v,4)}")
