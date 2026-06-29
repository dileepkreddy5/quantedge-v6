"""Adapter: extract everything the scanner needs from ONE bulk facts blob.

Reads from the local companyfacts.zip (no network). Produces the same shapes
the scanner consumes: annual facts {metric:{fy:val}}, quarterly revenue series,
and point-in-time {metric:[(fy,val,filed)]} — all from local data, no 429s.
"""
from __future__ import annotations
from datetime import date

ANNUAL_TAGS = {
    "net_income": ["NetIncomeLoss"],
    "op_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "assets": ["Assets"], "liabilities": ["Liabilities"],
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax","Revenues",
                "RevenueFromContractWithCustomerIncludingAssessedTax","SalesRevenueNet"],
    "gross_profit": ["GrossProfit"],
    "cost_of_rev": ["CostOfRevenue","CostOfGoodsAndServicesSold","CostOfGoodsSold"],
    "cur_assets": ["AssetsCurrent"], "cur_liab": ["LiabilitiesCurrent"],
    "shares": ["CommonStockSharesOutstanding","WeightedAverageNumberOfDilutedSharesOutstanding"],
}
REV_TAGS = ANNUAL_TAGS["revenue"]


def _usd_rows(gaap, tag):
    c = gaap.get(tag)
    if not c: return []
    units = c.get("units", {})
    # prefer USD; for shares use the shares unit
    for u in ("USD", "shares", "USD/shares"):
        if u in units: return units[u]
    return units[list(units.keys())[0]] if units else []


def pit_from_bulk(facts: dict) -> dict:
    """{metric:[(fy,val,filed_date)]} from a bulk facts blob (annual 10-K figures)."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    out = {}
    for metric, tags in ANNUAL_TAGS.items():
        merged = {}
        for tag in tags:
            for r in _usd_rows(gaap, tag):
                if r.get("fp") == "FY" and r.get("form") == "10-K" and r.get("fy") and r.get("filed"):
                    fy = int(r["fy"]); val = r["val"]; filed = date.fromisoformat(r["filed"])
                    cur = merged.get(fy)
                    if cur is None or (cur[0] == 0 and val != 0):
                        merged[fy] = (val, filed)
        out[metric] = [(fy, v, f) for fy, (v, f) in sorted(merged.items())]
    if not out["gross_profit"] and out["revenue"] and out["cost_of_rev"]:
        cost = {fy: (v, f) for fy, v, f in out["cost_of_rev"]}
        out["gross_profit"] = [(fy, rev - cost[fy][0], max(rf, cost[fy][1]))
                               for fy, rev, rf in out["revenue"] if fy in cost]
    return out


def quarterly_revenue_from_bulk(facts: dict):
    """[(period_end, value, filed)] quarterly revenue from a bulk facts blob."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    merged = {}
    for tag in REV_TAGS:
        for r in _usd_rows(gaap, tag):
            if not (r.get("start") and r.get("end") and r.get("filed")): continue
            try:
                s = date.fromisoformat(r["start"]); e = date.fromisoformat(r["end"])
            except Exception:
                continue
            if 80 <= (e - s).days <= 100:
                cur = merged.get(e)
                if cur is None or (cur[0] == 0 and r["val"] != 0):
                    merged[e] = (r["val"], date.fromisoformat(r["filed"]))
    return [(e, v, f) for e, (v, f) in sorted(merged.items())]
