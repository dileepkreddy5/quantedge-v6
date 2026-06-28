"""EDGAR fundamentals fetcher — merges across candidate XBRL tags by year.

Robustness lessons baked in:
  - Companies switch tags over time (old 'Revenues' -> new contract-revenue),
    so we MERGE all candidate tags by fiscal year instead of picking one.
  - A tag can exist but be ZERO/stale in recent years, so a real value from
    another tag must win. We prefer the largest-magnitude non-zero value.
  - GrossProfit is often not filed; derive it as Revenue - CostOfRevenue.
"""
from __future__ import annotations
import urllib.request, json, time

UA = "Dileep Kapu dileepkreddy5@gmail.com"

CONCEPTS = {
    "net_income":   ["NetIncomeLoss"],
    "op_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "assets":       ["Assets"],
    "liabilities":  ["Liabilities"],
    "revenue":      ["RevenueFromContractWithCustomerExcludingAssessedTax",
                     "Revenues", "RevenueFromContractWithCustomerIncludingAssessedTax",
                     "SalesRevenueNet"],
    "gross_profit": ["GrossProfit"],
    "cost_of_rev":  ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "cur_assets":   ["AssetsCurrent"],
    "cur_liab":     ["LiabilitiesCurrent"],
    "shares":       ["CommonStockSharesOutstanding",
                     "WeightedAverageNumberOfDilutedSharesOutstanding"],
}


def _fetch_one_tag(cik: str, tag: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    d = json.load(urllib.request.urlopen(req, timeout=20))
    units = d.get("units", {})
    if not units:
        return {}
    rows = units[list(units.keys())[0]]
    fy = {}
    for r in rows:
        if r.get("fp") == "FY" and r.get("form") == "10-K" and r.get("fy"):
            fy[int(r["fy"])] = r["val"]
    return fy


def _merge_tags(cik: str, tags: list, pause: float) -> dict:
    """Merge annual values across candidate tags; prefer non-zero magnitude."""
    merged = {}
    for tag in tags:
        try:
            fy = _fetch_one_tag(cik, tag)
        except Exception:
            fy = {}
        for y, v in fy.items():
            # Prefer a non-zero value, or a larger-magnitude one, over what we have.
            cur = merged.get(y)
            if cur is None or (cur == 0 and v != 0) or (abs(v) > abs(cur) and cur == 0):
                merged[y] = v
            elif cur is None:
                merged[y] = v
        time.sleep(pause)
    return merged


def fetch_annual_facts(cik: str, pause: float = 0.1) -> dict:
    out = {}
    for metric, tags in CONCEPTS.items():
        out[metric] = _merge_tags(cik, tags, pause)

    if not out["gross_profit"] and out["revenue"] and out["cost_of_rev"]:
        derived = {}
        for y, rev in out["revenue"].items():
            cor = out["cost_of_rev"].get(y)
            if cor is not None and rev:
                derived[y] = rev - cor
        out["gross_profit"] = derived
    return out
