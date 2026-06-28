"""Point-in-time EDGAR fetcher — every fact stamped with its FILING date.

The difference from edgar_facts.py: instead of collapsing to fiscal year, we
keep the date each number was FILED ('filed' field in EDGAR). That stamp is
what lets the backtest harness ask "what was knowable on date T" with zero
look-ahead. This is the discipline that separates a backtest from a fantasy.
"""
from __future__ import annotations
import urllib.request, json, time
from datetime import date

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


def _fetch_tag_pit(cik: str, tag: str) -> list:
    """Return [(fiscal_year, value, filed_date)] for annual 10-K figures."""
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    d = json.load(urllib.request.urlopen(req, timeout=20))
    units = d.get("units", {})
    if not units:
        return []
    rows = units[list(units.keys())[0]]
    out = []
    for r in rows:
        if r.get("fp") == "FY" and r.get("form") == "10-K" and r.get("fy") and r.get("filed"):
            out.append((int(r["fy"]), r["val"], date.fromisoformat(r["filed"])))
    return out


def fetch_pit_facts(cik: str, pause: float = 0.1) -> dict:
    """Return {metric: [(fy, value, filed_date), ...]} merged across tags.

    Each value carries the date it was FILED — the look-ahead guard.
    """
    out = {}
    for metric, tags in CONCEPTS.items():
        merged = {}  # fy -> (value, filed) preferring non-zero, earliest filing
        for tag in tags:
            try:
                rows = _fetch_tag_pit(cik, tag)
            except Exception:
                rows = []
            for fy, val, filed in rows:
                cur = merged.get(fy)
                if cur is None or (cur[0] == 0 and val != 0):
                    merged[fy] = (val, filed)
            time.sleep(pause)
        out[metric] = [(fy, v, f) for fy, (v, f) in sorted(merged.items())]

    # Derive gross profit from revenue - cost where not filed, keeping the
    # later of the two filing dates (the fact isn't knowable until both exist).
    if not out["gross_profit"] and out["revenue"] and out["cost_of_rev"]:
        cost_by_fy = {fy: (v, f) for fy, v, f in out["cost_of_rev"]}
        derived = []
        for fy, rev, rfiled in out["revenue"]:
            if fy in cost_by_fy:
                cval, cfiled = cost_by_fy[fy]
                derived.append((fy, rev - cval, max(rfiled, cfiled)))
        out["gross_profit"] = derived
    return out


def knowable_as_of(pit: dict, t: date) -> dict:
    """Collapse to {metric: {fy: value}} using ONLY facts filed on/before t.

    This is the look-ahead guard in action: pass a date, get back only what
    was public then. Feed this to the scorer to compute a historical score.
    """
    out = {}
    for metric, rows in pit.items():
        out[metric] = {fy: v for fy, v, filed in rows if filed <= t}
    return out
