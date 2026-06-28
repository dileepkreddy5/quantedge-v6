"""Quarterly point-in-time EDGAR fetcher — catches inflection before annuals.

Annual data arrives ~too late to find multibaggers early (proven by backtest).
Quarterly revenue, with filing dates, lets us measure growth ACCELERATION
quarter-over-quarter and year-over-year — the leading signal (manual 7.1).
"""
from __future__ import annotations
import urllib.request, json, time
from datetime import date

UA = "Dileep Kapu dileepkreddy5@gmail.com"
REV_TAGS = ["RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues", "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet"]


def _quarterly_revenue(cik: str, pause: float = 0.1):
    """Return [(period_end_date, value, filed_date)] for ~quarterly revenue.

    EDGAR rows have start/end; a quarter is ~85-95 days. We keep those, sorted
    by period end, each stamped with its filing date for PIT correctness.
    """
    merged = {}  # end_date -> (value, filed)
    for tag in REV_TAGS:
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            d = json.load(urllib.request.urlopen(req, timeout=20))
        except Exception:
            time.sleep(pause); continue
        units = d.get("units", {})
        if not units:
            time.sleep(pause); continue
        for r in units[list(units.keys())[0]]:
            if not (r.get("start") and r.get("end") and r.get("filed")):
                continue
            try:
                s = date.fromisoformat(r["start"]); e = date.fromisoformat(r["end"])
            except Exception:
                continue
            days = (e - s).days
            if 80 <= days <= 100:  # a single quarter, not a YTD or annual span
                key = e
                val = r["val"]
                cur = merged.get(key)
                if cur is None or (cur[0] == 0 and val != 0):
                    merged[key] = (val, date.fromisoformat(r["filed"]))
        time.sleep(pause)
    return [(e, v, f) for e, (v, f) in sorted(merged.items())]


def quarterly_growth_signal(cik: str, as_of: date):
    """Compute YoY quarterly revenue growth + acceleration, as knowable on as_of.

    Returns dict with the latest knowable quarter's YoY growth, the prior
    quarter's YoY growth, and whether growth is accelerating.
    """
    q = _quarterly_revenue(cik)
    # Only quarters whose FILING date is on/before as_of (PIT guard).
    visible = [(e, v) for e, v, f in q if f <= as_of]
    if len(visible) < 5:
        return {"ok": False, "reason": "insufficient quarterly history"}
    visible.sort()
    by_end = dict(visible)

    def yoy(end):
        # find the quarter ~1 year earlier (same period prior year)
        prior = date(end.year - 1, end.month, min(end.day, 28))
        # match nearest available end within ~45 days
        best = min(by_end.keys(), key=lambda k: abs((k - prior).days))
        if abs((best - prior).days) <= 45 and by_end[best]:
            return by_end[end] / by_end[best] - 1.0
        return None

    ends = sorted(by_end.keys())
    latest, prev = ends[-1], ends[-2]
    g_latest, g_prev = yoy(latest), yoy(prev)
    accel = (g_latest is not None and g_prev is not None and g_latest > g_prev)
    return {"ok": True, "latest_quarter": latest.isoformat(),
            "yoy_growth_latest": g_latest, "yoy_growth_prev": g_prev,
            "accelerating": accel}
