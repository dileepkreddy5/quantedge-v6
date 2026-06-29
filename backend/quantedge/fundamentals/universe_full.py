"""Full-universe builder — market cap for the WHOLE market in ~1 call.

The free Polygon tier can't fetch 3000 market caps individually (hours). But
ONE grouped-daily call returns every ticker's close. Combined with shares
outstanding (EDGAR, free), we compute market cap for the entire market, rank,
and take the top N per cap tier. This is the §7.2.4 'design the full scan from
the start' solution without a paid tier.
"""
from __future__ import annotations
import os, urllib.request, json, time
from datetime import date, timedelta

POLY = os.environ.get("POLYGON_API_KEY") or os.environ.get("POLYGON")
UA = "Dileep Kapu dileepkreddy5@gmail.com"


def _recent_weekday() -> date:
    d = date.today() - timedelta(days=1)
    while d.weekday() >= 5:  # skip Sat/Sun
        d -= timedelta(days=1)
    return d


def all_closes() -> dict:
    """{TICKER: close} for the whole US market in one grouped-daily call."""
    for back in range(1, 6):  # try last few days in case of holidays
        d0 = _recent_weekday() - timedelta(days=back-1)
        url = (f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/"
               f"{d0.isoformat()}?adjusted=true&apiKey={POLY}")
        try:
            data = json.load(urllib.request.urlopen(url, timeout=60))
            res = data.get("results", [])
            if res:
                return {r["T"]: r["c"] for r in res if r.get("c")}
        except Exception:
            time.sleep(1)
    return {}


def ticker_cik_map() -> dict:
    """{TICKER: cik(10-digit)} from Polygon's reference list (carries cik)."""
    out = {}
    url = (f"https://api.polygon.io/v3/reference/tickers?market=stocks&active=true"
           f"&type=CS&limit=1000&apiKey={POLY}")
    while url:
        d = json.load(urllib.request.urlopen(url, timeout=30))
        for r in d.get("results", []):
            cik = r.get("cik")
            if cik:
                out[r["ticker"]] = str(cik).zfill(10)
        nxt = d.get("next_url")
        url = (nxt + f"&apiKey={POLY}") if nxt else None
        time.sleep(0.1)
    return out


def shares_outstanding(cik: str) -> float | None:
    """Latest shares outstanding from EDGAR (free)."""
    for tag in ["CommonStockSharesOutstanding",
                "WeightedAverageNumberOfDilutedSharesOutstanding"]:
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            d = json.load(urllib.request.urlopen(req, timeout=15))
            rows = d["units"][list(d["units"].keys())[0]]
            vals = [r["val"] for r in rows if r.get("val")]
            if vals:
                return vals[-1]
        except Exception:
            continue
    return None
