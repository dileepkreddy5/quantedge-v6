"""Cap-tier universe builder — real data, SEC tickers + Polygon market caps.

Buckets the US universe into Small / Mid / Large using the thresholds in
params.yaml. The ticker->CIK map is SEC's authoritative file (we proved a bad
CIK silently scores the wrong company, so the source must be authoritative).
"""
from __future__ import annotations
import os, urllib.request, json, time
from datetime import date

UA = "Dileep Kapu dileepkreddy5@gmail.com"
POLY = os.environ.get("POLYGON_API_KEY") or os.environ.get("POLYGON")


def sec_ticker_cik_map() -> dict:
    """{TICKER: cik_str(10-digit)} from SEC's official file."""
    url = "https://www.sec.gov/files/company_tickers.json"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    d = json.load(urllib.request.urlopen(req, timeout=30))
    out = {}
    for row in d.values():
        out[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
    return out


def polygon_market_caps(tickers: list, pause: float = 0.02) -> dict:
    """{TICKER: market_cap} from Polygon ticker details. Paced for rate limit."""
    caps = {}
    for t in tickers:
        url = f"https://api.polygon.io/v3/reference/tickers/{t}?apiKey={POLY}"
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                d = json.load(r)
            mc = d.get("results", {}).get("market_cap")
            if mc:
                caps[t] = mc
        except Exception:
            pass
        time.sleep(pause)
    return caps


def tier_of(market_cap: float, params: dict) -> str | None:
    ct = params["cap_tiers"]
    if market_cap >= ct["large_cap_min_usd"]:
        return "large"
    if ct["mid_cap_usd"][0] <= market_cap < ct["mid_cap_usd"][1]:
        return "mid"
    if ct["small_cap_usd"][0] <= market_cap < ct["small_cap_usd"][1]:
        return "small"
    return None  # below small-cap floor (micro) — excluded
