"""Minimal Polygon daily-price fetcher for the CF reproduction.

Key is read from the environment (loaded from .env on the server), never
hardcoded and never logged. Returns a date-indexed list of adjusted closes.
"""
from __future__ import annotations
import os, time, urllib.request, urllib.parse, json
from datetime import date


def _key() -> str:
    k = os.environ.get("POLYGON_API_KEY") or os.environ.get("POLYGON")
    if not k:
        raise RuntimeError("No POLYGON key in environment")
    return k


def daily_closes(ticker: str, start: date, end: date, pause: float = 0.2):
    """List of (date, adjusted_close), ascending. Polygon free tier = 5 req/min."""
    url = (f"https://api.polygon.io/v2/aggs/ticker/{urllib.parse.quote(ticker)}"
           f"/range/1/day/{start.isoformat()}/{end.isoformat()}"
           f"?adjusted=true&sort=asc&limit=50000&apiKey={_key()}")
    with urllib.request.urlopen(url, timeout=30) as r:
        d = json.load(r)
    if d.get("status") not in ("OK", "DELAYED") or not d.get("results"):
        return []
    out = []
    for bar in d["results"]:
        # bar['t'] is epoch ms of the bar's day
        dt = date.fromtimestamp(bar["t"] / 1000)
        out.append((dt, bar["c"]))
    time.sleep(pause)
    return out


def monthly_return(closes, lookback_days: int = 21):
    """Trailing return over ~one trading month from a (date,close) series."""
    if len(closes) < lookback_days + 1:
        return None
    return closes[-1][1] / closes[-1 - lookback_days][1] - 1.0
