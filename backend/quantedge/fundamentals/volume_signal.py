"""Volume-change signal — accumulation confirmation (no fake buy/sell split).

True buy/sell volume is not derivable from daily bars (every trade has both).
What IS honest and computable:
  - vol_change_pct: recent (5d) avg volume vs trailing (60d) avg volume.
    A surge = the market starting to notice (accumulation), confirming a
    fundamentally-justified name. NOT a reason to chase weak fundamentals.
  - up_vol_ratio: share of recent volume on UP days vs down days — a real
    proxy for accumulation pressure (up-day volume = net buying interest).
"""
from __future__ import annotations
from datetime import date, timedelta
from quantedge.data.sources.polygon_prices import daily_closes


def volume_signal(ticker: str, as_of: date | None = None) -> dict:
    as_of = as_of or date.today()
    # need volume, so fetch raw bars via a small helper using daily_closes' source
    import os, urllib.request, json
    POLY = os.environ.get("POLYGON_API_KEY") or os.environ.get("POLYGON")
    start = (as_of - timedelta(days=95)).isoformat()
    url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
           f"{start}/{as_of.isoformat()}?adjusted=true&sort=asc&limit=200&apiKey={POLY}")
    try:
        d = json.load(urllib.request.urlopen(url, timeout=20))
        bars = d.get("results", [])
    except Exception:
        return {"vol_change_pct": None, "up_vol_ratio": None}
    if len(bars) < 20:
        return {"vol_change_pct": None, "up_vol_ratio": None}
    vols = [b.get("v", 0) for b in bars]
    recent = vols[-5:]
    base = vols[:-5] or vols
    rec_avg = sum(recent)/len(recent)
    base_avg = sum(base)/len(base)
    vol_change = (rec_avg/base_avg - 1.0) if base_avg else None
    # up-day vs down-day volume over the recent ~20 days
    up_v = dn_v = 0.0
    for i in range(max(1, len(bars)-20), len(bars)):
        if bars[i].get("c", 0) >= bars[i-1].get("c", 0):
            up_v += bars[i].get("v", 0)
        else:
            dn_v += bars[i].get("v", 0)
    tot = up_v + dn_v
    up_ratio = (up_v/tot) if tot else None
    return {"vol_change_pct": round(vol_change, 3) if vol_change is not None else None,
            "up_vol_ratio": round(up_ratio, 3) if up_ratio is not None else None}
