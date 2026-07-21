"""
Select the liquid US universe dynamically from the whole market.
Ranks all US-listed stocks by dollar volume (liquidity), returns the top N.
This is what makes the ML train on 'every US stock that matters' rather than a
hardcoded list — the universe is the real market, filtered to tradeable names.
"""
from __future__ import annotations
import os, json, time, urllib.request
from datetime import date, timedelta
from typing import List, Dict

POLY = os.environ.get("POLYGON_API_KEY", "")

def _recent_weekday(back=1) -> date:
    d = date.today() - timedelta(days=back)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d

def liquid_universe(top_n: int = 2500, min_price: float = 3.0,
                    min_dollar_vol: float = 5e6) -> List[str]:
    """Return top_n most-liquid US common stocks by dollar volume.
    Filters: price >= min_price (no penny stocks), dollar_vol >= min_dollar_vol
    (tradeable), averaged over several recent days for stability."""
    from collections import defaultdict
    dv = defaultdict(list)  # ticker -> [dollar_volume per day]
    days_fetched = 0
    for back in range(1, 12):  # average over ~7 trading days
        d0 = _recent_weekday(back)
        url = (f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/"
               f"{d0.isoformat()}?adjusted=true&apiKey={POLY}")
        try:
            data = json.load(urllib.request.urlopen(url, timeout=60))
            res = data.get("results", [])
            if not res:
                continue
            for r in res:
                t = r.get("T"); c = r.get("c"); v = r.get("v")
                if not t or not c or not v:
                    continue
                if c < min_price:
                    continue
                # skip obvious non-common (warrants/units/rights) by suffix
                if any(t.endswith(sfx) for sfx in [".W", ".U", ".R", "W", "R"]) and len(t) > 4:
                    continue
                dv[t].append(c * v)
            days_fetched += 1
            if days_fetched >= 7:
                break
        except Exception:
            time.sleep(1)
    # average dollar volume, filter, rank
    ranked = []
    for t, vols in dv.items():
        if len(vols) < 3:
            continue
        avg_dv = sum(vols) / len(vols)
        if avg_dv >= min_dollar_vol:
            ranked.append((t, avg_dv))
    ranked.sort(key=lambda x: -x[1])
    universe = [t for t, _ in ranked[:top_n]]
    return universe

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    u = liquid_universe(top_n=n)
    print(f"Liquid universe: {len(u)} tickers")
    print("Top 20 by liquidity:", u[:20])
    print("...")
    print("Bottom 10 of selection:", u[-10:])
