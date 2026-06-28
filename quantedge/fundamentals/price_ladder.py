"""Price-movement ladder — how far a name has moved across time windows.

Context for the fundamental score (manual 7.1 price/attention velocity):
a high score + flat ladder = window still open (early); a high score + a
rising ladder = market starting to notice; already-run = late. This is
CONTEXT for a fundamentally-justified name, never a reason to chase (Table 17).
"""
from __future__ import annotations
from datetime import date, timedelta
from quantedge.data.sources.polygon_prices import daily_closes

WINDOWS = [("1d",1),("3d",3),("1w",7),("2w",14),("1m",30),("2m",60),("3m",90)]


def price_ladder(ticker: str, as_of: date | None = None) -> dict:
    """Return {window_label: pct_move} across the ladder, ending at as_of."""
    as_of = as_of or date.today()
    # one fetch covering the longest window, then slice
    closes = daily_closes(ticker, as_of - timedelta(days=130), as_of)
    if len(closes) < 2:
        return {lbl: None for lbl, _ in WINDOWS}
    last_date, last_px = closes[-1]
    out = {}
    for lbl, days in WINDOWS:
        target = last_date - timedelta(days=days)
        # nearest close on/before target
        prior = [c for c in closes if c[0] <= target]
        if prior:
            out[lbl] = round(last_px / prior[-1][1] - 1.0, 4)
        else:
            out[lbl] = None
    return out
