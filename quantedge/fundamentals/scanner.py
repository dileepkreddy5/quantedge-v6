"""Tier scanner — scores a list of companies on the multibagger profile.

Combines what the backtests taught us:
  - quarterly YoY revenue growth MAGNITUDE (the real signal, not a flag)
  - Piotroski as a light quality sanity check
  - recent price move (for the 'quiet price' divergence — quiet = catchable)
Honest: this is a FILTER/shortlist, not a predictor. Tail inflections
(SMCI-2023) are unpredictable from prior fundamentals — proven by backtest.
"""
from __future__ import annotations
from datetime import date, timedelta
from quantedge.fundamentals.edgar_quarterly import quarterly_growth_signal
from quantedge.fundamentals.edgar_pit import fetch_pit_facts, knowable_as_of
from quantedge.fundamentals.multibagger_score import score
from quantedge.data.sources.polygon_prices import daily_closes
from quantedge.fundamentals.price_ladder import price_ladder


def scan_one(ticker: str, cik: str, as_of: date | None = None) -> dict | None:
    as_of = as_of or date.today()
    qs = quarterly_growth_signal(cik, as_of)
    growth = qs.get("yoy_growth_latest") if qs.get("ok") else None

    pit = fetch_pit_facts(cik)
    r = score(ticker, knowable_as_of(pit, as_of))

    # recent 6-month price move (the 'is the price still quiet?' read)
    closes = daily_closes(ticker, as_of - timedelta(days=190), as_of)
    px_move = (closes[-1][1] / closes[0][1] - 1.0) if len(closes) >= 2 else None

    if growth is None:
        return None

    # Composite: growth magnitude leads; piotroski light; quiet-price bonus.
    comp = min(max(growth, 0), 2.0) * 50          # up to 100 for 200% growth
    comp += (r.piotroski / 9) * 15
    quiet = (px_move is not None and px_move < 0.20)  # price hasn't run yet
    if quiet and growth and growth > 0.25:
        comp += 20                                # strong growth + quiet price
    return {
        "ticker": ticker, "score": round(comp, 1),
        "qtr_yoy_growth": round(growth, 3),
        "piotroski": r.piotroski,
        "price_move_6mo": round(px_move, 3) if px_move is not None else None,
        "quiet_price": quiet,
        "price_ladder": price_ladder(ticker, as_of),
    }


def scan_batch(items: list, as_of: date | None = None) -> list:
    """items = [(ticker, cik), ...]  -> ranked list of score dicts."""
    out = []
    for tk, cik in items:
        try:
            r = scan_one(tk, cik, as_of)
            if r:
                out.append(r)
        except Exception:
            pass
    out.sort(key=lambda x: x["score"], reverse=True)
    return out
