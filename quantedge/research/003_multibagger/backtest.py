"""Multibagger score backtest — does the score PREDICT forward returns?

For each company: compute its fundamental score AS OF a past date (using only
facts filed by then), then measure its ACTUAL forward return from that date.
High-scored names outperforming low-scored ones = predictive lift.
Small-sample directional check, NOT full validation (Phase 5 needs the
survivorship-free universe). But real PIT discipline.
"""
from __future__ import annotations
import os, sys
from datetime import date
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from quantedge.fundamentals.edgar_pit import fetch_pit_facts, knowable_as_of
from quantedge.fundamentals.multibagger_score import score
from quantedge.data.sources.polygon_prices import daily_closes


def composite(r) -> float:
    s = 0.0
    s += (r.piotroski / 9) * 40
    if r.gross_profitability:
        s += min(r.gross_profitability, 1.0) * 30
    if r.growth_accelerating:
        s += 30
    elif r.rev_growth_recent and r.rev_growth_recent > 0.15:
        s += 15
    return round(s, 1)


def backtest_company(ticker, cik, as_of, fwd_end):
    pit = fetch_pit_facts(cik)
    known = knowable_as_of(pit, as_of)
    r = score(ticker, known)
    comp = composite(r)
    closes = daily_closes(ticker, as_of, fwd_end)
    fwd_ret = (closes[-1][1] / closes[0][1] - 1.0) if len(closes) >= 2 else None
    return comp, r.piotroski, fwd_ret
