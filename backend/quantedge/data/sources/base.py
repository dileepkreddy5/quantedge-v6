"""Data-source adapters — one per source, a common interface (Manual Table 25).

The point-in-time discipline starts here: every fundamental fact carries the
date it was FILED (available_date), not just the period it describes. That
stamp is the look-ahead defense the whole harness depends on (§13, Table 26).
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Protocol


@dataclass(frozen=True)
class FundamentalFact:
    """Manual Table 26. available_date is the guard — when it was filed."""
    ticker: str
    metric: str            # e.g. 'gross_profit'
    fiscal_period: date    # the period it describes
    value: float
    available_date: date   # when it was FILED — the look-ahead guard
    source: str


@dataclass(frozen=True)
class Edge:
    """Manual Table 27. A supplier->customer economic link, time-stamped."""
    from_ticker: str       # supplier
    to_ticker: str         # customer
    weight: float          # % of supplier revenue
    tier: int              # 1=direct, 2/3=indirect
    source: str            # 10-K | BEA | text | commercial
    confirmed: bool        # 2-source rule (§8.1)
    as_of: date            # links change over time


class DataSource(Protocol):
    """Common interface. Each source implements what it does well (Table 24)."""
    name: str
    def fetch_fundamentals(self, ticker: str) -> list[FundamentalFact]: ...
    def fetch_prices(self, ticker: str, start: date, end: date): ...
