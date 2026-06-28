"""pit_loader — the single most important module (Manual Table 25, §13).

Given a date T, return ONLY facts with available_date <= T. No future data
reaches a model. Ever. This is the look-ahead trap's mechanical defense:
enforced here, in one place, so no unit can violate it by accident.

A fact's fiscal_period (the period it describes) is irrelevant to visibility
— only available_date (when it was FILED) decides whether T can see it. A Q4
result for period 2020-12-31 filed 2021-02-15 is INVISIBLE on 2021-01-10,
even though the quarter is over. That distinction is the entire point.
"""
from __future__ import annotations
from datetime import date
from typing import Iterable
from quantedge.data.sources.base import FundamentalFact


class LookAheadError(Exception):
    """Raised if a fact with available_date > as_of ever escapes the loader."""


class PITLoader:
    def __init__(self, facts: Iterable[FundamentalFact]):
        self._facts: tuple[FundamentalFact, ...] = tuple(facts)

    def as_of(self, t: date) -> list[FundamentalFact]:
        """Every fact knowable on date t — available_date <= t. Nothing else."""
        visible = [f for f in self._facts if f.available_date <= t]
        for f in visible:
            if f.available_date > t:
                raise LookAheadError(
                    f"{f.ticker}/{f.metric} available {f.available_date} > as_of {t}"
                )
        return visible

    def latest(self, ticker: str, metric: str, t: date) -> FundamentalFact | None:
        """Most recently FILED value of a metric knowable as of t."""
        candidates = [
            f for f in self.as_of(t)
            if f.ticker == ticker and f.metric == metric
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda f: f.available_date)
