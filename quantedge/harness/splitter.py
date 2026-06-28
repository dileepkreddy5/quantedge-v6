"""splitter — honest train/test splits by TIME, no leakage (Manual §13).

Split by time; score only on data after the training boundary. No random
shuffles — that leaks the future into the past in a time series.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Split:
    train_start: date
    train_end: date     # inclusive training boundary
    test_start: date    # must be strictly after train_end
    test_end: date

    def __post_init__(self):
        if not (self.train_start <= self.train_end < self.test_start <= self.test_end):
            raise ValueError(
                "Invalid time split: require "
                "train_start <= train_end < test_start <= test_end"
            )

    def is_train(self, t: date) -> bool:
        return self.train_start <= t <= self.train_end

    def is_test(self, t: date) -> bool:
        return self.test_start <= t <= self.test_end
