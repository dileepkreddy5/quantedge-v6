"""learning_loop — the system studies its own failures (Manual §16).

A system studied only on winners learns overconfidence; one that studies its
failures learns where not to trust itself. This is the edge a big black-box
shop usually skips: every miss and every false-positive becomes structured
research, not regret.

Three parts:
  - MissTaxonomy  : why a true winner was missed (§16.1) — fixed causes.
  - Graveyard     : companies the system LIKED that then FAILED (§16.2).
  - (timeline lives in company_timeline.py — the mission's direct test, §16.3)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import date
from enum import Enum
import json
import os


# §16.1 — the fixed taxonomy. A miss must map to exactly one cause, so misses
# become research (each category drives a specific fix), not vague regret.
class MissCause(str, Enum):
    MISSING_DATA = "missing_data"
    WRONG_INDUSTRY_READ = "wrong_industry_read"
    VALUATION_GATE_TOO_STRICT = "valuation_gate_too_strict"
    MODEL_BLIND_SPOT = "model_blind_spot"
    NEWS_MISINTERPRETATION = "news_misinterpretation"
    FALSE_ACCOUNTING_SIGNAL = "false_accounting_signal"


# §16.2 — autopsy causes for a company that was LIKED then FAILED.
class FailureCause(str, Enum):
    MANAGEMENT = "management"
    FRAUD = "fraud"
    COMPETITION = "competition"
    REGULATION = "regulation"
    DEBT = "debt"
    EXECUTION = "execution"


@dataclass
class Miss:
    ticker: str
    as_of: date              # when we should have flagged it
    cause: MissCause
    note: str                # the specific evidence that explains the miss


@dataclass
class GraveyardEntry:
    """A company the system LIKED that then FAILED — highest-information event."""
    ticker: str
    liked_on: date           # when the system rated it attractive
    failed_by: date          # when failure was confirmed
    cause: FailureCause
    score_at_like: float     # what we thought then
    note: str                # the autopsy — what we missed

    def as_negative_evidence(self) -> dict:
        """The pattern that should down-weight future look-alikes (§16.2)."""
        return {"ticker": self.ticker, "cause": self.cause.value,
                "score_at_like": self.score_at_like}


class Graveyard:
    """Permanent, append-only record. Failures are never silently forgotten."""
    def __init__(self, path: str = "quantedge/journal/graveyard/graveyard.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def bury(self, entry: GraveyardEntry) -> None:
        rec = asdict(entry)
        rec["liked_on"] = entry.liked_on.isoformat()
        rec["failed_by"] = entry.failed_by.isoformat()
        rec["cause"] = entry.cause.value
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def all(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def negative_evidence(self) -> list[dict]:
        """Feed for the down-weighting model — what we got wrong, by cause."""
        return [{"ticker": r["ticker"], "cause": r["cause"],
                 "score_at_like": r["score_at_like"]} for r in self.all()]
