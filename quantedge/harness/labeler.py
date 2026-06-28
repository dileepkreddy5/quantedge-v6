"""labeler — the most error-prone step (Manual §14, Table 30).

Companies exit the data for opposite reasons; mislabeling exits biases base
rates in opposite directions. Every exit type is handled explicitly.
Total-return labeling, not price-only: a company taken private at a premium
is correctly a WINNER; one delisted for failure is a NON_WINNER — a loss,
NOT a dropped row. Dropping it is the core survivorship error.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from enum import Enum


class Label(str, Enum):
    WINNER = "WINNER"
    NON_WINNER = "NON_WINNER"
    CENSORED = "CENSORED"
    FOLLOW_SUCCESSOR = "FOLLOW_SUCCESSOR"


class Exit(str, Enum):
    STILL_TRADING = "still_trading"
    BANKRUPT_DELISTED = "bankrupt_delisted"
    ACQUIRED_PREMIUM = "acquired_premium"
    ACQUIRED_AT_OR_BELOW = "acquired_at_or_below"
    MERGED_CONTINUES = "merged_continues"


@dataclass
class Outcome:
    exit_type: Exit
    horizon_reached: bool
    total_excess_return_vs_sector: float | None = None  # multiple, e.g. 2.3
    revenue_multiple: float | None = None
    roic_improving: bool | None = None
    terminal_dilution_pct: float | None = None
    bankrupt: bool = False


def label(outcome: Outcome, params: dict) -> Label:
    """Apply the §14 exit rules + §22.1 winner definition (params.yaml)."""
    if not outcome.horizon_reached:
        return Label.CENSORED  # unfinished window = look-ahead if used

    if outcome.exit_type is Exit.MERGED_CONTINUES:
        return Label.FOLLOW_SUCCESSOR
    if outcome.exit_type is Exit.BANKRUPT_DELISTED:
        return Label.NON_WINNER  # a loss, not a missing row — survivorship fix
    if outcome.exit_type is Exit.ACQUIRED_AT_OR_BELOW:
        return Label.NON_WINNER  # not value creation

    wd = params["winner_definition"]
    market_ok = (
        outcome.total_excess_return_vs_sector is not None
        and outcome.total_excess_return_vs_sector >= wd["market_test"]["multiple"]
    )

    if outcome.exit_type is Exit.ACQUIRED_PREMIUM:
        return Label.WINNER if market_ok else Label.NON_WINNER

    bt = wd["business_test"]
    business_ok = (
        outcome.revenue_multiple is not None
        and outcome.revenue_multiple >= bt["revenue_multiple"]
        and (outcome.roic_improving is True if bt["require_roic_improving"] else True)
    )
    st = wd["survival_test"]
    survival_ok = (
        not outcome.bankrupt
        and (outcome.terminal_dilution_pct is None
             or outcome.terminal_dilution_pct <= st["max_terminal_dilution_pct"])
    )
    return Label.WINNER if (market_ok and business_ok and survival_ok) else Label.NON_WINNER
