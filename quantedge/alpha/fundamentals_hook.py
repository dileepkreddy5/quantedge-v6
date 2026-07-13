"""Fundamentals feature hook (Stage F+) — wire PIT fundamentals into the panel.

The alpha model has only ever seen price/volume features. This adds the
orthogonal fundamentals we already extract (PIT, tested in the REBOUND engine):
value vs own history, quality (F-score), margin trend, accruals, growth. These
are NEW information for the model, tested under the SAME frozen gates — new
inputs, not a re-slice of the old signal.

Slow-moving factors are computed once per (ticker, quarter) and cached in
memory, so the monthly panel build stays fast. Everything filed <= as_of.
Returns None per-field when data is missing (model imputes; never fabricated).
"""
from __future__ import annotations
from datetime import date
from typing import Dict, Optional

from quantedge.fundamentals.edgar_bulk import company_facts_from_bulk
from quantedge.fundamentals.bulk_adapter import pit_from_bulk
from quantedge.fundamentals.edgar_pit import knowable_as_of
from quantedge.fundamentals.rebound import bulk_extra as bx
from quantedge.fundamentals.rebound.discount import valuation_vs_own_history
from quantedge.fundamentals.rebound.health import growth_streak, margin_trajectory
from quantedge.fundamentals.multibagger_score import score as piotroski_score
from quantedge.fundamentals.extra_signals import accruals


class FundamentalsProvider:
    """Callable (ticker, as_of) -> feature dict, cached by (cik, quarter)."""
    def __init__(self, cikmap: Dict[str, str], price_lookup):
        self.cikmap = cikmap
        self.price_lookup = price_lookup          # fn(ticker, as_of) -> closes list
        self._facts_cache: Dict[str, Optional[dict]] = {}
        self._feat_cache: Dict[str, Dict] = {}

    def _facts(self, cik: str) -> Optional[dict]:
        if cik not in self._facts_cache:
            try:
                self._facts_cache[cik] = company_facts_from_bulk(cik)
            except Exception:
                self._facts_cache[cik] = None
        return self._facts_cache[cik]

    def __call__(self, ticker: str, as_of: date) -> Optional[Dict]:
        cik = self.cikmap.get(ticker)
        if not cik:
            return None
        qkey = f"{cik}:{as_of.year}Q{(as_of.month-1)//3}"
        if qkey in self._feat_cache:
            return self._feat_cache[qkey]

        facts = self._facts(cik)
        if not facts:
            self._feat_cache[qkey] = None
            return None

        out: Dict[str, Optional[float]] = {}
        try:
            q_rev = bx.quarterly_revenue_complete(facts)
            q_gp = bx.quarterly_gross_profit(facts)
            shares = bx.quarterly_shares(facts)
            cash = bx.cash_series(facts)
            debt = bx.debt_series(facts)
            known = knowable_as_of(pit_from_bulk(facts), as_of)
            closes = self.price_lookup(ticker, as_of)

            val = valuation_vs_own_history(closes, q_rev, shares, cash, debt, as_of)
            out["ps_percentile_own"] = val.get("ps_percentile_own") if val.get("ok") else None

            gs = growth_streak(q_rev, as_of, 25_000_000)
            out["growth_streak"] = float(gs["streak"]) if gs.get("ok") else None
            out["latest_yoy"] = gs.get("latest_yoy") if gs.get("ok") else None

            mt = margin_trajectory(q_gp, q_rev, as_of)
            out["gross_margin_slope"] = mt.get("gross_margin_slope_per_q") if mt.get("ok") else None

            try:
                pio = piotroski_score(ticker, known)
                out["piotroski"] = float(pio.piotroski)
            except Exception:
                out["piotroski"] = None
            out["accruals"] = accruals(known)

            sh_now = bx.latest_knowable(shares, as_of)
            mktcap = closes[-1][1] * sh_now[1] if closes and sh_now and sh_now[1] > 0 else None
            rd = bx.knowable(bx.ttm_points(bx.rd_series(facts)), as_of)
            out["rd_to_mktcap"] = (rd[-1][1] / mktcap) if (rd and mktcap) else None
        except Exception:
            pass

        self._feat_cache[qkey] = out
        return out
