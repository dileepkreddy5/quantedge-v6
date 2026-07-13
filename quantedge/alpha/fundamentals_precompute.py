"""Memory-bounded fundamentals precompute — extract scalars, release blobs.

The OOM cause: holding full companyfacts JSON blobs in memory for thousands of
companies (~2.4GB) on a 4GB box running the live API. Fix: stream one company
at a time from the bulk file, extract the ~7 scalar features per quarter-end,
write to a small SQLite table, and RELEASE the blob before the next company.
Peak memory = one blob (~a few MB), not the whole universe.
"""
from __future__ import annotations
import os, sqlite3, time
from datetime import date, timedelta
from typing import Dict, List, Optional

from quantedge.data.price_store import PriceStore
from quantedge.fundamentals.universe_full import ticker_cik_map
from quantedge.fundamentals.edgar_bulk import company_facts_from_bulk
from quantedge.fundamentals.bulk_adapter import pit_from_bulk
from quantedge.fundamentals.edgar_pit import knowable_as_of
from quantedge.fundamentals.rebound import bulk_extra as bx
from quantedge.fundamentals.rebound.discount import valuation_vs_own_history
from quantedge.fundamentals.rebound.health import growth_streak, margin_trajectory
from quantedge.fundamentals.multibagger_score import score as piotroski_score
from quantedge.fundamentals.extra_signals import accruals

FIELDS = ["ps_percentile_own", "growth_streak", "latest_yoy",
          "gross_margin_slope", "piotroski", "accruals", "rd_to_mktcap"]


def _features_for(facts: dict, closes_fn, ticker: str, as_of: date) -> Dict:
    out: Dict[str, Optional[float]] = {k: None for k in FIELDS}
    try:
        q_rev = bx.quarterly_revenue_complete(facts)
        q_gp = bx.quarterly_gross_profit(facts)
        shares = bx.quarterly_shares(facts)
        cash = bx.cash_series(facts); debt = bx.debt_series(facts)
        known = knowable_as_of(pit_from_bulk(facts), as_of)
        closes = closes_fn(ticker, as_of)
        val = valuation_vs_own_history(closes, q_rev, shares, cash, debt, as_of)
        if val.get("ok"):
            out["ps_percentile_own"] = val.get("ps_percentile_own")
        gs = growth_streak(q_rev, as_of, 25_000_000)
        if gs.get("ok"):
            out["growth_streak"] = float(gs["streak"]); out["latest_yoy"] = gs.get("latest_yoy")
        mt = margin_trajectory(q_gp, q_rev, as_of)
        if mt.get("ok"):
            out["gross_margin_slope"] = mt.get("gross_margin_slope_per_q")
        try:
            out["piotroski"] = float(piotroski_score(ticker, known).piotroski)
        except Exception:
            pass
        out["accruals"] = accruals(known)
        sh_now = bx.latest_knowable(shares, as_of)
        mktcap = closes[-1][1]*sh_now[1] if closes and sh_now and sh_now[1] > 0 else None
        rd = bx.knowable(bx.ttm_points(bx.rd_series(facts)), as_of)
        out["rd_to_mktcap"] = (rd[-1][1]/mktcap) if (rd and mktcap) else None
    except Exception:
        pass
    return out


def precompute(price_db: str, out_db: str, as_ofs: List[date], verbose=True):
    store = PriceStore(price_db)
    cikmap = ticker_cik_map()

    def closes_fn(ticker, as_of):
        bars = store.series(ticker, as_of - timedelta(days=3*366), as_of)
        return [(d, c) for d, c, _ in bars]

    con = sqlite3.connect(out_db)
    con.execute("CREATE TABLE IF NOT EXISTS fund (cik TEXT, as_of TEXT, "
                + ", ".join(f"{f} REAL" for f in FIELDS)
                + ", PRIMARY KEY (cik, as_of))")
    con.commit()
    done = {(c, a) for c, a in con.execute("SELECT cik, as_of FROM fund")}

    tickers = sorted(set(cikmap))
    t0 = time.time()
    for i, tkr in enumerate(tickers):
        cik = cikmap[tkr]
        if all((cik, a.isoformat()) in done for a in as_ofs):
            continue
        try:
            facts = company_facts_from_bulk(cik)
        except Exception:
            facts = None
        if facts:
            for a in as_ofs:
                if (cik, a.isoformat()) in done:
                    continue
                f = _features_for(facts, closes_fn, tkr, a)
                con.execute(
                    "INSERT OR REPLACE INTO fund VALUES (?,?," + ",".join("?"*len(FIELDS)) + ")",
                    (cik, a.isoformat(), *[f[k] for k in FIELDS]))
        facts = None
        if i % 200 == 0:
            con.commit()
            if verbose:
                print(f"  {i}/{len(tickers)} companies ({time.time()-t0:.0f}s)", flush=True)
    con.commit(); con.close(); store.close()
    if verbose:
        print(f"precompute done: {len(tickers)} companies x {len(as_ofs)} dates", flush=True)


class FundTable:
    """Fast reader the panel build uses instead of parsing blobs."""
    def __init__(self, db: str, cikmap: Dict[str, str]):
        self.con = sqlite3.connect(db)
        self.cikmap = cikmap
    def __call__(self, ticker: str, as_of: date) -> Optional[Dict]:
        cik = self.cikmap.get(ticker)
        if not cik:
            return None
        row = self.con.execute(
            "SELECT " + ",".join(FIELDS) + " FROM fund WHERE cik=? AND as_of=?",
            (cik, as_of.isoformat())).fetchone()
        return dict(zip(FIELDS, row)) if row else None
