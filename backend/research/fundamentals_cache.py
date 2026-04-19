"""
QuantEdge v6.0 — Fundamentals Cache for Backtesting
=====================================================
Fetches historical quarterly financials (10 years) for the universe,
caches to local disk. Provides point-in-time query that respects
filing_date discipline.

CRITICAL: At rebalance T, returns ONLY filings where filing_date < T.
Default 45-day lag applied on top — no filing within 45 days of T is
considered "available" to avoid latency/processing assumptions.
"""

import os
import pickle
import asyncio
import aiohttp
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from loguru import logger


POLYGON_BASE = "https://api.polygon.io"
CACHE_DIR = Path.home() / ".quantedge_cache" / "fundamentals"
DEFAULT_FILING_LAG_DAYS = 45


@dataclass
class HistoricalQuarter:
    """One historical quarter, fetched once and cached."""
    fiscal_period: str
    fiscal_year: int
    filing_date: Optional[date]       # when it was filed with SEC
    period_end: Optional[date]        # quarter ending date
    # Income statement
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    eps_diluted: Optional[float] = None
    # Balance sheet
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    long_term_debt: Optional[float] = None
    cash: Optional[float] = None
    # Cash flow
    operating_cash_flow: Optional[float] = None
    capex: Optional[float] = None


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.pkl"


def _safe(d: dict, key: str) -> Optional[float]:
    if not d or key not in d:
        return None
    v = d[key]
    if isinstance(v, dict):
        v = v.get("value")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date()
    except (ValueError, TypeError):
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except Exception:
            return None


async def _fetch_one(
    ticker: str,
    api_key: str,
    session: aiohttp.ClientSession,
    limit: int = 40,
) -> List[HistoricalQuarter]:
    """Fetch up to `limit` quarters for one ticker."""
    url = f"{POLYGON_BASE}/vX/reference/financials"
    params = {
        "ticker": ticker.upper(),
        "timeframe": "quarterly",
        "order": "desc",
        "limit": limit,
        "sort": "period_of_report_date",
        "apiKey": api_key,
    }

    for attempt in range(3):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if resp.status != 200:
                    logger.warning(f"{ticker} fundamentals HTTP {resp.status}")
                    return []
                data = await resp.json()
                break
        except Exception as e:
            if attempt == 2:
                logger.warning(f"{ticker} fundamentals failed: {e}")
                return []
            await asyncio.sleep(1)
    else:
        return []

    results = data.get("results", [])
    if not results:
        return []

    quarters: List[HistoricalQuarter] = []
    for r in results:
        fin = r.get("financials", {}) or {}
        income = fin.get("income_statement", {}) or {}
        balance = fin.get("balance_sheet", {}) or {}
        cashflow = fin.get("cash_flow_statement", {}) or {}

        q = HistoricalQuarter(
            fiscal_period=r.get("fiscal_period", ""),
            fiscal_year=int(r.get("fiscal_year", 0) or 0),
            filing_date=_parse_date(r.get("filing_date")),
            period_end=_parse_date(r.get("end_date") or r.get("period_of_report_date")),
            revenue=_safe(income, "revenues"),
            gross_profit=_safe(income, "gross_profit"),
            operating_income=_safe(income, "operating_income_loss"),
            net_income=_safe(income, "net_income_loss"),
            eps_diluted=_safe(income, "diluted_earnings_per_share"),
            total_assets=_safe(balance, "assets"),
            current_assets=_safe(balance, "current_assets"),
            current_liabilities=_safe(balance, "current_liabilities"),
            total_liabilities=_safe(balance, "liabilities"),
            total_equity=_safe(balance, "equity"),
            long_term_debt=_safe(balance, "long_term_debt"),
            cash=_safe(balance, "cash"),
            operating_cash_flow=_safe(cashflow, "net_cash_flow_from_operating_activities"),
            capex=_safe(cashflow, "net_cash_flow_from_investing_activities"),
        )
        quarters.append(q)
    # Oldest to newest
    quarters.reverse()
    return quarters


class FundamentalsCache:
    """
    Point-in-time fundamentals store.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._memory: Dict[str, List[HistoricalQuarter]] = {}

    def _load_from_disk(self, ticker: str) -> Optional[List[HistoricalQuarter]]:
        p = _cache_path(ticker)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_to_disk(self, ticker: str, quarters: List[HistoricalQuarter]):
        p = _cache_path(ticker)
        try:
            with open(p, "wb") as f:
                pickle.dump(quarters, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning(f"Fundamentals cache save failed for {ticker}: {e}")

    async def populate(
        self,
        tickers: List[str],
        concurrency: int = 3,
        force_refresh: bool = False,
    ) -> Dict[str, int]:
        """Download fundamentals for all tickers. One-time cache build."""
        if not self.api_key:
            raise RuntimeError("POLYGON_API_KEY not set")

        to_fetch: List[str] = []
        for t in tickers:
            if not force_refresh and _cache_path(t).exists():
                continue
            to_fetch.append(t)

        logger.info(f"Fundamentals: {len(tickers) - len(to_fetch)} cached, {len(to_fetch)} to fetch")

        if not to_fetch:
            return {"cached": len(tickers), "fetched": 0, "failed": 0}

        sem = asyncio.Semaphore(concurrency)
        connector = aiohttp.TCPConnector(limit=10, force_close=False, enable_cleanup_closed=True)

        async def _task(ticker: str, session: aiohttp.ClientSession) -> Tuple[str, bool]:
            async with sem:
                quarters = await _fetch_one(ticker, self.api_key, session)
                if quarters and len(quarters) >= 4:
                    self._save_to_disk(ticker, quarters)
                    return ticker, True
                return ticker, False

        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [_task(t, session) for t in to_fetch]
            results = await asyncio.gather(*tasks, return_exceptions=False)

        fetched = sum(1 for _, ok in results if ok)
        failed = sum(1 for _, ok in results if not ok)
        logger.info(f"Fundamentals done: {fetched} fetched, {failed} failed")

        return {
            "cached": len(tickers) - len(to_fetch),
            "fetched": fetched,
            "failed": failed,
        }

    def get_quarters_as_of(
        self,
        ticker: str,
        as_of: date,
        lag_days: int = DEFAULT_FILING_LAG_DAYS,
    ) -> List[HistoricalQuarter]:
        """
        Point-in-time query. Returns quarters where filing_date < (as_of - lag_days).

        This is the CRITICAL function. It enforces that we only use financial
        data that was actually available `lag_days` before `as_of`.
        """
        if ticker not in self._memory:
            qs = self._load_from_disk(ticker)
            if qs is None:
                return []
            self._memory[ticker] = qs

        cutoff = as_of - timedelta(days=lag_days)
        return [
            q for q in self._memory[ticker]
            if q.filing_date is not None and q.filing_date <= cutoff
        ]

    def available_tickers(self) -> List[str]:
        if not CACHE_DIR.exists():
            return []
        return sorted([p.stem.upper() for p in CACHE_DIR.glob("*.pkl")])


# ══════════════════════════════════════════════════════════════
# POINT-IN-TIME QUALITY COMPUTATION
# ══════════════════════════════════════════════════════════════
def compute_quality_pit(
    quarters: List[HistoricalQuarter],
    min_quarters: int = 8,
) -> Optional[float]:
    """
    Point-in-time quality score from historical quarterly financials.

    Combines simplified versions of our production metrics:
      - ROIC (5-year average if available, else shorter)
      - Gross margin stability (coefficient of variation)
      - FCF conversion
      - Debt-to-equity (latest)
      - Piotroski F-Score (current vs prior year)

    Returns 0-100 score, or None if insufficient data.
    """
    if not quarters or len(quarters) < min_quarters:
        return None

    # Use last 20 quarters (5 years) for metrics — or fewer if unavailable
    recent = quarters[-20:] if len(quarters) >= 20 else quarters

    # ── ROIC ──
    roics = []
    for q in recent:
        if q.operating_income is None:
            continue
        invested = (q.total_equity or 0) + (q.long_term_debt or 0)
        if invested > 0:
            roics.append(q.operating_income / invested)
    if len(roics) < 4:
        return None
    roic_mean = float(pd.Series(roics).mean())
    roic_score = _score_linear(roic_mean, good=0.05, great=0.20)

    # ── Gross margin stability ──
    gms = []
    for q in recent:
        if q.revenue and q.gross_profit and q.revenue > 0:
            gms.append(q.gross_profit / q.revenue)
    if len(gms) >= 4:
        gm_series = pd.Series(gms)
        gm_mean = float(gm_series.mean())
        gm_cov = float(gm_series.std() / gm_mean) if gm_mean > 0 else 0.5
        margin_score = _score_linear(gm_cov, good=0.15, great=0.05, higher_better=False)
    else:
        margin_score = 50.0

    # ── FCF conversion ──
    fcfs = []
    for q in recent:
        if q.operating_cash_flow is not None and q.capex is not None and q.net_income:
            fcf = q.operating_cash_flow + q.capex
            if q.net_income > 0:
                fcfs.append(fcf / q.net_income)
    fcf_conv = float(pd.Series(fcfs).mean()) if fcfs else 0.5
    fcf_score = _score_linear(fcf_conv, good=0.6, great=1.2)

    # ── Debt-to-equity (latest) ──
    latest = recent[-1]
    if latest.total_equity and latest.total_equity > 0:
        d2e = (latest.long_term_debt or 0) / latest.total_equity
        debt_score = _score_linear(d2e, good=1.5, great=0.3, higher_better=False)
    else:
        debt_score = 50.0

    # ── Piotroski (simplified, last 8 quarters = curr yr vs prior yr) ──
    if len(recent) >= 8:
        curr = recent[-4:]
        prior = recent[-8:-4]
        f_score = _piotroski_simple(curr, prior)
    else:
        f_score = None
    piotroski_score = (f_score / 9.0 * 100) if f_score is not None else 50.0

    # Weighted composite
    composite = (
        0.30 * roic_score +
        0.20 * margin_score +
        0.20 * fcf_score +
        0.15 * debt_score +
        0.15 * piotroski_score
    )
    return float(max(0, min(100, composite)))


def _score_linear(value: float, good: float, great: float, higher_better: bool = True) -> float:
    if value is None:
        return 50.0
    if higher_better:
        if value >= great: return 100.0
        if value <= good - (great - good): return 0.0
        frac = (value - good) / (great - good) if great != good else 0.0
        return max(0, min(100, 50 + 50 * frac))
    else:
        if value <= great: return 100.0
        if value >= good + (good - great): return 0.0
        frac = (good - value) / (good - great) if good != great else 0.0
        return max(0, min(100, 50 + 50 * frac))


def _piotroski_simple(curr: List[HistoricalQuarter], prior: List[HistoricalQuarter]) -> Optional[int]:
    """Simplified 6-point Piotroski from TTM figures."""
    def _ttm_sum(qs, field: str) -> Optional[float]:
        vals = [getattr(q, field) for q in qs if getattr(q, field) is not None]
        return sum(vals) if len(vals) == len(qs) else None

    ni_c = _ttm_sum(curr, "net_income")
    ni_p = _ttm_sum(prior, "net_income")
    ocf_c = _ttm_sum(curr, "operating_cash_flow")
    assets_c = curr[-1].total_assets if curr else None
    assets_p = prior[-1].total_assets if prior else None
    ltd_c = curr[-1].long_term_debt if curr else None
    ltd_p = prior[-1].long_term_debt if prior else None
    rev_c = _ttm_sum(curr, "revenue")
    rev_p = _ttm_sum(prior, "revenue")
    gp_c = _ttm_sum(curr, "gross_profit")
    gp_p = _ttm_sum(prior, "gross_profit")

    score = 0
    if ni_c is not None and ni_c > 0: score += 1
    if ocf_c is not None and ocf_c > 0: score += 1
    if (ni_c is not None and assets_c and ni_p is not None and assets_p
        and (ni_c / assets_c) > (ni_p / assets_p)): score += 1
    if ocf_c is not None and ni_c is not None and ocf_c > ni_c: score += 1
    if ltd_c is not None and ltd_p is not None and ltd_c < ltd_p: score += 1
    if (gp_c and rev_c and gp_p and rev_p and rev_c > 0 and rev_p > 0
        and (gp_c / rev_c) > (gp_p / rev_p)): score += 1
    return score


# ══════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════
async def _test():
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set"); return

    tickers = ["AAPL", "MSFT", "NVDA", "KO", "F", "INTC", "META", "AMZN"]
    cache = FundamentalsCache(api_key=api_key)

    print(f"Populating fundamentals for {len(tickers)} tickers...")
    stats = await cache.populate(tickers)
    print(f"  Cached: {stats['cached']}, Fetched: {stats['fetched']}, Failed: {stats['failed']}")

    print(f"\n{'='*70}")
    print(f"  POINT-IN-TIME QUALITY SCORES")
    print(f"{'='*70}")
    print(f"  {'Ticker':<8} {'2023-01-01':>12} {'2024-01-01':>12} {'2025-01-01':>12} {'Most recent':>12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for t in tickers:
        row = [t]
        for as_of in [date(2023, 1, 1), date(2024, 1, 1), date(2025, 1, 1), date(2026, 4, 1)]:
            qs = cache.get_quarters_as_of(t, as_of)
            score = compute_quality_pit(qs) if qs else None
            row.append(f"{score:.1f}" if score is not None else "N/A")
        print(f"  {row[0]:<8} {row[1]:>12} {row[2]:>12} {row[3]:>12} {row[4]:>12}")


if __name__ == "__main__":
    asyncio.run(_test())
