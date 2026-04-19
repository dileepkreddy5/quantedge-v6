"""
QuantEdge v6.0 — Backtest Engine
=================================
Walk-forward backtest runner with strict point-in-time discipline.

At each rebalance date T:
  1. Compute factor scores using ONLY data available at T-1 close
  2. Rank universe by composite score for this horizon
  3. Select top_n names
  4. Compute turnover vs current portfolio
  5. Apply transaction costs (10 bps round-trip on turnover)
  6. Hold positions until next rebalance

Tracks daily P&L, drawdowns, turnover, alpha vs SPY benchmark.

CRITICAL: All factor computations use HistoricalDataStore.get_prices(end_date=T)
which enforces the no-look-ahead invariant.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from research.historical_data import HistoricalDataStore
from research.fundamentals_cache import FundamentalsCache, compute_quality_pit


# ══════════════════════════════════════════════════════════════
# FACTOR COMPUTATIONS (point-in-time versions of factor_engine.py)
# ══════════════════════════════════════════════════════════════
# We reimplement the core factor logic here to operate purely on
# cached historical data — no API calls during backtest. This is
# critical for speed (thousands of factor computations per backtest).


def _linear_score(value: float, low: float, high: float) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 50.0
    if high == low:
        return 50.0
    frac = (value - low) / (high - low)
    return float(np.clip(frac * 100, 0, 100))


def compute_momentum_score(df: pd.DataFrame) -> Tuple[float, Dict]:
    """Risk-adjusted price momentum from historical OHLCV."""
    if df is None or len(df) < 252:
        return 50.0, {}
    close = df["close"]
    p_12m = float(close.iloc[-252])
    p_1m = float(close.iloc[-21])
    p_6m = float(close.iloc[-126]) if len(close) >= 126 else float(close.iloc[0])
    p_3m = float(close.iloc[-63])
    p_now = float(close.iloc[-1])
    mom_12_1 = (p_1m / p_12m) - 1 if p_12m > 0 else 0.0
    mom_6m = (p_now / p_6m) - 1 if p_6m > 0 else 0.0
    mom_3m = (p_now / p_3m) - 1 if p_3m > 0 else 0.0
    returns = close.pct_change().dropna()
    vol_3m = float(returns.tail(63).std() * np.sqrt(252))
    sharpe_3m = mom_3m / vol_3m if vol_3m > 0 else 0.0
    score = (
        0.40 * _linear_score(mom_12_1, -0.20, 0.50) +
        0.25 * _linear_score(mom_6m, -0.15, 0.30) +
        0.20 * _linear_score(mom_3m, -0.10, 0.20) +
        0.15 * _linear_score(sharpe_3m, -1.0, 2.0)
    )
    return float(np.clip(score, 0, 100)), {
        "mom_12_1": mom_12_1, "mom_6m": mom_6m, "mom_3m": mom_3m, "sharpe_3m": sharpe_3m,
    }


def compute_accumulation_score(df: pd.DataFrame) -> Tuple[float, Dict]:
    """Volume-based accumulation signal."""
    if df is None or len(df) < 90:
        return 50.0, {}
    close = df["close"]
    volume = df["volume"]
    returns = close.pct_change().fillna(0.0)
    sign = np.sign(returns).replace(0, 0)
    obv = (volume * sign).cumsum()
    obv_recent = obv.tail(63)
    if len(obv_recent) >= 20 and obv_recent.std() > 0:
        x = np.arange(len(obv_recent), dtype=np.float64)
        obv_slope = float(np.polyfit(x, obv_recent.values.astype(np.float64), 1)[0]) / (abs(obv_recent.mean()) + 1e-9)
    else:
        obv_slope = 0.0
    vol_20 = float(volume.tail(20).mean())
    vol_90 = float(volume.tail(90).mean())
    volume_surge = vol_20 / vol_90 if vol_90 > 0 else 1.0
    dollar_vol = (close * volume).tail(60)
    abs_ret = returns.abs().tail(60)
    amihud = float((abs_ret / (dollar_vol + 1)).mean()) * 1e9 if dollar_vol.mean() > 0 else 100.0
    score = (
        0.40 * _linear_score(obv_slope, -0.02, 0.02) +
        0.35 * _linear_score(volume_surge, 0.7, 1.8) +
        0.25 * (100 - _linear_score(amihud, 0.1, 10.0))
    )
    return float(np.clip(score, 0, 100)), {
        "obv_slope": obv_slope, "volume_surge": volume_surge, "amihud": amihud,
    }


def compute_trend_score(df: pd.DataFrame) -> Tuple[float, Dict]:
    """Trend quality score."""
    if df is None or len(df) < 200:
        return 50.0, {}
    close = df["close"]
    returns = close.pct_change().dropna()
    ma50 = float(close.tail(50).mean())
    ma200 = float(close.tail(200).mean())
    price = float(close.iloc[-1])
    pct_ma50 = (price / ma50) - 1 if ma50 > 0 else 0.0
    pct_ma200 = (price / ma200) - 1 if ma200 > 0 else 0.0
    if price > ma50 > ma200:
        alignment = 1.0
    elif price < ma50 < ma200:
        alignment = -1.0
    else:
        alignment = 0.0
    alignment_score = 100.0 if alignment > 0 else 0.0 if alignment < 0 else 50.0
    vol_annual = float(returns.tail(252).std() * np.sqrt(252)) if len(returns) >= 252 else 0.0
    ret_12m = float((close.iloc[-1] / close.iloc[-252]) - 1) if len(close) >= 252 else 0.0
    vol_adj = ret_12m / vol_annual if vol_annual > 0 else 0.0
    score = (
        0.30 * _linear_score(pct_ma50, -0.10, 0.15) +
        0.30 * _linear_score(pct_ma200, -0.20, 0.30) +
        0.25 * alignment_score +
        0.15 * _linear_score(vol_adj, -0.5, 2.0)
    )
    return float(np.clip(score, 0, 100)), {
        "pct_ma50": pct_ma50, "pct_ma200": pct_ma200, "alignment": alignment, "vol_adj": vol_adj,
    }


def compute_quality_proxy(df: pd.DataFrame) -> float:
    """
    Price-based quality proxy for backtest speed.

    In production, quality uses 10yr SEC financials (slow). For the backtest
    we use a proxy: vol-adjusted Sharpe + trend consistency over 2 years.
    Not as good as true fundamentals-based quality, but computable from
    cached price data in milliseconds.
    """
    if df is None or len(df) < 500:
        return 50.0
    returns = df["close"].pct_change().dropna()
    # 2-year annualized Sharpe (return / vol)
    r_2y = returns.tail(504)
    if len(r_2y) < 252:
        return 50.0
    ann_return = float(r_2y.mean() * 252)
    ann_vol = float(r_2y.std() * np.sqrt(252))
    sharpe_2y = ann_return / ann_vol if ann_vol > 0 else 0.0
    # Drawdown from 2yr peak
    close_2y = df["close"].tail(504)
    rolling_max = close_2y.expanding().max()
    drawdown = (close_2y / rolling_max - 1).min()
    # Higher Sharpe + shallower drawdown = higher quality
    sharpe_score = _linear_score(sharpe_2y, -0.5, 1.5)
    dd_score = _linear_score(abs(float(drawdown)), 0.50, 0.10)  # Lower drawdown = better
    return 0.6 * sharpe_score + 0.4 * dd_score


# ══════════════════════════════════════════════════════════════
# HORIZON WEIGHTS (same as production)
# ══════════════════════════════════════════════════════════════
HORIZON_WEIGHTS = {
    "short_term":  {"momentum": 0.40, "accumulation": 0.30, "trend": 0.20, "quality": 0.10},
    "medium_term": {"quality":  0.35, "momentum": 0.25, "trend": 0.20, "accumulation": 0.20},
    "long_term":   {"quality":  0.55, "trend": 0.20, "momentum": 0.15, "accumulation": 0.10},
}


def compute_composite_score(scores: Dict[str, float], horizon: str) -> float:
    w = HORIZON_WEIGHTS[horizon]
    return (
        w["quality"] * scores["quality"] +
        w["momentum"] * scores["momentum"] +
        w["accumulation"] * scores["accumulation"] +
        w["trend"] * scores["trend"]
    )


# ══════════════════════════════════════════════════════════════
# BACKTEST STRUCTURES
# ══════════════════════════════════════════════════════════════
@dataclass
class RebalanceRecord:
    rebalance_date: date
    portfolio_tickers: List[str]
    weights: Dict[str, float]
    scores: Dict[str, float]
    entries: List[str]
    exits: List[str]
    turnover: float
    nav_before: float
    nav_after: float
    transaction_cost: float


@dataclass
class BacktestResult:
    horizon: str
    start_date: date
    end_date: date
    initial_capital: float
    final_nav: float
    total_return: float
    daily_nav: pd.Series
    daily_returns: pd.Series
    rebalance_records: List[RebalanceRecord]
    benchmark_daily_nav: pd.Series
    benchmark_total_return: float
    config: Dict


# ══════════════════════════════════════════════════════════════
# BACKTEST RUNNER
# ══════════════════════════════════════════════════════════════
def _rebalance_dates(
    start: date, end: date, frequency: str
) -> List[date]:
    """
    Generate rebalance dates. `frequency`:
      'monthly'   — first trading day of each month
      'quarterly' — first trading day of Jan/Apr/Jul/Oct
    """
    dates = []
    current = date(start.year, start.month, 1)
    while current <= end:
        if frequency == "monthly":
            dates.append(current)
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)
        elif frequency == "quarterly":
            if current.month in (1, 4, 7, 10):
                dates.append(current)
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)
        else:
            raise ValueError(f"Unknown frequency: {frequency}")
    return [d for d in dates if d <= end]


def _next_trading_day(store: HistoricalDataStore, ref: date, benchmark_ticker: str = "SPY") -> Optional[date]:
    """Find the next trading day on/after `ref` using benchmark's trading calendar."""
    df = store.get_prices(benchmark_ticker)
    if df is None:
        return None
    future = df[df.index.date >= ref]
    if len(future) == 0:
        return None
    return future.index[0].date()


class BacktestEngine:
    """
    Walk-forward backtest for one horizon.

    Usage:
      engine = BacktestEngine(store, universe)
      result = engine.run(
          start=date(2022, 1, 1),
          end=date(2026, 4, 1),
          horizon='long_term',
          top_n=20,
      )
    """

    def __init__(
        self,
        store: HistoricalDataStore,
        universe: List[str],
        fundamentals: Optional[FundamentalsCache] = None,
        benchmark: str = "SPY",
        transaction_cost_bps: float = 10.0,
    ):
        self.store = store
        self.universe = universe
        self.fundamentals = fundamentals
        self.benchmark = benchmark
        self.tc_bps = transaction_cost_bps

    def _compute_all_scores(self, as_of: date, horizon: str) -> Dict[str, Dict]:
        """Score entire universe at a single point in time. Returns {ticker: {...}}."""
        results: Dict[str, Dict] = {}
        for ticker in self.universe:
            df = self.store.get_prices(ticker, end_date=as_of)
            if df is None or len(df) < 252:
                continue

            # Price-based factors
            mom_score, _ = compute_momentum_score(df)
            acc_score, _ = compute_accumulation_score(df)
            trend_score, _ = compute_trend_score(df)

            # REAL point-in-time quality from fundamentals (if available)
            if self.fundamentals is not None:
                qs = self.fundamentals.get_quarters_as_of(ticker, as_of)
                pit_quality = compute_quality_pit(qs) if qs else None
                qual_score = pit_quality if pit_quality is not None else 50.0
            else:
                # Fallback to price proxy only if no fundamentals cache provided
                qual_score = compute_quality_proxy(df)

            scores = {
                "quality": qual_score,
                "momentum": mom_score,
                "accumulation": acc_score,
                "trend": trend_score,
            }
            composite = compute_composite_score(scores, horizon)
            scores["composite"] = composite
            results[ticker] = scores
        return results

    def run(
        self,
        start: date,
        end: date,
        horizon: str,
        top_n: int = 20,
        rebalance_frequency: Optional[str] = None,
        initial_capital: float = 100_000.0,
    ) -> BacktestResult:
        """Run the walk-forward backtest."""
        if horizon not in HORIZON_WEIGHTS:
            raise ValueError(f"Unknown horizon: {horizon}")

        # Default rebalance frequency per horizon
        if rebalance_frequency is None:
            rebalance_frequency = "quarterly" if horizon == "long_term" else "monthly"

        logger.info(
            f"Running backtest: {horizon}, {start} → {end}, "
            f"top_{top_n}, {rebalance_frequency} rebalance"
        )

        # Build rebalance calendar
        rebalance_dates = _rebalance_dates(start, end, rebalance_frequency)
        rebalance_dates = [
            _next_trading_day(self.store, d, self.benchmark) or d
            for d in rebalance_dates
        ]
        rebalance_dates = [d for d in rebalance_dates if d is not None and d >= start and d <= end]

        # Get benchmark price series for alignment
        bench_df = self.store.get_prices(self.benchmark, start_date=start, end_date=end)
        if bench_df is None:
            raise RuntimeError(f"No benchmark data for {self.benchmark}")
        all_trading_days = bench_df.index.date.tolist()

        # Track NAV day by day
        nav = initial_capital
        nav_series: Dict[date, float] = {}
        portfolio: Dict[str, float] = {}  # ticker -> shares held
        rebalance_records: List[RebalanceRecord] = []

        # Pre-cache: load all universe price frames to avoid re-loading
        # (big perf win — 507 tickers × N rebalances × 1 load otherwise)
        for t in self.universe:
            _ = self.store.get_prices(t)

        last_rebal_idx = 0

        for day_idx, day in enumerate(all_trading_days):
            # Check if today is a rebalance day
            is_rebalance = day in rebalance_dates

            # Mark-to-market: compute current NAV from shares
            if day_idx > 0 and portfolio:
                total_equity = 0.0
                for t, shares in portfolio.items():
                    price = self.store.get_price_at(t, day)
                    if price is not None:
                        total_equity += shares * price
                nav = total_equity

            if is_rebalance:
                nav_before = nav

                # Score the universe as of yesterday's close
                as_of = all_trading_days[day_idx - 1] if day_idx > 0 else day
                all_scores = self._compute_all_scores(as_of, horizon)

                if not all_scores:
                    nav_series[day] = nav
                    continue

                # Rank + pick top_n
                ranked = sorted(all_scores.items(), key=lambda kv: kv[1]["composite"], reverse=True)
                selected = [t for t, _ in ranked[:top_n]]

                # Equal-weight allocation
                target_weight = 1.0 / len(selected) if selected else 0.0
                target_dollars = {t: nav * target_weight for t in selected}

                # Compute actual shares to hold at today's open
                new_portfolio: Dict[str, float] = {}
                for t in selected:
                    px = self.store.get_price_at(t, day, column="open")
                    if px is None or px <= 0:
                        continue
                    new_portfolio[t] = target_dollars[t] / px

                # Compute turnover (fraction of portfolio that changed)
                # Turnover = sum of absolute dollar changes / 2 / nav
                old_dollars = {}
                for t, shares in portfolio.items():
                    px = self.store.get_price_at(t, day, column="open")
                    if px is not None:
                        old_dollars[t] = shares * px
                new_dollars = {}
                for t, shares in new_portfolio.items():
                    px = self.store.get_price_at(t, day, column="open")
                    if px is not None:
                        new_dollars[t] = shares * px

                all_names = set(old_dollars) | set(new_dollars)
                total_change = 0.0
                for t in all_names:
                    total_change += abs(new_dollars.get(t, 0) - old_dollars.get(t, 0))
                turnover = total_change / (2 * nav) if nav > 0 else 0.0

                # Transaction cost: tc_bps on notional turned over
                tc = total_change * (self.tc_bps / 10000.0)

                # Deduct transaction costs from NAV
                nav -= tc
                # Rescale shares slightly to reflect actual available capital
                if nav_before > 0:
                    for t in new_portfolio:
                        new_portfolio[t] *= (nav / nav_before)

                entries = [t for t in new_portfolio if t not in portfolio]
                exits = [t for t in portfolio if t not in new_portfolio]

                portfolio = new_portfolio

                rebalance_records.append(RebalanceRecord(
                    rebalance_date=day,
                    portfolio_tickers=list(new_portfolio.keys()),
                    weights={t: (target_weight) for t in new_portfolio},
                    scores={t: all_scores[t]["composite"] for t in new_portfolio if t in all_scores},
                    entries=entries,
                    exits=exits,
                    turnover=turnover,
                    nav_before=nav_before,
                    nav_after=nav,
                    transaction_cost=tc,
                ))

            nav_series[day] = nav
            last_rebal_idx = day_idx

        # Build output
        nav_index = pd.DatetimeIndex([pd.Timestamp(d) for d in nav_series.keys()])
        nav_pd = pd.Series(list(nav_series.values()), index=nav_index, name="nav").sort_index()
        daily_returns = nav_pd.pct_change().dropna()

        # Benchmark NAV: hold SPY from start with same initial capital
        bench_start_price = self.store.get_price_at(self.benchmark, start)
        bench_nav = (bench_df["close"] / bench_start_price) * initial_capital
        bench_nav.name = "benchmark_nav"

        total_return = (nav_pd.iloc[-1] / initial_capital) - 1 if len(nav_pd) > 0 else 0.0
        bench_total_return = (bench_nav.iloc[-1] / initial_capital) - 1 if len(bench_nav) > 0 else 0.0

        return BacktestResult(
            horizon=horizon,
            start_date=start,
            end_date=end,
            initial_capital=initial_capital,
            final_nav=float(nav_pd.iloc[-1]) if len(nav_pd) > 0 else initial_capital,
            total_return=float(total_return),
            daily_nav=nav_pd,
            daily_returns=daily_returns,
            rebalance_records=rebalance_records,
            benchmark_daily_nav=bench_nav,
            benchmark_total_return=float(bench_total_return),
            config={
                "horizon": horizon,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "initial_capital": initial_capital,
                "top_n": top_n,
                "rebalance_frequency": rebalance_frequency,
                "transaction_cost_bps": self.tc_bps,
                "benchmark": self.benchmark,
                "universe_size": len(self.universe),
            },
        )


# ══════════════════════════════════════════════════════════════
# STANDALONE QUICK TEST — single horizon
# ══════════════════════════════════════════════════════════════
async def _test():
    import os
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set"); return

    from research.universe import UniverseBuilder

    store = HistoricalDataStore(api_key=api_key)
    cached = store.available_tickers()
    print(f"Price tickers cached: {len(cached)}")
    if len(cached) < 100:
        print("Run: python -m research.historical_data full")
        return

    fundamentals = FundamentalsCache(api_key=api_key)
    fund_cached = fundamentals.available_tickers()
    print(f"Fundamentals tickers cached: {len(fund_cached)}")
    if len(fund_cached) < 100:
        print("Run fundamentals populate first!")
        return

    builder = UniverseBuilder(api_key=api_key)
    universe_entries = await builder.build()
    universe = [u.ticker for u in universe_entries if u.ticker in cached and u.ticker in fund_cached]
    print(f"Universe available for backtest: {len(universe)}")

    engine = BacktestEngine(
        store=store,
        universe=universe,
        fundamentals=fundamentals,
        benchmark="SPY",
    )

    # Run long-term backtest from April 2022 (need 1yr data buffer before that)
    print("\nRunning LONG_TERM backtest (quarterly rebalance, top 20)...")
    result = engine.run(
        start=date(2022, 4, 1),
        end=date(2026, 4, 1),
        horizon="long_term",
        top_n=20,
        initial_capital=100_000.0,
    )

    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULT — {result.horizon.upper()}")
    print(f"{'='*70}")
    print(f"  Period:              {result.start_date} → {result.end_date}")
    print(f"  Rebalances:          {len(result.rebalance_records)}")
    print(f"  Initial capital:     ${result.initial_capital:,.0f}")
    print(f"  Final NAV:           ${result.final_nav:,.0f}")
    print(f"  Total return:        {result.total_return*100:+.2f}%")
    print(f"  Benchmark (SPY):     {result.benchmark_total_return*100:+.2f}%")
    print(f"  Alpha vs SPY:        {(result.total_return - result.benchmark_total_return)*100:+.2f}%")

    if result.rebalance_records:
        last = result.rebalance_records[-1]
        print(f"\n  Most recent rebalance:")
        print(f"    Date:     {last.rebalance_date}")
        print(f"    Holdings: {', '.join(last.portfolio_tickers[:10])}...")
        print(f"    Turnover: {last.turnover*100:.1f}%")


if __name__ == "__main__":
    asyncio.run(_test())
