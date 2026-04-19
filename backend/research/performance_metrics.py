"""
QuantEdge v6.0 — Performance Metrics
=====================================
Institutional-grade performance analytics for backtest results.

Metrics computed:
  - Annualized return & volatility (geometric)
  - Sharpe ratio (excess return / vol)
  - Sortino ratio (excess return / downside vol)
  - Calmar ratio (annual return / max drawdown)
  - Maximum drawdown + duration
  - Alpha & Beta vs benchmark (CAPM regression)
  - Information ratio (alpha / tracking error)
  - Win rate, profit factor
  - Turnover
  - Deflated Sharpe Ratio (López de Prado 2014) — corrects for multiple testing

All return series assume daily frequency, 252 trading days/year.
"""

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.04  # 4% annualized (approx current T-bill)


@dataclass
class PerformanceReport:
    """Complete performance breakdown for one strategy."""
    # Identification
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_nav: float

    # Returns
    total_return: float
    annualized_return: float
    annualized_volatility: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdowns
    max_drawdown: float
    max_drawdown_duration_days: int
    time_in_drawdown_pct: float

    # vs Benchmark
    benchmark_total_return: float
    benchmark_annualized_return: float
    alpha_annualized: float             # CAPM alpha, annualized
    beta: float
    r_squared: float
    information_ratio: float
    tracking_error: float

    # Win/loss stats
    win_rate: float                      # % of days with positive return
    profit_factor: float                 # gross wins / gross losses
    avg_win: float                       # mean positive daily return
    avg_loss: float                      # mean negative daily return
    largest_win: float
    largest_loss: float

    # Trading stats
    n_rebalances: int
    avg_turnover: float
    total_transaction_cost_pct: float    # cumulative cost as % of initial capital

    # Statistical rigor
    deflated_sharpe: float               # Lopez de Prado (2014) — corrects for selection
    sharpe_significance: str             # narrative interpretation

    # Raw return stats (for deeper inspection)
    skewness: float
    kurtosis: float
    n_observations: int


# ══════════════════════════════════════════════════════════════
# CORE METRIC COMPUTATIONS
# ══════════════════════════════════════════════════════════════
def compute_annualized_return(returns: pd.Series) -> float:
    """Geometric annualized return — the correct way (not arithmetic mean)."""
    if len(returns) == 0:
        return 0.0
    cumulative = float((1 + returns).prod())
    years = len(returns) / TRADING_DAYS_PER_YEAR
    if years <= 0 or cumulative <= 0:
        return 0.0
    return cumulative ** (1.0 / years) - 1.0


def compute_annualized_volatility(returns: pd.Series) -> float:
    """Annualized standard deviation."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def compute_sharpe(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Annualized Sharpe ratio."""
    ann_ret = compute_annualized_return(returns)
    ann_vol = compute_annualized_volatility(returns)
    if ann_vol <= 0:
        return 0.0
    return (ann_ret - rf) / ann_vol


def compute_sortino(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Sortino uses downside deviation instead of total vol."""
    if len(returns) < 2:
        return 0.0
    ann_ret = compute_annualized_return(returns)
    daily_rf = rf / TRADING_DAYS_PER_YEAR
    downside = returns[returns < daily_rf] - daily_rf
    if len(downside) == 0:
        return 0.0
    downside_dev = float(downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    if downside_dev <= 0:
        return 0.0
    return (ann_ret - rf) / downside_dev


def compute_drawdown_stats(nav: pd.Series) -> Dict:
    """Max drawdown magnitude, duration, and time-in-drawdown."""
    if len(nav) < 2:
        return {"max_dd": 0.0, "max_dd_duration": 0, "time_in_dd_pct": 0.0}

    running_max = nav.expanding().max()
    dd_series = (nav / running_max) - 1.0
    max_dd = float(dd_series.min())

    # Duration of max drawdown: days from peak until recovery (or end)
    max_dd_end_idx = dd_series.idxmin()
    peak_before = nav[:max_dd_end_idx].idxmax()

    # Find recovery date (first date AFTER trough where nav >= peak)
    after_trough = nav[max_dd_end_idx:]
    peak_value = nav.loc[peak_before]
    recovered = after_trough[after_trough >= peak_value]
    if len(recovered) > 0:
        recovery_date = recovered.index[0]
        duration = (recovery_date - peak_before).days
    else:
        duration = (nav.index[-1] - peak_before).days  # still in drawdown at end

    # Time in drawdown: % of days where dd < 0
    time_in_dd = float((dd_series < -0.001).sum() / len(dd_series))

    return {
        "max_dd": max_dd,
        "max_dd_duration": int(duration),
        "time_in_dd_pct": time_in_dd,
    }


def compute_alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: float = RISK_FREE_RATE,
) -> Dict:
    """CAPM regression: R_strategy - Rf = alpha + beta*(R_bench - Rf)"""
    # Normalize both indices to plain DatetimeIndex for safe join
    s = strategy_returns.copy()
    b = benchmark_returns.copy()
    s.index = pd.to_datetime(s.index).normalize()
    b.index = pd.to_datetime(b.index).normalize()
    s.name = "strat"
    b.name = "bench"

    aligned = pd.concat([s, b], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return {"alpha": 0.0, "beta": 1.0, "r_squared": 0.0}

    daily_rf = rf / TRADING_DAYS_PER_YEAR
    excess_strat = aligned["strat"].values - daily_rf
    excess_bench = aligned["bench"].values - daily_rf

    slope, intercept, r_val, _, _ = stats.linregress(excess_bench, excess_strat)
    alpha_annualized = float(intercept) * TRADING_DAYS_PER_YEAR
    beta = float(slope)
    r_squared = float(r_val ** 2)

    return {"alpha": alpha_annualized, "beta": beta, "r_squared": r_squared}


def compute_information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Dict:
    """IR = annualized mean(excess return) / annualized std(excess return)"""
    s = strategy_returns.copy()
    b = benchmark_returns.copy()
    s.index = pd.to_datetime(s.index).normalize()
    b.index = pd.to_datetime(b.index).normalize()
    s.name = "strat"; b.name = "bench"

    aligned = pd.concat([s, b], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return {"ir": 0.0, "tracking_error": 0.0}

    excess = aligned["strat"].values - aligned["bench"].values
    mean_excess = float(np.mean(excess) * TRADING_DAYS_PER_YEAR)
    tracking_error = float(np.std(excess, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    ir = mean_excess / tracking_error if tracking_error > 0 else 0.0
    return {"ir": ir, "tracking_error": tracking_error}


def compute_win_loss_stats(returns: pd.Series) -> Dict:
    """Granular win/loss statistics."""
    if len(returns) == 0:
        return {
            "win_rate": 0.0, "profit_factor": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0,
            "largest_win": 0.0, "largest_loss": 0.0,
        }

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    gross_wins = float(wins.sum())
    gross_losses = float(abs(losses.sum()))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else (float("inf") if gross_wins > 0 else 0.0)

    return {
        "win_rate":     float(len(wins) / len(returns)) if len(returns) > 0 else 0.0,
        "profit_factor": profit_factor if profit_factor != float("inf") else 999.0,
        "avg_win":      float(wins.mean()) if len(wins) > 0 else 0.0,
        "avg_loss":     float(losses.mean()) if len(losses) > 0 else 0.0,
        "largest_win":  float(returns.max()),
        "largest_loss": float(returns.min()),
    }


def compute_deflated_sharpe(
    returns: pd.Series,
    n_trials: int = 1,
    rf: float = RISK_FREE_RATE,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    Corrects observed Sharpe for:
      1. Sample size (more data = less uncertainty)
      2. Number of strategies tested (n_trials) — multiple testing bias
      3. Skewness and kurtosis of returns

    Returns a probability (0-1) that the true Sharpe is > 0.
    Above 0.95 = high confidence the edge is real.
    Below 0.5  = edge is likely luck.

    Reference: Bailey & López de Prado (2014)
    "The Deflated Sharpe Ratio: Correcting for Selection Bias"
    """
    if len(returns) < 30:
        return 0.5

    # Canonical Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.
    # All Sharpe quantities in daily units. Formula 9 in the paper.

    daily_rf = rf / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    if excess.std() <= 0:
        return 0.5

    sr = float(excess.mean() / excess.std())  # daily Sharpe
    n = len(returns)
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=False))  # NOT excess kurtosis here

    # Expected maximum Sharpe under null hypothesis over n_trials
    # (Bailey & Lopez de Prado 2014, equation 9 in terms of daily SR)
    if n_trials > 1:
        euler = 0.5772156649
        z_n  = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z_ne = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        # E[max SR_null] = Gumbel approximation, in units of 1/sqrt(n-1)
        sr_star_standardized = (1.0 - euler) * z_n + euler * z_ne
        sr_star = sr_star_standardized / np.sqrt(n - 1)
    else:
        sr_star = 0.0

    # Denominator: standard error of Sharpe estimator (Opdyke 2007 / Mertens 2002)
    # sigma_SR = sqrt((1 - skew*SR + (kurt-1)/4 * SR^2) / (n-1))
    variance_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if variance_term <= 0 or n <= 1:
        return 0.5
    sigma_sr = np.sqrt(variance_term / (n - 1))

    # DSR = Phi((SR_observed - SR_star) / sigma_SR)
    z = (sr - sr_star) / sigma_sr
    dsr = float(stats.norm.cdf(z))
    return dsr


def interpret_sharpe(sharpe: float, deflated: float) -> str:
    """Narrative interpretation of Sharpe and deflated significance."""
    if deflated < 0.5:
        return "Very low confidence — likely lucky selection"
    if deflated < 0.8:
        return "Weak evidence — more data needed"
    if deflated < 0.95:
        return "Moderate confidence — signal likely real"
    if sharpe > 2.0:
        return "High confidence AND strong Sharpe — genuinely promising"
    return "High statistical confidence the edge is real"


# ══════════════════════════════════════════════════════════════
# TOP-LEVEL REPORT GENERATOR
# ══════════════════════════════════════════════════════════════
def generate_report(
    backtest_result,
    strategy_name: Optional[str] = None,
    n_trials_for_deflation: int = 3,  # we tested 3 horizons
) -> PerformanceReport:
    """
    Generate full performance report from a BacktestResult.
    """
    if strategy_name is None:
        strategy_name = backtest_result.horizon

    returns = backtest_result.daily_returns
    nav = backtest_result.daily_nav
    bench_nav = backtest_result.benchmark_daily_nav
    bench_returns = bench_nav.pct_change().dropna()

    # Align index types (backtest nav index can mix date/Timestamp)
    returns.index = pd.DatetimeIndex(returns.index)
    bench_returns.index = pd.DatetimeIndex(bench_returns.index)

    # Core return metrics
    total_return = backtest_result.total_return
    ann_return = compute_annualized_return(returns)
    ann_vol = compute_annualized_volatility(returns)

    # Risk-adjusted
    sharpe = compute_sharpe(returns)
    sortino = compute_sortino(returns)

    # Drawdown
    dd_stats = compute_drawdown_stats(nav)
    max_dd = dd_stats["max_dd"]
    calmar = ann_return / abs(max_dd) if max_dd < 0 else 0.0

    # vs Benchmark
    bench_total_ret = backtest_result.benchmark_total_return
    bench_ann_ret = compute_annualized_return(bench_returns)
    ab = compute_alpha_beta(returns, bench_returns)
    ir_stats = compute_information_ratio(returns, bench_returns)

    # Win/loss
    wl = compute_win_loss_stats(returns)

    # Trading stats
    n_rebal = len(backtest_result.rebalance_records)
    avg_turnover = (
        sum(r.turnover for r in backtest_result.rebalance_records) / n_rebal
        if n_rebal > 0 else 0.0
    )
    total_tc = sum(r.transaction_cost for r in backtest_result.rebalance_records)
    tc_pct = total_tc / backtest_result.initial_capital * 100

    # Statistical rigor
    dsr = compute_deflated_sharpe(returns, n_trials=n_trials_for_deflation)
    sharpe_interp = interpret_sharpe(sharpe, dsr)

    # Moments
    skew = float(stats.skew(returns)) if len(returns) >= 10 else 0.0
    kurt = float(stats.kurtosis(returns)) if len(returns) >= 10 else 0.0

    return PerformanceReport(
        strategy_name=strategy_name,
        start_date=str(backtest_result.start_date),
        end_date=str(backtest_result.end_date),
        initial_capital=backtest_result.initial_capital,
        final_nav=backtest_result.final_nav,
        total_return=total_return,
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration_days=dd_stats["max_dd_duration"],
        time_in_drawdown_pct=dd_stats["time_in_dd_pct"],
        benchmark_total_return=bench_total_ret,
        benchmark_annualized_return=bench_ann_ret,
        alpha_annualized=ab["alpha"],
        beta=ab["beta"],
        r_squared=ab["r_squared"],
        information_ratio=ir_stats["ir"],
        tracking_error=ir_stats["tracking_error"],
        win_rate=wl["win_rate"],
        profit_factor=wl["profit_factor"],
        avg_win=wl["avg_win"],
        avg_loss=wl["avg_loss"],
        largest_win=wl["largest_win"],
        largest_loss=wl["largest_loss"],
        n_rebalances=n_rebal,
        avg_turnover=avg_turnover,
        total_transaction_cost_pct=tc_pct,
        deflated_sharpe=dsr,
        sharpe_significance=sharpe_interp,
        skewness=skew,
        kurtosis=kurt,
        n_observations=len(returns),
    )


def print_report(report: PerformanceReport):
    """Pretty-print a PerformanceReport to terminal."""
    r = report
    print(f"\n{'='*80}")
    print(f"  PERFORMANCE REPORT — {r.strategy_name.upper()}")
    print(f"  {r.start_date} → {r.end_date}  ({r.n_observations} trading days)")
    print(f"{'='*80}")

    print(f"\n  RETURNS")
    print(f"    Initial capital:          ${r.initial_capital:,.0f}")
    print(f"    Final NAV:                ${r.final_nav:,.0f}")
    print(f"    Total return:             {r.total_return*100:+.2f}%")
    print(f"    Annualized return:        {r.annualized_return*100:+.2f}%")
    print(f"    Benchmark annualized:     {r.benchmark_annualized_return*100:+.2f}%")

    print(f"\n  RISK-ADJUSTED")
    print(f"    Volatility (annualized):  {r.annualized_volatility*100:.2f}%")
    print(f"    Sharpe ratio:             {r.sharpe_ratio:.3f}")
    print(f"    Sortino ratio:            {r.sortino_ratio:.3f}")
    print(f"    Calmar ratio:             {r.calmar_ratio:.3f}")

    print(f"\n  DRAWDOWNS")
    print(f"    Max drawdown:             {r.max_drawdown*100:.2f}%")
    print(f"    Max DD duration (days):   {r.max_drawdown_duration_days}")
    print(f"    Time in drawdown:         {r.time_in_drawdown_pct*100:.1f}%")

    print(f"\n  vs BENCHMARK (SPY)")
    print(f"    Alpha (annualized):       {r.alpha_annualized*100:+.2f}%")
    print(f"    Beta:                     {r.beta:.3f}")
    print(f"    R-squared:                {r.r_squared:.3f}")
    print(f"    Information Ratio:        {r.information_ratio:.3f}")
    print(f"    Tracking Error:           {r.tracking_error*100:.2f}%")

    print(f"\n  WIN/LOSS DISTRIBUTION")
    print(f"    Win rate (daily):         {r.win_rate*100:.1f}%")
    print(f"    Profit factor:            {r.profit_factor:.2f}")
    print(f"    Avg win:                  {r.avg_win*100:+.3f}%")
    print(f"    Avg loss:                 {r.avg_loss*100:+.3f}%")
    print(f"    Largest win:              {r.largest_win*100:+.2f}%")
    print(f"    Largest loss:             {r.largest_loss*100:+.2f}%")

    print(f"\n  TRADING")
    print(f"    Rebalances:               {r.n_rebalances}")
    print(f"    Avg turnover/rebalance:   {r.avg_turnover*100:.1f}%")
    print(f"    Total transaction costs:  {r.total_transaction_cost_pct:.2f}%")

    print(f"\n  STATISTICAL RIGOR")
    print(f"    Skewness:                 {r.skewness:.3f}  (0 = symmetric, + = right tail)")
    print(f"    Kurtosis (excess):        {r.kurtosis:.3f}  (0 = normal, + = fat tails)")
    print(f"    Deflated Sharpe p-value:  {r.deflated_sharpe:.3f}")
    print(f"    Interpretation:           {r.sharpe_significance}")


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════
async def _test():
    import os, sys
    import asyncio
    sys.path.insert(0, '.')
    from datetime import date
    from research.historical_data import HistoricalDataStore
    from research.fundamentals_cache import FundamentalsCache
    from research.universe import UniverseBuilder
    from research.backtest_engine import BacktestEngine

    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set"); return

    store = HistoricalDataStore(api_key=api_key)
    funds = FundamentalsCache(api_key=api_key)
    builder = UniverseBuilder(api_key=api_key)
    entries = await builder.build()
    price_cache = set(store.available_tickers())
    fund_cache = set(funds.available_tickers())
    universe = [u.ticker for u in entries if u.ticker in price_cache and u.ticker in fund_cache]

    engine = BacktestEngine(store=store, universe=universe, fundamentals=funds)

    for horizon in ("long_term", "medium_term", "short_term"):
        result = engine.run(
            start=date(2022, 4, 1),
            end=date(2026, 4, 1),
            horizon=horizon,
            top_n=20,
        )
        report = generate_report(result, strategy_name=horizon, n_trials_for_deflation=3)
        print_report(report)


if __name__ == "__main__":
    import asyncio
    asyncio.run(_test())
