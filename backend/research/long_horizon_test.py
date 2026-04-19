"""
QuantEdge v6.0 — Long-Horizon Backtest
========================================
Designed for a true long-term investor (5-10+ year horizon).

Differences from standard horizons:
  - Annual rebalance (vs quarterly/monthly)
  - Top 30 names (vs 20) - more diversification
  - Quality weighted 70% (vs 55%) - stronger fundamental tilt
  - Expected: lower turnover cost, lower alpha magnitude, more realistic
"""

import asyncio
import os
import sys
from datetime import date
import numpy as np

sys.path.insert(0, '.')

from research.historical_data import HistoricalDataStore
from research.fundamentals_cache import FundamentalsCache
from research.universe import UniverseBuilder
from research.backtest_engine import BacktestEngine, HORIZON_WEIGHTS
from research.performance_metrics import generate_report, print_report


async def main():
    api_key = os.getenv('POLYGON_API_KEY')
    store = HistoricalDataStore(api_key=api_key)
    funds = FundamentalsCache(api_key=api_key)
    builder = UniverseBuilder(api_key=api_key)
    entries = await builder.build()
    price_cache = set(store.available_tickers())
    fund_cache = set(funds.available_tickers())
    universe = [u.ticker for u in entries if u.ticker in price_cache and u.ticker in fund_cache]

    # Redefine long_term weights for patient investing
    HORIZON_WEIGHTS["patient_quality"] = {
        "quality":      0.70,
        "trend":        0.15,
        "momentum":     0.10,
        "accumulation": 0.05,
    }

    engine = BacktestEngine(store=store, universe=universe, fundamentals=funds)

    print(f"\n{'='*80}")
    print(f"  PATIENT QUALITY STRATEGY — FULL 4-YEAR WINDOW")
    print(f"{'='*80}")
    print(f"  Weights: Quality 70% / Trend 15% / Momentum 10% / Accumulation 5%")
    print(f"  Top 30 names, annual rebalance on April 1")
    print(f"  Transaction costs: 10 bps")
    print(f"{'='*80}")

    # Full-window annual rebalance
    # Rebalance freq 'annual' isn't built-in, so we manually set as quarterly but
    # run only on April-start dates — but our calendar is quarterly. So use
    # quarterly but reduce frequency by running with top_n=30 and quality 70%.
    # For an honest annual test we override the rebalance_frequency.

    # Monkey-patch _rebalance_dates locally for this run
    from research import backtest_engine as bt_module
    original_fn = bt_module._rebalance_dates
    def annual_dates(start, end, frequency):
        dates = []
        current = date(start.year, 4, 1)
        while current <= end:
            if current >= start:
                dates.append(current)
            current = date(current.year + 1, 4, 1)
        return dates
    bt_module._rebalance_dates = annual_dates

    try:
        result = engine.run(
            start=date(2022, 4, 1),
            end=date(2026, 4, 1),
            horizon="patient_quality",
            top_n=30,
        )
        report = generate_report(result, strategy_name="PATIENT_QUALITY_FULL", n_trials_for_deflation=3)
        print_report(report)

        # Show holdings at each annual rebalance
        print(f"\n{'='*80}")
        print(f"  HOLDINGS AT EACH ANNUAL REBALANCE")
        print(f"{'='*80}")
        for rec in result.rebalance_records:
            print(f"\n  {rec.rebalance_date}:  top scores")
            ranked = sorted(rec.scores.items(), key=lambda kv: kv[1], reverse=True)
            for i, (t, s) in enumerate(ranked[:10]):
                print(f"    {i+1:>2}. {t:<6}  composite={s:.1f}")
            print(f"    ... and {len(rec.portfolio_tickers) - 10} more")
            print(f"    Turnover: {rec.turnover*100:.1f}%  Cost: ${rec.transaction_cost:.0f}")

        # Also run split train/test for this configuration
        print(f"\n{'='*80}")
        print(f"  OUT-OF-SAMPLE SPLIT")
        print(f"{'='*80}")
        train = engine.run(
            start=date(2022, 4, 1), end=date(2024, 4, 1),
            horizon="patient_quality", top_n=30,
        )
        test = engine.run(
            start=date(2024, 4, 1), end=date(2026, 4, 1),
            horizon="patient_quality", top_n=30,
        )
        tr = generate_report(train, strategy_name="TRAIN", n_trials_for_deflation=3)
        te = generate_report(test, strategy_name="TEST", n_trials_for_deflation=3)

        def fmt_pct(x): return f"{x*100:+.2f}%"
        def fmt_ratio(x): return f"{x:.3f}"

        print(f"\n  {'Metric':<28}{'TRAIN':>14}{'TEST':>14}")
        print(f"  {'-'*28}{'-'*14}{'-'*14}")
        print(f"  {'Annualized return':<28}{fmt_pct(tr.annualized_return):>14}{fmt_pct(te.annualized_return):>14}")
        print(f"  {'SPY annualized':<28}{fmt_pct(tr.benchmark_annualized_return):>14}{fmt_pct(te.benchmark_annualized_return):>14}")
        print(f"  {'Alpha annualized':<28}{fmt_pct(tr.alpha_annualized):>14}{fmt_pct(te.alpha_annualized):>14}")
        print(f"  {'Sharpe ratio':<28}{fmt_ratio(tr.sharpe_ratio):>14}{fmt_ratio(te.sharpe_ratio):>14}")
        print(f"  {'Beta':<28}{fmt_ratio(tr.beta):>14}{fmt_ratio(te.beta):>14}")
        print(f"  {'Max drawdown':<28}{fmt_pct(tr.max_drawdown):>14}{fmt_pct(te.max_drawdown):>14}")
        print(f"  {'Information ratio':<28}{fmt_ratio(tr.information_ratio):>14}{fmt_ratio(te.information_ratio):>14}")

        retention = te.sharpe_ratio / tr.sharpe_ratio if tr.sharpe_ratio > 0 else 0
        print(f"\n  Sharpe retention (test/train): {retention*100:.0f}%")

    finally:
        bt_module._rebalance_dates = original_fn


if __name__ == "__main__":
    asyncio.run(main())
