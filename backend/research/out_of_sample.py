"""
QuantEdge v6.0 — Out-of-Sample Validation
==========================================
The gold-standard test for any systematic strategy.

Splits the 4-year backtest into train (first 2 years) and test (last 2 years).
Compares performance metrics between periods.

Interpretation:
  - Test Sharpe / Train Sharpe > 0.60:  edge likely generalizes
  - Test Sharpe / Train Sharpe 0.30-0.60: weak/uncertain generalization
  - Test Sharpe / Train Sharpe < 0.30:  likely overfit
  - Test Sharpe < 0:                    system fails out-of-sample
"""

import asyncio
import os
import sys
from datetime import date
from typing import Dict
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from research.historical_data import HistoricalDataStore
from research.fundamentals_cache import FundamentalsCache
from research.universe import UniverseBuilder
from research.backtest_engine import BacktestEngine
from research.performance_metrics import generate_report


def pct_str(x: float) -> str:
    return f"{x*100:+.2f}%"


def format_row(label: str, train_val: float, test_val: float, fmt="pct") -> str:
    """Format one row of comparison table."""
    if fmt == "pct":
        tr = pct_str(train_val); te = pct_str(test_val)
    elif fmt == "ratio":
        tr = f"{train_val:+.3f}"; te = f"{test_val:+.3f}"
    elif fmt == "int":
        tr = f"{int(train_val)}"; te = f"{int(test_val)}"
    else:
        tr = f"{train_val:.2f}"; te = f"{test_val:.2f}"
    return f"  {label:<28}{tr:>14}{te:>14}"


async def main():
    api_key = os.getenv('POLYGON_API_KEY')
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

    # CRITICAL: train and test periods are strictly non-overlapping
    train_start, train_end = date(2022, 4, 1), date(2024, 4, 1)
    test_start,  test_end  = date(2024, 4, 1), date(2026, 4, 1)

    print(f"\n{'='*90}")
    print(f"  OUT-OF-SAMPLE VALIDATION")
    print(f"{'='*90}")
    print(f"  TRAIN:  {train_start} → {train_end}  (2 years, ~500 trading days)")
    print(f"  TEST:   {test_start} → {test_end}  (2 years, ~500 trading days)")
    print(f"  NOTE: Model weights fixed; this tests whether signal generalizes in time")
    print(f"{'='*90}")

    results: Dict[str, Dict] = {}

    for horizon in ("long_term", "medium_term", "short_term"):
        print(f"\nRunning {horizon}...")

        train_result = engine.run(
            start=train_start, end=train_end,
            horizon=horizon, top_n=20,
        )
        train_report = generate_report(train_result, strategy_name=f"{horizon}_TRAIN", n_trials_for_deflation=3)

        test_result = engine.run(
            start=test_start, end=test_end,
            horizon=horizon, top_n=20,
        )
        test_report = generate_report(test_result, strategy_name=f"{horizon}_TEST", n_trials_for_deflation=3)

        results[horizon] = {"train": train_report, "test": test_report}

    # Summary table
    print(f"\n{'='*90}")
    print(f"  OUT-OF-SAMPLE COMPARISON")
    print(f"{'='*90}")

    for horizon in ("long_term", "medium_term", "short_term"):
        tr = results[horizon]["train"]
        te = results[horizon]["test"]
        print(f"\n  ── {horizon.upper()} ──")
        print(f"  {'':<28}{'TRAIN':>14}{'TEST':>14}")
        print(f"  {'-'*28}{'-'*14}{'-'*14}")
        print(format_row("Total return",           tr.total_return,             te.total_return,             "pct"))
        print(format_row("Annualized return",      tr.annualized_return,        te.annualized_return,        "pct"))
        print(format_row("Benchmark return (SPY)", tr.benchmark_annualized_return, te.benchmark_annualized_return, "pct"))
        print(format_row("Alpha (annualized)",     tr.alpha_annualized,         te.alpha_annualized,         "pct"))
        print(format_row("Sharpe ratio",           tr.sharpe_ratio,             te.sharpe_ratio,             "ratio"))
        print(format_row("Sortino ratio",          tr.sortino_ratio,            te.sortino_ratio,            "ratio"))
        print(format_row("Beta",                   tr.beta,                     te.beta,                     "ratio"))
        print(format_row("Max drawdown",           tr.max_drawdown,             te.max_drawdown,             "pct"))
        print(format_row("Information Ratio",      tr.information_ratio,        te.information_ratio,        "ratio"))

        # Generalization verdict
        if tr.sharpe_ratio > 0 and te.sharpe_ratio > 0:
            retention = te.sharpe_ratio / tr.sharpe_ratio
        elif te.sharpe_ratio > 0 and tr.sharpe_ratio <= 0:
            retention = 999  # test better than train — unusual but good
        else:
            retention = 0.0

        if retention > 0.80:
            verdict = "✓✓ STRONG — edge holds up cleanly out-of-sample"
        elif retention > 0.60:
            verdict = "✓ MODERATE — edge likely real, some degradation"
        elif retention > 0.30:
            verdict = "? WEAK — partial generalization, more data needed"
        elif retention > 0:
            verdict = "✗ POOR — significant degradation, likely overfit"
        else:
            verdict = "✗✗ FAIL — edge collapses out-of-sample"

        ret_str = f"{retention*100:.0f}%" if retention < 10 else ">>100%"
        print(f"\n  Sharpe retention (test/train): {ret_str}")
        print(f"  Verdict: {verdict}")

    # Overall summary
    print(f"\n{'='*90}")
    print(f"  INTERPRETATION GUIDE")
    print(f"{'='*90}")
    print(f"  Retention >80%: edge generalizes well, high confidence")
    print(f"  Retention 60-80%: typical for real edge (some shrinkage expected)")
    print(f"  Retention 30-60%: marginal — would need longer data")
    print(f"  Retention <30%: likely overfit, do not deploy")
    print(f"  Negative test Sharpe: system fails, abandon or redesign")


if __name__ == "__main__":
    asyncio.run(main())
