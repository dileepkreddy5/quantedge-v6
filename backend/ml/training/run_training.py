"""
QuantEdge v6.0 — Training Orchestrator
========================================
© 2026 Dileep Kumar Reddy Kapu. All Rights Reserved.

Runs complete training pipeline:
  1. Fetch and store data (data pipeline)
  2. Train LSTM models (walk-forward)
  3. Train XGBoost + LightGBM (CPCV)
  4. Generate training report

Usage:
    python run_training.py --ticker AAPL
    python run_training.py --universe sp500_top50
    python run_training.py --ticker AAPL --skip-data  (skip data fetch)
    python run_training.py --ticker AAPL --model lstm  (single model)
    python run_training.py --report   (show saved model stats)

Run this weekly to keep models fresh.
Estimated time: ~5 min per ticker (CPU), ~1 min (GPU)
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import time
from loguru import logger

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

TRAINING_DIR = Path(__file__).parent
DATA_PIPELINE_DIR = TRAINING_DIR.parent / "data_pipeline"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/ml/saved_models"))

SP500_TOP50 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "JPM",
    "AVGO", "XOM", "TSLA", "UNH", "V", "JNJ", "PG", "MA", "COST", "HD",
    "MRK", "ABBV", "CVX", "WMT", "BAC", "AMD", "KO", "PEP", "TMO", "ORCL",
    "ADBE", "CSCO", "CRM", "ACN", "MCD", "NFLX", "ABT", "TXN", "DHR", "WFC",
    "AMGN", "INTC", "VZ", "INTU", "QCOM", "IBM", "RTX", "GE", "NOW", "SPGI"
]


def show_report():
    """Show stats for all saved models."""
    print("\n" + "="*70)
    print("QUANTEDGE™ MODEL INVENTORY REPORT")
    print(f"Generated: {datetime.now().isoformat()}")
    print("="*70)

    # LSTM models
    lstm_dir = MODEL_DIR / "lstm"
    if lstm_dir.exists():
        print(f"\n{'LSTM MODELS':─<50}")
        print(f"{'Ticker':<12} {'Mean Acc':>10} {'Std Acc':>10} {'Folds':>8} {'Trained At':>25}")
        for ticker_dir in sorted(lstm_dir.iterdir()):
            metrics_file = ticker_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    m = json.load(f)
                print(f"{m.get('ticker','?'):<12} "
                      f"{m.get('mean_accuracy',0):>10.4f} "
                      f"{m.get('std_accuracy',0):>10.4f} "
                      f"{m.get('n_folds',0):>8} "
                      f"{m.get('trained_at','?')[:19]:>25}")

    # XGBoost models
    xgb_dir = MODEL_DIR / "xgboost"
    if xgb_dir.exists():
        print(f"\n{'XGB+LGB ENSEMBLE MODELS':─<50}")
        print(f"{'Ticker':<12} {'XGB AUC':>10} {'LGB AUC':>10} {'Features':>10} {'Trained At':>25}")
        for ticker_dir in sorted(xgb_dir.iterdir()):
            metrics_file = ticker_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    m = json.load(f)
                print(f"{m.get('ticker','?'):<12} "
                      f"{m.get('xgb_mean_auc',0):>10.4f} "
                      f"{m.get('lgb_mean_auc',0):>10.4f} "
                      f"{m.get('n_features',0):>10} "
                      f"{m.get('trained_at','?')[:19]:>25}")

    print("\n" + "="*70)


def train_single_ticker(ticker: str, days: int = 1000, skip_data: bool = False,
                         model: str = "all") -> dict:
    """Run full pipeline for a single ticker."""
    results = {'ticker': ticker, 'start_time': datetime.now().isoformat()}
    t0 = time.time()

    # Step 1: Data pipeline
    if not skip_data:
        logger.info(f"📡 [{ticker}] Fetching data...")
        try:
            sys.path.insert(0, str(DATA_PIPELINE_DIR))
            from fetch_and_store import fetch_ohlcv, store_ohlcv, fetch_fundamentals, store_fundamentals, get_engine, create_tables
            engine = get_engine()
            create_tables(engine)
            df = fetch_ohlcv(ticker, days)
            if df is not None:
                store_ohlcv(df, engine)
            fund = fetch_fundamentals(ticker)
            if fund:
                store_fundamentals(fund, engine)
            engine.dispose()
            results['data'] = 'ok'
        except Exception as e:
            logger.error(f"Data pipeline failed for {ticker}: {e}")
            results['data'] = f'error: {e}'

    # Step 2: LSTM
    if model in ("all", "lstm"):
        logger.info(f"🧠 [{ticker}] Training LSTM...")
        try:
            sys.path.insert(0, str(TRAINING_DIR))
            from train_lstm import train_ticker as train_lstm_ticker
            lstm_metrics = train_lstm_ticker(ticker, days)
            if lstm_metrics:
                results['lstm'] = {
                    'mean_accuracy': lstm_metrics['mean_accuracy'],
                    'n_folds': lstm_metrics['n_folds'],
                    'status': 'ok'
                }
            else:
                results['lstm'] = {'status': 'failed'}
        except Exception as e:
            logger.error(f"LSTM training failed for {ticker}: {e}")
            results['lstm'] = {'status': f'error: {e}'}

    # Step 3: XGBoost + LightGBM
    if model in ("all", "xgb"):
        logger.info(f"🌳 [{ticker}] Training XGBoost + LightGBM...")
        try:
            from train_xgboost import train_xgboost_lgbm
            xgb_metrics = train_xgboost_lgbm(ticker, days)
            if xgb_metrics:
                results['xgboost'] = {
                    'xgb_auc': xgb_metrics['xgb_mean_auc'],
                    'lgb_auc': xgb_metrics['lgb_mean_auc'],
                    'n_features': xgb_metrics['n_features'],
                    'status': 'ok'
                }
            else:
                results['xgboost'] = {'status': 'failed'}
        except Exception as e:
            logger.error(f"XGB training failed for {ticker}: {e}")
            results['xgboost'] = {'status': f'error: {e}'}

    elapsed = time.time() - t0
    results['elapsed_seconds'] = round(elapsed, 1)
    logger.info(f"✅ [{ticker}] Complete in {elapsed:.1f}s")
    return results


def train_universe(tickers: list, days: int = 1000, skip_data: bool = False,
                   model: str = "all") -> list:
    """Train all tickers in universe."""
    all_results = []
    logger.info(f"🚀 Training {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers):
        logger.info(f"\n[{i+1}/{len(tickers)}] === {ticker} ===")
        try:
            r = train_single_ticker(ticker, days, skip_data, model)
            all_results.append(r)
        except Exception as e:
            logger.error(f"Failed {ticker}: {e}")
            all_results.append({'ticker': ticker, 'status': f'error: {e}'})
        time.sleep(2)  # Rate limiting between tickers

    # Summary
    successful = sum(1 for r in all_results if r.get('lstm', {}).get('status') == 'ok'
                     or r.get('xgboost', {}).get('status') == 'ok')
    logger.info(f"\n🎉 Universe training complete: {successful}/{len(tickers)} successful")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantEdge Training Orchestrator")
    parser.add_argument("--ticker", type=str, help="Single ticker to train")
    parser.add_argument("--universe", choices=["sp500_top50"], help="Train full universe")
    parser.add_argument("--days", type=int, default=1000, help="Days of history")
    parser.add_argument("--skip-data", action="store_true", help="Skip data fetching")
    parser.add_argument("--model", choices=["all", "lstm", "xgb"], default="all")
    parser.add_argument("--report", action="store_true", help="Show model inventory report")
    args = parser.parse_args()

    if args.report:
        show_report()
    elif args.ticker:
        results = train_single_ticker(args.ticker, args.days, args.skip_data, args.model)
        print(json.dumps(results, indent=2, default=str))
    elif args.universe == "sp500_top50":
        results = train_universe(SP500_TOP50, args.days, args.skip_data, args.model)
        # Save report
        report_path = MODEL_DIR / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Report saved to {report_path}")
        print(json.dumps(results, indent=2, default=str))
    else:
        logger.error("Provide --ticker AAPL, --universe sp500_top50, or --report")
        sys.exit(1)
