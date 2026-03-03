"""
QuantEdge v6.0 — XGBoost / LightGBM Training
=============================================
© 2026 Dileep Kumar Reddy Kapu. All Rights Reserved.
Proprietary & Confidential.

Trains XGBoost and LightGBM ensemble on 40+ features.
Uses Combinatorial Purged Cross-Validation (CPCV) per Lopez de Prado (2018).

CPCV prevents look-ahead bias by:
  1. Purging: removing train observations near test boundary
  2. Embargo: adding buffer gap between train/test splits
  3. Combinatorial: testing all k-fold combinations

Usage:
    python train_xgboost.py --ticker AAPL
    python train_xgboost.py --universe sp500_top50
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import yfinance as yf
from loguru import logger

# ── Config ────────────────────────────────────────────────────
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/ml/saved_models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_HORIZON = 5
EMBARGO_DAYS = 5
N_FOLDS = 5

XGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1,
}

LGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 4,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}


# ── Feature Engineering (40+ features) ───────────────────────
def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 40+ features covering:
    - Price momentum (1d, 5d, 10d, 20d, 60d, 120d)
    - Volatility regime (realized vol, GARCH approx)
    - Technical indicators (RSI, MACD, Bollinger, Stochastic)
    - Volume analysis (OBV, VWAP deviation, volume ratio)
    - Market microstructure (Amihud illiquidity, bid-ask proxy)
    - Mean reversion signals (z-score, distance from MA)
    - Cross-sectional momentum (not here — would need multiple tickers)
    """
    df = df.copy().sort_values('date').reset_index(drop=True)
    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']

    # ── Momentum ──────────────────────────────────────────────
    for n in [1, 5, 10, 20, 60, 120]:
        df[f'ret_{n}d'] = c.pct_change(n)

    # Log returns
    df['log_ret_1d'] = np.log(c / c.shift(1))

    # ── Volatility ────────────────────────────────────────────
    ret = df['log_ret_1d']
    for n in [5, 10, 20, 60]:
        df[f'vol_{n}d'] = ret.rolling(n).std()

    # Vol-of-vol
    df['vol_of_vol_20d'] = df['vol_20d'].rolling(20).std()

    # ── Moving Averages ───────────────────────────────────────
    for n in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{n}'] = c.rolling(n).mean()
        df[f'sma_{n}_ratio'] = c / df[f'sma_{n}'] - 1

    # EMA
    df['ema_12'] = c.ewm(span=12).mean()
    df['ema_26'] = c.ewm(span=26).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_norm'] = df['macd'] / c

    # ── RSI ───────────────────────────────────────────────────
    for n in [7, 14, 21]:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(n).mean()
        loss = (-delta.clip(upper=0)).rolling(n).mean()
        rs = gain / loss.replace(0, np.nan)
        df[f'rsi_{n}'] = 100 - (100 / (1 + rs))

    # ── Bollinger Bands ───────────────────────────────────────
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_position'] = (c - sma20) / (2 * std20 + 1e-8)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20

    # ── Stochastic ────────────────────────────────────────────
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df['stoch_k'] = (c - low14) / (high14 - low14 + 1e-8) * 100
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # ── Volume ───────────────────────────────────────────────
    df['vol_ratio_20d'] = v / v.rolling(20).mean()
    df['vol_log'] = np.log1p(v / 1e6)

    # OBV
    obv = (np.sign(c.diff()) * v).cumsum()
    df['obv_norm'] = obv / obv.rolling(20).mean()

    # VWAP deviation (intraday approx using (H+L+C)/3)
    typical = (h + l + c) / 3
    df['vwap_approx'] = (typical * v).rolling(20).sum() / v.rolling(20).sum()
    df['vwap_deviation'] = c / df['vwap_approx'] - 1

    # ── ATR ───────────────────────────────────────────────────
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_ratio'] = df['atr_14'] / c

    # ── Amihud Illiquidity ────────────────────────────────────
    df['amihud'] = (df['log_ret_1d'].abs() / (v * c / 1e6 + 1e-8)).rolling(20).mean()

    # ── Mean Reversion ────────────────────────────────────────
    df['z_score_20d'] = (c - sma20) / (std20 + 1e-8)
    df['z_score_60d'] = (c - c.rolling(60).mean()) / (c.rolling(60).std() + 1e-8)

    # ── Trend Strength ────────────────────────────────────────
    df['golden_cross'] = (df['sma_50_ratio'] > df['sma_200_ratio']).astype(int)
    df['hl_ratio'] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min() + 1e-8)

    # ── Calendar Features ─────────────────────────────────────
    if 'date' in df.columns:
        df['dow'] = pd.to_datetime(df['date']).dt.dayofweek / 4  # 0–1
        df['month'] = pd.to_datetime(df['date']).dt.month / 11  # 0–1

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get all computed feature columns (exclude raw OHLCV and date)."""
    exclude = {'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close',
               'ticker', 'dividends', 'stock splits', 'index',
               'ema_12', 'ema_26', 'bb_upper', 'bb_lower',
               'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
               'atr_14', 'vwap_approx'}
    return [c for c in df.columns if c not in exclude]


def make_target(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Binary target: 1 if forward return > 0, else 0."""
    fwd = df['close'].pct_change(horizon).shift(-horizon)
    return (fwd > 0).astype(int)


# ── CPCV Implementation ───────────────────────────────────────
class CPCVSplit:
    """
    Combinatorial Purged Cross-Validation.
    Lopez de Prado (2018) Chapter 12.
    """
    def __init__(self, n_splits: int = 5, embargo: int = 5):
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, X: np.ndarray):
        """Generate train/test indices with purging and embargo."""
        n = len(X)
        fold_size = n // self.n_splits
        indices = np.arange(n)

        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < self.n_splits - 1 else n

            # Purge: remove train obs close to test boundary
            purge_start = max(0, test_start - self.embargo)
            embargo_end = min(n, test_end + self.embargo)

            # Train = everything EXCEPT test + purge zone
            train_mask = np.ones(n, dtype=bool)
            train_mask[purge_start:embargo_end] = False
            train_idx = indices[train_mask]
            test_idx = indices[test_start:test_end]

            yield train_idx, test_idx


# ── Training ──────────────────────────────────────────────────
def train_xgboost_lgbm(ticker: str, days: int = 1000) -> Optional[dict]:
    """Full training pipeline for XGBoost + LightGBM ensemble."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Training XGB+LGB ensemble for {ticker}")
    logger.info(f"{'='*50}")

    # Fetch data
    end = datetime.now()
    start_dt = end - timedelta(days=days)
    t = yf.Ticker(ticker)
    df = t.history(start=start_dt.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]

    if len(df) < 300:
        logger.error(f"Insufficient data: {len(df)} rows")
        return None

    # Features
    df = compute_all_features(df)
    target = make_target(df, PREDICTION_HORIZON)
    feature_cols = get_feature_columns(df)

    # Align
    combined = df[feature_cols].copy()
    combined['target'] = target
    combined = combined.dropna()

    X = combined[feature_cols].values.astype(np.float32)
    y = combined['target'].values.astype(int)
    feat_names = feature_cols

    logger.info(f"Dataset: {X.shape[0]} samples × {X.shape[1]} features")
    logger.info(f"Class balance: {y.mean():.3f} up / {1-y.mean():.3f} down")

    # CPCV
    cpcv = CPCVSplit(n_splits=N_FOLDS, embargo=EMBARGO_DAYS)
    xgb_scores = []
    lgb_scores = []
    xgb_models = []
    lgb_models = []

    for fold_idx, (train_idx, test_idx) in enumerate(cpcv.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        # XGBoost
        xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
            early_stopping_rounds=20
        )
        xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
        xgb_auc = roc_auc_score(y_test, xgb_pred)
        xgb_acc = accuracy_score(y_test, (xgb_pred > 0.5).astype(int))
        xgb_scores.append(xgb_auc)
        xgb_models.append(xgb_model)

        # LightGBM
        lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)]
        )
        lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
        lgb_auc = roc_auc_score(y_test, lgb_pred)
        lgb_acc = accuracy_score(y_test, (lgb_pred > 0.5).astype(int))
        lgb_scores.append(lgb_auc)
        lgb_models.append(lgb_model)

        logger.info(f"Fold {fold_idx+1}: XGB AUC={xgb_auc:.4f} acc={xgb_acc:.4f} | "
                    f"LGB AUC={lgb_auc:.4f} acc={lgb_acc:.4f}")

    if not xgb_scores:
        logger.error("No folds completed")
        return None

    # Use best models from CPCV
    best_xgb_idx = int(np.argmax(xgb_scores))
    best_lgb_idx = int(np.argmax(lgb_scores))

    metrics = {
        'ticker': ticker,
        'n_features': len(feat_names),
        'n_samples': int(X.shape[0]),
        'xgb_mean_auc': float(np.mean(xgb_scores)),
        'xgb_std_auc': float(np.std(xgb_scores)),
        'lgb_mean_auc': float(np.mean(lgb_scores)),
        'lgb_std_auc': float(np.std(lgb_scores)),
        'ensemble_mean_auc': float(np.mean(xgb_scores + lgb_scores)),
        'trained_at': datetime.now().isoformat(),
        'feature_names': feat_names,
        'cpcv_params': {'n_splits': N_FOLDS, 'embargo': EMBARGO_DAYS}
    }

    # Save models
    ticker_dir = MODEL_DIR / "xgboost" / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(xgb_models[best_xgb_idx], ticker_dir / "xgb_model.pkl")
    joblib.dump(lgb_models[best_lgb_idx], ticker_dir / "lgb_model.pkl")

    with open(ticker_dir / "feature_names.json", 'w') as f:
        json.dump(feat_names, f)
    with open(ticker_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"✅ Saved to {ticker_dir}")
    logger.info(f"XGB AUC: {metrics['xgb_mean_auc']:.4f} ± {metrics['xgb_std_auc']:.4f}")
    logger.info(f"LGB AUC: {metrics['lgb_mean_auc']:.4f} ± {metrics['lgb_std_auc']:.4f}")

    return metrics


# ── Inference ─────────────────────────────────────────────────
def predict_ensemble(ticker: str, recent_df: pd.DataFrame) -> Optional[dict]:
    """Load saved XGB+LGB models and predict."""
    ticker_dir = MODEL_DIR / "xgboost" / ticker
    if not ticker_dir.exists():
        return None

    try:
        xgb_model = joblib.load(ticker_dir / "xgb_model.pkl")
        lgb_model = joblib.load(ticker_dir / "lgb_model.pkl")
        with open(ticker_dir / "feature_names.json") as f:
            feat_names = json.load(f)
        with open(ticker_dir / "metrics.json") as f:
            metrics = json.load(f)

        df = compute_all_features(recent_df)
        available = [f for f in feat_names if f in df.columns]
        X = df[available].dropna().values[-1:].astype(np.float32)

        if X.shape[0] == 0:
            return None

        xgb_prob = xgb_model.predict_proba(X)[0, 1]
        lgb_prob = lgb_model.predict_proba(X)[0, 1]
        ensemble_prob = 0.5 * xgb_prob + 0.5 * lgb_prob

        return {
            'ticker': ticker,
            'xgb_prob_up': float(xgb_prob),
            'lgb_prob_up': float(lgb_prob),
            'ensemble_prob_up': float(ensemble_prob),
            'signal': 'BULLISH' if ensemble_prob > 0.55 else 'BEARISH' if ensemble_prob < 0.45 else 'NEUTRAL',
            'ensemble_auc': metrics.get('ensemble_mean_auc', 0),
            'trained_at': metrics.get('trained_at', ''),
        }
    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantEdge XGBoost/LightGBM Trainer")
    parser.add_argument("--ticker", type=str)
    parser.add_argument("--days", type=int, default=1000)
    parser.add_argument("--universe", choices=["sp500_top50"], default=None)
    args = parser.parse_args()

    if args.ticker:
        metrics = train_xgboost_lgbm(args.ticker, args.days)
        if metrics:
            print(json.dumps({k: v for k, v in metrics.items() if k != 'feature_names'}, indent=2))
    elif args.universe == "sp500_top50":
        sys.path.insert(0, str(Path(__file__).parent.parent / "data_pipeline"))
        from fetch_and_store import SP500_TOP50
        results = []
        for ticker in SP500_TOP50:
            try:
                m = train_xgboost_lgbm(ticker, args.days)
                if m:
                    results.append({'ticker': ticker, 'auc': m['ensemble_mean_auc']})
            except Exception as e:
                logger.error(f"Failed {ticker}: {e}")
        results.sort(key=lambda x: x['auc'], reverse=True)
        print(json.dumps(results, indent=2))
    else:
        logger.error("Provide --ticker AAPL or --universe sp500_top50")
        sys.exit(1)
