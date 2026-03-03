"""
QuantEdge v6.0 — LSTM Model Training
=====================================
© 2026 Dileep Kumar Reddy Kapu. All Rights Reserved.
Proprietary & Confidential.

Trains LSTM model on price sequences using walk-forward validation.
Saves trained model to disk. Loaded by price oracle at inference time.

Walk-forward validation (Lopez de Prado Ch. 7):
  - Training window: 252 days (1 year)
  - Test window: 63 days (1 quarter)
  - Step: 21 days (1 month)
  - Embargo: 5 days (prevent leakage at train/test boundary)

Usage:
    python train_lstm.py --ticker AAPL --days 1000
    python train_lstm.py --universe sp500_top50 --days 1000
    python train_lstm.py --ticker AAPL --evaluate  # evaluate only
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
from loguru import logger

# ── Config ────────────────────────────────────────────────────
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/ml/saved_models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 60       # 60 days input window
PREDICTION_HORIZON = 5     # Predict 5-day return
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOP_PATIENCE = 10

# Walk-forward params
TRAIN_WINDOW = 252         # 1 year
TEST_WINDOW = 63           # 1 quarter
STEP = 21                  # 1 month
EMBARGO = 5                # 5-day gap between train/test


# ── Feature Engineering ───────────────────────────────────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 20+ features from OHLCV data.
    These are the input features for the LSTM.
    """
    df = df.copy()
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']

    # Returns (core signal)
    df['ret_1d'] = close.pct_change()
    df['ret_5d'] = close.pct_change(5)
    df['ret_20d'] = close.pct_change(20)
    df['ret_60d'] = close.pct_change(60)

    # Volatility
    df['vol_10d'] = df['ret_1d'].rolling(10).std()
    df['vol_20d'] = df['ret_1d'].rolling(20).std()
    df['vol_60d'] = df['ret_1d'].rolling(60).std()

    # Moving Averages (normalized by close)
    df['sma_20_ratio'] = close.rolling(20).mean() / close - 1
    df['sma_50_ratio'] = close.rolling(50).mean() / close - 1
    df['sma_200_ratio'] = close.rolling(200).mean() / close - 1

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_14'] = df['rsi_14'] / 100  # Normalize to [0,1]

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['macd_norm'] = (ema12 - ema26) / close

    # Bollinger Band position
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_position'] = (close - sma20) / (2 * std20 + 1e-8)

    # Volume features
    df['vol_ratio'] = volume / (volume.rolling(20).mean() + 1)
    df['vol_log'] = np.log1p(volume / 1e6)

    # ATR (normalized)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr_ratio'] = tr.rolling(14).mean() / close

    # High/Low position
    df['hl_ratio'] = (close - low) / (high - low + 1e-8)

    # Momentum z-score
    ret20 = df['ret_20d']
    df['momentum_z'] = (ret20 - ret20.rolling(63).mean()) / (ret20.rolling(63).std() + 1e-8)

    return df


def make_target(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """
    Target: Sign of forward return over `horizon` days.
    Returns 1 (up) or 0 (down/flat).
    """
    fwd_return = df['close'].pct_change(horizon).shift(-horizon)
    return (fwd_return > 0).astype(int)


FEATURE_COLS = [
    'ret_1d', 'ret_5d', 'ret_20d', 'ret_60d',
    'vol_10d', 'vol_20d', 'vol_60d',
    'sma_20_ratio', 'sma_50_ratio', 'sma_200_ratio',
    'rsi_14', 'macd_norm', 'bb_position',
    'vol_ratio', 'vol_log', 'atr_ratio', 'hl_ratio',
    'momentum_z'
]


# ── LSTM Model ────────────────────────────────────────────────
class QuantEdgeLSTM(nn.Module):
    """
    Bidirectional LSTM with attention for price direction prediction.
    Architecture: LSTM → Attention → Dropout → FC → Sigmoid
    """
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Bidirectional for richer context
        )
        # Attention
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Output
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)     # (batch, seq_len, hidden*2)

        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)

        # Classify
        out = self.dropout(context)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return self.sigmoid(out).squeeze(-1)


# ── Data Preparation ──────────────────────────────────────────
def prepare_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Create sliding windows of sequences."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def fetch_data(ticker: str, days: int = 1000) -> pd.DataFrame:
    """Fetch OHLCV from yFinance."""
    end = datetime.now()
    start = end - timedelta(days=days)
    t = yf.Ticker(ticker)
    df = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    return df


# ── Walk-Forward Trainer ──────────────────────────────────────
def walk_forward_train(
    ticker: str,
    X: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler
) -> Tuple[Optional[QuantEdgeLSTM], Dict]:
    """
    Walk-forward validation: train on rolling window, test on next window.
    Returns best model and performance metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    n = len(X)
    scores = []
    fold = 0

    best_model = None
    best_acc = 0.0

    # Walk forward
    start = TRAIN_WINDOW
    while start + TEST_WINDOW + EMBARGO <= n:
        fold += 1
        train_end = start
        test_start = start + EMBARGO
        test_end = min(test_start + TEST_WINDOW, n)

        if test_end - test_start < 20:
            break

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        # Create sequences
        X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, SEQUENCE_LENGTH)
        X_test_seq, y_test_seq = prepare_sequences(X_test, y_test, SEQUENCE_LENGTH)

        if len(X_train_seq) < BATCH_SIZE or len(X_test_seq) < 10:
            start += STEP
            continue

        # Train
        model = QuantEdgeLSTM(input_size=X.shape[1])
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_seq).to(device),
            torch.FloatTensor(y_train_seq).to(device)
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            avg_loss = train_loss / len(train_loader)
            scheduler.step(avg_loss)

            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    break

        # Evaluate
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test_seq).to(device)
            preds = model(X_t).cpu().numpy()
            pred_labels = (preds > 0.5).astype(int)
            acc = (pred_labels == y_test_seq).mean()

        scores.append(acc)
        logger.info(f"Fold {fold}: acc={acc:.4f} (train_end={train_end}, test=[{test_start},{test_end}])")

        if acc > best_acc:
            best_acc = acc
            best_model = model

        start += STEP

    metrics = {
        'ticker': ticker,
        'n_folds': fold,
        'mean_accuracy': float(np.mean(scores)) if scores else 0.0,
        'std_accuracy': float(np.std(scores)) if scores else 0.0,
        'best_accuracy': float(best_acc),
        'trained_at': datetime.now().isoformat(),
        'sequence_length': SEQUENCE_LENGTH,
        'feature_count': X.shape[1],
        'walk_forward_params': {
            'train_window': TRAIN_WINDOW,
            'test_window': TEST_WINDOW,
            'step': STEP,
            'embargo': EMBARGO
        }
    }

    logger.info(f"Walk-forward complete: mean_acc={metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
    return best_model, metrics


# ── Save / Load Model ─────────────────────────────────────────
def save_model(model: QuantEdgeLSTM, scaler: StandardScaler, metrics: dict, ticker: str):
    """Save model, scaler, and metrics to disk."""
    ticker_dir = MODEL_DIR / "lstm" / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    # Save PyTorch model
    torch.save(model.state_dict(), ticker_dir / "model.pt")

    # Save model config
    config = {
        'input_size': model.lstm.input_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
    }
    with open(ticker_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Save scaler
    joblib.dump(scaler, ticker_dir / "scaler.pkl")

    # Save metrics
    with open(ticker_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"✅ Model saved to {ticker_dir}")
    return str(ticker_dir)


def load_model(ticker: str) -> Tuple[Optional[QuantEdgeLSTM], Optional[StandardScaler], Optional[dict]]:
    """Load trained model from disk."""
    ticker_dir = MODEL_DIR / "lstm" / ticker
    if not ticker_dir.exists():
        return None, None, None
    try:
        with open(ticker_dir / "config.json") as f:
            config = json.load(f)
        model = QuantEdgeLSTM(**config)
        model.load_state_dict(torch.load(ticker_dir / "model.pt", map_location='cpu'))
        model.eval()
        scaler = joblib.load(ticker_dir / "scaler.pkl")
        with open(ticker_dir / "metrics.json") as f:
            metrics = json.load(f)
        return model, scaler, metrics
    except Exception as e:
        logger.error(f"Load failed for {ticker}: {e}")
        return None, None, None


# ── Full Training Pipeline ────────────────────────────────────
def train_ticker(ticker: str, days: int = 1000) -> Optional[dict]:
    """Run full training pipeline for one ticker."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Training LSTM for {ticker}")
    logger.info(f"{'='*50}")

    # Fetch data
    df = fetch_data(ticker, days)
    if len(df) < 300:
        logger.error(f"Insufficient data for {ticker}: {len(df)} rows")
        return None

    # Feature engineering
    df = compute_features(df)
    target = make_target(df, horizon=PREDICTION_HORIZON)

    # Align features and target
    combined = df[FEATURE_COLS].copy()
    combined['target'] = target
    combined = combined.dropna()

    X_raw = combined[FEATURE_COLS].values
    y = combined['target'].values

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Walk-forward train
    model, metrics = walk_forward_train(ticker, X, y, scaler)

    if model is None:
        logger.error(f"Training failed for {ticker}")
        return None

    # Save
    save_model(model, scaler, metrics, ticker)
    return metrics


# ── Inference (used by price oracle) ─────────────────────────
def predict(ticker: str, recent_data: pd.DataFrame) -> Optional[dict]:
    """
    Load trained model and predict direction for recent data.
    Returns probability of upward move.
    """
    model, scaler, metrics = load_model(ticker)
    if model is None:
        return None

    # Feature engineering
    df = compute_features(recent_data)
    df = df[FEATURE_COLS].dropna()

    if len(df) < SEQUENCE_LENGTH:
        return None

    # Scale and prepare sequence
    X = scaler.transform(df.values[-SEQUENCE_LENGTH - 10:])
    X_seq = X[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, -1)

    # Predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_seq).to(device)
        prob = model(X_t).cpu().item()

    return {
        'ticker': ticker,
        'prob_up': prob,
        'signal': 'BULLISH' if prob > 0.55 else 'BEARISH' if prob < 0.45 else 'NEUTRAL',
        'model_accuracy': metrics.get('mean_accuracy', 0),
        'trained_at': metrics.get('trained_at', ''),
        'n_folds': metrics.get('n_folds', 0),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantEdge LSTM Trainer")
    parser.add_argument("--ticker", type=str, help="Single ticker to train")
    parser.add_argument("--universe", choices=["sp500_top50"], default=None)
    parser.add_argument("--days", type=int, default=1000)
    parser.add_argument("--evaluate", action="store_true", help="Just evaluate loaded model")
    args = parser.parse_args()

    if args.evaluate and args.ticker:
        model, scaler, metrics = load_model(args.ticker)
        if metrics:
            print(json.dumps(metrics, indent=2))
        else:
            print(f"No saved model for {args.ticker}")
    elif args.ticker:
        metrics = train_ticker(args.ticker, args.days)
        if metrics:
            print(json.dumps(metrics, indent=2))
    elif args.universe == "sp500_top50":
        from fetch_and_store import SP500_TOP50
        all_metrics = []
        for ticker in SP500_TOP50:
            try:
                m = train_ticker(ticker, args.days)
                if m:
                    all_metrics.append(m)
            except Exception as e:
                logger.error(f"Failed {ticker}: {e}")
        summary = {
            'trained': len(all_metrics),
            'mean_accuracy': float(np.mean([m['mean_accuracy'] for m in all_metrics])),
        }
        print(json.dumps(summary, indent=2))
    else:
        logger.error("Provide --ticker AAPL or --universe sp500_top50")
        sys.exit(1)
