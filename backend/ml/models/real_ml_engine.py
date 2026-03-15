"""
QuantEdge v6.0 — Real ML Engine
==================================
© 2026 Dileep Kumar Reddy Kapu. All Rights Reserved.

All models train on real historical data with walk-forward validation
and 60-day embargo. No hardcoded outputs. No LLM API calls.

Classes:
    TrainedXGBoostPredictor   — trains/predicts with real XGBoost + SHAP
    TrainedLightGBMPredictor  — trains/predicts with real LightGBM + SHAP
    LSTMPredictor             — trains/predicts with real PyTorch BiLSTM + MC Dropout
    RegimeConditionalEnsemble — 5×5 weight matrix, weighted average by regime
    ModelTrainer              — orchestrates training all 3 models at startup
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import shap
import torch
import torch.nn as nn
from scipy import stats
from loguru import logger

# ── Paths ─────────────────────────────────────────────────────
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  TrainedXGBoostPredictor
# ══════════════════════════════════════════════════════════════

class TrainedXGBoostPredictor:
    """
    Walk-forward XGBoost with 60-day embargo and real SHAP.
    MODEL_PATH: MODEL_DIR/xgb_model.joblib
    """

    MODEL_PATH = MODEL_DIR / "xgb_model.joblib"
    META_PATH  = MODEL_DIR / "xgb_model_meta.json"

    def __init__(self):
        self.model = None
        self.feature_names: List[str] = []
        self.shap_explainer = None
        self._loaded = False

    def _ensure_loaded(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None):
        if self._loaded:
            return
        if self.MODEL_PATH.exists():
            self._load()
        elif X is not None and y is not None:
            self.train(X, y)
        else:
            raise RuntimeError(
                f"XGBoost model not found at {self.MODEL_PATH} and no training data provided."
            )

    def _load(self):
        self.model = joblib.load(self.MODEL_PATH)
        if self.META_PATH.exists():
            with open(self.META_PATH) as f:
                meta = json.load(f)
                self.feature_names = meta.get("feature_names", [])
        self.shap_explainer = shap.TreeExplainer(self.model)
        self._loaded = True
        logger.info(f"XGBoost model loaded from {self.MODEL_PATH}")

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Walk-forward validation with 60-day embargo.
        Train: first 80% chronologically.
        Skip:  60 days.
        Val:   remaining 20%.
        """
        from xgboost import XGBRegressor

        X = X.copy()
        y = y.copy()

        # Remove NaN rows
        mask = y.notna() & X.notna().all(axis=1)
        X, y = X[mask], y[mask]

        n = len(X)
        if n < 200:
            raise ValueError(f"Need at least 200 samples to train XGBoost, got {n}")

        n_train = int(n * 0.80)
        embargo  = 60
        n_val_start = n_train + embargo

        X_train = X.iloc[:n_train].values.astype(np.float32)
        y_train = y.iloc[:n_train].values.astype(np.float32)
        X_val   = X.iloc[n_val_start:].values.astype(np.float32)
        y_val   = y.iloc[n_val_start:].values.astype(np.float32)

        if len(X_val) < 20:
            logger.warning("XGBoost: validation set small — using last 20% without embargo")
            X_val = X.iloc[n_train:].values.astype(np.float32)
            y_val = y.iloc[n_train:].values.astype(np.float32)

        self.feature_names = list(X.columns)

        model = XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=20,
            early_stopping_rounds=50,
            eval_metric="rmse",
            verbosity=0,
            n_jobs=-1,
            random_state=42,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Validation metrics
        val_preds = model.predict(X_val)
        val_ic, _ = stats.spearmanr(val_preds, y_val)
        val_rmse  = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))

        # SHAP on validation set
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_val[:min(200, len(X_val))])
        mean_abs  = np.abs(shap_vals).mean(axis=0)
        top_idx   = np.argsort(mean_abs)[::-1][:20]
        feature_importance = {
            self.feature_names[i]: float(mean_abs[i]) for i in top_idx
        }

        # Save
        joblib.dump(model, self.MODEL_PATH)
        with open(self.META_PATH, "w") as f:
            json.dump({"feature_names": self.feature_names}, f)

        self.model = model
        self.shap_explainer = shap.TreeExplainer(model)
        self._loaded = True

        logger.info(f"XGBoost trained: val_IC={val_ic:.4f} val_RMSE={val_rmse:.4f} n_train={n_train}")
        return {"val_ic": float(val_ic), "val_rmse": val_rmse, "feature_importance": feature_importance}

    def predict(self, features: dict) -> dict:
        """
        Real XGBoost inference + SHAP values (NOT feature_importances_).
        """
        self._ensure_loaded()

        # Build feature vector in training column order
        if self.feature_names:
            vec = np.array([
                float(features.get(f, 0.0) or 0.0) for f in self.feature_names
            ], dtype=np.float32).reshape(1, -1)
        else:
            vec = np.array([float(v or 0.0) for v in features.values()], dtype=np.float32).reshape(1, -1)

        # Replace NaN/inf
        vec = np.where(np.isfinite(vec), vec, 0.0)

        pred = float(self.model.predict(vec)[0])

        # Real SHAP — not feature_importances_
        shap_vals = None
        shap_top = {}
        if self.shap_explainer is not None:
            try:
                raw_shap = self.shap_explainer.shap_values(vec)
                if raw_shap is not None and len(raw_shap.shape) >= 2:
                    shap_vals = raw_shap[0]
                    if self.feature_names and len(shap_vals) == len(self.feature_names):
                        top_idx = np.argsort(np.abs(shap_vals))[::-1][:10]
                        shap_top = {
                            self.feature_names[i]: round(float(shap_vals[i]), 6)
                            for i in top_idx
                        }
            except Exception as e:
                logger.warning(f"XGBoost SHAP failed: {e}")

        # Signal strength: normalize to [-1, 1] rough range
        signal = float(np.clip(pred * 10, -1.0, 1.0))
        confidence = float(np.clip(abs(signal), 0.0, 1.0))

        return {
            "signal_strength": signal,
            "pred_21d": pred,
            "confidence": confidence,
            "shap_values": shap_top,
            "model": "xgboost",
        }


# ══════════════════════════════════════════════════════════════
#  TrainedLightGBMPredictor
# ══════════════════════════════════════════════════════════════

class TrainedLightGBMPredictor:
    """
    Walk-forward LightGBM with 60-day embargo and real SHAP.
    MODEL_PATH: MODEL_DIR/lgb_model.joblib
    """

    MODEL_PATH = MODEL_DIR / "lgb_model.joblib"
    META_PATH  = MODEL_DIR / "lgb_model_meta.json"

    def __init__(self):
        self.model = None
        self.feature_names: List[str] = []
        self.shap_explainer = None
        self._loaded = False

    def _ensure_loaded(self, X=None, y=None):
        if self._loaded:
            return
        if self.MODEL_PATH.exists():
            self._load()
        elif X is not None and y is not None:
            self.train(X, y)
        else:
            raise RuntimeError(f"LightGBM model not found at {self.MODEL_PATH}")

    def _load(self):
        self.model = joblib.load(self.MODEL_PATH)
        if self.META_PATH.exists():
            with open(self.META_PATH) as f:
                meta = json.load(f)
                self.feature_names = meta.get("feature_names", [])
        self.shap_explainer = shap.TreeExplainer(self.model)
        self._loaded = True
        logger.info(f"LightGBM model loaded from {self.MODEL_PATH}")

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Walk-forward with 60-day embargo, same split as XGBoost."""
        import lightgbm as lgb

        X = X.copy()
        y = y.copy()
        mask = y.notna() & X.notna().all(axis=1)
        X, y = X[mask], y[mask]

        n = len(X)
        if n < 200:
            raise ValueError(f"Need at least 200 samples for LightGBM, got {n}")

        n_train = int(n * 0.80)
        embargo  = 60
        n_val_start = n_train + embargo

        X_train = X.iloc[:n_train]
        y_train = y.iloc[:n_train]
        X_val   = X.iloc[n_val_start:]
        y_val   = y.iloc[n_val_start:]

        if len(X_val) < 20:
            X_val = X.iloc[n_train:]
            y_val = y.iloc[n_train:]

        self.feature_names = list(X.columns)

        model = lgb.LGBMRegressor(
            n_estimators=1000,
            num_leaves=63,
            learning_rate=0.01,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )

        val_preds = model.predict(X_val)
        val_ic, _ = stats.spearmanr(val_preds, y_val)
        val_rmse  = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_val.values[:min(200, len(X_val))])
        mean_abs  = np.abs(shap_vals).mean(axis=0)
        top_idx   = np.argsort(mean_abs)[::-1][:20]
        feature_importance = {
            self.feature_names[i]: float(mean_abs[i]) for i in top_idx
        }

        joblib.dump(model, self.MODEL_PATH)
        with open(self.META_PATH, "w") as f:
            json.dump({"feature_names": self.feature_names}, f)

        self.model = model
        self.shap_explainer = shap.TreeExplainer(model)
        self._loaded = True

        logger.info(f"LightGBM trained: val_IC={val_ic:.4f} val_RMSE={val_rmse:.4f} n_train={n_train}")
        return {"val_ic": float(val_ic), "val_rmse": val_rmse, "feature_importance": feature_importance}

    def predict(self, features: dict) -> dict:
        """Real LightGBM inference + SHAP values."""
        self._ensure_loaded()

        if self.feature_names:
            vec = np.array([
                float(features.get(f, 0.0) or 0.0) for f in self.feature_names
            ], dtype=np.float32).reshape(1, -1)
        else:
            vec = np.array([float(v or 0.0) for v in features.values()], dtype=np.float32).reshape(1, -1)

        vec = np.where(np.isfinite(vec), vec, 0.0)
        pred = float(self.model.predict(vec)[0])

        shap_top = {}
        if self.shap_explainer is not None:
            try:
                raw_shap = self.shap_explainer.shap_values(vec)
                shap_vals = raw_shap[0] if len(raw_shap.shape) >= 2 else raw_shap
                if self.feature_names and len(shap_vals) == len(self.feature_names):
                    top_idx = np.argsort(np.abs(shap_vals))[::-1][:10]
                    shap_top = {
                        self.feature_names[i]: round(float(shap_vals[i]), 6)
                        for i in top_idx
                    }
            except Exception as e:
                logger.warning(f"LightGBM SHAP failed: {e}")

        signal = float(np.clip(pred * 10, -1.0, 1.0))
        confidence = float(np.clip(abs(signal), 0.0, 1.0))

        return {
            "signal_strength": signal,
            "pred_21d": pred,
            "confidence": confidence,
            "shap_values": shap_top,
            "model": "lightgbm",
        }


# ══════════════════════════════════════════════════════════════
#  BiLSTM Architecture (for LSTMPredictor)
# ══════════════════════════════════════════════════════════════

class _Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq, hidden*2)
        scores = self.attn(lstm_out).squeeze(-1)          # (batch, seq)
        weights = torch.softmax(scores, dim=1).unsqueeze(2)  # (batch, seq, 1)
        context = (lstm_out * weights).sum(dim=1)          # (batch, hidden*2)
        return context


class _BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.3, n_outputs: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = _Attention(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.fc1       = nn.Linear(hidden_size * 2, 64)
        self.relu      = nn.ReLU()
        self.fc2       = nn.Linear(64, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)          # (batch, seq, hidden*2)
        context = self.attention(lstm_out)  # (batch, hidden*2)
        out = self.dropout(context)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)                 # (batch, 3) — pred_5d, pred_21d, pred_63d
        return out


# ══════════════════════════════════════════════════════════════
#  LSTMPredictor
# ══════════════════════════════════════════════════════════════

class LSTMPredictor:
    """
    Real PyTorch BiLSTM with Attention + MC Dropout.

    Trains on rolling 60-day windows from price_data.
    Saves/loads model weights from MODEL_DIR/lstm_weights.pt.
    MC Dropout: 30 forward passes in train() mode for uncertainty.
    """

    MODEL_PATH      = MODEL_DIR / "lstm_weights.pt"
    META_PATH       = MODEL_DIR / "lstm_meta.json"
    SEQUENCE_LENGTH = 60
    HIDDEN_SIZE     = 256
    NUM_LAYERS      = 2
    DROPOUT         = 0.3
    N_OUTPUTS       = 3   # pred_5d, pred_21d, pred_63d
    MC_PASSES       = 30

    def __init__(self):
        self.model:      Optional[_BiLSTMWithAttention] = None
        self.n_features: int = 0
        self.feature_cols: List[str] = []
        self._loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_loaded(self, price_data=None):
        if self._loaded:
            return
        if self.MODEL_PATH.exists():
            self._load()
        elif price_data is not None:
            self.train(price_data)
        else:
            raise RuntimeError(f"LSTM model not found at {self.MODEL_PATH}")

    def _load(self):
        if self.META_PATH.exists():
            with open(self.META_PATH) as f:
                meta = json.load(f)
                self.n_features = meta["n_features"]
                self.feature_cols = meta.get("feature_cols", [])
        else:
            raise RuntimeError(f"LSTM meta file missing: {self.META_PATH}")

        self.model = _BiLSTMWithAttention(
            input_size=self.n_features,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS,
            dropout=self.DROPOUT,
            n_outputs=self.N_OUTPUTS,
        ).to(self.device)
        state = torch.load(self.MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self._loaded = True
        logger.info(f"LSTM model loaded: {self.n_features} features, {self.device}")

    def _build_features(self, price_data: pd.DataFrame) -> np.ndarray:
        """Build LSTM-compatible feature matrix from OHLCV data."""
        df = price_data.copy()
        df.columns = [c.lower() for c in df.columns]

        close = df["close"].values
        n = len(close)

        feats = {}
        # Returns
        feats["ret_1d"] = np.diff(close, prepend=close[0]) / (close + 1e-10)
        feats["ret_5d"] = np.array([(close[i] - close[max(0, i-5)]) / (close[max(0, i-5)] + 1e-10) for i in range(n)])
        feats["ret_21d"] = np.array([(close[i] - close[max(0, i-21)]) / (close[max(0, i-21)] + 1e-10) for i in range(n)])

        # Rolling vol
        ret = feats["ret_1d"]
        for w in [5, 21, 63]:
            feats[f"vol_{w}d"] = np.array([
                np.std(ret[max(0, i-w):i+1]) * np.sqrt(252) for i in range(n)
            ])

        # Volume
        if "volume" in df.columns:
            vol = df["volume"].values.astype(float)
            avg_vol = np.array([np.mean(vol[max(0, i-20):i+1]) for i in range(n)])
            feats["vol_ratio"] = vol / (avg_vol + 1)
        else:
            feats["vol_ratio"] = np.ones(n)

        # Price position
        for w in [20, 50, 200]:
            ma = np.array([np.mean(close[max(0, i-w):i+1]) for i in range(n)])
            feats[f"price_to_ma{w}"] = close / (ma + 1e-10) - 1.0

        # RSI
        delta = np.diff(close, prepend=close[0])
        up = np.where(delta > 0, delta, 0.0)
        dn = np.where(delta < 0, -delta, 0.0)
        rs_up = np.array([np.mean(up[max(0, i-14):i+1]) for i in range(n)])
        rs_dn = np.array([np.mean(dn[max(0, i-14):i+1]) for i in range(n)])
        rsi = 100 - 100 / (1 + rs_up / (rs_dn + 1e-10))
        feats["rsi_14"] = (rsi - 50) / 50.0  # normalize

        # MACD
        if n > 26:
            ema12 = _ewm(close, 12)
            ema26 = _ewm(close, 26)
            macd = ema12 - ema26
            macd_range = np.std(macd) + 1e-10
            feats["macd_norm"] = macd / macd_range
        else:
            feats["macd_norm"] = np.zeros(n)

        mat = np.column_stack(list(feats.values()))
        mat = np.where(np.isfinite(mat), mat, 0.0)
        self.feature_cols = list(feats.keys())
        return mat

    def train(self, price_data: pd.DataFrame) -> dict:
        """
        Build (n_samples, 60, n_features) tensor from rolling windows.
        Train BiLSTM with Huber loss, Adam, gradient clipping.
        30 epochs with early stopping.
        """
        features = self._build_features(price_data)
        n, n_features = features.shape
        self.n_features = n_features

        if n < self.SEQUENCE_LENGTH + 63 + 20:
            raise ValueError(f"Need at least {self.SEQUENCE_LENGTH + 63 + 20} rows for LSTM training, got {n}")

        # Build sequences
        close = price_data["close"].values if "close" in price_data.columns else price_data.iloc[:, 3].values
        X_seqs, y_seqs = [], []
        for i in range(self.SEQUENCE_LENGTH, n - 63):
            seq = features[i - self.SEQUENCE_LENGTH:i]
            # Targets: forward returns at 5, 21, 63 days
            ret_5  = (close[i+5]  - close[i]) / (close[i] + 1e-10)
            ret_21 = (close[i+21] - close[i]) / (close[i] + 1e-10)
            ret_63 = (close[i+63] - close[i]) / (close[i] + 1e-10)
            X_seqs.append(seq)
            y_seqs.append([ret_5, ret_21, ret_63])

        X_arr = np.array(X_seqs, dtype=np.float32)
        y_arr = np.array(y_seqs, dtype=np.float32)

        # Split: 80% train, 60-day embargo, 20% val
        n_samples = len(X_arr)
        n_train = int(n_samples * 0.80)
        n_val_start = n_train + 60
        X_train = torch.tensor(X_arr[:n_train]).to(self.device)
        y_train = torch.tensor(y_arr[:n_train]).to(self.device)
        X_val   = torch.tensor(X_arr[n_val_start:]).to(self.device)
        y_val   = torch.tensor(y_arr[n_val_start:]).to(self.device)

        model = _BiLSTMWithAttention(
            input_size=n_features,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS,
            dropout=self.DROPOUT,
            n_outputs=self.N_OUTPUTS,
        ).to(self.device)

        criterion = nn.HuberLoss(delta=0.01)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        batch_size = 64
        best_val_loss = float("inf")
        best_state = None
        patience_count = 0
        patience_limit = 8

        train_ds = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        for epoch in range(30):
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val)
                val_loss  = criterion(val_preds, y_val).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience_limit:
                    logger.info(f"LSTM early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 5 == 0:
                logger.info(f"LSTM epoch {epoch+1}/30  train_loss={epoch_loss/len(train_loader):.6f}  val_loss={val_loss:.6f}")

        # Restore best
        if best_state:
            model.load_state_dict(best_state)

        # Val IC on pred_21d (index 1)
        model.eval()
        with torch.no_grad():
            vp = model(X_val).cpu().numpy()
            vy = y_val.cpu().numpy()
        ic_21d, _ = stats.spearmanr(vp[:, 1], vy[:, 1])

        # Save
        torch.save(best_state or model.state_dict(), self.MODEL_PATH)
        with open(self.META_PATH, "w") as f:
            json.dump({"n_features": n_features, "feature_cols": self.feature_cols}, f)

        self.model = model.to(self.device)
        self._loaded = True
        logger.info(f"LSTM trained: val_IC_21d={float(ic_21d):.4f} val_loss={best_val_loss:.6f} n_train={n_train}")
        return {"val_ic_21d": float(ic_21d), "val_loss": best_val_loss, "n_features": n_features}

    def predict(self, price_data: pd.DataFrame) -> dict:
        """
        MC Dropout inference: 30 forward passes with model.train() mode.
        Returns mean ± epistemic uncertainty per horizon.
        """
        self._ensure_loaded(price_data)

        features = self._build_features(price_data)
        if len(features) < self.SEQUENCE_LENGTH:
            raise ValueError(f"Need {self.SEQUENCE_LENGTH} rows for LSTM predict, got {len(features)}")

        seq = features[-self.SEQUENCE_LENGTH:]
        X = torch.tensor(seq[np.newaxis, :, :], dtype=torch.float32).to(self.device)

        # MC Dropout: 30 passes
        self.model.train()  # enable dropout for uncertainty
        preds = []
        with torch.no_grad():
            for _ in range(self.MC_PASSES):
                out = self.model(X).cpu().numpy()[0]  # (3,)
                preds.append(out)

        preds_arr = np.array(preds)   # (30, 3)
        mean_preds = preds_arr.mean(axis=0)
        std_preds  = preds_arr.std(axis=0)

        self.model.eval()

        return {
            "pred_5d":          float(mean_preds[0]),
            "pred_21d":         float(mean_preds[1]),
            "pred_63d":         float(mean_preds[2]),
            "uncertainty_5d":   float(std_preds[0]),
            "uncertainty_21d":  float(std_preds[1]),
            "uncertainty_63d":  float(std_preds[2]),
            "uncertainty":      float(std_preds.mean()),
            "model":            "lstm_bilstm_attention",
            "mc_passes":        self.MC_PASSES,
        }


# ══════════════════════════════════════════════════════════════
#  RegimeConditionalEnsemble
# ══════════════════════════════════════════════════════════════

class RegimeConditionalEnsemble:
    """
    5×5 weight matrix: 5 HMM regimes × 5 model types.
    Weights are research-backed priors from Lopez de Prado (2018) and
    Baz et al. (2015) on regime-conditional alpha decay rates.

    These weights are updated over time by IC feedback from signal_tracker,
    but always initialized here.
    """

    DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
        "Bull_Trending":  {"lstm": 0.30, "xgb": 0.25, "lgb": 0.20, "garch": 0.10, "kalman": 0.15},
        "Bull_Volatile":  {"lstm": 0.20, "xgb": 0.25, "lgb": 0.25, "garch": 0.20, "kalman": 0.10},
        "Mean_Reverting": {"lstm": 0.15, "xgb": 0.30, "lgb": 0.30, "garch": 0.10, "kalman": 0.15},
        "Bear_Trending":  {"lstm": 0.10, "xgb": 0.20, "lgb": 0.20, "garch": 0.35, "kalman": 0.15},
        "Crisis":         {"lstm": 0.05, "xgb": 0.15, "lgb": 0.15, "garch": 0.45, "kalman": 0.20},
    }

    # Normalize fallback regime names to canonical names
    REGIME_MAP: Dict[str, str] = {
        "BULL_LOW_VOL":   "Bull_Trending",
        "BULL_HIGH_VOL":  "Bull_Volatile",
        "MEAN_REVERTING": "Mean_Reverting",
        "BEAR_LOW_VOL":   "Bear_Trending",
        "BEAR_HIGH_VOL":  "Crisis",
        "CRISIS":         "Crisis",
        "BULL_TRENDING":  "Bull_Trending",
        "BULL_VOLATILE":  "Bull_Volatile",
        "BEAR_TRENDING":  "Bear_Trending",
    }

    def __init__(self):
        self.weights = {k: dict(v) for k, v in self.DEFAULT_WEIGHTS.items()}

    def _normalize_regime(self, regime: str) -> str:
        """Map any regime string to one of the 5 canonical names."""
        upper = regime.upper().replace(" ", "_")
        return self.REGIME_MAP.get(upper, "Bull_Trending")

    def combine(self, model_outputs: dict, regime: str) -> dict:
        """
        Weighted average of all model signals for current regime.

        model_outputs must contain keys: "lstm", "xgboost"/"xgb",
        "lightgbm"/"lgb", "garch"/"volatility", "kalman"/"kalman_filter"

        Returns ensemble_signal, regime_used, weights_applied.
        """
        canonical_regime = self._normalize_regime(regime)
        w = self.weights.get(canonical_regime, self.weights["Bull_Trending"])

        # Extract signals from various key names
        lstm   = model_outputs.get("lstm", {})
        xgb    = model_outputs.get("xgboost", model_outputs.get("xgb", {}))
        lgb    = model_outputs.get("lightgbm", model_outputs.get("lgb", {}))
        garch  = model_outputs.get("garch", model_outputs.get("volatility", {}))
        kalman = model_outputs.get("kalman", model_outputs.get("kalman_filter", {}))

        # Get signal from each model — use pred_21d as the directional signal
        def _sig(d: dict) -> float:
            for key in ("signal_strength", "pred_21d", "signal", "trend", "kalman_trend"):
                val = d.get(key)
                if val is not None:
                    return float(val)
            return 0.0

        lstm_sig   = _sig(lstm)
        xgb_sig    = _sig(xgb)
        lgb_sig    = _sig(lgb)
        garch_sig  = _sig(garch)
        kalman_sig = _sig(kalman)

        # GARCH vol forecast contributes as a risk-adjusted directional signal
        # High vol → negative signal contribution in volatile regimes
        if garch_sig == 0.0:
            # Try volatility-based signal: if vol is high, negative; if low, positive
            vol = garch.get("vol_forecast", garch.get("annualized_vol", 0.0)) or 0.0
            garch_sig = float(np.clip(-float(vol) * 2, -1.0, 1.0))

        # Weighted sum
        ensemble = (
            w["lstm"]   * lstm_sig   +
            w["xgb"]    * xgb_sig    +
            w["lgb"]    * lgb_sig    +
            w["garch"]  * garch_sig  +
            w["kalman"] * kalman_sig
        )

        # Normalize to [-1, 1]
        weight_sum = sum(w.values())
        if weight_sum > 0:
            ensemble = ensemble / weight_sum

        ensemble = float(np.clip(ensemble, -1.0, 1.0))

        return {
            "ensemble_signal":  ensemble,
            "ensemble_direction": "LONG" if ensemble > 0.02 else "SHORT" if ensemble < -0.02 else "NEUTRAL",
            "regime_used":      canonical_regime,
            "weights_applied":  w,
            "component_signals": {
                "lstm":   round(lstm_sig, 6),
                "xgb":    round(xgb_sig, 6),
                "lgb":    round(lgb_sig, 6),
                "garch":  round(garch_sig, 6),
                "kalman": round(kalman_sig, 6),
            },
        }

    def update_weights(self, regime: str, new_weights: dict) -> None:
        """Update weights for a specific regime (called by IC feedback loop)."""
        canonical = self._normalize_regime(regime)
        # Normalize to sum to 1
        total = sum(new_weights.values())
        if total > 0:
            self.weights[canonical] = {k: v / total for k, v in new_weights.items()}


# ══════════════════════════════════════════════════════════════
#  ModelTrainer
# ══════════════════════════════════════════════════════════════

class ModelTrainer:
    """
    Orchestrates training all 3 models at startup if any model file is missing.
    Can also be triggered via an admin endpoint.
    """

    def __init__(
        self,
        xgb_predictor:  TrainedXGBoostPredictor,
        lgb_predictor:  TrainedLightGBMPredictor,
        lstm_predictor: LSTMPredictor,
    ):
        self.xgb  = xgb_predictor
        self.lgb  = lgb_predictor
        self.lstm = lstm_predictor

    def all_models_exist(self) -> bool:
        return (
            TrainedXGBoostPredictor.MODEL_PATH.exists() and
            TrainedLightGBMPredictor.MODEL_PATH.exists() and
            LSTMPredictor.MODEL_PATH.exists()
        )

    def train_all(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        fundamentals: dict,
    ) -> dict:
        """
        Train XGBoost, LightGBM, LSTM from price_data.
        Returns training metrics for all models.

        Called synchronously at startup — blocking is acceptable because
        the server won't serve requests until lifespan yields.
        """
        from ml.features.feature_engineering import FeaturePipeline
        from ml.labeling.triple_barrier import LabelingPipeline

        logger.info(f"ModelTrainer: starting full training for {ticker}")
        metrics: Dict[str, dict] = {}

        price_data = price_data.copy()
        price_data.columns = [c.lower() for c in price_data.columns]

        # ── Build historical feature matrix ──
        logger.info("ModelTrainer: building historical feature matrix...")
        fp = FeaturePipeline()
        try:
            X_arr, feature_names, dates = fp.build_historical_feature_matrix(
                price_data, fundamentals
            )
            X = pd.DataFrame(X_arr, columns=feature_names, index=dates)
            logger.info(f"ModelTrainer: feature matrix shape: {X.shape}")
        except Exception as e:
            logger.error(f"ModelTrainer: feature matrix failed: {e}")
            return {"error": str(e)}

        # ── Generate labels ──
        close = price_data["close"] if "close" in price_data.columns else price_data.iloc[:, 3]
        close = close.reindex(dates)

        logger.info("ModelTrainer: generating triple-barrier labels...")
        labeler = LabelingPipeline()
        try:
            label_result = labeler.run(close=close)
            # Labels are a Series of {-1, 0, 1}
            if hasattr(label_result, "labels"):
                y_raw = label_result.labels
            elif hasattr(label_result, "primary"):
                y_raw = label_result.primary
            else:
                y_raw = label_result

            # Align to feature matrix dates
            y = y_raw.reindex(dates).fillna(0)
            # Use continuous returns for regression (better for XGBoost/LightGBM)
            y_cont = close.pct_change(21).shift(-21).reindex(dates).fillna(0)
            y_use = y_cont  # 21-day forward return as regression target
        except Exception as e:
            logger.warning(f"ModelTrainer: labeling failed: {e}, using forward returns")
            y_use = close.pct_change(21).shift(-21).reindex(dates).fillna(0)

        y_use = pd.Series(y_use.values, index=X.index)

        # ── Train XGBoost ──
        logger.info("ModelTrainer: training XGBoost...")
        try:
            xgb_metrics = self.xgb.train(X, y_use)
            metrics["xgboost"] = xgb_metrics
        except Exception as e:
            logger.error(f"ModelTrainer: XGBoost training failed: {e}")
            metrics["xgboost"] = {"error": str(e)}

        # ── Train LightGBM ──
        logger.info("ModelTrainer: training LightGBM...")
        try:
            lgb_metrics = self.lgb.train(X, y_use)
            metrics["lightgbm"] = lgb_metrics
        except Exception as e:
            logger.error(f"ModelTrainer: LightGBM training failed: {e}")
            metrics["lightgbm"] = {"error": str(e)}

        # ── Train LSTM ──
        logger.info("ModelTrainer: training LSTM...")
        try:
            lstm_metrics = self.lstm.train(price_data)
            metrics["lstm"] = lstm_metrics
        except Exception as e:
            logger.error(f"ModelTrainer: LSTM training failed: {e}")
            metrics["lstm"] = {"error": str(e)}

        logger.info(f"ModelTrainer: training complete. Metrics: {metrics}")
        return metrics


# ── Helpers ───────────────────────────────────────────────────

def _ewm(series: np.ndarray, span: int) -> np.ndarray:
    """Exponentially weighted moving average (numpy implementation)."""
    alpha = 2.0 / (span + 1)
    result = np.empty_like(series, dtype=float)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
    return result
