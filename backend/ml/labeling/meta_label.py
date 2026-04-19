"""
QuantEdge v6.0 — Meta-Labeling Layer
=====================================
Implements López de Prado (2018) Ch.3 meta-labeling.

Primary model: XGBoost+LightGBM ensemble predicts expected return.
Secondary model: LightGBM classifier predicts P(primary is correct).

Output per horizon:
  - Primary prediction (expected return %)
  - Meta-confidence (0-1 probability primary is right)
  - Position size recommendation (magnitude × confidence)

Reference: Advances in Financial Machine Learning, Chapter 3.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import lightgbm as lgb
from loguru import logger


class MetaLabeler:
    """
    Per-horizon meta-labeling layer trained on val set residuals.
    """

    def __init__(
        self,
        horizons: List[int] = [5, 10, 21, 63, 252],
        conviction_threshold: float = 0.005,
    ):
        self.horizons = horizons
        self.conviction_threshold = conviction_threshold
        self.classifiers: Dict[int, lgb.LGBMClassifier] = {}

    def compute_meta_labels(
        self,
        primary_preds: np.ndarray,
        realized_returns: np.ndarray,
    ) -> np.ndarray:
        """
        Binary meta-labels:
          1 = primary got direction right AND move was meaningful
          0 = primary was wrong OR move was noise
        """
        primary_preds = np.asarray(primary_preds, dtype=np.float64)
        realized_returns = np.asarray(realized_returns, dtype=np.float64)

        correct_direction = np.sign(primary_preds) == np.sign(realized_returns)
        meaningful_move = np.abs(realized_returns) > self.conviction_threshold
        return (correct_direction & meaningful_move).astype(int)

    def fit(
        self,
        X_val: np.ndarray,
        primary_preds: np.ndarray,
        y_val: np.ndarray,
        horizon: int,
        feature_names: Optional[List[str]] = None,
    ) -> Optional[lgb.LGBMClassifier]:
        """
        Train meta-classifier for this horizon using val-set data.

        Features: [original features, primary_pred, |primary_pred|]
        Target: binary meta-label

        Returns None if:
          - insufficient samples (<30)
          - class imbalance too extreme (<5 positives or <5 negatives)
        """
        if len(X_val) < 15 or len(primary_preds) != len(X_val) or len(y_val) != len(X_val):
            logger.warning(f"MetaLabeler[h={horizon}]: insufficient/mismatched val data, skipping")
            return None

        # Construct meta-features
        pred_col = primary_preds.reshape(-1, 1)
        abs_pred_col = np.abs(primary_preds).reshape(-1, 1)
        X_meta = np.concatenate([X_val, pred_col, abs_pred_col], axis=1)
        meta_labels = self.compute_meta_labels(primary_preds, y_val)

        n_pos = int(meta_labels.sum())
        n_neg = len(meta_labels) - n_pos
        if n_pos < 5 or n_neg < 5:
            logger.warning(
                f"MetaLabeler[h={horizon}]: class imbalance (pos={n_pos}, neg={n_neg}), skipping"
            )
            return None

        clf = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        clf.fit(X_meta, meta_labels)
        self.classifiers[horizon] = clf

        # Training accuracy for telemetry (not model-selection; just informative)
        train_acc = float((clf.predict(X_meta) == meta_labels).mean())
        logger.info(
            f"MetaLabeler[h={horizon}]: trained on n={len(X_meta)} "
            f"(pos={n_pos}, neg={n_neg}, acc={train_acc:.3f})"
        )
        return clf

    def predict_confidence(
        self,
        today_features: np.ndarray,
        primary_pred: float,
        horizon: int,
    ) -> float:
        """
        Returns P(primary prediction is correct) for the given horizon.
        Falls back to 0.5 if no model trained for this horizon.
        """
        if horizon not in self.classifiers:
            return 0.5

        today_features = np.asarray(today_features, dtype=np.float64).flatten()
        meta_feat = np.concatenate([today_features, [primary_pred, abs(primary_pred)]]).reshape(1, -1)
        try:
            proba = self.classifiers[horizon].predict_proba(meta_feat)[0, 1]
            return float(proba)
        except Exception as e:
            logger.warning(f"MetaLabeler[h={horizon}]: predict failed: {e}")
            return 0.5

    def position_size_recommendation(
        self,
        primary_pred: float,
        confidence: float,
        magnitude_cap: float = 0.10,
    ) -> float:
        """
        Scalar 0-1 representing recommended position fraction.
          size = clip(|pred| / magnitude_cap, 0, 1) * confidence

        magnitude_cap=0.10 means a predicted 10% move maxes out the magnitude
        component at 1.0. Smaller moves scale proportionally.
        """
        pred = abs(primary_pred) / magnitude_cap
        magnitude = float(np.clip(pred, 0.0, 1.0))
        return float(magnitude * confidence)

    def conviction_level(self, confidence: float) -> str:
        if confidence >= 0.65:
            return "HIGH"
        if confidence >= 0.55:
            return "MEDIUM"
        if confidence >= 0.45:
            return "LOW"
        return "VERY_LOW"


def build_summary(
    horizons: List[int],
    primary_preds_by_horizon: Dict[int, float],
    confidences_by_horizon: Dict[int, float],
    meta_labeler: MetaLabeler,
) -> Dict:
    """
    Build response dict:
      meta_confidence_5d, meta_confidence_10d, ...
      position_size_recommendation (avg across horizons, weighted by |pred|)
      conviction_level (derived from primary horizon 21d)
    """
    out = {}
    for h in horizons:
        conf = confidences_by_horizon.get(h, 0.5)
        out[f"meta_confidence_{h}d"] = round(conf, 4)

    # Position size = weighted average across horizons
    sizes = []
    weights = []
    for h in horizons:
        pred = primary_preds_by_horizon.get(h, 0.0)
        conf = confidences_by_horizon.get(h, 0.5)
        if abs(pred) < 1e-6:
            continue
        sizes.append(meta_labeler.position_size_recommendation(pred, conf))
        weights.append(abs(pred))

    if sizes:
        weights_arr = np.asarray(weights)
        sizes_arr = np.asarray(sizes)
        out["position_size_recommendation"] = float(
            np.average(sizes_arr, weights=weights_arr)
        )
    else:
        out["position_size_recommendation"] = 0.0

    # Conviction level anchored on 21-day (primary horizon)
    primary_conf = confidences_by_horizon.get(21, 0.5)
    out["conviction_level"] = meta_labeler.conviction_level(primary_conf)
    out["primary_confidence"] = round(primary_conf, 4)

    return out
