"""
QuantEdge v6.0 — Walk-Forward Out-of-Fold Predictions for Meta-Labeling
=======================================================================

Purpose:
  Generate OOF predictions from primary models so meta-labeling has
  proper per-sample training data (each sample was predicted by a model
  that did NOT see it during training).

Why OOF instead of simple train/val split:
  - Train/val split leaves only 15-30 val samples for meta training → garbage
  - OOF walk-forward uses ALL samples for meta training (~100+) → real signal
  - Respects time-ordering (walk-forward, not random K-fold)
  - Includes embargo between train and OOF slice to prevent label leakage

Reference: Lopez de Prado (2018) — Combinatorial Purged K-Fold,
            Chapter 7 "Cross-Validation in Finance"
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from loguru import logger


def walk_forward_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    build_model: Callable[[], object],
    n_folds: int = 5,
    embargo: int = 5,
    min_train: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Walk-forward expanding-window OOF predictions.

    For fold k in [1..n_folds]:
      train_end   = min_train + (k-1) * fold_size
      oof_start   = train_end + embargo
      oof_end     = oof_start + fold_size
      Train model on X[:train_end], y[:train_end]
      Predict on X[oof_start:oof_end]

    Args:
        X: feature matrix, shape (n_samples, n_features)
        y: target vector, shape (n_samples,)
        feature_names: list of feature names
        build_model: zero-arg factory returning a fresh primary model
                     (must have .fit(X, y, feature_names) and .predict(X))
        n_folds: number of walk-forward folds
        embargo: gap in samples between train end and OOF start
        min_train: minimum train set size before first OOF prediction

    Returns:
        (oof_preds, oof_indices):
          oof_preds:   array of shape (n_oof,), OOF predictions
          oof_indices: array of shape (n_oof,), sample indices in original X/y

    If insufficient data, returns empty arrays.
    """
    n = len(X)
    if n < min_train + embargo + 10:
        logger.warning(f"walk_forward_oof: n={n} too small (need ≥{min_train + embargo + 10})")
        return np.array([]), np.array([], dtype=int)

    available_for_oof = n - min_train - embargo
    fold_size = max(5, available_for_oof // n_folds)

    all_preds: List[np.ndarray] = []
    all_indices: List[np.ndarray] = []

    for k in range(n_folds):
        train_end = min_train + k * fold_size
        oof_start = train_end + embargo
        oof_end = min(oof_start + fold_size, n)

        if oof_start >= n or oof_end - oof_start < 2:
            break
        if train_end < min_train:
            continue

        X_tr = X[:train_end]
        y_tr = y[:train_end]
        X_oof = X[oof_start:oof_end]

        try:
            # Carve a 15% slice at the END of the train window for early stopping validation.
            # Not a true val set — just a stop-signal for boosted trees.
            val_frac = max(0.10, min(0.20, 15.0 / len(X_tr)))
            val_cut = int(len(X_tr) * (1 - val_frac))
            X_train_inner = X_tr[:val_cut]
            y_train_inner = y_tr[:val_cut]
            X_val_inner = X_tr[val_cut:]
            y_val_inner = y_tr[val_cut:]

            model = build_model()
            # Models differ in whether they accept X_val/y_val kwargs. Try the richer call first,
            # fall back to plain .fit() if the model doesn\'t accept them.
            try:
                model.fit(X_train_inner, y_train_inner, feature_names,
                          X_val=X_val_inner, y_val=y_val_inner)
            except TypeError:
                model.fit(X_train_inner, y_train_inner, feature_names)

            preds = model.predict(X_oof)
            preds = np.asarray(preds, dtype=np.float64).flatten()

            if len(preds) != len(X_oof):
                continue

            all_preds.append(preds)
            all_indices.append(np.arange(oof_start, oof_end, dtype=int))
        except Exception as e:
            logger.warning(f"walk_forward_oof fold {k}: {e}")
            continue

    if not all_preds:
        return np.array([]), np.array([], dtype=int)

    oof_preds = np.concatenate(all_preds)
    oof_indices = np.concatenate(all_indices)
    logger.info(
        f"walk_forward_oof: {n_folds} folds → {len(oof_preds)} OOF predictions "
        f"(coverage {len(oof_preds) / n:.0%})"
    )
    return oof_preds, oof_indices


def walk_forward_oof_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_builders: Dict[str, Callable[[], object]],
    weights: Optional[Dict[str, float]] = None,
    n_folds: int = 5,
    embargo: int = 5,
    min_train: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    OOF predictions from an ensemble of primary models.

    Each model produces its own OOF predictions via walk_forward_oof_predictions.
    They are averaged (optionally weighted) at each OOF sample.

    Args:
        model_builders: dict {name: factory} e.g., {"xgb": lambda: ..., "lgb": lambda: ...}
        weights: optional dict {name: weight}; if None, equal-weighted

    Returns:
        (ensemble_oof_preds, oof_indices) — same shape as single-model version
    """
    if weights is None:
        weights = {name: 1.0 / len(model_builders) for name in model_builders}

    per_model_oof: Dict[str, np.ndarray] = {}
    per_model_idx: Dict[str, np.ndarray] = {}

    for name, builder in model_builders.items():
        preds, indices = walk_forward_oof_predictions(
            X=X, y=y, feature_names=feature_names,
            build_model=builder,
            n_folds=n_folds, embargo=embargo, min_train=min_train,
        )
        per_model_oof[name] = preds
        per_model_idx[name] = indices

    # Align on common indices (all models must have predicted the same samples)
    # Drop models that produced zero OOF predictions
    active_models = {name: preds for name, preds in per_model_oof.items() if len(preds) > 0}
    if not active_models:
        return np.array([]), np.array([], dtype=int)

    if len(active_models) < len(per_model_oof):
        dropped = set(per_model_oof.keys()) - set(active_models.keys())
        logger.warning(f"OOF ensemble: dropping failed models {dropped}, using {list(active_models.keys())}")

    # Intersection only over the models that actually ran
    names = list(active_models.keys())
    common_idx = per_model_idx[names[0]]
    for name in names[1:]:
        common_idx = np.intersect1d(common_idx, per_model_idx[name])

    if len(common_idx) == 0:
        return np.array([]), np.array([], dtype=int)

    # Weighted average over common indices (only active models contribute)
    total_weight = sum(weights.get(n, 0) for n in active_models.keys())
    if total_weight == 0:
        total_weight = 1.0  # degenerate safety
    ensemble_preds = np.zeros(len(common_idx), dtype=np.float64)
    for name, preds in active_models.items():
        idx = per_model_idx[name]
        mask = np.isin(idx, common_idx)
        aligned_preds = preds[mask]
        # Re-order to match common_idx ordering
        order = np.argsort(idx[mask])
        aligned_preds = aligned_preds[order]
        w = weights.get(name, 0) / total_weight
        ensemble_preds += w * aligned_preds

    return ensemble_preds, np.sort(common_idx)
