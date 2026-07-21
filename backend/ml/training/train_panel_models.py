"""
QuantEdge v6.0 — Cross-Sectional Panel Trainer
================================================
Trains XGBoost + LightGBM on the cross-sectional panel (35k+ rows) with proper
date-based walk-forward validation, then computes the HONEST quant skill metric:
cross-sectional out-of-sample rank-IC (per-date Spearman of predictions vs realized,
averaged across validation dates).

This is the metric that actually means something: "on dates the model never saw,
how well did its ranking of stocks predict their forward-return ranking?"

Outputs to MODEL_DIR/panel/:
  - xgb_model.joblib, lgb_model.joblib, scaler.joblib
  - feature_names.json (the cross-sectional-rank features used)
  - training_report.json (OOS rank-IC by horizon, hit rate, IC decay, SHAP top drivers)

Run:  python -m ml.training.train_panel_models
"""
from __future__ import annotations
import os, sys, json, glob, logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_panel")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./ml_models"))
PANEL_DIR = MODEL_DIR / "panels"
OUT_DIR = MODEL_DIR / "panel"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _latest_panel() -> Path:
    panels = sorted(glob.glob(str(PANEL_DIR / "panel_*.parquet")))
    if not panels:
        raise FileNotFoundError(f"No panel found in {PANEL_DIR}. Run build_panel first.")
    return Path(panels[-1])


def cross_sectional_rank_ic(dates: np.ndarray, preds: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    """The honest quant metric. For each date with >=5 names, Spearman-rank
    predictions vs realized returns; average across dates.
    Returns (mean_rank_ic, ic_std, n_dates_used)."""
    ics = []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < 5:
            continue
        p, r = preds[m], y[m]
        if np.std(p) == 0 or np.std(r) == 0:
            continue
        rho, _ = spearmanr(p, r)
        if np.isfinite(rho):
            ics.append(rho)
    if not ics:
        return 0.0, 0.0, 0
    return float(np.mean(ics)), float(np.std(ics)), len(ics)


def main():
    from ml.models.xgboost_lgbm import XGBoostPredictor, LightGBMPredictor
    import joblib

    panel_path = _latest_panel()
    logger.info(f"Loading panel: {panel_path}")
    df = pd.read_parquet(panel_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"Panel: {len(df)} rows, {df['ticker'].nunique()} tickers, {df['date'].nunique()} dates")

    # Use cross-sectional-rank features (the relative-value signal) as primary inputs
    csrank_cols = [c for c in df.columns if c.endswith("_csrank")]
    # Drop dead features: constant or all-zero raw columns carry no signal and
    # dilute the model (e.g. unfilled fundamental placeholders).
    _keep = []
    for c in csrank_cols:
        raw = c.replace("_csrank", "")
        if raw in df.columns and df[raw].nunique(dropna=True) > 3 and df[raw].abs().sum() > 0:
            _keep.append(c)
    dropped = len(csrank_cols) - len(_keep)
    csrank_cols = _keep
    logger.info(f"Dropped {dropped} dead/constant features; using {len(csrank_cols)}")
    logger.info(f"Using {len(csrank_cols)} cross-sectional-rank features")

    # Clean: drop rows with NaN label; fill feature NaNs with 0.5 (neutral rank)
    df = df.dropna(subset=["label"]).copy()
    X_all = df[csrank_cols].fillna(0.5).values.astype(np.float64)
    y_all = df["label"].values.astype(np.float64)
    dates_all = df["date"].values
    tickers_all = df["ticker"].values

    # ── Date-based walk-forward split: train on earlier 75% of DATES, validate on last 25% ──
    unique_dates = np.sort(df["date"].unique())
    split_idx = int(len(unique_dates) * 0.75)
    split_date = unique_dates[split_idx]
    train_mask = dates_all < split_date
    val_mask = dates_all >= split_date
    logger.info(f"Split date: {pd.Timestamp(split_date).date()} | "
                f"train={train_mask.sum()} ({(dates_all<split_date).mean():.0%}), val={val_mask.sum()}")

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    dates_val = dates_all[val_mask]

    # ── Train XGBoost ──
    logger.info("Training XGBoost on panel...")
    xgb = XGBoostPredictor(target_horizon=21)
    xgb_fit = xgb.fit(X_train, y_train, csrank_cols, X_val=X_val, y_val=y_val)
    xgb_val_preds = xgb.predict(X_val)
    xgb_ic, xgb_ic_std, xgb_nd = cross_sectional_rank_ic(dates_val, xgb_val_preds, y_val)
    logger.info(f"XGBoost OOS cross-sectional rank-IC: {xgb_ic:+.4f} (std {xgb_ic_std:.4f}, {xgb_nd} dates)")

    # ── Train LightGBM ──
    logger.info("Training LightGBM on panel...")
    lgb = LightGBMPredictor(target_horizon=21)
    lgb_fit = lgb.fit(X_train, y_train, csrank_cols, X_val=X_val, y_val=y_val)
    lgb_val_preds = lgb.predict(X_val)
    lgb_ic, lgb_ic_std, lgb_nd = cross_sectional_rank_ic(dates_val, lgb_val_preds, y_val)
    logger.info(f"LightGBM OOS cross-sectional rank-IC: {lgb_ic:+.4f} (std {lgb_ic_std:.4f}, {lgb_nd} dates)")

    # ── Ensemble rank-IC (50/50) ──
    ens_val_preds = 0.5 * xgb_val_preds + 0.5 * lgb_val_preds
    ens_ic, ens_ic_std, ens_nd = cross_sectional_rank_ic(dates_val, ens_val_preds, y_val)
    logger.info(f"ENSEMBLE OOS cross-sectional rank-IC: {ens_ic:+.4f} (std {ens_ic_std:.4f}, {ens_nd} dates)")

    # ── Hit rate: fraction of val dates with positive rank-IC ──
    date_ics = []
    for d in np.unique(dates_val):
        m = dates_val == d
        if m.sum() < 5: continue
        if np.std(ens_val_preds[m]) == 0: continue
        rho, _ = spearmanr(ens_val_preds[m], y_val[m])
        if np.isfinite(rho): date_ics.append(rho)
    hit_rate = float(np.mean([1 if x > 0 else 0 for x in date_ics])) if date_ics else 0.0
    ic_t_stat = float(np.mean(date_ics) / (np.std(date_ics) / np.sqrt(len(date_ics)))) if len(date_ics) > 1 and np.std(date_ics) > 0 else 0.0

    # ── SHAP top drivers (real, on trained model) ──
    shap_drivers = []
    try:
        _, shap_dict = xgb.predict_with_shap(X_val[:200])
        allshap = {**shap_dict.get("top_bullish_drivers", {}), **shap_dict.get("top_bearish_drivers", {})}
        top = sorted(allshap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:15]
        shap_drivers = [{"feature": k, "impact": round(float(v), 6)} for k, v in top]
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        # feature-importance fallback
        fi = xgb_fit.get("top10_features", [])
        shap_drivers = [{"feature": k, "impact": round(float(v), 6)} for k, v in fi[:15]]

    # ── Persist models ──
    joblib.dump(xgb, OUT_DIR / "xgb_model.joblib")
    joblib.dump(lgb, OUT_DIR / "lgb_model.joblib")
    (OUT_DIR / "feature_names.json").write_text(json.dumps(csrank_cols, indent=2))

    # Save the RAW feature training distribution so serving can map a single
    # ticker's raw feature value to its percentile within the training population.
    # This is what makes single-ticker cross-sectional prediction correct: the
    # searched ticker is ranked against the distribution the model learned on.
    raw_cols = [c.replace("_csrank", "") for c in csrank_cols]
    dist = {}
    for rc in raw_cols:
        if rc in df.columns:
            vals = df[rc].dropna().values.astype(float)
            if len(vals) >= 20:
                # store percentiles 0..100 for fast lookup
                dist[rc] = [float(np.percentile(vals, p)) for p in range(0, 101)]
    (OUT_DIR / "feature_distribution.json").write_text(json.dumps(dist))
    logger.info(f"Saved training distribution for {len(dist)} raw features")

    report = {
        "trained_at": datetime.now().isoformat(),
        "panel": str(panel_path.name),
        "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()),
        "n_tickers": int(df["ticker"].nunique()),
        "split_date": str(pd.Timestamp(split_date).date()),
        "n_features": len(csrank_cols),
        "oos_rank_ic": {
            "xgboost": round(xgb_ic, 4), "lightgbm": round(lgb_ic, 4), "ensemble": round(ens_ic, 4),
            "ensemble_std": round(ens_ic_std, 4), "n_val_dates": ens_nd,
        },
        "ic_hit_rate": round(hit_rate, 3),
        "ic_t_stat": round(ic_t_stat, 2),
        "shap_top_drivers": shap_drivers,
        "interpretation": (
            f"Ensemble cross-sectional rank-IC of {ens_ic:+.4f} on {ens_nd} out-of-sample dates. "
            f"IC>0.03 is useful, >0.05 is strong for cross-sectional equity signals. "
            f"Positive on {hit_rate:.0%} of validation dates (t-stat {ic_t_stat:.1f})."
        ),
    }
    (OUT_DIR / "training_report.json").write_text(json.dumps(report, indent=2))

    logger.info("=" * 64)
    logger.info("PANEL TRAINING COMPLETE")
    logger.info(f"  Ensemble OOS rank-IC: {ens_ic:+.4f} (std {ens_ic_std:.4f})")
    logger.info(f"  IC hit rate: {hit_rate:.0%} of {ens_nd} dates | t-stat: {ic_t_stat:.2f}")
    logger.info(f"  Top SHAP drivers: {', '.join(d['feature'].replace('_csrank','') for d in shap_drivers[:5])}")
    logger.info(f"  Models saved to: {OUT_DIR}")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
