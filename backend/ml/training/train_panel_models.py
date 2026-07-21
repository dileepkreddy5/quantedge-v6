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


def cross_sectional_rank_ic(dates: np.ndarray, preds: np.ndarray, y: np.ndarray,
                            horizon_days: int = 0) -> Tuple[float, float, int]:
    """Honest cross-sectional rank-IC with OVERLAP CORRECTION.
    For each date with >=5 names, Spearman-rank predictions vs realized returns.
    CRITICAL: for long horizons, overlapping forward-return windows are NOT
    independent — using all dates inflates IC and t-stats spuriously. We therefore
    only keep validation dates spaced >= horizon_days apart, so each measured IC
    comes from a non-overlapping (independent) forward window. This gives the
    honest number, which for long horizons will be lower and noisier — that's the
    truth of the data, not a bug."""
    uniq = np.sort(np.unique(dates))
    # enforce non-overlapping spacing for long horizons
    if horizon_days and horizon_days > 5:
        kept = []
        last = None
        for d in uniq:
            if last is None or (pd.Timestamp(d) - pd.Timestamp(last)).days >= horizon_days:
                kept.append(d); last = d
        uniq = np.array(kept)
    ics = []
    for d in uniq:
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

    HORIZONS = [5, 10, 21, 63, 126, 252]
    HORIZON_LABELS = {5:"1wk", 10:"2wk", 21:"1mo", 63:"3mo", 126:"6mo", 252:"1yr"}

    # Shared date-based walk-forward split (same split across horizons for comparability)
    dates_all_full = df["date"].values
    unique_dates = np.sort(df["date"].unique())
    split_idx = int(len(unique_dates) * 0.75)
    split_date = unique_dates[split_idx]
    logger.info(f"Split date: {pd.Timestamp(split_date).date()}")

    # Save shared feature distribution once (same features across horizons)
    raw_cols = [c.replace("_csrank", "") for c in csrank_cols]
    dist = {}
    for rc in raw_cols:
        if rc in df.columns:
            vals = df[rc].dropna().values.astype(float)
            if len(vals) >= 20:
                dist[rc] = [float(np.percentile(vals, p)) for p in range(0, 101)]
    (OUT_DIR / "feature_distribution.json").write_text(json.dumps(dist))
    (OUT_DIR / "feature_names.json").write_text(json.dumps(csrank_cols, indent=2))
    logger.info(f"Saved training distribution for {len(dist)} raw features")

    horizon_reports = {}
    shap_drivers_21d = []

    for h in HORIZONS:
        label_col = f"label_{h}d"
        if label_col not in df.columns:
            logger.warning(f"{label_col} not in panel, skipping horizon {h}")
            continue
        sub = df.dropna(subset=[label_col]).copy()
        if len(sub) < 200:
            logger.warning(f"Horizon {h}: only {len(sub)} rows, skipping")
            continue
        X_all = sub[csrank_cols].fillna(0.5).values.astype(np.float64)
        y_all = sub[label_col].values.astype(np.float64)
        dates_all = sub["date"].values
        train_mask = dates_all < split_date
        val_mask = dates_all >= split_date
        if train_mask.sum() < 100 or val_mask.sum() < 30:
            logger.warning(f"Horizon {h}: insufficient train/val ({train_mask.sum()}/{val_mask.sum()}), skipping")
            continue
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_val, y_val = X_all[val_mask], y_all[val_mask]
        dates_val = dates_all[val_mask]

        xgb = XGBoostPredictor(target_horizon=h)
        xgb_fit = xgb.fit(X_train, y_train, csrank_cols, X_val=X_val, y_val=y_val)
        xgb_val = xgb.predict(X_val)
        xgb_ic, _, _ = cross_sectional_rank_ic(dates_val, xgb_val, y_val, horizon_days=h)

        lgb = LightGBMPredictor(target_horizon=h)
        lgb.fit(X_train, y_train, csrank_cols, X_val=X_val, y_val=y_val)
        lgb_val = lgb.predict(X_val)
        lgb_ic, _, _ = cross_sectional_rank_ic(dates_val, lgb_val, y_val, horizon_days=h)

        ens_val = 0.5 * xgb_val + 0.5 * lgb_val
        ens_ic, ens_ic_std, ens_nd = cross_sectional_rank_ic(dates_val, ens_val, y_val, horizon_days=h)

        # hit rate + t-stat
        # non-overlapping dates for honest hit-rate / t-stat
        _uniq = np.sort(np.unique(dates_val)); _kept=[]; _last=None
        for _d in _uniq:
            if _last is None or (pd.Timestamp(_d)-pd.Timestamp(_last)).days >= h:
                _kept.append(_d); _last=_d
        date_ics = []
        for d in _kept:
            m = dates_val == d
            if m.sum() < 5 or np.std(ens_val[m]) == 0: continue
            rho, _ = spearmanr(ens_val[m], y_val[m])
            if np.isfinite(rho): date_ics.append(rho)
        hit_rate = float(np.mean([1 if x > 0 else 0 for x in date_ics])) if date_ics else 0.0
        t_stat = float(np.mean(date_ics) / (np.std(date_ics) / np.sqrt(len(date_ics)))) if len(date_ics) > 1 and np.std(date_ics) > 0 else 0.0

        joblib.dump(xgb, OUT_DIR / f"xgb_{h}d.joblib")
        joblib.dump(lgb, OUT_DIR / f"lgb_{h}d.joblib")

        horizon_reports[str(h)] = {
            "horizon_label": HORIZON_LABELS[h],
            "oos_rank_ic": {"xgboost": round(xgb_ic,4), "lightgbm": round(lgb_ic,4), "ensemble": round(ens_ic,4)},
            "ic_std": round(ens_ic_std,4), "n_val_dates": ens_nd,
            "ic_hit_rate": round(hit_rate,3), "ic_t_stat": round(t_stat,2),
            "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()),
        }
        logger.info(f"  [{HORIZON_LABELS[h]:>4} / {h:3}d] ensemble rank-IC {ens_ic:+.4f} | hit {hit_rate:.0%} | t {t_stat:+.2f} | n_val {val_mask.sum()}")

        # SHAP from the 21d (primary) model
        if h == 21:
            try:
                _, sd = xgb.predict_with_shap(X_val[:200])
                allshap = {**sd.get("top_bullish_drivers", {}), **sd.get("top_bearish_drivers", {})}
                top = sorted(allshap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:15]
                shap_drivers_21d = [{"feature": k.replace("_csrank",""), "impact": round(float(v),6)} for k, v in top]
            except Exception as e:
                logger.warning(f"SHAP failed: {e}")

    report = {
        "trained_at": datetime.now().isoformat(),
        "panel": str(panel_path.name),
        "n_tickers": int(df["ticker"].nunique()),
        "split_date": str(pd.Timestamp(split_date).date()),
        "n_features": len(csrank_cols),
        "horizons": horizon_reports,
        "shap_top_drivers": shap_drivers_21d,
        "interpretation": (
            "Multi-horizon cross-sectional gradient-boosted ensemble. Each horizon (1wk-1yr) "
            "independently trained + validated out-of-sample. Rank-IC >0.03 useful, >0.05 strong. "
            "Longer horizons typically show higher IC as fundamentals dominate."
        ),
    }
    (OUT_DIR / "training_report.json").write_text(json.dumps(report, indent=2))

    logger.info("=" * 64)
    logger.info("MULTI-HORIZON PANEL TRAINING COMPLETE")
    for h in HORIZONS:
        r = horizon_reports.get(str(h))
        if r: logger.info(f"  {r['horizon_label']:>4}: rank-IC {r['oos_rank_ic']['ensemble']:+.4f} | hit {r['ic_hit_rate']:.0%}")
    logger.info(f"  Top drivers (21d): {', '.join(d['feature'] for d in shap_drivers_21d[:5])}")
    logger.info(f"  Models saved to: {OUT_DIR}")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
