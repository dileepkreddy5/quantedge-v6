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


def per_date_ics(dates: np.ndarray, preds: np.ndarray, y: np.ndarray,
                 keep_dates=None) -> Tuple[List[float], List]:
    """Per-date Spearman IC across the cross-section. Returns (ics, dates_used)."""
    uniq = np.sort(np.unique(dates)) if keep_dates is None else np.sort(np.asarray(keep_dates))
    ics, used = [], []
    for d in uniq:
        m = dates == d
        if m.sum() < 5:
            continue
        p, r = preds[m], y[m]
        if np.std(p) == 0 or np.std(r) == 0:
            continue
        rho, _ = spearmanr(p, r)
        if np.isfinite(rho):
            ics.append(float(rho)); used.append(d)
    return ics, used


def hac_t_stat(ics: List[float], lag: int) -> Tuple[float, float, bool]:
    """Newey-West (HAC) t-stat for a mean IC series with OVERLAPPING windows.

    The previous approach kept only non-overlapping dates, which at 252d left a
    single observation — one day's Spearman is not an IC, and its t-stat was
    reported as 0.00. The standard econometric treatment is to use every date and
    widen the standard error to absorb the induced autocorrelation (Newey & West
    1987), which yields a valid, conservative t on all of the data.

    lag is the overlap measured in OBSERVATIONS, not calendar days: the panel
    samples every `step` days, so a horizon of h days overlaps ceil(h/step) rows.

    Returns (t_stat, se, reliable). HAC needs the sample to be long relative to
    the lag; below n >= 4*lag the correction itself is unstable, and we say so
    rather than printing a number that looks measured."""
    n = len(ics)
    if n < 3:
        return 0.0, 0.0, False
    a = np.asarray(ics, dtype=float)
    mu = float(a.mean())
    dev = a - mu
    L = max(0, min(int(lag), n - 2))
    gamma0 = float(np.dot(dev, dev) / n)
    var = gamma0
    for l in range(1, L + 1):
        g = float(np.dot(dev[l:], dev[:-l]) / n)
        var += 2.0 * (1.0 - l / (L + 1.0)) * g
    if var <= 0:
        return 0.0, 0.0, False
    se = float(np.sqrt(var / n))
    if se == 0:
        return 0.0, 0.0, False
    return float(mu / se), se, bool(n >= 4 * max(L, 1))


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

    import xgboost as _xgbmod, lightgbm as _lgbmod, sklearn as _skmod
    # Models trained under one XGBoost major version and loaded under another
    # restore base_score differently: a 3.x model auto-fits base_score from the
    # target mean (~0.003 here), while 2.x falls back to its own 0.5 default.
    # Every prediction then sits ~0.497 too high, and nothing raises. Record the
    # versions so PanelPredictor can refuse a mismatched artifact at load.
    _env = {"xgboost": _xgbmod.__version__, "lightgbm": _lgbmod.__version__,
            "sklearn": _skmod.__version__}
    logger.info(f"training environment: {_env}")
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

        # A fixed 50/50 blend assumes both models found the same signal. At 2wk
        # they did not: XGB ranked at -0.129 while LGB ranked at +0.164, i.e.
        # near-opposite orderings, and averaging them gave -0.098 — worse than
        # either component. Weight by IC and give a model that scored negative no
        # weight at all.
        #
        # But weights fit on the same rows the IC is then measured on leak: the
        # blend is chosen knowing which model won there. Split validation in two —
        # weights from the first half, reported skill from the second, which the
        # weighting never saw.
        _vdates = np.sort(np.unique(dates_val))
        _mid = _vdates[len(_vdates) // 2] if len(_vdates) >= 4 else None
        if _mid is not None:
            _fitm, _scorem = dates_val < _mid, dates_val >= _mid
            wx_ic, _, _ = cross_sectional_rank_ic(dates_val[_fitm], xgb_val[_fitm], y_val[_fitm], horizon_days=h)
            wl_ic, _, _ = cross_sectional_rank_ic(dates_val[_fitm], lgb_val[_fitm], y_val[_fitm], horizon_days=h)
        else:
            _fitm = np.zeros(len(dates_val), dtype=bool); _scorem = np.ones(len(dates_val), dtype=bool)
            wx_ic, wl_ic = xgb_ic, lgb_ic
        _wx, _wl = max(wx_ic, 0.0), max(wl_ic, 0.0)
        if _wx + _wl <= 0:
            ens_val = 0.5 * xgb_val + 0.5 * lgb_val   # both useless; nothing to prefer
            _blend_desc = "50/50 (neither model scored positive IC on the weight-fitting half)"
        else:
            ens_val = (_wx * xgb_val + _wl * lgb_val) / (_wx + _wl)
            _blend_desc = f"IC-weighted {_wx/(_wx+_wl):.0%} XGB / {_wl/(_wx+_wl):.0%} LGB (weights fit on the first half of validation)"
        # Per-model diagnostics: do XGB and LGB actually disagree?
        try:
            from scipy.stats import spearmanr as _sp
            _rho, _ = _sp(xgb_val, lgb_val)
            _pred_corr = float(_rho) if np.isfinite(_rho) else None
        except Exception:
            _pred_corr = None
        _per_model_ic = {"xgboost": round(float(xgb_ic), 4),
                         "lightgbm": round(float(lgb_ic), 4)}

        # headline IC is measured only on the held-out scoring half
        ens_ic, ens_ic_std, ens_nd = cross_sectional_rank_ic(
            dates_val[_scorem], ens_val[_scorem], y_val[_scorem], horizon_days=h)

        # Skill is measured on EVERY date in the scoring half, with the standard
        # error widened by Newey-West to absorb the overlap. Dropping overlapping
        # dates instead left 1 observation at 63d/126d/252d and a t-stat of 0.00.
        _sdates = np.sort(np.unique(dates_val[_scorem]))
        date_ics, _ic_dates = per_date_ics(dates_val[_scorem], ens_val[_scorem], y_val[_scorem])
        hit_rate = float(np.mean([1 if x > 0 else 0 for x in date_ics])) if date_ics else 0.0
        _step_days = 5
        if len(_sdates) > 1:
            _gaps = [(pd.Timestamp(_sdates[i+1]) - pd.Timestamp(_sdates[i])).days for i in range(len(_sdates)-1)]
            _step_days = max(1, int(np.median(_gaps)))
        _lag = int(np.ceil(h / _step_days))
        t_stat, _se, _hac_ok = hac_t_stat(date_ics, _lag)
        n_dates_used = len(date_ics)

        joblib.dump(xgb, OUT_DIR / f"xgb_{h}d.joblib")
        joblib.dump(lgb, OUT_DIR / f"lgb_{h}d.joblib")

        # A horizon needs enough INDEPENDENT (non-overlapping) validation dates to
        # trust its IC. With <5 independent dates the number is statistically
        # meaningless (t-stat undefined). We mark those as low-confidence rather
        # than reporting an inflated IC — this is the honest thing to do.
        # 'abs(t_stat) > 0' is true of any non-zero t, so this passed horizons at
        # t=1.53 on 13 windows and t=1.71 on 5. A skill claim needs significance
        # AND enough independent windows for the t to mean anything.
        MIN_T = 2.0
        # A skill claim needs three things: a HAC t past 2, enough dates for the
        # HAC correction itself to be stable (n >= 4*lag), and a positive IC.
        reliable = bool(_hac_ok and abs(t_stat) >= MIN_T and ens_ic > 0)
        horizon_reports[str(h)] = {
            "horizon_label": HORIZON_LABELS[h],
            "oos_rank_ic": {"xgboost": round(xgb_ic,4), "lightgbm": round(lgb_ic,4), "ensemble": round(ens_ic,4)},
            "per_model_ic": _per_model_ic,
            "xgb_lgb_pred_corr": _pred_corr,
            "ic_std": round(ens_ic_std,4),
            "n_independent_val_dates": ens_nd,
            "n_scoring_dates": n_dates_used,
            "hac_lag_obs": _lag,
            "hac_stable": _hac_ok,
            "blend": _blend_desc,
            "ic_hit_rate": round(hit_rate,3), "ic_t_stat": round(t_stat,2),
            "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()),
            "reliable": reliable,
            "confidence_note": (
                f"Validated: held-out rank-IC {ens_ic:+.3f}, Newey-West t={t_stat:+.2f} across {n_dates_used} scoring dates (overlap lag {_lag})."
                if reliable else
                f"Not validated: held-out rank-IC {ens_ic:+.3f}, Newey-West t={t_stat:+.2f} across {n_dates_used} scoring dates. "
                + ("" if _hac_ok else f"The {h}-day window overlaps {_lag} consecutive samples and {n_dates_used} dates is too short for that correction to be stable — this horizon cannot be measured on a 5-year price history. ")
                + (f"Needs |t|>=2 (has {abs(t_stat):.2f}). " if _hac_ok and ens_ic > 0 else "")
                + (f"IC is negative, so the ranking was inverted on held-out data. " if ens_ic <= 0 else "")
                + f"Treat the {HORIZON_LABELS[h]} figure as directional only."),
        }
        _tag = "OK " if reliable else "LOW"
        logger.info(f"  [{HORIZON_LABELS[h]:>4} / {h:3}d] {_tag} held-out rank-IC {ens_ic:+.4f} | NW t {t_stat:+.2f} (lag {_lag}, {n_dates_used} dates, stable={_hac_ok}) | {_blend_desc}")

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
        "environment": _env,
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
        if r:
            tag = "reliable" if r.get("reliable") else "LOW-CONF (insufficient independent windows)"
            logger.info(f"  {r['horizon_label']:>4}: rank-IC {r['oos_rank_ic']['ensemble']:+.4f} | t {r['ic_t_stat']:+.2f} | indep_dates {r['n_independent_val_dates']} | {tag}")
    logger.info(f"  Top drivers (21d): {', '.join(d['feature'] for d in shap_drivers_21d[:5])}")
    logger.info(f"  Models saved to: {OUT_DIR}")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
