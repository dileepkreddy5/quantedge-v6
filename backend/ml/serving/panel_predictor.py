"""
Panel Predictor — serves cross-sectional ML predictions for ANY US ticker.
Loads pre-trained panel models (XGBoost + LightGBM, trained on the rolling
whole-universe panel) once, then scores any searched ticker against them.

This solves the 81-sample problem: instead of training a fresh model per ticker
at request time (too little data → flat predictions), we load models trained on
the full universe (~35k samples) and apply them to the searched ticker's features.

Works for any US-listed ticker, any time (rolling window is relative to today).
"""
from __future__ import annotations
import os, json, logging
from pathlib import Path
from datetime import date
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger("panel_predictor")
# Panel models live in ml_models/panel. We check a few candidate roots because
# the container's MODEL_DIR env may point elsewhere (used for other model types).
def _find_panel_dir() -> Path:
    candidates = [
        Path(os.environ.get("MODEL_DIR", "/app/models")) / "panel",
        Path("/app/models/panel"),
        Path("/app/ml_models/panel"),
        Path("./ml_models/panel"),
        Path.home() / "Desktop/QuantEdge_V6/ml_models/panel",
    ]
    for c in candidates:
        if (c / "xgb_model.joblib").exists():
            return c
    return candidates[0]
PANEL_MODEL_DIR = _find_panel_dir()


class PanelPredictor:
    """Singleton-style loader for the trained cross-sectional panel models."""
    _instance = None

    def __init__(self):
        self.xgb = None
        self.lgb = None
        self.feature_names: List[str] = []
        self.report: Dict = {}
        self.distribution: Dict = {}
        self.horizons: Dict = {}
        self.loaded = False

    @classmethod
    def get(cls) -> "PanelPredictor":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load()
        return cls._instance

    def load(self) -> bool:
        try:
            import joblib
            xgb_p = PANEL_MODEL_DIR / "xgb_model.joblib"
            lgb_p = PANEL_MODEL_DIR / "lgb_model.joblib"
            feat_p = PANEL_MODEL_DIR / "feature_names.json"
            rep_p = PANEL_MODEL_DIR / "training_report.json"
            has_horizon = any((PANEL_MODEL_DIR / f"xgb_{h}d.joblib").exists() for h in [5,10,21,63,126,252])
            if not (has_horizon or (xgb_p.exists() and lgb_p.exists())) or not feat_p.exists():
                logger.warning("Panel models not found — predictor unavailable until trained")
                return False
            # Load per-horizon models: xgb_{h}d.joblib / lgb_{h}d.joblib
            self.horizons = {}
            for h in [5, 10, 21, 63, 126, 252]:
                xp = PANEL_MODEL_DIR / f"xgb_{h}d.joblib"
                lp = PANEL_MODEL_DIR / f"lgb_{h}d.joblib"
                if xp.exists() and lp.exists():
                    self.horizons[h] = (joblib.load(xp), joblib.load(lp))
            # Back-compat: if only the old single 21d model exists
            if not self.horizons and xgb_p.exists() and lgb_p.exists():
                self.horizons[21] = (joblib.load(xgb_p), joblib.load(lgb_p))
            self.feature_names = json.loads(feat_p.read_text())
            if rep_p.exists():
                self.report = json.loads(rep_p.read_text())
            dist_p = PANEL_MODEL_DIR / "feature_distribution.json"
            if dist_p.exists():
                self.distribution = json.loads(dist_p.read_text())
            # An XGBoost model trained under 3.x and loaded under 2.x restores
            # base_score from the 2.x default of 0.5 rather than the fitted value
            # (~0.003 for forward returns). Predictions come back ~0.497 too high
            # with no exception raised — a 25% one-week forecast that looks like a
            # model output. Refuse the artifact instead of serving it.
            _env = (self.report or {}).get("environment") or {}
            _trained = _env.get("xgboost")
            if _trained:
                import xgboost as _x
                if _trained.split(".")[0] != _x.__version__.split(".")[0]:
                    logger.error(
                        f"Panel models trained with xgboost {_trained} but running "
                        f"{_x.__version__} — major version mismatch changes base_score "
                        f"handling and silently offsets every prediction. Panel disabled.")
                    self.loaded = False
                    self.horizons = {}
                    return False
            else:
                logger.warning("Panel report has no environment stamp — cannot verify "
                               "the training library versions match this runtime.")
            self.loaded = len(self.horizons) > 0
            logger.info(f"Panel models loaded: {len(self.feature_names)} features, "
                        f"trained with xgboost {_trained or 'unknown'}, "
                        f"OOS rank-IC {self.report.get('oos_rank_ic',{}).get('ensemble','?')}")
            return True
        except Exception as e:
            logger.warning(f"Panel model load failed: {e}")
            return False

    def available(self) -> bool:
        return self.loaded and len(self.horizons) > 0

    def predict(self, feature_dict: Dict[str, float]) -> Optional[Dict]:
        """Predict for one ticker given its computed features (raw feature dict).
        The panel models expect CROSS-SECTIONAL RANK features (_csrank). Since a
        single ticker has no cross-section, we map its raw features onto the
        training distribution: each feature's value is converted to its percentile
        within the training population (stored at train time). If that mapping
        isn't available, we fall back to the raw feature ranked at 0.5 (neutral)."""
        if not self.available():
            return None
        # Build the model input vector in the exact feature order.
        # feature_names are '<raw>_csrank'; strip suffix to look up the raw value.
        vec = []
        for fn in self.feature_names:
            raw_key = fn[:-7] if fn.endswith("_csrank") else fn
            v = feature_dict.get(raw_key)
            if v is None:
                vec.append(0.5)  # feature unavailable -> neutral rank
            else:
                # Map the raw value to its percentile [0,1] within the training
                # distribution — this is the correct cross-sectional rank for a
                # single searched ticker (ranked against what the model learned on).
                pcts = self.distribution.get(raw_key)
                if pcts and len(pcts) == 101:
                    # binary-search the percentile position
                    import bisect
                    pos = bisect.bisect_left(pcts, float(v))
                    vec.append(min(1.0, max(0.0, pos / 100.0)))
                else:
                    vec.append(0.5)
        X = np.array(vec, dtype=np.float64).reshape(1, -1)
        H_LABELS = {5:"1wk", 10:"2wk", 21:"1mo", 63:"3mo", 126:"6mo", 252:"1yr"}
        try:
            horizon_preds = {}
            hz_report = self.report.get("horizons", {})
            for h, (xm, lm) in sorted(self.horizons.items()):
                xp = float(xm.predict(X)[0]); lp = float(lm.predict(X)[0])
                ens = 0.5 * xp + 0.5 * lp
                rep_h = hz_report.get(str(h), {})
                horizon_preds[f"{h}d"] = {
                    "label": H_LABELS.get(h, f"{h}d"),
                    "pred_pct": round(ens * 100, 3),
                    "xgb_pct": round(xp * 100, 3),
                    "lgb_pct": round(lp * 100, 3),
                    "agreement": round(1.0 - abs(xp - lp) / (abs(ens) + 1e-6), 3) if ens != 0 else None,
                    "oos_rank_ic": rep_h.get("oos_rank_ic", {}).get("ensemble"),
                    "ic_hit_rate": rep_h.get("ic_hit_rate"),
                    "ic_t_stat": rep_h.get("ic_t_stat"),
                    "n_independent_val_dates": rep_h.get("n_independent_val_dates"),
                    "reliable": rep_h.get("reliable"),
                    "confidence_note": rep_h.get("confidence_note"),
                    "n_train": rep_h.get("n_train"),
                    "n_val": rep_h.get("n_val"),
                }
            # SHAP drivers from the 21d model (primary)
            drivers = []
            if 21 in self.horizons:
                try:
                    _, sd = self.horizons[21][0].predict_with_shap(X)
                    allshap = {**sd.get("top_bullish_drivers", {}), **sd.get("top_bearish_drivers", {})}
                    top = sorted(allshap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
                    drivers = [{"feature": k.replace("_csrank", ""), "impact": round(float(v), 5)} for k, v in top]
                except Exception:
                    pass
            primary = horizon_preds.get("21d", {})
            return {
                "available": True,
                "horizons": horizon_preds,
                "pred_21d_pct": primary.get("pred_pct"),
                "model_agreement": primary.get("agreement"),
                "shap_drivers": drivers,
                "oos_rank_ic": primary.get("oos_rank_ic"),
                "trained_at": self.report.get("trained_at"),
                "n_tickers_trained": self.report.get("n_tickers"),
                "methodology": "Multi-horizon cross-sectional gradient-boosted ensemble on a rolling universe panel with point-in-time fundamentals. Separate validated model per horizon (1wk-1yr).",
            }
        except Exception as e:
            logger.warning(f"Panel predict failed: {e}")
            return None
