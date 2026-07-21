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
        Path("/app/ml_models/panel"),
        Path(os.environ.get("MODEL_DIR", "/app/ml_models")) / "panel",
        Path("/app/models/panel"),
        Path("./ml_models/panel"),
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
            if not (xgb_p.exists() and lgb_p.exists() and feat_p.exists()):
                logger.warning("Panel models not found — predictor unavailable until trained")
                return False
            self.xgb = joblib.load(xgb_p)
            self.lgb = joblib.load(lgb_p)
            self.feature_names = json.loads(feat_p.read_text())
            if rep_p.exists():
                self.report = json.loads(rep_p.read_text())
            dist_p = PANEL_MODEL_DIR / "feature_distribution.json"
            if dist_p.exists():
                self.distribution = json.loads(dist_p.read_text())
            self.loaded = True
            logger.info(f"Panel models loaded: {len(self.feature_names)} features, "
                        f"OOS rank-IC {self.report.get('oos_rank_ic',{}).get('ensemble','?')}")
            return True
        except Exception as e:
            logger.warning(f"Panel model load failed: {e}")
            return False

    def available(self) -> bool:
        return self.loaded and self.xgb is not None

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
        try:
            xgb_pred = float(self.xgb.predict(X)[0])
            lgb_pred = float(self.lgb.predict(X)[0])
            ens = 0.5 * xgb_pred + 0.5 * lgb_pred
            # SHAP drivers for this prediction
            drivers = []
            try:
                _, sd = self.xgb.predict_with_shap(X)
                allshap = {**sd.get("top_bullish_drivers", {}), **sd.get("top_bearish_drivers", {})}
                top = sorted(allshap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
                drivers = [{"feature": k.replace("_csrank", ""), "impact": round(float(v), 5)} for k, v in top]
            except Exception:
                pass
            ic = self.report.get("oos_rank_ic", {})
            return {
                "available": True,
                "pred_21d_pct": round(ens * 100, 3),
                "xgb_pred_pct": round(xgb_pred * 100, 3),
                "lgb_pred_pct": round(lgb_pred * 100, 3),
                "model_agreement": round(1.0 - abs(xgb_pred - lgb_pred) / (abs(ens) + 1e-6), 3) if ens != 0 else None,
                "shap_drivers": drivers,
                "oos_rank_ic": ic.get("ensemble"),
                "ic_hit_rate": self.report.get("ic_hit_rate"),
                "trained_at": self.report.get("trained_at"),
                "n_train": self.report.get("n_train"),
                "methodology": "Cross-sectional gradient-boosted ensemble trained on a rolling universe panel with point-in-time fundamentals. Predicts 21-day forward return.",
            }
        except Exception as e:
            logger.warning(f"Panel predict failed: {e}")
            return None
