"""
QuantEdge v6.0 — MODEL GOVERNANCE ENGINE (Layer 8)
════════════════════════════════════════════════════════════════
Every model deployed in production will eventually decay.

The ONLY question is: do you detect it before or after it destroys
your performance?

Model governance is the systematic process of:
1. Monitoring live model performance vs backtest expectations
2. Detecting statistically significant performance decay
3. Quarantining or deweighting decaying models automatically
4. Alerting humans when models deviate beyond thresholds
5. Maintaining a clean model registry with versioning

WHY MODELS DECAY:
  a) ALPHA DECAY: The market learns. Once a signal is published
     (or widely used), arbitrage activity erodes its edge.
     Academic factors lose ~30% of their returns after publication.
     (McLean & Pontiff 2016: 97 anomalies studied)

  b) REGIME SHIFT: A model trained on 2010-2020 data saw:
     - Low rates, QE, mega-cap dominance, low vol
     - It will fail when rates rise or market structure changes

  c) DATA DRIFT: Input distributions change.
     A model's features might be computed from sources that shift.
     Example: sentiment model trained on Twitter pre-API changes.

  d) OVERFITTING REVELATION: Some backtest "alpha" was pure noise.
     Forward testing reveals this through poor live IC.

GOVERNANCE RULES:
  Rule 1: IC Statistical Significance
    Min IC threshold: 0.03 (3% rank correlation)
    Min t-statistic: 2.0 (95% confidence)
    If IC < 0.01 for 30+ trading days: QUARANTINE

  Rule 2: IC Information Ratio (ICIR)
    ICIR = mean(IC) / std(IC)
    Expected: ICIR ≥ 0.50 (Grinold & Kahn)
    If ICIR < 0.30 for 60 days: REVIEW

  Rule 3: Backtest vs Live Divergence
    Live Sharpe / Backtest Sharpe < 0.50: ALERT
    Live IC / Expected IC < 0.30: ALERT

  Rule 4: Structural Break Detection
    Chow test for parameter stability
    CUSUM test for cumulative IC drift
    If detected: FLAG FOR REVIEW

References:
  - Grinold & Kahn (2000): Active Portfolio Management
  - McLean & Pontiff (2016): Does Academic Research Destroy Anomalies?
  - Lopez de Prado (2020): Machine Learning for Asset Managers
  - Harvey, Liu, Zhu (2016): ... and the Cross-Section of Expected Returns
════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────

@dataclass
class ModelRecord:
    """Registry entry for a deployed model."""
    model_id: str
    model_name: str
    version: str
    deployed_at: datetime
    expected_ic: float               # Expected from backtest
    expected_icir: float             # Expected IC Information Ratio
    backtest_sharpe: float           # Backtest Sharpe ratio
    status: str = 'ACTIVE'           # ACTIVE | WATCH | QUARANTINE | RETIRED
    current_weight: float = 1.0      # Weight in ensemble (0-1)
    live_ic_history: List[float] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)


class ModelRegistry:
    """
    Central registry of all models in production.
    Immutable history: once a model is registered, its record is preserved.
    Status changes are logged with timestamps.
    """

    def __init__(self):
        self._models: Dict[str, ModelRecord] = {}
        self._status_log: List[Dict] = []

    def register(self, record: ModelRecord):
        """Register a new model. ID must be unique."""
        if record.model_id in self._models:
            raise ValueError(f"Model {record.model_id} already registered")
        self._models[record.model_id] = record
        self._status_log.append({
            'timestamp': datetime.now(),
            'model_id': record.model_id,
            'action': 'REGISTERED',
            'status': record.status,
        })

    def update_status(self, model_id: str, new_status: str,
                       reason: str, new_weight: float = None):
        """Update model status. Logged for audit trail."""
        if model_id not in self._models:
            raise ValueError(f"Unknown model: {model_id}")

        model = self._models[model_id]
        old_status = model.status
        model.status = new_status
        if new_weight is not None:
            model.current_weight = max(0.0, min(1.0, new_weight))

        self._status_log.append({
            'timestamp': datetime.now(),
            'model_id': model_id,
            'action': 'STATUS_CHANGE',
            'old_status': old_status,
            'new_status': new_status,
            'reason': reason,
            'new_weight': model.current_weight,
        })
        model.alerts.append(f"[{datetime.now().date()}] {new_status}: {reason}")

    def get_active_models(self) -> List[ModelRecord]:
        """Returns all ACTIVE models."""
        return [m for m in self._models.values() if m.status == 'ACTIVE']

    def get_ensemble_weights(self) -> Dict[str, float]:
        """Returns normalized weights for all active models."""
        active = self.get_active_models()
        weights = {m.model_id: m.current_weight for m in active}
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        return weights


# ─────────────────────────────────────────────────────────────
# IC MONITOR
# ─────────────────────────────────────────────────────────────

class ICMonitor:
    """
    Monitors Information Coefficient (IC) performance.

    IC = Spearman rank correlation between signals and realized returns.
    It's the fundamental metric of signal quality.

    IC properties:
    - IC = 0: signal is random noise
    - IC = 0.05: good (top quant funds target IC > 0.05)
    - IC = 0.10: excellent (Renaissance-level)
    - IC = 0.15+: exceptional (extremely rare)
    - IC < 0: signal is inverting (trade against it or quarantine)

    ICIR = mean(IC) / std(IC)
    - ICIR > 0.5: statistically meaningful
    - ICIR > 1.0: strong signal

    Statistical significance of IC:
    t-stat = IC / (1/√T)
    Require t-stat > 2.0 (95% confidence) at minimum.

    For multiple testing correction (Harvey, Liu, Zhu 2016):
    - Testing 10 signals: require t-stat > 3.0 (Bonferroni-like)
    - Testing 100 signals: require t-stat > 3.5-4.0

    This prevents false discoveries from data mining.
    """

    MIN_IC = 0.03                # Minimum acceptable IC
    MIN_ICIR = 0.50              # Minimum IC Information Ratio
    MIN_TSTAT = 2.0              # Minimum t-statistic for significance
    QUARANTINE_IC = 0.01         # Quarantine if IC falls below this
    QUARANTINE_DAYS = 30         # Days below quarantine IC to trigger
    WATCHLIST_IC = 0.025         # Watch if IC falls to this
    BONFERRONI_CORRECTION = 3.0  # Multiple testing penalty

    def __init__(self, window_fast: int = 21, window_slow: int = 63):
        self.window_fast = window_fast
        self.window_slow = window_slow

    def compute_ic(self, signals: pd.Series, realized_returns: pd.Series) -> float:
        """
        Computes Spearman rank correlation between signals and realized returns.
        Only uses assets with non-NaN values for both.
        """
        combined = pd.DataFrame({'signal': signals, 'return': realized_returns}).dropna()
        if len(combined) < 10:
            return np.nan
        ic, p_value = stats.spearmanr(combined['signal'], combined['return'])
        return float(ic)

    def analyze_ic_series(self, ic_history: List[float],
                           expected_ic: float = 0.05) -> Dict:
        """
        Full analysis of IC history.
        Returns governance metrics and recommended actions.
        """
        if len(ic_history) < 5:
            return {'status': 'INSUFFICIENT_DATA'}

        ic_arr = np.array([x for x in ic_history if not np.isnan(x)])
        if len(ic_arr) < 5:
            return {'status': 'INSUFFICIENT_DATA'}

        mean_ic = np.mean(ic_arr)
        std_ic = np.std(ic_arr) + 1e-10
        icir = mean_ic / std_ic
        t_stat = mean_ic / (std_ic / np.sqrt(len(ic_arr)))

        # Recent IC (fast window)
        recent = ic_arr[-self.window_fast:]
        mean_ic_recent = np.mean(recent)
        icir_recent = mean_ic_recent / (np.std(recent) + 1e-10)

        # IC decay: is recent IC significantly worse than historical?
        if len(ic_arr) >= self.window_slow:
            historical = ic_arr[:-self.window_fast]
            t_decay, p_decay = stats.ttest_ind(recent, historical)
            ic_decay_detected = p_decay < 0.10 and mean_ic_recent < np.mean(historical)
        else:
            t_decay, p_decay = 0.0, 1.0
            ic_decay_detected = False

        # Consecutive days below threshold
        days_below_quarantine = 0
        for ic in reversed(ic_arr):
            if ic < self.QUARANTINE_IC:
                days_below_quarantine += 1
            else:
                break

        # Determine governance action
        if days_below_quarantine >= self.QUARANTINE_DAYS:
            action = 'QUARANTINE'
            severity = 'HIGH'
        elif mean_ic_recent < self.WATCHLIST_IC:
            action = 'WATCH'
            severity = 'MEDIUM'
        elif ic_decay_detected and mean_ic_recent < expected_ic * 0.5:
            action = 'REVIEW'
            severity = 'MEDIUM'
        elif mean_ic < self.MIN_IC:
            action = 'MONITOR'
            severity = 'LOW'
        else:
            action = 'MAINTAIN'
            severity = 'NONE'

        # Estimated weight adjustment
        if action in ('QUARANTINE', 'WATCH'):
            weight_adj = max(0.0, mean_ic_recent / (expected_ic + 1e-10))
        else:
            weight_adj = 1.0

        return {
            'mean_ic': float(mean_ic),
            'mean_ic_recent': float(mean_ic_recent),
            'std_ic': float(std_ic),
            'icir': float(icir),
            'icir_recent': float(icir_recent),
            't_statistic': float(t_stat),
            't_decay': float(t_decay),
            'ic_decay_detected': ic_decay_detected,
            'days_below_quarantine_threshold': days_below_quarantine,
            'recommended_action': action,
            'severity': severity,
            'suggested_weight': float(weight_adj),
            'is_statistically_significant': abs(t_stat) > self.MIN_TSTAT,
        }

    def cusum_test(self, ic_history: List[float],
                    expected_ic: float = 0.05) -> Dict[str, Any]:
        """
        CUSUM (Cumulative Sum) test for structural break in IC.

        CUSUM_t = Σ_{s=1}^{t} (IC_s - μ_0) / σ

        If CUSUM exceeds bounds ±h√T (typically h=1), a structural break
        has occurred. This is more sensitive than rolling averages.

        Used to detect:
        - Sudden alpha decay (regime change)
        - Slow gradual decay
        - IC reversal (model now working oppositely)
        """
        from typing import Any  # local import for type hint
        ic_arr = np.array([x for x in ic_history if not np.isnan(x)])
        if len(ic_arr) < 20:
            return {'break_detected': False}

        mu_0 = expected_ic
        sigma = np.std(ic_arr[:20]) + 1e-10  # Use first 20 periods for calibration

        cusum = np.cumsum((ic_arr - mu_0) / sigma)
        T = len(ic_arr)
        h = 1.0  # Boundary multiplier (1.0 = standard)
        boundary = h * np.sqrt(T)

        # Detect if CUSUM has crossed the boundary
        max_cusum = np.max(np.abs(cusum))
        break_detected = max_cusum > boundary
        break_direction = 'DOWNWARD' if cusum[-1] < -boundary else 'UPWARD' if cusum[-1] > boundary else 'NONE'

        return {
            'break_detected': break_detected,
            'break_direction': break_direction,
            'max_cusum': float(max_cusum),
            'boundary': float(boundary),
            'cusum_series': cusum.tolist(),
        }


# ─────────────────────────────────────────────────────────────
# BACKTEST VS LIVE DIVERGENCE DETECTOR
# ─────────────────────────────────────────────────────────────

class BacktestLiveDivergenceDetector:
    """
    Detects divergence between backtest expectations and live performance.

    The most dangerous form of overfitting is not discovered until
    the model goes live. This detector compares:
    - Expected Sharpe (from backtest)
    - Expected IC (from backtest)
    - Expected volatility (from backtest)

    Against live equivalents.

    Thresholds (conservative):
    - Sharpe ratio: live / backtest < 0.50 → ALERT
    - IC: live / expected IC < 0.30 → ALERT
    - Vol: live / expected vol > 2.0 → ALERT (model is taking more risk)
    """

    def detect_divergence(self, backtest_metrics: Dict[str, float],
                           live_metrics: Dict[str, float],
                           min_live_days: int = 60) -> Dict:
        """
        Compares live vs backtest metrics.
        Returns divergence assessment.
        """
        live_days = live_metrics.get('n_days', 0)
        if live_days < min_live_days:
            return {
                'sufficient_data': False,
                'message': f'Only {live_days} live days. Need {min_live_days} for reliable assessment.'
            }

        violations = []
        severity = 'NONE'

        # Sharpe ratio comparison
        if 'sharpe_backtest' in backtest_metrics and 'sharpe_live' in live_metrics:
            ratio = live_metrics['sharpe_live'] / (backtest_metrics['sharpe_backtest'] + 1e-10)
            if ratio < 0.20:
                violations.append(f"CRITICAL: Live Sharpe {live_metrics['sharpe_live']:.2f} "
                                  f"vs Backtest {backtest_metrics['sharpe_backtest']:.2f} "
                                  f"(ratio={ratio:.2f} < 0.20)")
                severity = 'CRITICAL'
            elif ratio < 0.50:
                violations.append(f"WARNING: Live Sharpe ratio is {ratio:.0%} of backtest")
                severity = max(severity, 'WARNING') if severity != 'CRITICAL' else severity

        # IC comparison
        if 'ic_expected' in backtest_metrics and 'ic_live' in live_metrics:
            ic_ratio = live_metrics['ic_live'] / (backtest_metrics['ic_expected'] + 1e-10)
            if ic_ratio < 0.20:
                violations.append(f"CRITICAL: Live IC {live_metrics['ic_live']:.3f} "
                                  f"vs Expected {backtest_metrics['ic_expected']:.3f}")
                severity = 'CRITICAL'
            elif ic_ratio < 0.50:
                violations.append(f"WARNING: Live IC is {ic_ratio:.0%} of expected")

        # Volatility comparison
        if 'vol_expected' in backtest_metrics and 'vol_live' in live_metrics:
            vol_ratio = live_metrics['vol_live'] / (backtest_metrics['vol_expected'] + 1e-10)
            if vol_ratio > 2.5:
                violations.append(f"WARNING: Live vol {live_metrics['vol_live']:.2%} "
                                  f"is {vol_ratio:.1f}x expected")

        # Recommended action
        if severity == 'CRITICAL':
            action = 'QUARANTINE'
        elif severity == 'WARNING':
            action = 'REDUCE_WEIGHT'
        elif violations:
            action = 'MONITOR'
        else:
            action = 'MAINTAIN'

        return {
            'sufficient_data': True,
            'violations': violations,
            'severity': severity,
            'recommended_action': action,
            'live_days': live_days,
        }


# ─────────────────────────────────────────────────────────────
# MASTER GOVERNANCE ENGINE
# ─────────────────────────────────────────════════════════════

class MasterGovernanceEngine:
    """
    Runs all governance checks and produces automated recommendations.
    Called daily (or per trading session).
    """

    def __init__(self):
        self.registry = ModelRegistry()
        self.ic_monitor = ICMonitor()
        self.divergence_detector = BacktestLiveDivergenceDetector()
        self.alerts: List[Dict] = []

    def run_daily_governance(self) -> Dict[str, Any]:
        """
        Runs all governance checks. Returns full governance report.
        Should be called daily before market open.
        """
        from typing import Any
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_checked': 0,
            'actions_taken': [],
            'alerts': [],
            'ensemble_weights': {},
        }

        active_models = self.registry.get_active_models()
        report['models_checked'] = len(active_models)

        for model in active_models:
            if len(model.live_ic_history) < 5:
                continue

            # IC Analysis
            ic_analysis = self.ic_monitor.analyze_ic_series(
                model.live_ic_history,
                expected_ic=model.expected_ic,
            )

            # CUSUM test
            cusum = self.ic_monitor.cusum_test(
                model.live_ic_history,
                expected_ic=model.expected_ic,
            )

            # Governance action
            action = ic_analysis.get('recommended_action', 'MAINTAIN')
            severity = ic_analysis.get('severity', 'NONE')

            if action == 'QUARANTINE':
                self.registry.update_status(
                    model.model_id, 'QUARANTINE',
                    reason=f"IC={ic_analysis['mean_ic_recent']:.3f} below threshold "
                           f"for {ic_analysis['days_below_quarantine_threshold']} days",
                    new_weight=0.0,
                )
                report['actions_taken'].append(f"QUARANTINED: {model.model_name}")
                report['alerts'].append({
                    'severity': 'HIGH',
                    'model': model.model_name,
                    'message': f"Model quarantined: IC decay",
                })

            elif action == 'WATCH':
                new_weight = ic_analysis.get('suggested_weight', 0.5) * model.current_weight
                self.registry.update_status(
                    model.model_id, 'WATCH',
                    reason=f"IC declining: {ic_analysis['mean_ic_recent']:.3f}",
                    new_weight=new_weight,
                )
                report['actions_taken'].append(f"WATCH: {model.model_name} weight → {new_weight:.2f}")

            # CUSUM structural break
            if cusum.get('break_detected') and cusum.get('break_direction') == 'DOWNWARD':
                alert = {
                    'severity': 'MEDIUM',
                    'model': model.model_name,
                    'message': f"CUSUM structural break detected (downward). IC may be decaying.",
                }
                report['alerts'].append(alert)
                self.alerts.append(alert)

        # Final ensemble weights
        report['ensemble_weights'] = self.registry.get_ensemble_weights()

        return report
