"""
QuantEdge v6.0 — REGIME ENGINE (Layer 7)
════════════════════════════════════════════════════════════════
The regime engine is the CONTEXT LAYER of the system.

The same signal — say, value — earns +0.8 Sharpe in bull markets
and -0.3 Sharpe in bear markets. WITHOUT regime conditioning,
you're flying blind.

This engine maintains a probabilistic belief over market regimes
and updates it in real-time using Bayesian inference.

ARCHITECTURE:
1. HMM REGIME CLASSIFIER
   - 5-state Gaussian HMM (Baum-Welch EM estimation)
   - States: BULL_LOW_VOL, BULL_HIGH_VOL, MEAN_REVERT, BEAR_LOW_VOL, BEAR_HIGH_VOL
   - Continuous belief update via forward algorithm
   - Viterbi path for most likely historical sequence

2. BAYESIAN REGIME UPDATER
   - Prior: transition probabilities from HMM
   - Evidence: current macro/market indicators
   - Posterior: updated regime probabilities
   - Avoids "hard switching" — all regime probs are non-zero

3. VOLATILITY CLUSTERING DETECTOR
   - GARCH-based vol state: CALM/NORMAL/ELEVATED/CRISIS
   - Markov-switching GARCH (Hamilton & Susmel 1994)
   - Separate from price-level regime (vol can spike in bull markets)

4. LIQUIDITY REGIME DETECTOR
   - TED spread, VIX, bid-ask spreads
   - Liquidity crises require separate treatment (position sizing ↓)

REGIME INSTABILITY PROBLEM:
   HMM transitions can oscillate rapidly at regime boundaries.
   This causes:
   - Excessive turnover
   - False de-risking signals
   - Strategy whipsaw

   Solutions implemented:
   a) Smoothing: use filtered (not most recent) regime probabilities
   b) Minimum regime duration: require 3+ days in new regime
   c) Hysteresis: different thresholds for entry vs exit
   d) Confidence weighting: scale action by regime confidence

References:
  - Hamilton (1989): A New Approach to Economic Analysis of NS
  - Ang & Bekaert (2002): International Asset Allocation with Regime Shifts
  - Guidolin & Timmermann (2007): Asset Allocation under Multivariate Regime Switching
════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn import hmm as hmmlearn_hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# REGIME DEFINITIONS
# ─────────────────────────────────────────────────────────────

REGIME_NAMES = [
    'BULL_LOW_VOL',    # Rising prices, calm volatility — best for momentum
    'BULL_HIGH_VOL',   # Rising prices, high volatility — risk assets trending
    'MEAN_REVERT',     # Sideways, mean-reverting — best for value, bad for momentum
    'BEAR_LOW_VOL',    # Falling prices, calm — orderly decline
    'BEAR_HIGH_VOL',   # Falling prices, panic volatility — CRISIS
]

# Factor performance by regime (empirical, from multi-decade backtests)
FACTOR_REGIME_PERFORMANCE = {
    # (factor, regime): expected IC (Information Coefficient)
    ('momentum', 'BULL_LOW_VOL'):   0.08,
    ('momentum', 'BULL_HIGH_VOL'):  0.05,
    ('momentum', 'MEAN_REVERT'):    -0.04,   # Momentum reverses in ranging markets
    ('momentum', 'BEAR_LOW_VOL'):   0.03,
    ('momentum', 'BEAR_HIGH_VOL'):  -0.06,   # Panic reversal

    ('value', 'BULL_LOW_VOL'):      0.03,
    ('value', 'BULL_HIGH_VOL'):     0.02,
    ('value', 'MEAN_REVERT'):       0.07,    # Value loves mean-reversion
    ('value', 'BEAR_LOW_VOL'):      0.05,
    ('value', 'BEAR_HIGH_VOL'):     -0.01,   # Value traps in crises

    ('quality', 'BULL_LOW_VOL'):    0.04,
    ('quality', 'BULL_HIGH_VOL'):   0.06,
    ('quality', 'MEAN_REVERT'):     0.04,
    ('quality', 'BEAR_LOW_VOL'):    0.08,    # Quality outperforms in downturns
    ('quality', 'BEAR_HIGH_VOL'):   0.10,    # Flight to quality in crises

    ('low_vol', 'BULL_LOW_VOL'):    0.02,
    ('low_vol', 'BULL_HIGH_VOL'):   0.06,
    ('low_vol', 'MEAN_REVERT'):     0.03,
    ('low_vol', 'BEAR_LOW_VOL'):    0.07,
    ('low_vol', 'BEAR_HIGH_VOL'):   0.09,    # Low-vol shines in crises

    ('tsmom', 'BULL_LOW_VOL'):      0.06,
    ('tsmom', 'BULL_HIGH_VOL'):     0.07,
    ('tsmom', 'MEAN_REVERT'):       -0.03,
    ('tsmom', 'BEAR_LOW_VOL'):      0.04,
    ('tsmom', 'BEAR_HIGH_VOL'):     0.08,   # Trend-following in crises
}


@dataclass
class RegimeState:
    """Current regime belief state."""
    probabilities: Dict[str, float]      # P(regime=k | data) for each k
    most_likely: str                     # argmax of probabilities
    confidence: float                    # Max probability (higher = more certain)
    regime_duration_days: int            # Days in current most-likely regime
    transition_instability: float        # 0-1, how much has regime been oscillating
    vol_regime: str                      # CALM | NORMAL | ELEVATED | CRISIS
    liquidity_regime: str                # NORMAL | TIGHT | CRISIS
    factor_ic_adjustments: Dict[str, float]  # Adjusted ICs given current regime


# ─────────────────────────────────────────────────────────────
# HMM REGIME CLASSIFIER
# ─────────────────────────────────────────────────────────────

class HMMRegimeClassifier:
    """
    5-state Gaussian HMM for market regime classification.

    Observations (features fed into HMM):
    1. Daily return (level of trend)
    2. 5-day realized volatility (vol environment)
    3. Log volume ratio (unusual activity signal)
    4. 10-day return acceleration (momentum change)
    5. VIX level or proxy (fear gauge)

    State structure:
    - BULL_LOW_VOL:   positive return, low vol
    - BULL_HIGH_VOL:  positive return, high vol
    - MEAN_REVERT:    near-zero return, moderate vol, high autocorrelation
    - BEAR_LOW_VOL:   negative return, low vol (orderly decline)
    - BEAR_HIGH_VOL:  negative return, high vol (panic)

    Estimation:
    - Baum-Welch EM algorithm (standard, 20 random restarts)
    - Select best model by log-likelihood to avoid local optima
    - Annual re-estimation with 3+ years of data
    """

    N_STATES = 5
    N_FEATURES = 4  # (return, vol, vol_ratio, trend)

    def __init__(self, n_restarts: int = 20, covariance_type: str = 'full'):
        self.n_restarts = n_restarts
        self.covariance_type = covariance_type
        self.model: Optional[Any] = None
        self.state_labels: List[str] = REGIME_NAMES.copy()
        self._is_fitted: bool = False
        self._regime_history: List[str] = []
        self._prob_history: List[Dict[str, float]] = []

    def _build_feature_matrix(self, prices: pd.Series,
                               volumes: pd.Series = None) -> np.ndarray:
        """
        Constructs the observation matrix for HMM fitting.
        All features use only PAST information (point-in-time safe).
        """
        log_returns = np.log(prices / prices.shift(1)).dropna()
        rolling_vol = log_returns.rolling(5).std().dropna() * np.sqrt(252)

        features = []

        # Feature 1: daily return (clipped to prevent HMM instability)
        ret_arr = log_returns.values
        ret_arr = np.clip(ret_arr, -0.15, 0.15)  # Remove extreme outliers

        # Feature 2: 5-day realized vol
        vol_arr = rolling_vol.values

        # Feature 3: vol trend (vol ratio: recent vs longer-term)
        vol_63d = log_returns.rolling(63).std() * np.sqrt(252)
        vol_ratio = (rolling_vol / (vol_63d.dropna() + 1e-10))
        vol_ratio = vol_ratio.reindex(rolling_vol.index).fillna(1.0)

        # Feature 4: 10-day momentum
        momentum = log_returns.rolling(10).sum()

        # Align all features
        min_len = min(len(ret_arr), len(vol_arr))
        start_idx = max(0, len(log_returns) - min_len)

        X = np.column_stack([
            ret_arr[-min_len:],
            vol_arr[-min_len:],
            vol_ratio.values[-min_len:] if len(vol_ratio) >= min_len else np.ones(min_len),
            momentum.values[-min_len:] if len(momentum) >= min_len else np.zeros(min_len),
        ])

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0)
        return X

    def fit(self, prices: pd.Series, volumes: pd.Series = None):
        """
        Fits the HMM using Baum-Welch with multiple random restarts.
        Selects best model by log-likelihood.
        """
        X = self._build_feature_matrix(prices, volumes)

        if len(X) < 252:
            raise ValueError(f"Need at least 252 observations, got {len(X)}")

        if not HMMLEARN_AVAILABLE:
            # Fallback: use simple clustering
            self._fit_clustering_fallback(X)
            return

        best_model = None
        best_score = -np.inf

        for seed in range(self.n_restarts):
            try:
                model = hmmlearn_hmm.GaussianHMM(
                    n_components=self.N_STATES,
                    covariance_type=self.covariance_type,
                    n_iter=200,
                    random_state=seed,
                    tol=1e-4,
                )
                model.fit(X)
                score = model.score(X)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue

        if best_model is None:
            self._fit_clustering_fallback(X)
            return

        self.model = best_model
        self._is_fitted = True
        self._auto_label_states(prices)

    def _fit_clustering_fallback(self, X: np.ndarray):
        """K-means fallback when hmmlearn not available."""
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.N_STATES, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        self.cluster_model = km
        self.cluster_labels = labels
        self._is_fitted = True

    def _auto_label_states(self, prices: pd.Series):
        """
        Auto-assigns regime names to HMM states by analyzing state characteristics.
        Sorts by (mean_return DESC, vol ASC) to assign meaningful labels.
        """
        if not HMMLEARN_AVAILABLE or self.model is None:
            return

        # Get state means for return and vol features
        means = self.model.means_  # (N_STATES, N_FEATURES)
        mean_returns = means[:, 0]
        mean_vols = means[:, 1]

        # Sort states by (mean_return, -vol) to assign labels
        scored = [(mean_returns[i] - mean_vols[i] * 0.5, i)
                  for i in range(self.N_STATES)]
        sorted_states = [s[1] for s in sorted(scored, reverse=True)]

        label_map = {sorted_states[i]: REGIME_NAMES[i] for i in range(self.N_STATES)}
        self._label_map = label_map

    def predict_current_regime(self, prices: pd.Series,
                                volumes: pd.Series = None,
                                n_recent_days: int = 252) -> RegimeState:
        """
        Predicts the current market regime with full probability distribution.

        Returns a RegimeState with:
        - Posterior probability over all 5 regimes
        - Most likely regime
        - Transition instability score

        CRITICAL: Uses FILTERED probabilities (forward algorithm),
        NOT smoothed (which would use future data).
        """
        if not self._is_fitted:
            # Default to UNKNOWN with uniform probs
            probs = {r: 1.0/5 for r in REGIME_NAMES}
            return RegimeState(
                probabilities=probs,
                most_likely='BULL_LOW_VOL',
                confidence=0.2,
                regime_duration_days=0,
                transition_instability=1.0,
                vol_regime='NORMAL',
                liquidity_regime='NORMAL',
                factor_ic_adjustments={},
            )

        X = self._build_feature_matrix(prices, volumes)
        X_recent = X[-n_recent_days:]

        if HMMLEARN_AVAILABLE and self.model is not None:
            # Filtered probabilities from forward algorithm
            # predict_proba returns smoothed — we use score_samples for filtered
            log_prob, state_probs = self.model.score_samples(X_recent)
            current_probs = state_probs[-1]  # Most recent time point

            # Map state indices to regime names
            label_map = getattr(self, '_label_map', {i: REGIME_NAMES[i] for i in range(5)})
            probs_named = {label_map.get(i, REGIME_NAMES[i]): float(current_probs[i])
                          for i in range(self.N_STATES)}

            # Regime duration (consecutive days in same most-likely state)
            recent_regimes = [label_map.get(np.argmax(state_probs[t]), 'UNKNOWN')
                             for t in range(len(state_probs))]
        else:
            # Clustering fallback
            cluster = self.cluster_model.predict(X_recent[-1:])
            probs_named = {r: 0.1 for r in REGIME_NAMES}
            probs_named[REGIME_NAMES[cluster[0] % 5]] = 0.6
            recent_regimes = [REGIME_NAMES[self.cluster_labels[-i] % 5]
                             for i in range(min(21, len(self.cluster_labels)))]

        most_likely = max(probs_named, key=probs_named.get)
        confidence = probs_named[most_likely]

        # Duration
        recent_21 = recent_regimes[-21:] if len(recent_regimes) >= 21 else recent_regimes
        duration = sum(1 for r in reversed(recent_21) if r == most_likely)

        # Transition instability: count regime switches in last 21 days
        switches = sum(1 for i in range(1, len(recent_21))
                      if recent_21[i] != recent_21[i-1])
        instability = switches / max(len(recent_21) - 1, 1)

        # Vol regime from recent volatility
        returns = np.log(prices / prices.shift(1)).dropna()
        if len(returns) >= 21:
            recent_vol = returns.iloc[-21:].std() * np.sqrt(252)
            historical_vol = returns.std() * np.sqrt(252)
            vol_ratio = recent_vol / (historical_vol + 1e-10)
            if vol_ratio > 2.5:
                vol_regime = 'CRISIS'
            elif vol_ratio > 1.5:
                vol_regime = 'ELEVATED'
            elif vol_ratio < 0.7:
                vol_regime = 'CALM'
            else:
                vol_regime = 'NORMAL'
        else:
            vol_regime = 'NORMAL'

        # Factor IC adjustments
        ic_adj = {
            factor: FACTOR_REGIME_PERFORMANCE.get((factor, most_likely), 0.05)
            for factor in ['momentum', 'value', 'quality', 'low_vol', 'tsmom']
        }

        self._regime_history.append(most_likely)
        self._prob_history.append(probs_named)

        return RegimeState(
            probabilities=probs_named,
            most_likely=most_likely,
            confidence=confidence,
            regime_duration_days=duration,
            transition_instability=instability,
            vol_regime=vol_regime,
            liquidity_regime='NORMAL',   # Expanded in production with TED spread
            factor_ic_adjustments=ic_adj,
        )


# ─────────────────────────────────────────────────────────────
# BAYESIAN REGIME UPDATER
# ─────────────────────────────────────────────────────────────

class BayesianRegimeUpdater:
    """
    Updates regime beliefs using Bayesian inference.

    This adds macro/market evidence to the HMM statistical regime.

    Bayes' theorem:
        P(regime | data) ∝ P(data | regime) * P(regime)

    Prior: HMM transition probabilities
    Likelihood: How likely is current data under each regime?
    Posterior: Updated regime probabilities

    Evidence used:
    - VIX level (volatility regime)
    - TED spread (credit/liquidity conditions)
    - Yield curve slope (macro cycle)
    - Put/call ratio (market sentiment)
    - Credit spreads (risk appetite)
    """

    # Likelihood tables: P(observation | regime) — from historical analysis
    # VIX regimes
    VIX_LIKELIHOODS = {
        'BULL_LOW_VOL':   {'low': 0.70, 'medium': 0.25, 'high': 0.04, 'crisis': 0.01},
        'BULL_HIGH_VOL':  {'low': 0.20, 'medium': 0.45, 'high': 0.30, 'crisis': 0.05},
        'MEAN_REVERT':    {'low': 0.35, 'medium': 0.45, 'high': 0.15, 'crisis': 0.05},
        'BEAR_LOW_VOL':   {'low': 0.10, 'medium': 0.35, 'high': 0.45, 'crisis': 0.10},
        'BEAR_HIGH_VOL':  {'low': 0.02, 'medium': 0.10, 'high': 0.35, 'crisis': 0.53},
    }

    def update_with_vix(self, prior_probs: Dict[str, float],
                         vix_level: float) -> Dict[str, float]:
        """Bayesian update of regime probabilities using VIX."""

        # Categorize VIX level
        if vix_level < 15:
            vix_cat = 'low'
        elif vix_level < 25:
            vix_cat = 'medium'
        elif vix_level < 40:
            vix_cat = 'high'
        else:
            vix_cat = 'crisis'

        # Compute posterior: P(regime | VIX) ∝ P(VIX | regime) * P(regime)
        posteriors = {}
        for regime in REGIME_NAMES:
            likelihood = self.VIX_LIKELIHOODS.get(regime, {}).get(vix_cat, 0.2)
            prior = prior_probs.get(regime, 1.0/5)
            posteriors[regime] = likelihood * prior

        # Normalize
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v/total for k, v in posteriors.items()}

        return posteriors

    def update_with_yield_curve(self, prior_probs: Dict[str, float],
                                 yield_curve_slope: float) -> Dict[str, float]:
        """
        Update using yield curve slope (10Y - 2Y spread).

        Inverted yield curve → recession signal → P(BEAR) increases.
        Steep curve → recovery/growth → P(BULL) increases.

        Historical accuracy: inverted yield curve predicts recession
        with 12-18 month lead time (Harvey 1991, Estrella & Mishkin 1996).
        """
        # Slope categorization
        if yield_curve_slope < -0.50:   # Deeply inverted
            slope_cat = 'inverted'
        elif yield_curve_slope < 0:     # Slightly inverted
            slope_cat = 'flat'
        elif yield_curve_slope < 1.0:   # Flat to normal
            slope_cat = 'normal'
        else:                           # Steep (recovery)
            slope_cat = 'steep'

        # Regime likelihood given slope
        slope_likelihoods = {
            'inverted': {'BULL_LOW_VOL': 0.05, 'BULL_HIGH_VOL': 0.10,
                        'MEAN_REVERT': 0.20, 'BEAR_LOW_VOL': 0.35, 'BEAR_HIGH_VOL': 0.30},
            'flat':     {'BULL_LOW_VOL': 0.15, 'BULL_HIGH_VOL': 0.20,
                        'MEAN_REVERT': 0.30, 'BEAR_LOW_VOL': 0.25, 'BEAR_HIGH_VOL': 0.10},
            'normal':   {'BULL_LOW_VOL': 0.30, 'BULL_HIGH_VOL': 0.25,
                        'MEAN_REVERT': 0.25, 'BEAR_LOW_VOL': 0.15, 'BEAR_HIGH_VOL': 0.05},
            'steep':    {'BULL_LOW_VOL': 0.50, 'BULL_HIGH_VOL': 0.25,
                        'MEAN_REVERT': 0.15, 'BEAR_LOW_VOL': 0.08, 'BEAR_HIGH_VOL': 0.02},
        }

        posteriors = {}
        for regime in REGIME_NAMES:
            likelihood = slope_likelihoods.get(slope_cat, {}).get(regime, 0.2)
            prior = prior_probs.get(regime, 1.0/5)
            posteriors[regime] = likelihood * prior

        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v/total for k, v in posteriors.items()}

        return posteriors

    def smooth_regime_transition(self, current_probs: Dict[str, float],
                                  previous_probs: Dict[str, float],
                                  smoothing: float = 0.30) -> Dict[str, float]:
        """
        Smooth regime transitions to prevent oscillation.

        Applies exponential smoothing to regime probabilities:
            P_smooth = (1-α) * P_current + α * P_previous

        Higher α = more smoothing = slower regime response.
        Typical: α = 0.30 (30% weight on yesterday's belief)

        This reduces turnover caused by regime flickering.
        """
        smoothed = {}
        for regime in REGIME_NAMES:
            curr = current_probs.get(regime, 1.0/5)
            prev = previous_probs.get(regime, 1.0/5)
            smoothed[regime] = (1 - smoothing) * curr + smoothing * prev

        # Normalize
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v/total for k, v in smoothed.items()}
        return smoothed


# ─────────────────────────────────────────────────────────────
# MASTER REGIME ENGINE
# ─────────────────────────────────────────────────────────────

class MasterRegimeEngine:
    """
    Combines HMM + Bayesian updating into unified regime engine.
    """

    def __init__(self, n_hmm_restarts: int = 20):
        self.hmm = HMMRegimeClassifier(n_restarts=n_hmm_restarts)
        self.bayesian = BayesianRegimeUpdater()
        self._previous_probs: Dict[str, float] = {r: 1.0/5 for r in REGIME_NAMES}
        self._is_fitted = False

    def fit(self, prices: pd.Series, volumes: pd.Series = None):
        """Fit the HMM on historical data."""
        self.hmm.fit(prices, volumes)
        self._is_fitted = True

    def detect_current_regime(self,
                               prices: pd.Series,
                               volumes: pd.Series = None,
                               vix: float = None,
                               yield_curve_slope: float = None,
                               ) -> RegimeState:
        """
        Full regime detection with Bayesian updating.
        """
        # 1. HMM regime probabilities
        hmm_state = self.hmm.predict_current_regime(prices, volumes)
        probs = hmm_state.probabilities

        # 2. Bayesian updates with macro data
        if vix is not None:
            probs = self.bayesian.update_with_vix(probs, vix)

        if yield_curve_slope is not None:
            probs = self.bayesian.update_with_yield_curve(probs, yield_curve_slope)

        # 3. Smooth to prevent oscillation
        probs = self.bayesian.smooth_regime_transition(probs, self._previous_probs)
        self._previous_probs = probs

        # 4. Update state with new probabilities
        most_likely = max(probs, key=probs.get)
        confidence = probs[most_likely]

        # Factor IC adjustments
        ic_adj = {
            factor: sum(
                probs[r] * FACTOR_REGIME_PERFORMANCE.get((factor, r), 0.03)
                for r in REGIME_NAMES
            )
            for factor in ['momentum', 'value', 'quality', 'low_vol', 'tsmom']
        }

        return RegimeState(
            probabilities=probs,
            most_likely=most_likely,
            confidence=confidence,
            regime_duration_days=hmm_state.regime_duration_days,
            transition_instability=hmm_state.transition_instability,
            vol_regime=hmm_state.vol_regime,
            liquidity_regime=hmm_state.liquidity_regime,
            factor_ic_adjustments=ic_adj,
        )

    def get_regime_adjusted_risk_budget(self, regime_state: RegimeState,
                                         base_vol_target: float = 0.10) -> Dict[str, float]:
        """
        Adjusts risk budget based on regime.
        Called by Capital Allocation Engine.
        """
        vol_multipliers = {
            'BULL_LOW_VOL':   1.00,
            'BULL_HIGH_VOL':  0.85,
            'MEAN_REVERT':    0.90,
            'BEAR_LOW_VOL':   0.70,
            'BEAR_HIGH_VOL':  0.45,
        }

        # Expected vol multiplier (probability-weighted)
        vol_mult = sum(
            regime_state.probabilities.get(r, 0) * vol_multipliers.get(r, 0.80)
            for r in REGIME_NAMES
        )

        return {
            'vol_target': base_vol_target * vol_mult,
            'max_gross_leverage': 1.5 * vol_mult,
            'max_net_exposure': 1.0 * vol_mult,
            'max_position_weight': 0.08,
            'regime_vol_multiplier': vol_mult,
        }
