"""
QuantEdge v6.0 — Portfolio Construction Engine
================================================
Implements institutional-grade portfolio optimization, completely
SEPARATE from both the alpha engine and the risk engine.

Data flow:
  Alpha Engine → signals (expected returns, confidence)
  Risk Engine  → constraints (CVaR, drawdown, vol limits)
  THIS MODULE  → weights (how to combine signals under constraints)

Algorithms implemented:
  1. Hierarchical Risk Parity (HRP) — Lopez de Prado 2016
  2. CVaR-minimizing optimization — Rockafellar & Uryasev 2000
  3. Risk Budgeting (Equal Risk Contribution)
  4. Regime-aware blending (switches between algorithms by regime)
  5. Turnover penalty (realistic transaction costs)
  6. Liquidity constraints (max % of ADV)

WHY HRP over Mean-Variance Optimization:
  MVO (Markowitz): requires covariance matrix inversion → numerically unstable
  HRP: no matrix inversion → no instability, no "Markowitz curse"
  MVO: concentrates in assets with extreme estimated returns (overfitting)
  HRP: diversifies hierarchically → robust to estimation errors
  MVO: collapses in crisis (all correlations → 1 → concentrated)
  HRP: explicitly handles correlation structure → better crisis behavior

Mathematical framework:
  1. Compute correlation matrix C
  2. Convert to distance: D_ij = √(0.5 × (1 - C_ij))
  3. Hierarchical clustering: Ward/single linkage on D
  4. Recursive bisection: allocate risk by cluster variance
  5. Apply position limits from risk engine
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
# HIERARCHICAL RISK PARITY
# ══════════════════════════════════════════════════════════════
class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity (Lopez de Prado 2016).

    Step 1: DISTANCE MATRIX
      D_ij = √(0.5 × (1 - ρ_ij))
      Maps correlation [-1,1] to distance [0,1]
      ρ = 1 → D = 0 (identical)
      ρ = 0 → D = 0.707 (uncorrelated)
      ρ = -1 → D = 1 (perfectly inverse)

    Step 2: HIERARCHICAL CLUSTERING
      Group similar assets (low distance = high correlation)
      Using Ward's linkage: minimize within-cluster variance
      Result: dendrogram (tree structure of asset relationships)

    Step 3: QUASI-DIAGONAL COVARIANCE
      Reorder covariance matrix so similar assets are adjacent
      Allows recursive bisection to work correctly

    Step 4: RECURSIVE BISECTION (weights)
      - Start: all weight = 1 / N
      - For each cluster pair (left L, right R):
          var_L = w_L^T Σ_L w_L
          var_R = w_R^T Σ_R w_R
          α = var_R / (var_L + var_R)
          w_L *= α  (give less weight to higher-variance cluster)
          w_R *= (1 - α)

    Result: weights that are diversified across correlation clusters.
    No matrix inversion. No expected returns needed. Robust.
    """

    def __init__(self, linkage_method: str = 'single'):
        """
        linkage_method: 'single', 'complete', 'ward', 'average'
        Lopez de Prado uses 'single' (most flexible)
        Ward's is empirically stronger (groups minimizing within-cluster var)
        """
        self.linkage_method = linkage_method

    def _correlation_to_distance(self, corr: np.ndarray) -> np.ndarray:
        """D_ij = √(0.5 × (1 - ρ_ij))"""
        dist = np.sqrt(np.maximum(0.5 * (1 - corr), 0))
        np.fill_diagonal(dist, 0)
        return dist

    def _cluster_assets(self, dist_matrix: np.ndarray, n_assets: int) -> np.ndarray:
        """
        Apply hierarchical clustering to distance matrix.
        Returns sorted asset order (quasi-diagonal reordering).
        """
        # Convert to condensed form for scipy
        condensed = squareform(dist_matrix, checks=False)

        # Hierarchical clustering
        link = linkage(condensed, method=self.linkage_method)

        # Extract leaf order from dendrogram (quasi-diagonal)
        dend = dendrogram(link, no_plot=True)
        sorted_order = dend['leaves']

        return np.array(sorted_order)

    def _get_cluster_variance(self, cov: np.ndarray, items: List[int]) -> float:
        """Variance of an inverse-variance-weighted cluster."""
        sub_cov = cov[np.ix_(items, items)]
        ivp = 1.0 / np.diag(sub_cov)
        ivp /= ivp.sum()
        return float(ivp @ sub_cov @ ivp)

    def _recursive_bisection(
        self, cov: np.ndarray, sorted_items: List[int]
    ) -> np.ndarray:
        """
        Recursive bisection algorithm.
        Allocates weight by splitting clusters based on their variance.
        """
        w = pd.Series(1.0, index=sorted_items)
        cluster_items = [sorted_items]

        while cluster_items:
            new_clusters = []
            for cluster in cluster_items:
                if len(cluster) > 1:
                    # Split in half
                    mid = len(cluster) // 2
                    left = cluster[:mid]
                    right = cluster[mid:]
                    new_clusters.append(left)
                    new_clusters.append(right)

            cluster_items = new_clusters

            # Pairwise rebalancing
            for i in range(0, len(cluster_items) - 1, 2):
                left = cluster_items[i]
                right = cluster_items[i + 1]

                var_l = self._get_cluster_variance(cov, left)
                var_r = self._get_cluster_variance(cov, right)

                alpha = var_r / (var_l + var_r)  # Left gets more weight if lower var

                w[left] *= alpha
                w[right] *= (1 - alpha)

        return w.values

    def compute_weights(
        self,
        returns: pd.DataFrame,
        covariance: Optional[np.ndarray] = None,
        apply_shrinkage: bool = True,
    ) -> pd.Series:
        """
        Compute HRP weights.

        Parameters
        ----------
        returns: T×N DataFrame of historical returns
        covariance: Pre-computed covariance (optional; computed internally if None)
        apply_shrinkage: Apply Ledoit-Wolf shrinkage to covariance

        Returns
        -------
        pd.Series of portfolio weights (sum to 1, all positive for long-only)
        """
        n_assets = returns.shape[1]
        assets = list(returns.columns)

        # Step 1: Covariance matrix
        if covariance is None:
            if apply_shrinkage:
                lw = LedoitWolf()
                lw.fit(returns.values)
                cov = lw.covariance_
            else:
                cov = returns.cov().values
        else:
            cov = covariance

        # Ensure positive definiteness
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Step 2: Correlation matrix → distance matrix
        vols = np.sqrt(np.diag(cov))
        D = np.diag(1.0 / vols)
        corr = D @ cov @ D
        np.clip(corr, -1, 1, out=corr)
        np.fill_diagonal(corr, 1.0)

        dist = self._correlation_to_distance(corr)

        # Step 3: Hierarchical clustering → sorted order
        sorted_idx = self._cluster_assets(dist, n_assets)

        # Step 4: Recursive bisection on quasi-diagonal covariance
        sorted_items = list(sorted_idx)
        weights_raw = self._recursive_bisection(cov, sorted_items)

        # Map back to original asset order
        weights = np.zeros(n_assets)
        for rank, original_idx in enumerate(sorted_items):
            weights[original_idx] = weights_raw[rank]

        # Normalize (should already sum to 1, but numerical safety)
        weights = np.maximum(weights, 0)
        weights /= weights.sum()

        return pd.Series(weights, index=assets)


# ══════════════════════════════════════════════════════════════
# CVAR-MINIMIZING OPTIMIZATION
# ══════════════════════════════════════════════════════════════
class CVaROptimizer:
    """
    CVaR-minimizing portfolio optimization.
    Rockafellar & Uryasev (2000): Optimization of Conditional Value-at-Risk.

    minimize   CVaR_α(w) = VaR_α + 1/((1-α)T) × Σ_t max(0, -r_t^T w - VaR_α)
    subject to:
      Σ w_i = 1 (fully invested)
      w_i ≥ 0 (long-only)
      w_i ≤ w_max (position cap)
      E[r^T w] ≥ μ_target (minimum expected return)
      turnover ≤ turnover_max (transaction cost control)

    Why CVaR vs variance?
    - CVaR minimizes the TAIL risk (what happens when things go wrong)
    - Variance minimization just minimizes all volatility (including upside)
    - CVaR is coherent → respects portfolio diversification
    - CVaR handles fat tails correctly (variance assumes Gaussian)
    """

    def __init__(
        self,
        confidence: float = 0.95,
        max_weight: float = 0.10,
        min_weight: float = 0.0,
        turnover_penalty: float = 0.001,  # Cost per unit of turnover
        return_target: Optional[float] = None,
    ):
        self.alpha = confidence
        self.max_w = max_weight
        self.min_w = min_weight
        self.turnover_penalty = turnover_penalty
        self.return_target = return_target

    def compute_weights(
        self,
        returns: pd.DataFrame,
        prev_weights: Optional[np.ndarray] = None,
        expected_returns: Optional[np.ndarray] = None,
    ) -> pd.Series:
        """
        Solve CVaR minimization.

        Uses scipy minimize with CVaR computed via auxiliary variable method.
        """
        T, N = returns.shape
        R = returns.values

        if prev_weights is None:
            prev_weights = np.ones(N) / N

        # Expected returns (equal if not provided)
        mu = expected_returns if expected_returns is not None else R.mean(axis=0)

        def portfolio_cvar(weights: np.ndarray) -> float:
            """CVaR objective function."""
            port_ret = R @ weights

            # Sort losses
            losses = -port_ret
            sorted_losses = np.sort(losses)[::-1]

            # VaR = alpha-quantile loss
            var_idx = int(self.alpha * T)
            var = sorted_losses[var_idx] if var_idx < T else sorted_losses[-1]

            # CVaR = mean of losses above VaR
            tail_losses = sorted_losses[:var_idx + 1]
            cvar = float(tail_losses.mean())

            # Turnover penalty
            turnover = float(np.sum(np.abs(weights - prev_weights)))
            penalty = self.turnover_penalty * turnover

            return cvar + penalty

        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        if self.return_target is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: float(mu @ w) - self.return_target
            })

        # Bounds
        bounds = [(self.min_w, self.max_w)] * N

        # Optimize
        result = minimize(
            portfolio_cvar,
            x0=prev_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-9},
        )

        if result.success:
            w = np.maximum(result.x, 0)
            w /= w.sum()
        else:
            # Fallback: equal weight
            w = np.ones(N) / N

        return pd.Series(w, index=returns.columns)


# ══════════════════════════════════════════════════════════════
# EQUAL RISK CONTRIBUTION (Risk Parity)
# ══════════════════════════════════════════════════════════════
class EqualRiskContribution:
    """
    Equal Risk Contribution (Risk Parity) portfolio.
    Each asset contributes equally to total portfolio variance.

    Risk contribution of asset i:
    RC_i = w_i × (Σw)_i / (w^T Σ w)

    ERC condition: RC_i = 1/N for all i

    This means:
    - Low-vol assets get MORE weight (they contribute less risk per dollar)
    - High-vol assets get LESS weight
    - Correlations matter: highly correlated assets get less weight

    Bridgewater's "All Weather" fund is essentially risk parity.
    """

    def __init__(self, max_iter: int = 500, tol: float = 1e-8):
        self.max_iter = max_iter
        self.tol = tol

    def compute_weights(
        self,
        covariance: np.ndarray,
        asset_names: List[str],
        risk_budget: Optional[np.ndarray] = None,
    ) -> pd.Series:
        """
        Compute ERC weights via Newton's method.

        Parameters
        ----------
        covariance: N×N covariance matrix
        risk_budget: Desired risk contributions (default: equal 1/N)
        """
        N = covariance.shape[0]

        if risk_budget is None:
            risk_budget = np.ones(N) / N  # Equal risk budget
        else:
            risk_budget = risk_budget / risk_budget.sum()

        # Initial weights: inverse volatility
        vols = np.sqrt(np.diag(covariance))
        w = 1.0 / vols
        w /= w.sum()

        # Newton's method (Maillard, Roncalli, Teiletche 2010)
        for iteration in range(self.max_iter):
            # Current risk contributions
            port_var = float(w @ covariance @ w)
            marginal_rc = covariance @ w  # ∂(port_var)/∂w_i = 2(Σw)_i
            rc = w * marginal_rc / port_var  # Normalized contributions

            # Gradient: how far from equal risk
            gradient = rc - risk_budget

            if np.max(np.abs(gradient)) < self.tol:
                break

            # Jacobian of risk contributions w.r.t. weights (approximation)
            J = (np.diag(marginal_rc) + covariance @ np.diag(w)) / port_var

            # Newton step
            try:
                delta_w = np.linalg.solve(J, gradient)
                step_size = 0.5 / (iteration + 1)  # Diminishing step
                w -= step_size * delta_w
            except np.linalg.LinAlgError:
                break

            # Project to simplex (positive, sum to 1)
            w = np.maximum(w, 1e-6)
            w /= w.sum()

        return pd.Series(w, index=asset_names)


# ══════════════════════════════════════════════════════════════
# REGIME-AWARE PORTFOLIO BLENDER
# ══════════════════════════════════════════════════════════════
class RegimeAwarePortfolioBlender:
    """
    Blends portfolio construction methods based on detected market regime.

    Different regimes favor different portfolio approaches:

    BULL_LOW_VOL:  HRP works well (trends are real, diversify across them)
    BULL_HIGH_VOL: ERC reduces risk (high vol → need risk parity)
    MEAN_REVERT:   CVaR min (protect tail risk in choppy markets)
    BEAR_LOW_VOL:  ERC or min vol (defensive positioning)
    BEAR_HIGH_VOL: CVaR min + max de-risk (protect capital first)

    The blend is weighted by regime probability, not hard-switching.
    Hard-switching creates instability at regime boundaries.
    """

    def __init__(self):
        self.hrp = HierarchicalRiskParity()
        self.cvar_opt = CVaROptimizer()
        self.erc = EqualRiskContribution()

    # Regime → method weights mapping
    REGIME_BLEND = {
        'BULL_LOW_VOL':   {'hrp': 0.5, 'erc': 0.3, 'cvar': 0.2},
        'BULL_HIGH_VOL':  {'hrp': 0.3, 'erc': 0.5, 'cvar': 0.2},
        'MEAN_REVERT':    {'hrp': 0.2, 'erc': 0.3, 'cvar': 0.5},
        'BEAR_LOW_VOL':   {'hrp': 0.2, 'erc': 0.5, 'cvar': 0.3},
        'BEAR_HIGH_VOL':  {'hrp': 0.1, 'erc': 0.3, 'cvar': 0.6},
        'UNKNOWN':        {'hrp': 0.4, 'erc': 0.3, 'cvar': 0.3},
    }

    def compute_weights(
        self,
        returns: pd.DataFrame,
        covariance: np.ndarray,
        regime: str = 'UNKNOWN',
        regime_probabilities: Optional[Dict] = None,
        alpha_signals: Optional[np.ndarray] = None,
        prev_weights: Optional[np.ndarray] = None,
        risk_limits: Optional[Dict] = None,
    ) -> Dict:
        """
        Compute regime-aware portfolio weights.

        Parameters
        ----------
        returns: Historical returns T×N
        covariance: Current covariance estimate
        regime: Current regime label
        regime_probabilities: Soft regime probabilities (weighted blending)
        alpha_signals: Expected returns from alpha engine (optional)
        prev_weights: Previous weights (for turnover calculation)
        risk_limits: Position limits from risk engine

        Returns
        -------
        Dict with weights + diagnostics
        """
        N = returns.shape[1]
        assets = list(returns.columns)
        max_w = (risk_limits or {}).get('max_per_asset', 0.10)

        # Compute all method weights
        hrp_weights = self.hrp.compute_weights(returns, covariance).values
        erc_weights = self.erc.compute_weights(covariance, assets).values

        expected_returns = alpha_signals if alpha_signals is not None else returns.mean().values
        cvar_weights = self.cvar_opt.compute_weights(
            returns,
            prev_weights=prev_weights,
            expected_returns=expected_returns,
        ).values

        # Determine blend weights
        # Use soft probability weighting if available (smoother transitions)
        if regime_probabilities:
            # Weight blend by regime probability
            blended_regime_weights = {'hrp': 0.0, 'erc': 0.0, 'cvar': 0.0}
            for reg, prob in regime_probabilities.items():
                blend = self.REGIME_BLEND.get(reg, self.REGIME_BLEND['UNKNOWN'])
                for method, w in blend.items():
                    blended_regime_weights[method] += prob * w
        else:
            blended_regime_weights = self.REGIME_BLEND.get(
                regime, self.REGIME_BLEND['UNKNOWN']
            )

        # Blend portfolio weights
        final_weights = (
            blended_regime_weights['hrp'] * hrp_weights +
            blended_regime_weights['erc'] * erc_weights +
            blended_regime_weights['cvar'] * cvar_weights
        )

        # Apply risk engine constraints (position caps)
        final_weights = np.clip(final_weights, 0, max_w)
        final_weights /= final_weights.sum()

        # Turnover from previous weights
        turnover = float(np.sum(np.abs(final_weights - (prev_weights if prev_weights is not None else np.ones(N)/N))))

        return {
            'weights': pd.Series(final_weights, index=assets),
            'hrp_weights': pd.Series(hrp_weights, index=assets),
            'erc_weights': pd.Series(erc_weights, index=assets),
            'cvar_weights': pd.Series(cvar_weights, index=assets),
            'blend_used': blended_regime_weights,
            'regime': regime,
            'turnover': turnover,
            'effective_n': float(1.0 / np.sum(final_weights**2)),  # HHI-based
            'max_weight': float(final_weights.max()),
            'min_weight': float(final_weights.min()),
            'concentration': float(np.sum(final_weights**2)),  # Herfindahl index
        }


# ══════════════════════════════════════════════════════════════
# MODEL GOVERNANCE ENGINE (Drift Detection)
# ══════════════════════════════════════════════════════════════
class ModelGovernanceEngine:
    """
    Monitors model performance and detects degradation/decay.

    Institutional requirement: models must be automatically monitored.
    If a model underperforms → de-weight it. Don't wait for quarterly review.

    Implemented:
    1. Information Coefficient (IC) rolling monitoring
       IC = Spearman(predictions, realized_returns)
       IC_IR = rolling_mean(IC) / rolling_std(IC)
       Good IC_IR: > 0.5. Concerning: < 0.3. De-weight: < 0.1.

    2. CUSUM drift detection on prediction errors
       Detects systematic bias (model predicting wrong sign)

    3. Backtest vs live divergence
       If live Sharpe < 50% of backtest Sharpe → quarantine model

    4. Statistical significance (t-test on rolling IC)
       De-weight if t-stat < 2 (p > 0.05, not significant)

    5. Automatic de-weighting schedule
       Weight = f(IC_IR, t_stat, backtest_vs_live_ratio)
    """

    def __init__(
        self,
        ic_window: int = 63,        # Rolling window for IC computation
        decay_threshold: float = 0.1,  # IC_IR below this → de-weight
        quarantine_threshold: float = -0.1,  # IC_IR below this → quarantine
        backtest_live_min_ratio: float = 0.5,  # Live/backtest Sharpe floor
    ):
        self.ic_window = ic_window
        self.decay_thresh = decay_threshold
        self.quarantine_thresh = quarantine_threshold
        self.bt_live_min = backtest_live_min_ratio

    def compute_ic(
        self,
        predictions: pd.Series,
        realized_returns: pd.Series,
    ) -> pd.Series:
        """
        Rolling Information Coefficient (IC).
        Spearman rank correlation between predictions and outcomes.
        IC > 0 means model has positive predictive power.
        """
        from scipy.stats import spearmanr

        aligned = pd.concat([predictions, realized_returns], axis=1).dropna()
        if len(aligned) < 10:
            return pd.Series(dtype=float)

        aligned.columns = ['pred', 'realized']
        rolling_ic = pd.Series(index=aligned.index, dtype=float)

        for i in range(self.ic_window, len(aligned)):
            window = aligned.iloc[i - self.ic_window: i]
            if window['pred'].std() > 0 and window['realized'].std() > 0:
                ic, _ = spearmanr(window['pred'], window['realized'])
                rolling_ic.iloc[i] = ic

        return rolling_ic.dropna()

    def compute_ic_ir(self, rolling_ic: pd.Series) -> float:
        """IC Information Ratio: mean(IC) / std(IC). Target > 0.5."""
        if len(rolling_ic) < 5:
            return 0.0
        return float(rolling_ic.mean() / (rolling_ic.std() + 1e-6))

    def detect_drift(self, errors: pd.Series, threshold: float = 2.0) -> Dict:
        """
        CUSUM test for systematic prediction error drift.
        Detects if model has developed a persistent bias.
        """
        if len(errors) < 20:
            return {'drift_detected': False, 'cusum_stat': 0.0}

        mu = errors.mean()
        sigma = errors.std()
        if sigma == 0:
            return {'drift_detected': False, 'cusum_stat': 0.0}

        standardized = (errors - mu) / sigma
        s_pos = 0.0
        s_neg = 0.0
        max_stat = 0.0

        for z in standardized:
            s_pos = max(0, s_pos + z)
            s_neg = min(0, s_neg + z)
            max_stat = max(max_stat, s_pos, abs(s_neg))

        drift_detected = max_stat > threshold

        return {
            'drift_detected': drift_detected,
            'cusum_stat': float(max_stat),
            'drift_direction': 'positive' if s_pos > abs(s_neg) else 'negative',
        }

    def assess_model_health(
        self,
        model_name: str,
        predictions: pd.Series,
        realized_returns: pd.Series,
        backtest_sharpe: float,
        live_sharpe: float,
    ) -> Dict:
        """
        Full model health assessment.
        Returns weight multiplier to apply to model's signals.
        """
        rolling_ic = self.compute_ic(predictions, realized_returns)
        ic_ir = self.compute_ic_ir(rolling_ic)

        # t-statistic on IC (is it significantly positive?)
        ic_values = rolling_ic.values
        if len(ic_values) > 5:
            from scipy.stats import ttest_1samp
            t_stat, p_value = ttest_1samp(ic_values, 0)
        else:
            t_stat, p_value = 0.0, 1.0

        # Backtest vs live divergence
        bt_live_ratio = (live_sharpe / backtest_sharpe) if backtest_sharpe != 0 else 1.0

        # Error drift
        errors = realized_returns - predictions
        drift_result = self.detect_drift(errors.reindex(realized_returns.index).dropna())

        # Model status determination
        if ic_ir < self.quarantine_thresh or bt_live_ratio < 0.25:
            status = 'QUARANTINED'
            weight_multiplier = 0.0
        elif ic_ir < self.decay_thresh or bt_live_ratio < self.bt_live_min:
            status = 'DEGRADED'
            weight_multiplier = max(0.1, ic_ir / self.decay_thresh * 0.5)
        elif t_stat < 1.5:
            status = 'WEAK'
            weight_multiplier = 0.7
        elif drift_result['drift_detected']:
            status = 'DRIFTING'
            weight_multiplier = 0.5
        else:
            status = 'HEALTHY'
            weight_multiplier = 1.0

        return {
            'model_name': model_name,
            'status': status,
            'weight_multiplier': weight_multiplier,
            'ic_ir': ic_ir,
            'mean_ic': float(rolling_ic.mean()) if len(rolling_ic) > 0 else 0.0,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'backtest_live_ratio': float(bt_live_ratio),
            'drift_detected': drift_result['drift_detected'],
            'n_observations': len(rolling_ic),
            'recommendation': self._get_recommendation(status, ic_ir),
        }

    def _get_recommendation(self, status: str, ic_ir: float) -> str:
        if status == 'QUARANTINED':
            return 'Remove from ensemble. Investigate data/feature changes. Retrain from scratch.'
        elif status == 'DEGRADED':
            return f'De-weight to {max(0.1, ic_ir):.1%}. Retrain with recent data. Check for regime change.'
        elif status == 'DRIFTING':
            return 'Model has systematic bias. Check features for structural break. Add recent data to training.'
        elif status == 'WEAK':
            return 'IC not statistically significant. Gather more data or strengthen signal.'
        else:
            return 'Model performing as expected. Continue monitoring.'
