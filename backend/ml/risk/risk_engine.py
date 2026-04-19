"""
QuantEdge v6.0 — Independent Risk Engine
==========================================
CRITICAL DESIGN PRINCIPLE: This engine runs INDEPENDENTLY of the alpha engine.
Risk must constrain alpha. Alpha must never constrain risk.

If alpha engine fails → risk engine still runs.
If risk engine fails → no positions allowed. Full stop.

Implements (from the institutional architecture brief):
  ✓ Dynamic covariance matrix (EWMA + Ledoit-Wolf shrinkage)
  ✓ Correlation stress expansion (crisis-adjusted)
  ✓ CVaR / Expected Shortfall (Rockafellar & Uryasev 2000)
  ✓ Tail co-movement risk (lower tail dependence)
  ✓ Liquidity-adjusted risk (Almgren-Chriss 2001)
  ✓ Factor crowding risk (dispersion of factor loadings)
  ✓ Volatility targeting (TargetVol = 10% annualized)
  ✓ Drawdown governor (auto de-risk triggers)
  ✓ Portfolio-level VaR decomposition

Mathematical foundations:
  - EWMA covariance: Σ_t = λ·Σ_{t-1} + (1-λ)·r_{t-1}r_{t-1}^T
  - Ledoit-Wolf shrinkage: Σ̂ = (1-α)Σ_sample + α·(μ_vol·I)
  - CVaR: E[L | L > VaR_α] = (1/(1-α))·∫_{VaR}^∞ x·f(x)dx
  - Cornish-Fisher expansion for non-normal VaR
  - Factor crowding: std(β_factor across stocks) → low std = crowded
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from scipy.linalg import sqrtm
from sklearn.covariance import LedoitWolf
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
# DYNAMIC COVARIANCE ENGINE
# ══════════════════════════════════════════════════════════════
class DynamicCovarianceEngine:
    """
    Dynamic covariance estimation using EWMA with Ledoit-Wolf shrinkage.

    Standard sample covariance: collapses during crisis (all correlations → 1)
    EWMA: weights recent observations more → faster regime adaptation
    Ledoit-Wolf: mathematically optimal linear shrinkage
    Combined: adaptive + robust + well-conditioned

    EWMA update:
      Σ_t = λ·Σ_{t-1} + (1-λ)·r_{t-1}·r_{t-1}^T
    where λ = 0.94 (RiskMetrics standard for daily data)

    Shrinkage:
      Σ̂ = (1-α)·Σ_EWMA + α·Σ_target
    where Σ_target = diagonal (vol²) matrix (Ledoit-Wolf optimal α)
    """

    def __init__(
        self,
        ewma_lambda: float = 0.94,        # RiskMetrics daily decay
        min_periods: int = 252,            # Min observations before trusting estimate
        shrinkage_method: str = 'ledoit_wolf',  # or 'constant', 'oracle'
        stress_correlation_floor: float = 0.0,  # Min correlation in stress (0 = no floor)
        stress_vol_multiplier: float = 1.5,     # Multiply vols by this in stress
    ):
        self.lam = ewma_lambda
        self.min_periods = min_periods
        self.shrinkage = shrinkage_method
        self.stress_vol_mult = stress_vol_multiplier
        self.stress_corr_floor = stress_correlation_floor

    def ewma_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute EWMA covariance matrix.
        More weight to recent returns → adapts to volatility regimes.
        """
        n, k = returns.shape
        Sigma = np.cov(returns.values[:63].T)  # Seed with first 3 months

        for i in range(63, n):
            r = returns.values[i].reshape(-1, 1)
            Sigma = self.lam * Sigma + (1 - self.lam) * r @ r.T

        return Sigma

    def ledoit_wolf_shrinkage(self, returns: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Ledoit-Wolf (2004) analytical shrinkage estimator.
        Optimal linear combination of sample covariance and diagonal target.
        Returns (shrunk_covariance, shrinkage_alpha)
        """
        lw = LedoitWolf()
        lw.fit(returns.values)
        return lw.covariance_, lw.shrinkage_

    def compute(
        self,
        returns: pd.DataFrame,
        stressed: bool = False,
    ) -> Dict:
        """
        Full covariance estimation pipeline.

        Parameters
        ----------
        returns: T×N DataFrame of asset returns
        stressed: If True, apply crisis correlation expansion

        Returns dict with:
          covariance: N×N shrunk covariance matrix
          correlation: N×N correlation matrix
          volatilities: N-vector of annualized vols
          shrinkage_alpha: Ledoit-Wolf shrinkage intensity
          condition_number: Matrix conditioning (low = well-conditioned)
        """
        if len(returns) < self.min_periods:
            # Not enough data: use equal-weight simple estimate
            cov = returns.cov().values
            shrinkage = 0.5
        else:
            # EWMA covariance
            ewma_cov = self.ewma_covariance(returns)

            # Ledoit-Wolf shrinkage on top of EWMA
            if self.shrinkage == 'ledoit_wolf':
                lw_cov, shrinkage = self.ledoit_wolf_shrinkage(returns)
                # Combine EWMA and Ledoit-Wolf
                cov = 0.7 * ewma_cov + 0.3 * lw_cov
            else:
                cov = ewma_cov
                shrinkage = 0.0

        # Ensure positive definiteness
        cov = self._make_psd(cov)

        # Extract vols and correlations
        vols = np.sqrt(np.diag(cov)) * np.sqrt(252)  # Annualized
        D = np.diag(np.sqrt(np.diag(cov)))
        D_inv = np.diag(1.0 / np.sqrt(np.diag(cov)))
        corr = D_inv @ cov @ D_inv

        # Stress scenario: expand correlations toward crisis levels
        if stressed:
            cov, corr = self._stress_covariance(vols, corr)

        return {
            'covariance': cov,
            'correlation': corr,
            'volatilities': vols,
            'shrinkage_alpha': shrinkage,
            'condition_number': float(np.linalg.cond(cov)),
            'avg_correlation': float(np.mean(corr[np.triu_indices_from(corr, k=1)])) if corr.shape[0] > 1 else 0.0,
            'max_correlation': float(np.max(np.abs(corr[np.triu_indices_from(corr, k=1)]))) if corr.shape[0] > 1 else 0.0,
            'is_stressed': stressed,
        }

    def _make_psd(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix is positive semi-definite via nearest PSD."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def _stress_covariance(
        self, vols: np.ndarray, normal_corr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crisis scenario: correlations increase toward 1 (correlation breakdown).
        During 2008/2020, cross-asset correlations surged from 0.3 → 0.85+.
        Stress test: blend with perfect correlation matrix.
        """
        n = len(vols)
        stressed_vols = vols * self.stress_vol_mult

        # Crisis correlation: blend toward all-ones matrix
        crisis_corr = np.ones((n, n))
        np.fill_diagonal(crisis_corr, 1.0)

        # Stress blend: 50% normal + 50% crisis
        stressed_corr = 0.5 * normal_corr + 0.5 * crisis_corr
        np.fill_diagonal(stressed_corr, 1.0)

        # Rebuild covariance from stressed vols + stressed corr
        daily_vols = stressed_vols / np.sqrt(252)
        D = np.diag(daily_vols)
        stressed_cov = D @ stressed_corr @ D

        return self._make_psd(stressed_cov), stressed_corr


# ══════════════════════════════════════════════════════════════
# CVaR / EXPECTED SHORTFALL ENGINE
# ══════════════════════════════════════════════════════════════
class CVaREngine:
    """
    Conditional Value at Risk (Expected Shortfall) computation.

    CVaR at confidence α = E[L | L > VaR_α]
    = Average loss when loss exceeds the α-quantile.

    CVaR is SUPERIOR to VaR because:
    1. VaR doesn't tell you HOW BAD the bad days are
    2. CVaR is coherent (sub-additive) → can be added across positions
    3. CVaR is convex → can be minimized with convex optimization
    4. Basel III/IV and Solvency II require Expected Shortfall, not VaR

    Methods implemented:
    1. Historical simulation (non-parametric, no distribution assumption)
    2. Gaussian analytical (fast, assumes normality)
    3. Cornish-Fisher (adjusts for skewness/kurtosis)
    4. Student-t (fat tails)
    5. Portfolio-level CVaR decomposition (marginal contributions)
    """

    def __init__(self, confidence: float = 0.95):
        self.alpha = confidence

    def historical_cvar(self, returns: np.ndarray) -> Dict:
        """
        Non-parametric CVaR from historical returns.
        Most robust: no distribution assumption needed.
        """
        sorted_ret = np.sort(returns)
        n = len(sorted_ret)
        var_idx = int((1 - self.alpha) * n)

        var = -sorted_ret[var_idx]
        cvar = -np.mean(sorted_ret[:var_idx])

        return {
            'var': float(var),
            'cvar': float(cvar),
            'method': 'historical',
            'n_tail_obs': var_idx,
        }

    def gaussian_cvar(self, mu: float, sigma: float) -> Dict:
        """
        Analytical Gaussian CVaR.
        CVaR_α = μ + σ × φ(z_α) / (1-α)
        where φ = normal PDF, z_α = Φ⁻¹(α)
        """
        z = norm.ppf(self.alpha)
        var = -(mu - sigma * z)
        cvar = -(mu - sigma * norm.pdf(z) / (1 - self.alpha))

        return {
            'var': float(var),
            'cvar': float(cvar),
            'method': 'gaussian',
        }

    def cornish_fisher_cvar(self, mu: float, sigma: float, skew: float, kurt: float) -> Dict:
        """
        Cornish-Fisher VaR/CVaR with skewness and kurtosis adjustment.
        More accurate than Gaussian when returns are fat-tailed/skewed.

        Modified z-score:
        z_CF = z + (z²-1)/6 × γ₁ + (z³-3z)/24 × (γ₂-3) - (2z³-5z)/36 × γ₁²
        where γ₁ = skewness, γ₂ = kurtosis
        """
        z = norm.ppf(self.alpha)

        # Cornish-Fisher expansion
        cf_z = (
            z +
            (z**2 - 1) / 6 * skew +
            (z**3 - 3*z) / 24 * (kurt - 3) -
            (2*z**3 - 5*z) / 36 * skew**2
        )

        var = -(mu - sigma * cf_z)
        # CVaR: integrate beyond VaR point
        cvar_factor = norm.pdf(cf_z) / (1 - self.alpha)
        cvar = -(mu - sigma * cvar_factor)

        return {
            'var': float(var),
            'cvar': float(cvar),
            'cf_z': float(cf_z),
            'method': 'cornish_fisher',
        }

    def student_t_cvar(self, mu: float, sigma: float, df: float = 5.0) -> Dict:
        """
        Student-t CVaR (fat tails).
        CVaR_α(t_ν) = (f(t_α,ν) / (1-α)) × ((ν + t²_α,ν)/(ν-1)) × σ - μ
        """
        t_alpha = student_t.ppf(1 - self.alpha, df=df)
        var = -(mu + sigma * t_alpha)

        # CVaR for Student-t
        cvar_multiplier = (
            student_t.pdf(t_alpha, df=df) / (1 - self.alpha) *
            (df + t_alpha**2) / (df - 1)
        )
        cvar = -(mu - sigma * cvar_multiplier)

        return {
            'var': float(var),
            'cvar': float(cvar),
            'df': df,
            'method': 'student_t',
        }

    def portfolio_cvar_decomposition(
        self,
        weights: np.ndarray,       # Portfolio weights
        returns: pd.DataFrame,      # Historical returns T×N
    ) -> Dict:
        """
        Decompose portfolio CVaR into asset-level contributions.
        Marginal CVaR = ∂CVaR/∂w_i (sensitivity of portfolio CVaR to weight i)
        Component CVaR = w_i × Marginal CVaR_i  (sum = total CVaR)
        """
        # Portfolio returns
        port_ret = returns.values @ weights
        port_cvar_dict = self.historical_cvar(port_ret)
        port_cvar = port_cvar_dict['cvar']

        # Marginal CVaR via finite difference
        eps = 0.001
        marginal_cvars = np.zeros(len(weights))
        for i in range(len(weights)):
            w_up = weights.copy()
            w_up[i] += eps
            w_up /= w_up.sum()
            port_up = returns.values @ w_up
            cvar_up = self.historical_cvar(port_up)['cvar']
            marginal_cvars[i] = (cvar_up - port_cvar) / eps

        # Component CVaR
        component_cvars = weights * marginal_cvars

        return {
            'portfolio_cvar': port_cvar,
            'marginal_cvars': dict(zip(returns.columns, marginal_cvars)),
            'component_cvars': dict(zip(returns.columns, component_cvars)),
            'pct_contributions': dict(zip(returns.columns, component_cvars / port_cvar)),
            'diversification_ratio': float(np.sum(np.abs(component_cvars)) / port_cvar),
        }


# ══════════════════════════════════════════════════════════════
# TAIL CO-MOVEMENT RISK
# ══════════════════════════════════════════════════════════════
class TailCoMovementAnalyzer:
    """
    Lower tail dependence and co-movement analysis.

    During crises, correlation structure breaks down AND
    assets move together in tails even when unconditional
    correlation is moderate. Standard Gaussian copula misses this.

    Lower Tail Dependence Coefficient (LTDC):
    λ_L(i,j) = lim_{u→0} P(F_i(X_i) < u | F_j(X_j) < u)
    = probability both assets crash simultaneously

    High LTDC: dangerous portfolio (joint crash risk)
    Low LTDC: good diversification even in extremes

    Empirically: equity portfolio LTDC ≈ 0.3-0.6 (much higher than Gaussian implies)
    """

    def __init__(self, confidence: float = 0.05):
        self.alpha = confidence  # Tail threshold

    def tail_correlation(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute pairwise lower tail correlation.
        Correlation conditional on both assets being below alpha-quantile.
        """
        n_assets = returns.shape[1]
        tail_corr = np.eye(n_assets)

        # Quantile threshold
        thresholds = returns.quantile(self.alpha)

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                # Joint tail events (both below threshold)
                mask_i = returns.iloc[:, i] < thresholds.iloc[i]
                mask_j = returns.iloc[:, j] < thresholds.iloc[j]
                joint_mask = mask_i & mask_j

                if joint_mask.sum() > 5:
                    # Conditional correlation given joint tail
                    cond_returns = returns[joint_mask][[returns.columns[i], returns.columns[j]]]
                    c = np.corrcoef(cond_returns.T)[0, 1]
                    tail_corr[i, j] = tail_corr[j, i] = c

        return tail_corr

    def lower_tail_dependence(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Empirical lower tail dependence coefficient.
        LTDC_ij = P(U_i < α | U_j < α) where U = uniform scores
        """
        n_assets = returns.shape[1]
        ltdc = np.zeros((n_assets, n_assets))

        # Convert to uniform scores (ranks)
        scores = returns.rank() / (len(returns) + 1)

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                # P(U_i < alpha AND U_j < alpha) / alpha
                joint = ((scores.iloc[:, i] < self.alpha) & (scores.iloc[:, j] < self.alpha)).mean()
                ltdc[i, j] = ltdc[j, i] = joint / self.alpha

        np.fill_diagonal(ltdc, 1.0)
        return ltdc

    def portfolio_tail_risk(
        self, weights: np.ndarray, returns: pd.DataFrame
    ) -> Dict:
        """Aggregate portfolio tail risk metrics."""
        port_ret = returns.values @ weights

        # Tail statistics
        tail_threshold = np.percentile(port_ret, 5)
        tail_obs = port_ret[port_ret < tail_threshold]

        return {
            'tail_var_5pct': float(-tail_threshold),
            'tail_cvar_5pct': float(-tail_obs.mean()) if len(tail_obs) > 0 else 0,
            'tail_skewness': float(pd.Series(tail_obs).skew()) if len(tail_obs) > 3 else 0,
            'n_tail_obs': int(len(tail_obs)),
            'tail_sharpe': float(
                tail_obs.mean() / tail_obs.std() * np.sqrt(252)
            ) if len(tail_obs) > 1 else 0,
        }


# ══════════════════════════════════════════════════════════════
# LIQUIDITY-ADJUSTED RISK
# ══════════════════════════════════════════════════════════════
class LiquidityRiskEngine:
    """
    Liquidity-adjusted VaR (LaVaR).

    Standard VaR assumes you can exit at current prices.
    Reality: large positions create market impact → actual exit price
    is worse than quoted price.

    Almgren-Chriss (2001) execution cost model:
    - Permanent impact: g(v) = γ·σ·(v/V)^(1/2)  (permanent price movement)
    - Temporary impact: h(v) = η·(v/V)          (bid-ask + temporary pressure)

    Liquidity Horizon = T_liq = position / ADV × multiplier
    Higher position → longer time to exit → more market impact

    LaVaR = Standard VaR × √(T_liq / T_normal)
    (scales VaR for the actual holding period required to exit)
    """

    def __init__(
        self,
        liquidation_horizon_days: int = 10,  # Standard Basel requirement
        adv_fraction: float = 0.20,           # Max 20% of ADV per day
        permanent_impact: float = 0.314,      # Almgren-Chriss γ
        temporary_impact: float = 0.142,      # Almgren-Chriss η
    ):
        self.T_liq = liquidation_horizon_days
        self.adv_frac = adv_fraction
        self.gamma = permanent_impact
        self.eta = temporary_impact

    def execution_cost(
        self,
        position_value: float,  # $ value of position
        adv: float,              # Average daily volume in $
        sigma: float,            # Daily vol of asset
    ) -> Dict:
        """
        Almgren-Chriss optimal execution cost.
        Assumes TWAP (time-weighted average price) execution.
        """
        if adv <= 0:
            return {'cost_pct': 0.05, 'days_to_exit': 999}

        # Participation rate
        participation = position_value / adv

        # Days required to exit (at max adv_frac daily)
        days_to_exit = participation / self.adv_frac

        # Permanent market impact
        perm_impact = self.gamma * sigma * np.sqrt(participation)

        # Temporary market impact per day
        temp_impact = self.eta * participation

        # Total cost as fraction of position
        total_cost_pct = perm_impact + temp_impact / 2

        return {
            'cost_pct': float(total_cost_pct),
            'permanent_impact_pct': float(perm_impact),
            'temporary_impact_pct': float(temp_impact),
            'days_to_exit': float(days_to_exit),
            'participation_rate': float(participation),
        }

    def liquidity_adjusted_var(
        self,
        position_value: float,
        adv: float,
        sigma: float,
        standard_var: float,
    ) -> Dict:
        """
        LaVaR = Standard VaR adjusted for actual liquidation time.

        Basel III formula:
        LaVaR = VaR(1d) × √(T_liq / 1d) × (1 + execution_cost)
        """
        exec_cost = self.execution_cost(position_value, adv, sigma)
        days = max(exec_cost['days_to_exit'], 1)

        # Time-scaled VaR
        lavar = standard_var * np.sqrt(days) * (1 + exec_cost['cost_pct'])

        return {
            'standard_var': standard_var,
            'liquidity_adjusted_var': float(lavar),
            'liquidity_multiplier': float(lavar / max(standard_var, 1e-6)),
            'days_to_exit': float(days),
            'execution_cost_pct': exec_cost['cost_pct'],
        }


# ══════════════════════════════════════════════════════════════
# FACTOR CROWDING DETECTOR
# ══════════════════════════════════════════════════════════════
class FactorCrowdingDetector:
    """
    Factor crowding risk: too many funds loading same factors.

    When a factor is crowded:
    - Many funds hold same positions
    - Small shock → simultaneous unwinding → amplified price move
    - "Quant quake" August 2007: momentum crowding caused 10σ moves

    Crowding signals:
    1. Cross-sectional dispersion of factor loadings (low = crowded)
    2. Factor return autocorrelation (negative → unwinding)
    3. Factor Sharpe ratio momentum (too high = too many following it)
    4. Short interest in factor portfolios

    If crowding detected → reduce factor exposure in risk budget.
    """

    def __init__(self, crowding_threshold: float = 0.3):
        self.crowding_threshold = crowding_threshold

    def factor_dispersion(self, factor_loadings: pd.DataFrame) -> Dict:
        """
        Compute cross-sectional dispersion of factor loadings.
        Low dispersion → all stocks loaded same way → crowded.
        """
        dispersions = {}
        for factor in factor_loadings.columns:
            loadings = factor_loadings[factor].dropna()
            dispersion = float(loadings.std() / (abs(loadings.mean()) + 1e-6))
            dispersions[factor] = dispersion

        # Crowding score: how many factors have low dispersion
        crowded_factors = [f for f, d in dispersions.items() if d < self.crowding_threshold]

        return {
            'dispersions': dispersions,
            'crowded_factors': crowded_factors,
            'crowding_score': len(crowded_factors) / max(len(dispersions), 1),
            'is_crowded': len(crowded_factors) > 0,
        }

    def factor_return_momentum(self, factor_returns: pd.DataFrame, window: int = 63) -> Dict:
        """
        Factor return Sharpe ratio momentum.
        Too-high recent Sharpe → likely crowded → mean reversion risk.
        """
        recent = factor_returns.tail(window)
        sharpes = {}
        for factor in factor_returns.columns:
            ret = recent[factor].dropna()
            if len(ret) > 10:
                sr = float(ret.mean() / ret.std() * np.sqrt(252))
                sharpes[factor] = sr

        # High Sharpe factors are likely crowded
        high_sharpe = {f: s for f, s in sharpes.items() if abs(s) > 2.0}

        return {
            'factor_sharpes': sharpes,
            'potentially_crowded': list(high_sharpe.keys()),
            'crowding_risk_level': 'HIGH' if len(high_sharpe) > 2 else 'MODERATE' if len(high_sharpe) > 0 else 'LOW',
        }


# ══════════════════════════════════════════════════════════════
# VOLATILITY TARGETING ENGINE
# ══════════════════════════════════════════════════════════════
class VolatilityTargetingEngine:
    """
    Volatility targeting for portfolio-level risk control.

    Core formula:
      TargetVol = 10% annualized (user-configurable)
      ScaleFactor = TargetVol / RealizedVol
      Scaled_weight = original_weight × ScaleFactor

    This ensures portfolio always runs at same risk level regardless
    of market regime. In high-vol environments: reduce leverage.
    In low-vol environments: increase leverage.

    Why 10%?
    - Typical institutional target for equity long/short strategies
    - High enough to generate meaningful alpha
    - Low enough to survive 3σ drawdowns without blowing up

    Drawdown governor:
      If current drawdown > threshold → reduce ScaleFactor proportionally
      If max drawdown > hard limit → halt all positions
    """

    def __init__(
        self,
        target_vol: float = 0.10,        # 10% annualized target
        max_leverage: float = 2.0,        # Maximum scale factor
        min_leverage: float = 0.2,        # Minimum scale factor
        vol_window: int = 21,             # Days for realized vol estimate
        drawdown_halt_threshold: float = 0.15,  # 15% drawdown → halt
        drawdown_reduce_threshold: float = 0.08, # 8% drawdown → reduce
        smoothing: float = 0.5,           # EMA smoothing on scale factor
    ):
        self.target_vol = target_vol
        self.max_lev = max_leverage
        self.min_lev = min_leverage
        self.vol_window = vol_window
        self.dd_halt = drawdown_halt_threshold
        self.dd_reduce = drawdown_reduce_threshold
        self.smoothing = smoothing

    def compute_scale_factor(
        self,
        portfolio_returns: pd.Series,
        current_drawdown: Optional[float] = None,
    ) -> Dict:
        """
        Compute current volatility scaling factor.

        ScaleFactor = TargetVol / RealizedVol
        Clipped to [min_lev, max_lev]
        """
        if len(portfolio_returns) < self.vol_window:
            return {
                'scale_factor': 1.0,
                'realized_vol': self.target_vol,
                'target_vol': self.target_vol,
                'leverage_signal': 'NORMAL',
                'governor_active': False,
            }

        # Realized vol (EWMA)
        realized_vol = float(
            portfolio_returns.tail(self.vol_window).std() * np.sqrt(252)
        )

        if realized_vol < 0.001:
            realized_vol = self.target_vol

        # Base scale factor
        scale = self.target_vol / realized_vol
        scale = np.clip(scale, self.min_lev, self.max_lev)

        # Drawdown governor
        governor_active = False
        leverage_signal = 'NORMAL'

        if current_drawdown is not None:
            if current_drawdown >= self.dd_halt:
                scale = 0.0  # HALT: no positions
                governor_active = True
                leverage_signal = 'HALT'
            elif current_drawdown >= self.dd_reduce:
                # Linear reduction: 0 at dd_halt, full at dd_reduce
                reduction = 1 - (current_drawdown - self.dd_reduce) / (self.dd_halt - self.dd_reduce)
                scale *= reduction
                governor_active = True
                leverage_signal = 'REDUCING'
            elif scale > 1.2:
                leverage_signal = 'LEVERING_UP'
            elif scale < 0.8:
                leverage_signal = 'DE-RISKING'

        return {
            'scale_factor': float(scale),
            'realized_vol': float(realized_vol),
            'target_vol': self.target_vol,
            'leverage_signal': leverage_signal,
            'governor_active': governor_active,
            'vol_ratio': float(realized_vol / self.target_vol),
        }

    def compute_drawdown(self, nav_series: pd.Series) -> float:
        """Current drawdown from all-time high NAV."""
        if len(nav_series) == 0:
            return 0.0
        peak = nav_series.cummax()
        drawdown = (nav_series - peak) / peak
        return float(-drawdown.iloc[-1])  # Positive number


# ══════════════════════════════════════════════════════════════
# MASTER RISK ENGINE (Orchestrator)
# ══════════════════════════════════════════════════════════════
class MasterRiskEngine:
    """
    The single source of risk truth.

    This engine runs INDEPENDENTLY.
    Alpha engine output → Risk engine → Position constraints.
    Risk engine NEVER uses alpha predictions.
    Risk engine sets limits. Alpha works within limits.

    Outputs:
    - Per-asset position limits
    - Portfolio-level VaR/CVaR
    - Risk budget allocation
    - Volatility scale factor
    - Regime-adjusted risk budget
    - Stop triggers
    """

    def __init__(
        self,
        target_vol: float = 0.10,
        max_position_pct: float = 0.10,   # Max 10% per stock
        max_sector_pct: float = 0.25,     # Max 25% per sector
        max_factor_exposure: float = 1.0,  # Max 1σ factor exposure
        cvar_confidence: float = 0.95,
        liquidity_constraint: float = 0.20,  # Max 20% of ADV
    ):
        self.cov_engine = DynamicCovarianceEngine()
        self.cvar_engine = CVaREngine(confidence=cvar_confidence)
        self.tail_engine = TailCoMovementAnalyzer()
        self.liq_engine = LiquidityRiskEngine()
        self.crowding = FactorCrowdingDetector()
        self.vol_target = VolatilityTargetingEngine(target_vol=target_vol)

        self.max_pos = max_position_pct
        self.max_sector = max_sector_pct
        self.max_factor_exp = max_factor_exposure
        self.liq_constraint = liquidity_constraint

    def full_risk_assessment(
        self,
        returns: pd.DataFrame,              # Historical returns T×N
        current_weights: Optional[np.ndarray] = None,
        portfolio_nav: Optional[pd.Series] = None,
        stressed: bool = False,
    ) -> Dict:
        """
        Complete risk assessment. Single entry point.
        Called BEFORE any portfolio construction.
        """
        N = returns.shape[1]
        w = current_weights if current_weights is not None else np.ones(N) / N

        # 1. Dynamic covariance
        cov_result = self.cov_engine.compute(returns, stressed=stressed)

        # 2. CVaR multiple methods
        port_ret_series = returns.values @ w
        hist_cvar = self.cvar_engine.historical_cvar(port_ret_series)
        mu = float(np.mean(port_ret_series))  # mean of weighted portfolio return series
        sigma = float(np.sqrt(w @ cov_result['covariance'] @ w))
        skew = float(pd.Series(port_ret_series).skew())
        kurt = float(pd.Series(port_ret_series).kurtosis() + 3)

        gaussian_cvar = self.cvar_engine.gaussian_cvar(mu, sigma)
        cf_cvar = self.cvar_engine.cornish_fisher_cvar(mu, sigma, skew, kurt)
        t_cvar = self.cvar_engine.student_t_cvar(mu, sigma, df=5.0)

        # Worst-case CVaR (conservative)
        worst_cvar = max(
            hist_cvar['cvar'],
            gaussian_cvar['cvar'],
            cf_cvar['cvar'],
            t_cvar['cvar'],
        )

        # 3. CVaR decomposition
        cvar_decomp = self.cvar_engine.portfolio_cvar_decomposition(w, returns)

        # 4. Tail co-movement
        tail_risk = self.tail_engine.portfolio_tail_risk(w, returns)

        # 5. Volatility targeting
        port_ret = pd.Series(port_ret_series, index=returns.index)
        current_dd = self.vol_target.compute_drawdown(
            (1 + port_ret).cumprod() if portfolio_nav is None else portfolio_nav
        )
        vol_target_result = self.vol_target.compute_scale_factor(port_ret, current_dd)

        # 6. Stressed scenario
        if not stressed:
            stressed_result = self.cov_engine.compute(returns, stressed=True)
            stressed_port_var = float(np.sqrt(w @ stressed_result['covariance'] @ w * 252))
        else:
            stressed_port_var = float(np.sqrt(w @ cov_result['covariance'] @ w * 252))

        return {
            # Covariance
            'covariance': cov_result,

            # CVaR (multiple methods)
            'cvar': {
                'historical': hist_cvar,
                'gaussian': gaussian_cvar,
                'cornish_fisher': cf_cvar,
                'student_t': t_cvar,
                'worst_case': worst_cvar,
                'decomposition': cvar_decomp,
            },

            # Tail risk
            'tail_risk': tail_risk,

            # Volatility targeting
            'vol_targeting': vol_target_result,

            # Drawdown
            'current_drawdown': current_dd,

            # Stressed scenario
            'stressed_portfolio_vol': stressed_port_var,

            # Position limits (risk-engine output that constrains alpha)
            'position_limits': {
                'max_per_asset': self.max_pos * vol_target_result['scale_factor'],
                'max_per_sector': self.max_sector,
                'max_factor_exposure': self.max_factor_exp,
                'liquidity_constraint_adv_pct': self.liq_constraint,
            },

            # Risk budget (regime-adjusted)
            'risk_budget': {
                'scale_factor': vol_target_result['scale_factor'],
                'target_vol': self.vol_target.target_vol,
                'realized_vol': vol_target_result['realized_vol'],
                'signal': vol_target_result['leverage_signal'],
                'governor_active': vol_target_result['governor_active'],
            },

            # Key metrics (single numbers for dashboard)
            'summary': {
                'portfolio_vol_annualized': float(sigma * np.sqrt(252)),
                'worst_case_cvar_daily': float(worst_cvar),
                'worst_case_cvar_annual': float(worst_cvar * np.sqrt(252)),
                'current_drawdown': float(current_dd),
                'stressed_vol': stressed_port_var,
                'avg_correlation': cov_result['avg_correlation'],
                'is_halted': vol_target_result['leverage_signal'] == 'HALT',
            },
        }
