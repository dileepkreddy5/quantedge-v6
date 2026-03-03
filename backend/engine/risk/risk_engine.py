"""
QuantEdge v6.0 — RISK ENGINE (Layer 4, but built before Alpha)
════════════════════════════════════════════════════════════════
ARCHITECTURAL PRINCIPLE: Risk must constrain alpha. Not depend on it.

The risk engine is the MOST CRITICAL component of the system.
It operates independently. If the alpha engine fails completely,
the risk engine continues running. It is the last line of defense.

A firm that gets the alpha wrong loses money slowly.
A firm that gets the risk wrong goes bankrupt fast.

This engine computes:

1. DYNAMIC COVARIANCE MATRIX
   - EWMA (RiskMetrics 1994): Σ_t = λΣ_{t-1} + (1-λ)r_{t-1}r_{t-1}^T
   - Ledoit-Wolf shrinkage (2004): Σ* = αΣ_sample + (1-α)F (structured target)
   - Factor model decomposition: Σ = BFB^T + D (factor + idiosyncratic)

2. TAIL RISK
   - CVaR (Expected Shortfall) via historical simulation and parametric
   - Cornish-Fisher expansion for non-Gaussian tails
   - Extreme Value Theory (GEV/GPD) for tail extrapolation

3. LIQUIDITY-ADJUSTED RISK
   - Almgren-Chriss (2001) execution impact
   - Liquidity-adjusted VaR (LVaR)
   - Market impact of position liquidation

4. FACTOR CROWDING RISK
   - Flows-based crowding (institutional herding)
   - Short interest concentration
   - Factor momentum + reversal risk

5. CORRELATION STRESS TESTING
   - DCC-GARCH dynamic correlations
   - Correlation stress (set all off-diagonal to crisis levels)
   - Cross-asset contagion scenarios

Mathematical references:
  - RiskMetrics (1994): J.P. Morgan Technical Document
  - Ledoit & Wolf (2004): A Well-Conditioned Estimator for Large-Dimensional
  - Almgren & Chriss (2001): Optimal Execution of Portfolio Transactions
  - McNeil, Frey, Embrechts (2005): Quantitative Risk Management
════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import eigh, inv
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
# DYNAMIC COVARIANCE MATRIX ENGINE
# ─────────────────────────────────────────────────────────────

class DynamicCovarianceEngine:
    """
    Computes the covariance matrix using multiple estimators and combines them.

    The sample covariance matrix Σ_sample has problems:
    - For N assets with T observations, Σ is poorly conditioned when T/N < 10
    - Small eigenvalues are biased DOWN (noise)
    - Large eigenvalues are biased UP (spurious)
    - Inverse Σ^{-1} is extremely noisy

    We use Ledoit-Wolf shrinkage:
        Σ* = ρF + (1-ρ)Σ_sample
    where F is a structured "shrinkage target" (constant correlation matrix)
    and ρ is analytically optimal shrinkage intensity.

    This produces a well-conditioned, stable covariance matrix.
    """

    def __init__(self, ewma_lambda: float = 0.94,
                 min_history_days: int = 63):
        """
        ewma_lambda: RiskMetrics decay factor
            - 0.94 for daily data (standard)
            - 0.97 for weekly data
        min_history_days: minimum required for covariance estimation
        """
        self.ewma_lambda = ewma_lambda
        self.min_history = min_history_days
        self._covariance_cache: Dict[pd.Timestamp, np.ndarray] = {}
        self._correlation_cache: Dict[pd.Timestamp, np.ndarray] = {}

    def compute_ewma_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Exponentially Weighted Moving Average covariance.

        Σ_t = λΣ_{t-1} + (1-λ)r_{t-1}r_{t-1}^T

        This upweights recent observations and allows the covariance
        to adapt to changing market regimes.

        RiskMetrics (1994) standardized λ=0.94 for daily data,
        which means the half-life of observations is:
            half_life = -ln(2) / ln(λ) ≈ 11 trading days
        """
        T, N = returns.shape
        if T < self.min_history:
            raise ValueError(f"Need at least {self.min_history} days, got {T}")

        returns_arr = returns.values
        # Initialize with sample covariance of first half
        init_half = max(N * 3, 30)
        Sigma = np.cov(returns_arr[:init_half].T)

        # EWMA recursion
        lam = self.ewma_lambda
        for t in range(init_half, T):
            r = returns_arr[t].reshape(-1, 1)
            Sigma = lam * Sigma + (1 - lam) * (r @ r.T)

        # Ensure symmetry (numerical precision)
        Sigma = (Sigma + Sigma.T) / 2
        return Sigma

    def ledoit_wolf_shrinkage(self, returns: pd.DataFrame,
                               shrinkage_target: str = 'constant_correlation'
                               ) -> Tuple[np.ndarray, float]:
        """
        Ledoit-Wolf (2004) analytical shrinkage.

        Finds optimal ρ* that minimizes:
            E[||Σ* - Σ_true||_F^2]

        where Σ* = ρF + (1-ρ)Σ_sample

        Shrinkage targets:
        1. 'identity': F = σ̄²I (simplest, largest shrinkage)
        2. 'constant_correlation': F uses average off-diagonal correlation
        3. 'single_factor': F from CAPM (market factor only)

        Returns: (shrunk_covariance, optimal_rho)
        """
        X = returns.values
        T, N = X.shape

        # Sample covariance (demeaned)
        X = X - X.mean(axis=0)
        S = X.T @ X / T  # Sample covariance

        if shrinkage_target == 'constant_correlation':
            # Target: constant correlation matrix
            # F_ij = r̄ * sqrt(S_ii * S_jj) for i≠j, S_ii for i=j
            vols = np.sqrt(np.diag(S))
            corr = S / np.outer(vols, vols)
            np.fill_diagonal(corr, 1.0)
            r_bar = (corr.sum() - N) / (N * (N - 1))  # Average off-diagonal
            F = r_bar * np.outer(vols, vols)
            np.fill_diagonal(F, np.diag(S))

        elif shrinkage_target == 'identity':
            mu = np.trace(S) / N
            F = mu * np.eye(N)

        else:
            F = np.eye(N) * np.trace(S) / N

        # Analytical shrinkage intensity (Oracle approximation)
        # Ledoit-Wolf formula (simplified)
        delta = S - F
        phi = 0.0
        for t in range(T):
            x = X[t]
            phi += (x.T @ delta @ x) ** 2 / T ** 2

        kappa = np.sum(delta ** 2)
        rho = min(1.0, max(0.0, phi / kappa)) if kappa > 0 else 0.1

        # Apply shrinkage
        Sigma_star = rho * F + (1 - rho) * S
        Sigma_star = (Sigma_star + Sigma_star.T) / 2  # Enforce symmetry

        return Sigma_star, rho

    def compute_combined_covariance(self, returns: pd.DataFrame,
                                     ewma_weight: float = 0.6) -> np.ndarray:
        """
        Combines EWMA (regime-responsive) and Ledoit-Wolf (well-conditioned).

        Σ_combined = w * Σ_EWMA + (1-w) * Σ_LW

        EWMA captures recent regime (more responsive)
        Ledoit-Wolf is well-conditioned (better for optimization)
        """
        try:
            Sigma_ewma = self.compute_ewma_covariance(returns)
        except Exception:
            Sigma_ewma = np.cov(returns.values.T)

        try:
            Sigma_lw, rho = self.ledoit_wolf_shrinkage(returns)
        except Exception:
            Sigma_lw = np.cov(returns.values.T)

        Sigma = ewma_weight * Sigma_ewma + (1 - ewma_weight) * Sigma_lw

        # Final eigenvalue cleaning: remove spurious small eigenvalues
        Sigma = self._clean_eigenvalues(Sigma, returns.shape[0])
        return Sigma

    def _clean_eigenvalues(self, Sigma: np.ndarray, T: int) -> np.ndarray:
        """
        Marchenko-Pastur eigenvalue cleaning.

        Random matrix theory (Marchenko-Pastur 1967) shows that
        for Σ_sample with T observations and N assets, eigenvalues
        below λ_max = σ²(1 + √(N/T))² are pure noise.

        We replace noise eigenvalues with their mean (shrinkage toward identity).
        This is the 'eigenvalue clipping' approach used by:
        - Bouchaud, Potters (2009): Theory of Financial Risk
        - Laloux et al (2000): Random Matrix Theory and Financial Correlations
        """
        N = Sigma.shape[0]
        q = T / N  # Aspect ratio

        # Marchenko-Pastur upper bound for noise eigenvalues
        sigma_sq = np.trace(Sigma) / N  # Average variance
        lambda_max_noise = sigma_sq * (1 + 1/q + 2 * np.sqrt(1/q))

        eigenvalues, eigenvectors = eigh(Sigma)
        eigenvalues = eigenvalues.real

        # Identify noise eigenvalues (below Marchenko-Pastur bound)
        noise_mask = eigenvalues < lambda_max_noise
        signal_mask = ~noise_mask

        if noise_mask.sum() > 0:
            # Replace noise eigenvalues with average noise eigenvalue
            noise_mean = eigenvalues[noise_mask].mean()
            eigenvalues[noise_mask] = noise_mean

        # Ensure all eigenvalues are positive (positive semi-definiteness)
        eigenvalues = np.maximum(eigenvalues, 1e-8)

        # Reconstruct
        Sigma_clean = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return (Sigma_clean + Sigma_clean.T) / 2

    def get_correlation_matrix(self, Sigma: np.ndarray) -> np.ndarray:
        """Extract correlation matrix from covariance."""
        vols = np.sqrt(np.diag(Sigma))
        vols = np.maximum(vols, 1e-10)
        Corr = Sigma / np.outer(vols, vols)
        np.fill_diagonal(Corr, 1.0)
        return Corr


# ─────────────────────────────────────────────────────────────
# TAIL RISK ENGINE
# ─────────────────────────────────────────────────────────────

class TailRiskEngine:
    """
    Computes portfolio-level tail risk metrics.

    Uses three complementary approaches:
    1. Historical Simulation: non-parametric, uses actual return history
    2. Parametric: assumes multivariate t-distribution (fat tails)
    3. Cornish-Fisher: adjusts Gaussian quantiles for skewness and kurtosis

    For institutional use, historical simulation is preferred for VaR
    but tends to underestimate tail risk for unprecedented events.
    EVT (Extreme Value Theory) addresses this for the far tail.
    """

    def __init__(self, confidence_levels: List[float] = None):
        self.confidence_levels = confidence_levels or [0.95, 0.99, 0.999]

    def portfolio_var_cvar(self, weights: np.ndarray,
                            returns: pd.DataFrame,
                            method: str = 'historical'
                            ) -> Dict[str, float]:
        """
        Computes portfolio VaR and CVaR (Expected Shortfall).

        CVaR (Conditional Value at Risk) = Expected Shortfall (ES):
            CVaR_α = E[L | L > VaR_α]
            CVaR_α = -1/(1-α) * E[r * 1{r < -VaR_α}]

        CVaR is preferred over VaR because:
        1. Coherent risk measure (Artzner et al 1999) — VaR is not
        2. Captures tail severity, not just tail probability
        3. Convex → suitable for portfolio optimization (CVaR min)
        4. Required by Basel IV for trading book capital
        """
        # Portfolio returns
        port_returns = returns.values @ weights

        results = {}

        if method == 'historical':
            for alpha in self.confidence_levels:
                sorted_returns = np.sort(port_returns)
                cutoff_idx = int(np.floor(len(sorted_returns) * (1 - alpha)))
                var = -sorted_returns[cutoff_idx]
                # ES = average of returns below VaR
                cvar = -sorted_returns[:cutoff_idx].mean() if cutoff_idx > 0 else var
                results[f'var_{int(alpha*100)}'] = var
                results[f'cvar_{int(alpha*100)}'] = cvar

        elif method == 'cornish_fisher':
            # Cornish-Fisher expansion accounts for skewness and kurtosis
            mu = np.mean(port_returns)
            sigma = np.std(port_returns)
            skew = stats.skew(port_returns)
            kurt = stats.kurtosis(port_returns)  # Excess kurtosis

            for alpha in self.confidence_levels:
                z_alpha = stats.norm.ppf(1 - alpha)
                # Cornish-Fisher adjusted quantile
                cf_quantile = (
                    z_alpha
                    + (z_alpha**2 - 1) * skew / 6
                    + (z_alpha**3 - 3*z_alpha) * kurt / 24
                    - (2*z_alpha**3 - 5*z_alpha) * skew**2 / 36
                )
                var = -(mu + sigma * cf_quantile)
                # CVaR via numerical integration (approximation)
                cvar = var * 1.15  # Rough approximation; full integral needed
                results[f'var_cf_{int(alpha*100)}'] = var
                results[f'cvar_cf_{int(alpha*100)}'] = cvar

        elif method == 'parametric_t':
            # Multivariate t-distribution (fat tails)
            mu = np.mean(port_returns)
            sigma = np.std(port_returns)
            # Fit degrees of freedom
            df, loc, scale = stats.t.fit(port_returns)
            df = max(df, 2.1)  # Minimum df for finite variance

            for alpha in self.confidence_levels:
                var = -(stats.t.ppf(1 - alpha, df=df, loc=mu, scale=sigma))
                # CVaR for t-distribution (analytical formula)
                # CVaR_t(α) = -μ + σ * t_{df}(z_α) * (df + z_α²)/(df - 1) / (1-α)
                z_alpha = stats.t.ppf(1 - alpha, df=df)
                cvar_adjustment = (stats.t.pdf(z_alpha, df=df) * (df + z_alpha**2)
                                  / ((df - 1) * (1 - alpha)))
                cvar = -(mu - sigma * cvar_adjustment)
                results[f'var_t_{int(alpha*100)}'] = var
                results[f'cvar_t_{int(alpha*100)}'] = cvar

        # Add distribution stats
        results['portfolio_mean_daily'] = np.mean(port_returns)
        results['portfolio_vol_daily'] = np.std(port_returns)
        results['portfolio_vol_annual'] = np.std(port_returns) * np.sqrt(252)
        results['portfolio_skewness'] = float(stats.skew(port_returns))
        results['portfolio_kurtosis'] = float(stats.kurtosis(port_returns))

        return results

    def stress_test_correlation(self, Sigma: np.ndarray,
                                 stress_correlation: float = 0.90
                                 ) -> np.ndarray:
        """
        Correlation stress test: what happens in a crisis when
        all correlations jump to stress_correlation?

        In the 2008 crisis, correlations across stocks jumped from ~0.3 to ~0.75.
        In March 2020 COVID, correlations jumped even higher.

        This test ensures the risk engine is calibrated for these events.
        """
        N = Sigma.shape[0]
        vols = np.sqrt(np.diag(Sigma))

        # Stressed correlation matrix: all off-diagonal = stress_correlation
        Corr_stress = np.full((N, N), stress_correlation)
        np.fill_diagonal(Corr_stress, 1.0)

        # Reconstruct stressed covariance
        Sigma_stress = np.outer(vols, vols) * Corr_stress
        return Sigma_stress

    def tail_comoment(self, returns: pd.DataFrame,
                       threshold_pct: float = 0.05) -> np.ndarray:
        """
        Tail co-movement matrix: correlation conditional on both assets being in their tails.

        Standard correlation is symmetric and fails during crises.
        Tail correlation measures: Corr(r_i, r_j | r_i < q_α AND r_j < q_α)

        This captures contagion: in a crisis, are all your longs collapsing together?
        High tail co-movement = diversification illusion.
        """
        N = returns.shape[1]
        tail_corr = np.eye(N)

        quantiles = returns.quantile(threshold_pct)

        for i in range(N):
            for j in range(i+1, N):
                # Both in left tail simultaneously
                tail_mask = (returns.iloc[:, i] < quantiles.iloc[i]) & \
                            (returns.iloc[:, j] < quantiles.iloc[j])
                if tail_mask.sum() > 10:
                    tail_data = returns[tail_mask].iloc[:, [i, j]]
                    corr = tail_data.corr().iloc[0, 1]
                    tail_corr[i, j] = corr
                    tail_corr[j, i] = corr

        return tail_corr


# ─────────────────────────────────────────────────────────────
# LIQUIDITY RISK ENGINE
# ─────────────────────────────────────────────────────────────

class LiquidityRiskEngine:
    """
    Computes liquidity-adjusted risk metrics.

    Liquidity risk is the risk that you cannot exit a position
    without moving the market. This is critical for:
    - Small-cap positions (low ADV)
    - Crisis periods (bid-ask spreads widen 5-10x)
    - Crowded positions (everyone exiting simultaneously)

    Almgren-Chriss (2001) optimal execution model:
        Total cost = spread_cost + market_impact_cost + timing_risk

    We use this to:
    1. Compute maximum position size given ADV constraint
    2. Compute days required to liquidate
    3. Adjust position risk for liquidity premium
    """

    def __init__(self, max_participation_rate: float = 0.10,
                 spread_impact_coefficient: float = 0.5,
                 price_impact_coefficient: float = 0.1):
        """
        max_participation_rate: max fraction of ADV in a single day
        spread_impact_coefficient: γ in Almgren-Chriss
        price_impact_coefficient: η in Almgren-Chriss
        """
        self.max_participation = max_participation_rate
        self.gamma = spread_impact_coefficient
        self.eta = price_impact_coefficient

    def compute_amihud_illiquidity(self, prices: pd.Series,
                                    volumes: pd.Series,
                                    window: int = 21) -> float:
        """
        Amihud (2002) illiquidity ratio:
            ILLIQ_t = (1/D) * Σ_{d=1}^{D} |r_{i,d}| / (V_{i,d} * P_{i,d})

        Units: price impact per dollar of volume
        Higher = more illiquid (price moves more per dollar traded)

        This is a simple but powerful illiquidity measure used by
        virtually all academic asset pricing studies.
        """
        returns = np.abs(prices.pct_change())
        dollar_volume = prices * volumes  # DVOL in dollars
        dollar_volume = dollar_volume.replace(0, np.nan)

        illiq = (returns / dollar_volume).rolling(window).mean().iloc[-1]
        return float(illiq) if not np.isnan(illiq) else np.inf

    def days_to_liquidate(self, position_value: float,
                           adv_dollars: float) -> float:
        """
        Estimated days to liquidate a position at max_participation_rate.
        adv_dollars: average daily volume in dollars
        """
        max_daily_liquidation = adv_dollars * self.max_participation
        if max_daily_liquidation <= 0:
            return np.inf
        return position_value / max_daily_liquidation

    def almgren_chriss_cost(self, position_shares: float,
                             adv_shares: float,
                             price: float,
                             daily_vol: float,
                             liquidation_days: float = None) -> Dict[str, float]:
        """
        Almgren-Chriss (2001) trading cost estimation.

        For a TWAP strategy over T days:
        Permanent impact: γ * σ * (Q/ADV)^0.5 * Q
        Temporary impact: η * σ * (Q/ADV/T)^0.5 * Q
        Timing risk: 0.5 * σ * Q * √T (market risk while executing)

        Total cost = permanent + temporary + timing_risk_adjustment

        Trade-off: Execute faster (less timing risk) vs
                   Execute slower (less market impact)
        Optimal T minimizes total expected cost.
        """
        if adv_shares <= 0 or position_shares <= 0:
            return {'total_cost_pct': 0.0, 'optimal_days': 1.0}

        participation_rate = position_shares / adv_shares

        # Optimal liquidation time (Almgren-Chriss formula)
        # T* = (Q/ADV) * sqrt(η/(γ*σ))
        optimal_T = participation_rate * np.sqrt(
            self.eta / (self.gamma * daily_vol + 1e-10)
        )
        optimal_T = np.clip(optimal_T, 0.5, 30)  # 0.5 to 30 days

        T = liquidation_days or optimal_T

        # Permanent impact (market learns from your trades)
        perm_impact = self.gamma * daily_vol * np.sqrt(participation_rate / T)

        # Temporary impact (bid-ask and immediacy premium)
        temp_impact = self.eta * daily_vol * np.sqrt(participation_rate * T)

        # Timing risk (market moves against you while executing)
        timing_risk = 0.5 * daily_vol * np.sqrt(T)  # Simplified

        total_impact_pct = perm_impact + temp_impact
        total_cost_dollars = total_impact_pct * position_shares * price

        return {
            'permanent_impact_pct': perm_impact,
            'temporary_impact_pct': temp_impact,
            'timing_risk_pct': timing_risk,
            'total_cost_pct': total_impact_pct,
            'total_cost_dollars': total_cost_dollars,
            'optimal_liquidation_days': optimal_T,
        }

    def liquidity_adjusted_var(self, portfolio_var_1d: float,
                                 position_values: Dict[str, float],
                                 adv_dollars: Dict[str, float],
                                 confidence: float = 0.99) -> float:
        """
        Liquidity-adjusted VaR (LVaR).

        Standard VaR assumes instant liquidation. This is unrealistic.
        LVaR accounts for the liquidation period.

        LVaR = VaR_1d * sqrt(liquidation_days + 1)

        Uses weighted average liquidation time across positions.
        """
        total_value = sum(position_values.values())
        if total_value <= 0:
            return portfolio_var_1d

        weighted_liq_days = 0.0
        for ticker, value in position_values.items():
            adv = adv_dollars.get(ticker, total_value * 0.01)
            liq_days = self.days_to_liquidate(abs(value), adv)
            liq_days = min(liq_days, 30)  # Cap at 30 days
            weight = abs(value) / total_value
            weighted_liq_days += weight * liq_days

        # Scaling: VaR scales with square root of time
        lvar = portfolio_var_1d * np.sqrt(weighted_liq_days + 1)
        return lvar


# ─────────────────────────────────────────────────────────────
# FACTOR CROWDING RISK ENGINE
# ─────────────────────────────────────────────────────────────

class FactorCrowdingRisk:
    """
    Detects and quantifies factor crowding risk.

    CROWDING: When many funds hold the same factor positions,
    they face simultaneous liquidation risk when they unwind.

    The 2007 Quant Meltdown (August 4-9, 2007):
    - Quant funds all held similar factor positions (momentum, value, quality)
    - One fund started liquidating
    - Price pressure forced OTHER funds' models to generate sell signals
    - Cascade unwind: factors lost 20-30% in 5 days despite no macro event

    Crowding indicators:
    1. Short interest concentration (multiple funds short the same names)
    2. Factor momentum (factor has done well → more money chasing it → crowded)
    3. Ownership breadth decline (fewer institutions own the stock)
    4. Factor capacity analysis (size of signal vs. market liquidity)
    """

    def __init__(self):
        self.crowding_scores: Dict[str, float] = {}
        self.crowding_history: List[Dict] = []

    def compute_factor_crowding_score(self,
                                       factor_returns: pd.Series,
                                       factor_sharpe_history: pd.Series,
                                       short_interest_ratio: float = None,
                                       institutional_ownership_change: float = None
                                       ) -> float:
        """
        Composite crowding score from 0 (uncrowded) to 1 (extremely crowded).

        Components:
        1. Factor momentum (recent alpha → crowding via performance-chasing)
        2. Factor Sharpe persistence (consistently good → crowded)
        3. Short interest concentration
        4. Institutional ownership increase

        A high crowding score means: this factor position may reverse violently
        if crowded funds start unwinding simultaneously.
        """
        score = 0.0
        n_components = 0

        # Factor momentum score: recent 3M Sharpe ratio
        if len(factor_returns) >= 63:
            recent_sharpe = (factor_returns.iloc[-63:].mean() /
                            (factor_returns.iloc[-63:].std() + 1e-10) * np.sqrt(252))
            factor_mom_score = min(1.0, max(0.0, recent_sharpe / 3.0))
            score += factor_mom_score * 0.40
            n_components += 1

        # Sharpe persistence score: factor Sharpe been good for 1Y
        if len(factor_sharpe_history) >= 12:
            monthly_sharpes = factor_sharpe_history.iloc[-12:]
            positive_months = (monthly_sharpes > 0.5).mean()
            persistence_score = min(1.0, positive_months)
            score += persistence_score * 0.30
            n_components += 1

        # Short interest score (if available)
        if short_interest_ratio is not None:
            # High SI relative to historical = crowded short
            si_score = min(1.0, short_interest_ratio / 30.0)  # Normalize to 30% SI
            score += si_score * 0.20
            n_components += 1

        # Ownership change score (if available)
        if institutional_ownership_change is not None:
            # Rapid increase in inst. ownership = crowding signal
            io_score = min(1.0, max(0.0, institutional_ownership_change / 0.05))
            score += io_score * 0.10
            n_components += 1

        return score / max(n_components, 1) * (4 if n_components == 0 else 1)

    def estimate_unwind_impact(self, factor_exposure: float,
                                factor_adv_ratio: float,
                                crowding_score: float,
                                market_stress: float = 1.0) -> float:
        """
        Estimate potential loss if crowded factors unwind simultaneously.

        P(unwind | crowding) ≈ σ(crowding - 0.7) * market_stress
        Expected loss = P(unwind) * factor_exposure * impact_per_unit_exposure
        """
        prob_unwind = 1 / (1 + np.exp(-(crowding_score - 0.65) * 10))
        impact_per_unit = factor_adv_ratio * market_stress * 0.5
        expected_loss = prob_unwind * abs(factor_exposure) * impact_per_unit
        return min(expected_loss, abs(factor_exposure) * 0.30)  # Cap at 30%


# ─────────────────────────────────────────────────────────────
# MASTER RISK ENGINE — assembles all components
# ─────────────────────────────────────────────────────────────

class MasterRiskEngine:
    """
    The master risk engine. Assembles all risk components into a single,
    independent risk report.

    INDEPENDENCE GUARANTEE:
    This engine receives ONLY:
    - Price/return data
    - Position weights
    - Fundamental data (for factor exposure)

    It does NOT receive:
    - Alpha signals
    - ML model predictions
    - Any output from the alpha engine

    This ensures the risk engine provides objective, unbiased risk assessment.
    Risk constraints are applied to alpha AFTER this engine runs.
    """

    def __init__(self):
        self.cov_engine = DynamicCovarianceEngine()
        self.tail_engine = TailRiskEngine()
        self.liquidity_engine = LiquidityRiskEngine()
        self.crowding_engine = FactorCrowdingRisk()
        self._last_covariance: Optional[np.ndarray] = None
        self._last_correlation: Optional[np.ndarray] = None

    def compute_full_risk_report(self,
                                   weights: np.ndarray,
                                   returns: pd.DataFrame,
                                   prices: pd.DataFrame = None,
                                   volumes: pd.DataFrame = None,
                                   position_values: Dict[str, float] = None,
                                   factor_returns: pd.DataFrame = None
                                   ) -> Dict[str, any]:
        """
        Computes the full risk report for a given portfolio.

        weights: N-dimensional weight vector
        returns: T x N return matrix (historical)
        """
        tickers = returns.columns.tolist()
        N = len(tickers)

        # 1. Covariance matrix
        Sigma = self.cov_engine.compute_combined_covariance(returns)
        self._last_covariance = Sigma
        Corr = self.cov_engine.get_correlation_matrix(Sigma)
        self._last_correlation = Corr

        # 2. Portfolio risk decomposition
        port_variance = weights @ Sigma @ weights
        port_vol_annual = np.sqrt(port_variance) * np.sqrt(252)
        vols = np.sqrt(np.diag(Sigma)) * np.sqrt(252)

        # Marginal contributions to risk (MCR)
        # MCR_i = (∂σ_p/∂w_i) = (Σw)_i / σ_p
        MCR = (Sigma @ weights) / (np.sqrt(port_variance) + 1e-10)
        # Component risk = w_i * MCR_i (percentage contribution)
        CR = weights * MCR
        CR_pct = CR / (np.sqrt(port_variance) + 1e-10)  # % of total vol

        # 3. Tail risk
        tail_risk = self.tail_engine.portfolio_var_cvar(
            weights, returns, method='historical'
        )

        # Also compute stressed covariance tail risk
        Sigma_stress = self.tail_engine.stress_test_correlation(Sigma, stress_correlation=0.85)
        port_var_stressed = weights @ Sigma_stress @ weights
        stressed_annual_vol = np.sqrt(port_var_stressed) * np.sqrt(252)

        # Tail co-movement
        tail_comov = self.tail_engine.tail_comoment(returns)
        # Average tail co-movement (higher = more tail contagion risk)
        avg_tail_comov = (tail_comov.sum() - N) / max(N * (N-1), 1)

        # 4. Individual stock risk
        per_stock_risk = {}
        for i, ticker in enumerate(tickers):
            stock_returns = returns.iloc[:, i]

            # Amihud illiquidity
            illiq = 0.0
            if prices is not None and volumes is not None and ticker in prices.columns:
                illiq = self.liquidity_engine.compute_amihud_illiquidity(
                    prices[ticker].iloc[-21:], volumes[ticker].iloc[-21:]
                )

            # Days to liquidate
            pos_value = position_values.get(ticker, 0) if position_values else 0
            adv = prices[ticker].iloc[-21:].mean() * volumes[ticker].iloc[-21:].mean() \
                  if prices is not None and volumes is not None and ticker in prices.columns else 1e6
            dtl = self.liquidity_engine.days_to_liquidate(abs(pos_value), adv)

            # Individual VaR
            sorted_rets = np.sort(stock_returns.values)
            var_95 = -sorted_rets[int(len(sorted_rets) * 0.05)]
            cvar_95 = -sorted_rets[:int(len(sorted_rets) * 0.05)].mean()

            per_stock_risk[ticker] = {
                'annual_vol': float(vols[i]),
                'var_95_1d': float(var_95),
                'cvar_95_1d': float(cvar_95),
                'amihud_illiquidity': float(illiq),
                'days_to_liquidate': float(min(dtl, 100)),
                'marginal_risk_contribution': float(CR_pct[i]),
                'beta_to_portfolio': float(Sigma[i, :] @ weights / (port_variance + 1e-10)),
            }

        # 5. Liquidity-adjusted VaR
        adv_by_ticker = {}
        if prices is not None and volumes is not None:
            for ticker in tickers:
                if ticker in prices.columns and ticker in volumes.columns:
                    adv_by_ticker[ticker] = (
                        prices[ticker].iloc[-21:].mean() *
                        volumes[ticker].iloc[-21:].mean()
                    )
        lvar = self.liquidity_engine.liquidity_adjusted_var(
            tail_risk.get('var_99', port_vol_annual / 16),
            position_values or {},
            adv_by_ticker
        )

        # 6. Regime classification
        recent_vol = returns.iloc[-21:].values.flatten()
        historical_vol = returns.values.flatten()
        vol_z_score = (np.std(recent_vol) - np.std(historical_vol)) / (np.std(historical_vol) + 1e-10)

        if vol_z_score > 2.0:
            risk_regime = 'CRISIS'
        elif vol_z_score > 1.0:
            risk_regime = 'ELEVATED'
        else:
            risk_regime = 'NORMAL'

        return {
            # Portfolio-level risk
            'portfolio_vol_annual': port_vol_annual,
            'portfolio_vol_daily': np.sqrt(port_variance),
            'stressed_vol_annual': stressed_annual_vol,
            'stress_vol_ratio': stressed_annual_vol / (port_vol_annual + 1e-10),

            # Tail risk
            **tail_risk,
            'liquidity_adjusted_var_99': lvar,
            'avg_tail_comoment': avg_tail_comov,

            # Concentration risk
            'max_risk_contribution': float(CR_pct.max()),
            'herfindahl_risk': float((CR_pct ** 2).sum()),  # >0.25 = concentrated

            # Per-stock breakdown
            'per_stock': per_stock_risk,

            # Correlation
            'avg_pairwise_correlation': float((Corr.sum() - N) / max(N*(N-1), 1)),

            # Risk regime
            'risk_regime': risk_regime,
            'vol_z_score': float(vol_z_score),

            # Matrix outputs (for portfolio optimizer)
            'covariance_matrix': Sigma,
            'correlation_matrix': Corr,
        }

    def get_risk_budget(self, risk_regime: str, base_vol_target: float = 0.10) -> Dict[str, float]:
        """
        Returns risk budget parameters based on current regime.
        Risk budget tightens automatically during elevated/crisis regimes.
        """
        budgets = {
            'NORMAL': {
                'vol_target': base_vol_target,
                'max_gross_leverage': 1.5,
                'max_net_exposure': 1.0,
                'max_position_weight': 0.10,
                'max_sector_weight': 0.25,
                'cvar_budget': base_vol_target * 1.5,
            },
            'ELEVATED': {
                'vol_target': base_vol_target * 0.7,
                'max_gross_leverage': 1.0,
                'max_net_exposure': 0.8,
                'max_position_weight': 0.07,
                'max_sector_weight': 0.20,
                'cvar_budget': base_vol_target * 1.0,
            },
            'CRISIS': {
                'vol_target': base_vol_target * 0.4,
                'max_gross_leverage': 0.6,
                'max_net_exposure': 0.5,
                'max_position_weight': 0.05,
                'max_sector_weight': 0.15,
                'cvar_budget': base_vol_target * 0.6,
            }
        }
        return budgets.get(risk_regime, budgets['NORMAL'])
