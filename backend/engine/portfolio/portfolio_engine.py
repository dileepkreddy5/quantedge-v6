"""
QuantEdge v6.0 — PORTFOLIO CONSTRUCTION ENGINE (Layer 5)
════════════════════════════════════════════════════════════════
This layer takes alpha signals and risk estimates and constructs
an OPTIMAL PORTFOLIO subject to real-world constraints.

Portfolio construction is SEPARATE from alpha generation.
The optimizer doesn't know how alpha signals were computed.
It only sees:
  - Expected returns: μ vector
  - Covariance matrix: Σ
  - Current weights: w_0
  - Constraints: positions, sectors, liquidity, risk budget

This separation matters because:
  - The optimizer is a pure mathematical problem solver
  - Alpha researchers can change signals without touching optimizer
  - Risk team can tighten constraints without touching alpha
  - Governance can add new constraints without code changes

OPTIMIZATION FORMULATION:
══════════════════════════════════════════════════════

Primary objective: Minimize CVaR (Conditional Value at Risk)
    CVaR_α(w) = min_v { v + 1/(n(1-α)) * Σ_t max(-r_p_t - v, 0) }

Subject to:
    (1) Expected return ≥ μ_target
    (2) Σ w_i = 1 (fully invested, or w_cash ≥ 0)
    (3) |w_i| ≤ position_cap (individual stock limit)
    (4) |Σ_{i∈S} w_i| ≤ sector_cap (sector concentration)
    (5) |w_i - w_i^{t-1}| ≤ turnover_cap (trading cost control)
    (6) w_i ≤ f_i * ADV_i / portfolio_size (liquidity constraint)
    (7) w^T Σ w ≤ vol_budget² (total variance budget)
    (8) e_k^T Σ w / (w^T Σ w) ≤ risk_budget_k (factor risk budget)

The CVaR formulation is a LINEAR PROGRAM in the scenarios:
    min_w CVaR(w) = min_{w,v,u} v + 1/(T*(1-α)) * Σ u_t
    subject to:
        u_t ≥ -r_t^T w - v  for all t
        u_t ≥ 0

This makes it convex and efficiently solvable.

Mathematics references:
  - Rockafellar & Uryasev (2000): CVaR Optimization
  - Markowitz (1952): Mean-Variance Optimization (baseline)
  - Black & Litterman (1992): BL Model
  - Ledoit & Wolf (2017): Nonlinear Shrinkage for Optimization
  - DeMiguel et al (2009): 1/N beats mean-variance (estimation error)
════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("Warning: cvxpy not installed. Using scipy optimizer fallback.")


# ─────────────────────────────────────────────────────────────
# PORTFOLIO CONSTRAINTS
# ─────────────────────────────────────────────────────────────

@dataclass
class PortfolioConstraints:
    """
    All constraints for portfolio construction.
    Derived from the capital allocation engine (which is regime-aware).
    """
    # Position constraints
    max_position_weight: float = 0.05      # 5% max single position
    min_position_weight: float = -0.05     # -5% min (short side)
    max_positions: int = 50                # Maximum holdings

    # Sector constraints
    max_sector_weight: float = 0.20        # 20% max sector
    sector_map: Dict[str, str] = field(default_factory=dict)  # ticker → sector

    # Exposure constraints
    max_gross_leverage: float = 1.5        # Gross exposure
    max_net_exposure: float = 1.0          # Net long/short
    max_factor_exposure: float = 0.30      # Max exposure to any factor

    # Turnover constraints
    max_daily_turnover: float = 0.10       # 10% daily turnover max
    max_one_way_turnover: float = 0.05     # 5% one-way per day

    # Liquidity constraints
    max_adv_participation: float = 0.10    # 10% of ADV max position
    adv_dollars: Dict[str, float] = field(default_factory=dict)  # ticker → ADV

    # Risk budget
    vol_budget: float = 0.10              # 10% annual vol target
    cvar_budget: float = 0.015            # 1.5% daily CVaR budget

    # Return target
    min_expected_return: float = 0.0       # Return floor (usually 0)


@dataclass
class OptimizationResult:
    """Result from portfolio optimizer."""
    weights: Dict[str, float]
    expected_return: float
    expected_vol: float
    expected_cvar: float
    turnover_from_current: float
    gross_leverage: float
    net_exposure: float
    optimization_status: str             # 'optimal' | 'feasible' | 'infeasible'
    objective_value: float
    constraint_violations: List[str]     # Any violated constraints
    risk_contributions: Dict[str, float] # Per-asset risk contribution %


# ─────────────────────────────────────────────────────────────
# CVaR OPTIMIZER (primary)
# ─────────────────────────────────────────────────────────────

class CVaROptimizer:
    """
    Portfolio optimizer using CVaR (Conditional Value at Risk) objective.

    CVaR minimization is superior to mean-variance because:
    1. CVaR is a coherent risk measure (Artzner et al 1999)
    2. Naturally handles non-Gaussian return distributions
    3. Directly optimizes what matters: expected loss in bad scenarios
    4. The linear formulation is robust to estimation error

    The Rockafellar-Uryasev (2000) linear formulation:

    min_{w, v} v + 1/((1-α)*T) * Σ_t u_t
    s.t. u_t ≥ -r_t^T w - v    [scenario losses]
         u_t ≥ 0                [non-negative exceedances]
         Aw ≤ b                 [linear constraints]
         w^T Σ w ≤ σ²_budget   [quadratic variance constraint]

    When CVXPY is available, this is solved exactly.
    When not available, we use scipy with a mean-variance approximation.
    """

    def __init__(self, alpha: float = 0.95, n_scenarios: int = None):
        """
        alpha: confidence level for CVaR (0.95 = 95th percentile)
        n_scenarios: number of historical scenarios to use
        """
        self.alpha = alpha
        self.n_scenarios = n_scenarios

    def optimize_cvar(self,
                       expected_returns: pd.Series,
                       scenario_returns: pd.DataFrame,
                       covariance: np.ndarray,
                       current_weights: pd.Series,
                       constraints: PortfolioConstraints
                       ) -> OptimizationResult:
        """
        Solves the CVaR minimization problem.

        expected_returns: mu vector (N,) — ticker → expected return
        scenario_returns: T x N DataFrame of historical returns
        covariance: N x N covariance matrix
        current_weights: current portfolio weights (for turnover constraint)
        constraints: PortfolioConstraints object
        """
        tickers = expected_returns.index.tolist()
        N = len(tickers)

        # Align scenario returns with tickers
        scenario_returns_aligned = scenario_returns[tickers].dropna()
        T = min(len(scenario_returns_aligned), self.n_scenarios or 504)  # 2 years
        scenarios = scenario_returns_aligned.iloc[-T:].values

        current_w = current_weights.reindex(tickers).fillna(0).values
        mu = expected_returns.values

        if CVXPY_AVAILABLE:
            result = self._optimize_cvxpy(
                mu, scenarios, covariance, current_w, constraints, tickers
            )
        else:
            result = self._optimize_scipy_mv(
                mu, covariance, current_w, constraints, tickers
            )

        return result

    def _optimize_cvxpy(self, mu: np.ndarray, scenarios: np.ndarray,
                         Sigma: np.ndarray, current_w: np.ndarray,
                         constraints: PortfolioConstraints,
                         tickers: List[str]) -> OptimizationResult:
        """CVaR optimization via CVXPY (exact solution)."""
        N = len(tickers)
        T = len(scenarios)

        # Decision variables
        w = cp.Variable(N, name='weights')       # Portfolio weights
        v = cp.Variable(name='var_threshold')     # VaR threshold
        u = cp.Variable(T, name='cvar_slack')     # CVaR auxiliary vars

        # Objective: minimize CVaR
        cvar = v + (1 / (T * (1 - self.alpha))) * cp.sum(u)
        objective = cp.Minimize(cvar)

        # Constraints
        constrs = []

        # CVaR scenario constraints: u_t ≥ -r_t^T w - v
        for t in range(T):
            constrs.append(u[t] >= -scenarios[t] @ w - v)
        constrs.append(u >= 0)

        # Position limits
        constrs.append(w <= constraints.max_position_weight)
        constrs.append(w >= constraints.min_position_weight)

        # Budget constraint (fully invested or less)
        constrs.append(cp.sum(w) == 1.0)

        # Gross leverage (sum of absolute weights)
        constrs.append(cp.norm1(w) <= constraints.max_gross_leverage)

        # Expected return floor
        constrs.append(mu @ w >= constraints.min_expected_return)

        # Turnover constraint: Σ |w_i - w0_i| ≤ 2 * max_one_way_turnover
        turnover = cp.norm1(w - current_w)
        constrs.append(turnover <= 2 * constraints.max_daily_turnover)

        # Volatility budget: w^T Σ w ≤ vol_budget^2 / 252 (daily)
        daily_vol_budget = (constraints.vol_budget ** 2) / 252
        constrs.append(cp.quad_form(w, Sigma) <= daily_vol_budget)

        # Sector constraints
        if constraints.sector_map:
            sectors = list(set(constraints.sector_map.values()))
            for sector in sectors:
                sector_tickers = [t for t in tickers
                                  if constraints.sector_map.get(t) == sector]
                if sector_tickers:
                    sector_idx = [tickers.index(t) for t in sector_tickers]
                    sector_weight = cp.sum(w[sector_idx])
                    constrs.append(sector_weight <= constraints.max_sector_weight)
                    constrs.append(sector_weight >= -constraints.max_sector_weight)

        # Liquidity constraints: position ≤ ADV_i * participation_rate / portfolio_AUM
        # (We normalize: assume portfolio AUM = 1 for weight purposes)
        if constraints.adv_dollars:
            total_adv = sum(constraints.adv_dollars.values())
            if total_adv > 0:
                for i, ticker in enumerate(tickers):
                    if ticker in constraints.adv_dollars:
                        max_liq_weight = (
                            constraints.adv_dollars[ticker]
                            * constraints.max_adv_participation
                            / total_adv * N  # Scale to portfolio
                        )
                        max_liq_weight = min(max_liq_weight,
                                             constraints.max_position_weight)
                        constrs.append(cp.abs(w[i]) <= max_liq_weight)

        # Solve
        prob = cp.Problem(objective, constrs)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            try:
                prob.solve(solver=cp.SCS, verbose=False)
            except Exception:
                prob.solve(verbose=False)

        if prob.status in ['optimal', 'optimal_inaccurate']:
            opt_weights = w.value
            cvar_value = float(prob.value)
            status = 'optimal'
        elif prob.status == 'infeasible':
            # Fallback: use equal-weight portfolio
            opt_weights = current_w.copy()
            cvar_value = 0.0
            status = 'infeasible'
        else:
            opt_weights = current_w.copy()
            cvar_value = 0.0
            status = 'failed'

        return self._build_result(opt_weights, mu, Sigma, current_w,
                                   tickers, cvar_value, status, constraints)

    def _optimize_scipy_mv(self, mu: np.ndarray, Sigma: np.ndarray,
                             current_w: np.ndarray,
                             constraints: PortfolioConstraints,
                             tickers: List[str]) -> OptimizationResult:
        """
        Mean-variance optimization fallback when CVXPY not available.
        Maximizes Sharpe ratio subject to constraints.
        """
        N = len(tickers)

        def negative_sharpe(w):
            port_ret = mu @ w
            port_var = w @ Sigma @ w
            return -(port_ret / (np.sqrt(port_var) + 1e-10))

        def portfolio_vol(w):
            return np.sqrt(w @ Sigma @ w)

        bounds = [(constraints.min_position_weight,
                   constraints.max_position_weight)] * N

        scipy_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'ineq', 'fun': lambda w: constraints.vol_budget/np.sqrt(252)
                                              - portfolio_vol(w)},
            {'type': 'ineq', 'fun': lambda w: 2*constraints.max_daily_turnover
                                              - np.sum(np.abs(w - current_w))},
        ]

        w0 = current_w.copy()
        w0 = w0 / (np.sum(np.abs(w0)) + 1e-10)

        try:
            result = minimize(negative_sharpe, w0, method='SLSQP',
                             bounds=bounds, constraints=scipy_constraints,
                             options={'maxiter': 1000, 'ftol': 1e-8})
            opt_weights = result.x
            status = 'optimal' if result.success else 'feasible'
        except Exception:
            opt_weights = np.ones(N) / N
            status = 'failed'

        # Estimate CVaR from portfolio vol (approximation)
        port_vol = float(np.sqrt(opt_weights @ Sigma @ opt_weights))
        cvar_approx = port_vol * 1.65  # Gaussian approximation at 95%

        return self._build_result(opt_weights, mu, Sigma, current_w,
                                   tickers, cvar_approx, status, constraints)

    def _build_result(self, weights: np.ndarray, mu: np.ndarray,
                       Sigma: np.ndarray, current_w: np.ndarray,
                       tickers: List[str], cvar: float, status: str,
                       constraints: PortfolioConstraints) -> OptimizationResult:
        """Constructs the OptimizationResult from raw optimization output."""

        # Clean weights: zero out tiny positions (< 0.1%)
        weights = np.where(np.abs(weights) < 0.001, 0.0, weights)

        # Re-normalize
        if np.sum(np.abs(weights)) > 0:
            total = np.sum(weights)
            if abs(total) > 0.01:
                weights = weights / abs(total) * np.sign(total)

        port_ret = float(mu @ weights)
        port_var = float(weights @ Sigma @ weights)
        port_vol = float(np.sqrt(port_var) * np.sqrt(252))
        turnover = float(np.sum(np.abs(weights - current_w)))
        gross_lev = float(np.sum(np.abs(weights)))
        net_exp = float(np.sum(weights))

        # Risk contributions
        MCR = (Sigma @ weights) / (np.sqrt(port_var) + 1e-10)
        CR = weights * MCR
        CR_pct = CR / (np.sqrt(port_var) + 1e-10) if port_var > 0 else CR

        risk_contribs = {tickers[i]: float(CR_pct[i]) for i in range(len(tickers))}

        # Check constraint violations
        violations = []
        if gross_lev > constraints.max_gross_leverage * 1.01:
            violations.append(f"Gross leverage {gross_lev:.2f} > {constraints.max_gross_leverage}")
        if abs(net_exp) > constraints.max_net_exposure * 1.01:
            violations.append(f"Net exposure {net_exp:.2f}")
        if np.max(np.abs(weights)) > constraints.max_position_weight * 1.01:
            violations.append(f"Max position {np.max(np.abs(weights)):.3f}")
        if turnover > 2 * constraints.max_daily_turnover * 1.01:
            violations.append(f"Turnover {turnover:.3f}")

        return OptimizationResult(
            weights={tickers[i]: float(weights[i]) for i in range(len(tickers))},
            expected_return=port_ret * 252,   # Annualized
            expected_vol=port_vol,
            expected_cvar=float(cvar),
            turnover_from_current=turnover,
            gross_leverage=gross_lev,
            net_exposure=net_exp,
            optimization_status=status,
            objective_value=float(cvar),
            constraint_violations=violations,
            risk_contributions=risk_contribs,
        )


# ─────────────────────────────────────────────────────────────
# BLACK-LITTERMAN MODEL
# ─────────────────────────────────────────────────────────────

class BlackLittermanModel:
    """
    Black-Litterman (1992) model: combines market equilibrium with views.

    PROBLEM WITH MARKOWITZ:
    Mean-variance optimization with sample estimates produces portfolios
    that are overly concentrated and unstable (DeMiguel et al 2009).
    The "error maximizer" problem: optimization amplifies estimation errors.

    BLACK-LITTERMAN SOLUTION:
    1. Start with equilibrium expected returns (CAPM implied)
    2. Express views as: Pμ = Q + ε, ε ~ N(0, Ω)
    3. Bayesian update: μ_BL = [(τΣ)^{-1} + P^T Ω^{-1} P]^{-1} * [...]
    4. Optimize mean-variance on μ_BL

    This produces:
    - Diversified, stable portfolios
    - Incorporates model views with appropriate uncertainty
    - Reverts to market portfolio when views have no conviction

    References: Black & Litterman (1992): Global Portfolio Optimization
    """

    def __init__(self, risk_aversion: float = 3.0, tau: float = 0.05):
        """
        risk_aversion: λ in CAPM equilibrium (typically 2-4 for equities)
        tau: uncertainty in prior (typically 0.05 = 5%)
        """
        self.risk_aversion = risk_aversion
        self.tau = tau

    def compute_equilibrium_returns(self, market_weights: np.ndarray,
                                    Sigma: np.ndarray) -> np.ndarray:
        """
        CAPM implied equilibrium returns: Π = λ * Σ * w_market
        These are the "center of gravity" for BL.
        """
        return self.risk_aversion * Sigma @ market_weights

    def combine_views(self, Sigma: np.ndarray,
                       equilibrium_returns: np.ndarray,
                       view_matrix: np.ndarray,        # P matrix (K x N)
                       view_returns: np.ndarray,        # Q vector (K,)
                       view_confidence: np.ndarray      # Ω diagonal (K,)
                       ) -> np.ndarray:
        """
        BL posterior expected returns.

        μ_BL = [(τΣ)^{-1} + P^T Ω^{-1} P]^{-1} * [(τΣ)^{-1}Π + P^T Ω^{-1} Q]

        view_matrix P: rows are views, columns are assets
            Example view: "AAPL will outperform MSFT by 2%"
            P row: [1, -1, 0, 0, ...] for (AAPL long, MSFT short, others 0)
        """
        tau_Sigma = self.tau * Sigma
        try:
            tau_Sigma_inv = np.linalg.inv(tau_Sigma)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            tau_Sigma_inv = np.linalg.pinv(tau_Sigma)

        # Omega: diagonal matrix of view uncertainties
        Omega = np.diag(view_confidence)
        try:
            Omega_inv = np.diag(1.0 / (view_confidence + 1e-10))
        except Exception:
            Omega_inv = np.eye(len(view_confidence))

        # BL update
        M1 = tau_Sigma_inv + view_matrix.T @ Omega_inv @ view_matrix
        M2 = tau_Sigma_inv @ equilibrium_returns + view_matrix.T @ Omega_inv @ view_returns

        try:
            mu_BL = np.linalg.solve(M1, M2)
        except np.linalg.LinAlgError:
            mu_BL = np.linalg.lstsq(M1, M2, rcond=None)[0]

        return mu_BL

    def views_from_alpha(self, alpha_signals: pd.Series,
                          tickers: List[str],
                          Sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts alpha signals into BL view matrices.

        For each signal, we create a long-short view:
        "Top quintile stocks will outperform bottom quintile by x%"

        This is more sophisticated than just plugging signals into μ directly,
        because it accounts for covariance structure.
        """
        N = len(tickers)
        signals = alpha_signals.reindex(tickers).fillna(0).values

        # Rank signals
        ranks = pd.Series(signals).rank(pct=True).values

        # Top 20% vs Bottom 20% view
        long_mask = ranks >= 0.80
        short_mask = ranks <= 0.20

        if long_mask.sum() == 0 or short_mask.sum() == 0:
            # No strong views
            return np.zeros((1, N)), np.zeros(1), np.ones(1) * 0.01

        # View: long top quintile, short bottom quintile
        P = np.zeros((1, N))
        P[0, long_mask] = 1.0 / long_mask.sum()
        P[0, short_mask] = -1.0 / short_mask.sum()

        # Expected return of this long-short: based on IC and vol
        cross_vol = np.sqrt(np.diag(Sigma)).mean() * np.sqrt(252)
        ic = 0.05  # Conservative prior
        expected_ls_return = 2 * ic * cross_vol  # Rough approximation

        Q = np.array([expected_ls_return])

        # View uncertainty: inversely proportional to signal dispersion
        signal_dispersion = np.std(signals[long_mask]) + np.std(signals[short_mask])
        Omega = np.array([cross_vol ** 2 / max(signal_dispersion, 0.1)])

        return P, Q, Omega


# ─────────────────────────────────────────────────────────────
# RISK PARITY (alternative portfolio construction)
# ─────────────────────────────────────────────────────────────

class RiskParityPortfolio:
    """
    Risk Parity: equalize risk contributions across all assets.

    Motivation: Equal-weight portfolios are dominated by high-vol assets.
    Market-cap weighted portfolios are dominated by largest names.
    Risk parity ensures each asset contributes equally to total portfolio risk.

    Formulation:
        Find w such that: w_i * (Σw)_i = σ_p² / N for all i

    This is a nonlinear equation system. Solved via Newton's method.
    Bridgewater's All Weather strategy is based on risk parity.

    Advantage: Works even without expected return estimates
    (avoids estimation error problem entirely)
    """

    def optimize(self, Sigma: np.ndarray, tickers: List[str],
                  risk_budget: np.ndarray = None) -> OptimizationResult:
        """
        Solve for risk parity weights.

        risk_budget: optional target risk contribution per asset (default: equal)
        """
        N = len(tickers)
        if risk_budget is None:
            risk_budget = np.ones(N) / N  # Equal risk contribution

        def risk_parity_objective(w):
            w = np.maximum(w, 1e-6)  # Ensure positive weights
            port_var = w @ Sigma @ w
            port_vol = np.sqrt(port_var)
            MCR = Sigma @ w / (port_vol + 1e-10)  # Marginal contribution
            CR = w * MCR / (port_vol + 1e-10)      # % contribution
            # Minimize squared deviation from target
            return np.sum((CR - risk_budget) ** 2)

        def grad(w):
            w = np.maximum(w, 1e-6)
            port_var = w @ Sigma @ w
            port_vol = np.sqrt(port_var)
            MCR = Sigma @ w / (port_vol + 1e-10)
            CR = w * MCR / (port_vol + 1e-10)
            # Gradient via chain rule (simplified)
            return 2 * (CR - risk_budget) * MCR / (port_vol + 1e-10)

        w0 = np.ones(N) / N
        bounds = [(0.001, 0.20)] * N  # Long-only, max 20%
        constraints_sp = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        try:
            result = minimize(risk_parity_objective, w0, jac=grad,
                             method='SLSQP', bounds=bounds,
                             constraints=constraints_sp,
                             options={'maxiter': 2000, 'ftol': 1e-10})
            opt_w = result.x
            status = 'optimal' if result.success else 'feasible'
        except Exception:
            opt_w = np.ones(N) / N
            status = 'failed'

        # Normalize
        opt_w = opt_w / opt_w.sum()

        port_var = float(opt_w @ Sigma @ opt_w)
        port_vol = float(np.sqrt(port_var) * np.sqrt(252))
        MCR = Sigma @ opt_w / (np.sqrt(port_var) + 1e-10)
        CR = opt_w * MCR / (np.sqrt(port_var) + 1e-10)

        return OptimizationResult(
            weights={tickers[i]: float(opt_w[i]) for i in range(N)},
            expected_return=0.0,  # Risk parity doesn't use expected returns
            expected_vol=port_vol,
            expected_cvar=port_vol * 1.65 / np.sqrt(252),  # Approx
            turnover_from_current=1.0,  # Unknown without current portfolio
            gross_leverage=float(np.sum(np.abs(opt_w))),
            net_exposure=float(np.sum(opt_w)),
            optimization_status=status,
            objective_value=float(risk_parity_objective(opt_w)),
            constraint_violations=[],
            risk_contributions={tickers[i]: float(CR[i]) for i in range(N)},
        )
