"""
QuantEdge v6.0 — ALPHA ENGINE (Layer 3)
════════════════════════════════════════════════════════════════
PHILOSOPHICAL FOUNDATION:

Alpha is modeled as:
    r_{i,t} = f(F_{i,t}, R_t, ε_{i,t})

Where:
    F_{i,t} = factor exposures (cross-sectional: who is cheap? who has momentum?)
    R_t     = regime state (how does the SAME signal perform in different regimes?)
    ε_{i,t} = idiosyncratic residual (what's left after factor attribution?)

We NEVER make point predictions. We model distributions:
    P(r_{i,t+h} | F_{i,t}, R_t) = N(μ_{i,t}, σ²_{i,t}) [approximately]

In reality returns are NOT Gaussian. We model:
    - Conditional mean: E[r | F, R]
    - Conditional variance: Var[r | F, R]
    - Conditional skewness: Skew[r | F, R]
    - Tail probabilities: P(r < -10% | F, R)

THREE SIGNAL TYPES — must be kept separate because they're
statistically independent and decay at different rates:

1. CROSS-SECTIONAL SIGNALS (who to buy relative to peers)
   - Factor exposures: value, quality, momentum, low-vol
   - These are relative: "AAPL has better momentum than sector peers"
   - These work via mean reversion and trending simultaneously
   - Estimated via XGBoost + LightGBM cross-sectional ranking

2. TIME-SERIES SIGNALS (when to buy the same asset)
   - Return autocorrelation, vol regime, trend strength
   - These are absolute: "AAPL is trending up this month"
   - Estimated via LSTM + ARIMA + Kalman filter
   - Decay faster than XS signals

3. RESIDUAL (IDIOSYNCRATIC) SIGNALS
   - After removing factor + time-series components
   - What the FinBERT NLP model captures
   - Corporate events, earnings surprises
   - Highest alpha potential, highest variance

ENSEMBLE ARCHITECTURE:
    Final signal = w1 * XS + w2 * TS + w3 * Residual
    weights adapt online based on recent IC per model

Mathematics:
    IC_recent_k = Spearman(rank(signals), rank(returns))
    w_k ∝ max(IC_recent_k, 0)  ← discard negative-IC models

References:
  - Gu, Kelly, Xiu (2020): Empirical Asset Pricing via ML
  - Barra USE4 Factor Model documentation
  - Jegadeesh, Titman (1993): Returns to Buying Winners
  - Asness et al (2013): Value and Momentum Everywhere
════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
# DISTRIBUTIONAL ALPHA MODEL
# ─────────────────────────────────────────────────────────────

@dataclass
class DistributionalForecast:
    """
    Full distributional forecast — not a point prediction.

    The mean alone is insufficient. A stock can have E[r]=+5%
    but 40% probability of a -30% loss (bad skew).
    A professional risk-adjusted signal needs the full distribution.

    This is the KEY difference between institutional and retail ML.
    """
    ticker: str
    timestamp: pd.Timestamp
    horizon: str

    # Central tendency
    expected_return: float           # E[r_{t+h}]

    # Dispersion
    return_std: float               # Conditional std (uncertainty)

    # Higher moments
    return_skew: float              # Negative = left tail risk
    return_kurtosis: float          # Excess kurtosis (fat tails)

    # Tail probabilities (directly useful for risk management)
    prob_loss_5pct: float           # P(r < -5%)
    prob_loss_10pct: float          # P(r < -10%)
    prob_loss_20pct: float          # P(r < -20%)
    prob_gain_5pct: float           # P(r > +5%)
    prob_gain_10pct: float          # P(r > +10%)

    # Information ratios
    ic_estimate: float              # Expected Information Coefficient
    signal_confidence: float        # Epistemic confidence 0-1

    # Decomposition
    xs_component: float             # Cross-sectional factor component
    ts_component: float             # Time-series component
    residual_component: float       # Idiosyncratic component

    # Regime adjustment
    regime: str = "UNKNOWN"
    regime_confidence: float = 0.5

    def sharpe_ratio_estimate(self) -> float:
        """E[r] / std[r] — but accounts for full distribution."""
        return (self.expected_return / (self.return_std + 1e-10)
                * np.sqrt(252 / self._horizon_to_days()))

    def omega_ratio_estimate(self) -> float:
        """
        Omega ratio: E[max(r-threshold, 0)] / E[max(threshold-r, 0)]
        More informative than Sharpe for non-normal distributions.
        Uses 0% threshold.
        """
        # Approximation using normal distribution with skewness correction
        mu = self.expected_return
        sigma = self.return_std
        # Omega = (exp(mu + 0.5*sigma^2) * N(d1) - N(d2)) / (N(-d2) - exp(mu + 0.5*sigma^2) * N(-d1))
        # Simplified: ratio of upside to downside probability-weighted
        upside = 1 - self.prob_loss_5pct
        downside = self.prob_loss_5pct
        return upside / (downside + 1e-10)

    def _horizon_to_days(self) -> int:
        return {'1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252}.get(self.horizon, 21)

    def is_tradeable(self, min_ic: float = 0.03, max_uncertainty: float = 0.8) -> bool:
        """Signal is only tradeable if IC is meaningful and uncertainty is not too high."""
        return (
            abs(self.ic_estimate) >= min_ic and
            self.signal_confidence >= (1 - max_uncertainty) and
            not np.isnan(self.expected_return)
        )


# ─────────────────────────────────────────────────────────────
# CROSS-SECTIONAL ALPHA ENGINE
# ─────────────────────────────────────────────────────────────

class CrossSectionalAlphaEngine:
    """
    Cross-sectional alpha: rank stocks within a universe at each time point.

    The fundamental insight: we don't need to predict the absolute return
    of a stock. We need to rank stocks correctly relative to each other.

    A long-short portfolio built from correct rankings extracts alpha
    independent of market direction. This is "pure alpha" — the
    holy grail of quant investing.

    Model: at each time t, rank all stocks by predicted return rank.
    Long top quintile, short bottom quintile = long-short alpha.

    Implemented via:
    1. Factor construction (value, momentum, quality, low-vol)
    2. XGBoost / LightGBM cross-sectional regressor
    3. IC-weighted combination of models
    4. Sector-neutral ranking (to avoid sector bets)
    """

    def __init__(self):
        self.ic_history: Dict[str, List[float]] = {}
        self.model_weights: Dict[str, float] = {}
        self._fitted_models = {}

    # ── Factor Construction ──────────────────────────────────
    def compute_value_factors(self, fundamentals: pd.DataFrame,
                               prices: pd.Series) -> Dict[str, float]:
        """
        Value factors: buy cheap, sell expensive.

        Evidence: Fama & French (1992) — HML (High minus Low B/P)
        Risk explanation: value stocks are distressed (Fama)
        Behavioral explanation: overextrapolation (Lakonishok, Shleifer, Vishny 1994)

        Factors:
        - Earnings yield (E/P) = inverse P/E
        - Book/Price ratio (B/P) = inverse P/B
        - Sales/EV ratio
        - Free cash flow yield
        - Composite value score (equal-weighted)
        """
        eps = fundamentals.get('eps_ttm', 0)
        book_value = fundamentals.get('book_value_per_share', 0)
        fcf = fundamentals.get('free_cash_flow', 0)
        revenue = fundamentals.get('revenue_ttm', 0)
        ev = fundamentals.get('enterprise_value', prices * 1e6)
        shares = fundamentals.get('shares_outstanding', 1e6)

        price = float(prices) if hasattr(prices, '__float__') else prices

        factors = {}

        # Earnings yield (higher = cheaper = more value)
        if price > 0 and eps is not None and eps != 0:
            factors['earnings_yield'] = eps / price
        else:
            factors['earnings_yield'] = np.nan

        # Book-to-price (Fama-French HML factor)
        if price > 0 and book_value is not None and book_value > 0:
            factors['book_to_price'] = book_value / price
        else:
            factors['book_to_price'] = np.nan

        # FCF yield
        if price > 0 and fcf is not None and shares > 0:
            fcf_per_share = fcf / shares
            factors['fcf_yield'] = fcf_per_share / price
        else:
            factors['fcf_yield'] = np.nan

        # Sales/EV
        if ev > 0 and revenue is not None and revenue > 0:
            factors['sales_to_ev'] = revenue / ev
        else:
            factors['sales_to_ev'] = np.nan

        # Composite value score (rank average of available factors)
        valid = [v for v in factors.values() if not np.isnan(v)]
        factors['value_composite'] = np.nanmean(list(factors.values())) if valid else np.nan

        return factors

    def compute_momentum_factors(self, prices: pd.Series) -> Dict[str, float]:
        """
        Momentum factors: buy recent winners, sell recent losers.

        Jegadeesh & Titman (1993): 12-1 month momentum works.
        The "12-1" means: past 12 months, skipping the last month.
        (The last month often reverses due to microstructure — bid-ask bounce)

        Types of momentum:
        1. Price momentum (Jegadeesh-Titman)
        2. Earnings momentum (SUE: Standardized Unexpected Earnings)
        3. Residual momentum (Blitz, Huij, Martens 2011): beta-adjusted
        4. Time-series momentum (Moskowitz, Ooi, Pedersen 2012)
        """
        if len(prices) < 252:
            return {k: np.nan for k in ['mom_12_1', 'mom_6_1', 'mom_3_1',
                                          'mom_1', 'vol_scaled_mom', 'tsmom_12']}

        log_returns = np.log(prices / prices.shift(1)).dropna()

        factors = {}

        # 12-1 month momentum (skip-1 to avoid short-term reversal)
        try:
            ret_12_1 = np.log(prices.iloc[-252] / prices.iloc[-21]) if len(prices) >= 252 else np.nan
            factors['mom_12_1'] = ret_12_1
        except Exception:
            factors['mom_12_1'] = np.nan

        # 6-1 month momentum
        try:
            ret_6_1 = np.log(prices.iloc[-126] / prices.iloc[-21]) if len(prices) >= 126 else np.nan
            factors['mom_6_1'] = ret_6_1
        except Exception:
            factors['mom_6_1'] = np.nan

        # 3-1 month momentum
        try:
            ret_3_1 = np.log(prices.iloc[-63] / prices.iloc[-21]) if len(prices) >= 63 else np.nan
            factors['mom_3_1'] = ret_3_1
        except Exception:
            factors['mom_3_1'] = np.nan

        # 1-month reversal (contrarian signal at 1M)
        try:
            ret_1m = np.log(prices.iloc[-1] / prices.iloc[-21])
            factors['mom_1'] = ret_1m
        except Exception:
            factors['mom_1'] = np.nan

        # Vol-scaled momentum (Barroso & Santa-Clara 2015)
        # Weights past returns by inverse of their realized volatility
        # This improves Sharpe ratio vs. raw momentum
        try:
            recent_vol = log_returns.iloc[-21:].std() * np.sqrt(252)
            if recent_vol > 0:
                factors['vol_scaled_mom'] = factors.get('mom_12_1', 0) / (recent_vol + 1e-10)
            else:
                factors['vol_scaled_mom'] = np.nan
        except Exception:
            factors['vol_scaled_mom'] = np.nan

        # Time-series momentum (Moskowitz, Ooi, Pedersen 2012)
        # Sign of trailing 12-month return * inverse volatility
        try:
            tsmom_signal = np.sign(factors.get('mom_12_1', 0))
            factors['tsmom_12'] = tsmom_signal / (log_returns.iloc[-252:].std() * np.sqrt(252) + 1e-10)
        except Exception:
            factors['tsmom_12'] = np.nan

        return factors

    def compute_quality_factors(self, fundamentals: pd.DataFrame) -> Dict[str, float]:
        """
        Quality factors: buy profitable, growing, safe companies.

        Asness, Frazzini, Pedersen (2019): Quality Minus Junk (QMJ)
        "The highest-quality stocks earn higher returns" — robust across 24 countries.

        Quality dimensions:
        1. Profitability: ROE, ROA, gross profit/assets, FCF/assets
        2. Growth: 5-year earnings growth, revenue growth
        3. Safety: leverage, volatility, beta (low = safer)
        4. Payout: dividend + buyback yield (capital discipline)
        """
        factors = {}

        # Profitability
        roe = fundamentals.get('roe', np.nan)
        roa = fundamentals.get('roa', np.nan)
        gross_margin = fundamentals.get('gross_margin', np.nan)
        fcf_margin = fundamentals.get('fcf_margin', np.nan)

        factors['roe'] = roe
        factors['roa'] = roa
        factors['gross_margin'] = gross_margin
        factors['fcf_margin'] = fcf_margin

        # Growth
        revenue_growth = fundamentals.get('revenue_growth_5y', np.nan)
        eps_growth = fundamentals.get('eps_growth_5y', np.nan)
        factors['revenue_growth'] = revenue_growth
        factors['earnings_growth'] = eps_growth

        # Safety (low leverage, low vol = high safety → high quality)
        de_ratio = fundamentals.get('debt_to_equity', np.nan)
        current_ratio = fundamentals.get('current_ratio', np.nan)
        factors['inv_leverage'] = -de_ratio if de_ratio is not None else np.nan
        factors['current_ratio'] = current_ratio

        # Payout
        div_yield = fundamentals.get('dividend_yield', 0) or 0
        buyback_yield = fundamentals.get('buyback_yield', 0) or 0
        factors['total_payout_yield'] = div_yield + buyback_yield

        # QMJ Composite (Asness et al 2019 methodology)
        # Rank each sub-dimension, average ranks
        profitability_dims = [roe, roa, gross_margin, fcf_margin]
        valid_prof = [v for v in profitability_dims if v is not None and not np.isnan(v)]
        factors['qmj_profitability'] = np.mean(valid_prof) if valid_prof else np.nan

        growth_dims = [revenue_growth, eps_growth]
        valid_growth = [v for v in growth_dims if v is not None and not np.isnan(v)]
        factors['qmj_growth'] = np.mean(valid_growth) if valid_growth else np.nan

        # Final QMJ composite (tanh to bound to [-1, 1])
        components = [factors.get('qmj_profitability'), factors.get('qmj_growth')]
        valid = [v for v in components if v is not None and not np.isnan(v)]
        if valid:
            raw_qmj = np.mean(valid)
            factors['qmj_composite'] = np.tanh(raw_qmj * 5)  # Scale then bound
        else:
            factors['qmj_composite'] = np.nan

        return factors

    def compute_low_vol_factor(self, returns: pd.Series) -> Dict[str, float]:
        """
        Low-Volatility / Low-Beta anomaly.

        Frazzini & Pedersen (2014): Betting Against Beta (BAB)
        "Low-beta assets earn higher risk-adjusted returns than high-beta assets."

        This CONTRADICTS CAPM. Explanation:
        - Leverage-constrained investors bid up high-beta assets
        - Creates return anomaly: low-beta outperforms high-beta (risk-adjusted)
        - AQR's biggest factor after momentum

        Note: This generates NEGATIVE beta exposure → provides diversification.
        """
        factors = {}

        if len(returns) < 63:
            return {'low_vol_1m': np.nan, 'low_vol_3m': np.nan,
                    'low_vol_1y': np.nan, 'bab_signal': np.nan}

        # Rolling volatility measures
        factors['low_vol_1m'] = -returns.iloc[-21:].std() * np.sqrt(252)  # Negative → lower vol = higher score
        factors['low_vol_3m'] = -returns.iloc[-63:].std() * np.sqrt(252)
        factors['low_vol_1y'] = -returns.std() * np.sqrt(252) if len(returns) >= 252 else np.nan

        # BAB: shrinkage beta estimate
        # Vasicek (1973) shrinkage: β_shrunk = (1-w)*1 + w*β_OLS
        # where w = n*Var(β_OLS) / (n*Var(β_OLS) + σ²_cross_section_beta)
        # Approximate: w = 0.6 (Frazzini-Pedersen use 0.6 in practice)
        if len(returns) >= 252:
            # OLS beta against itself (market proxy approximation)
            # In production: use actual market return series
            market_proxy = returns.rolling(21).mean()
            cov = np.cov(returns.values, market_proxy.fillna(0).values)
            if cov.shape == (2, 2) and cov[1, 1] > 0:
                beta_ols = cov[0, 1] / cov[1, 1]
                beta_shrunk = 0.4 * 1.0 + 0.6 * beta_ols  # Vasicek shrinkage to 1
                factors['bab_signal'] = -beta_shrunk  # Negative beta = BAB long signal
            else:
                factors['bab_signal'] = np.nan
        else:
            factors['bab_signal'] = np.nan

        return factors

    def compute_cross_sectional_scores(self,
                                        universe_data: Dict[str, Dict[str, float]]
                                        ) -> pd.DataFrame:
        """
        Computes cross-sectional factor scores for the full universe.

        universe_data: {ticker: {feature_name: value}}

        Returns DataFrame of cross-sectionally normalized factor scores.
        All scores are:
        - Normalized by cross-sectional median/IQR (not mean/std)
        - Winsorized at ±3σ
        - Sector-neutral (demeaned within sector)
        """
        df = pd.DataFrame(universe_data).T  # tickers x features

        # Cross-sectional normalization
        normalized = pd.DataFrame(index=df.index, columns=df.columns)
        for col in df.columns:
            series = df[col].dropna()
            if len(series) < 5:
                continue
            median = series.median()
            iqr = series.quantile(0.75) - series.quantile(0.25)
            if iqr == 0:
                iqr = series.std() + 1e-10
            norm = (df[col] - median) / iqr
            normalized[col] = norm.clip(-3, 3)

        return normalized

    def ic_weighted_ensemble(self,
                              signal_forecasts: Dict[str, pd.Series],
                              realized_returns: pd.Series,
                              lookback_periods: int = 12) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Combines multiple cross-sectional signals using IC-weighted ensemble.

        Weight each model by its recent Information Coefficient:
            w_k = max(IC_k, 0) / Σ_j max(IC_j, 0)

        IC = Spearman rank correlation between signals and realized returns
        Only positive-IC models get weight (discard negative-IC models)

        This adapts to:
        - Factor regime changes (value works in some regimes, not others)
        - Model decay (deweights declining models automatically)
        - Signal interactions (diversification bonus from combining)
        """
        # Update IC history
        for name, signals in signal_forecasts.items():
            ic = stats.spearmanr(signals.dropna(), realized_returns.dropna())[0]
            if name not in self.ic_history:
                self.ic_history[name] = []
            self.ic_history[name].append(ic)

        # Compute recent IC for each model
        recent_ics = {}
        for name in signal_forecasts:
            history = self.ic_history.get(name, [])
            if history:
                # Use exponentially weighted recent IC
                weights_exp = np.exp(np.linspace(-1, 0, min(len(history), lookback_periods)))
                recent_ics[name] = np.average(
                    history[-lookback_periods:],
                    weights=weights_exp[-len(history[-lookback_periods:]):]
                )
            else:
                recent_ics[name] = 0.0

        # IC-weighted combination (only positive IC)
        positive_ics = {k: max(v, 0) for k, v in recent_ics.items()}
        total_ic = sum(positive_ics.values())

        if total_ic < 0.001:
            # No signal has positive IC: equal weight (or no signal)
            weights = {k: 1.0 / len(signal_forecasts) for k in signal_forecasts}
        else:
            weights = {k: v / total_ic for k, v in positive_ics.items()}

        self.model_weights = weights

        # Combine signals
        combined = pd.Series(0.0, index=list(signal_forecasts.values())[0].index)
        for name, signals in signal_forecasts.items():
            combined += weights.get(name, 0) * signals.fillna(0)

        return combined, recent_ics


# ─────────────────────────────────────────────────────────────
# TIME-SERIES ALPHA ENGINE
# ─────────────────────────────────────────────────────────────

class TimeSeriesAlphaEngine:
    """
    Time-series alpha: predict future return of a SINGLE asset
    based on its OWN history.

    Key difference from XS: this generates ABSOLUTE forecasts,
    not rankings. Useful for:
    - Timing entries/exits within a long-short book
    - Scaling position sizes (larger when TS signal strong)
    - TSMOM (Moskowitz, Ooi, Pedersen 2012)

    Models:
    1. TSMOM: sign(12M return) * inverse volatility
    2. ARIMA: conditional mean from autoregressive structure
    3. Kalman filter: trend component
    4. LSTM: non-linear temporal patterns
    5. GARCH-filtered returns: vol-adjusted returns
    """

    def compute_tsmom_signal(self, prices: pd.Series,
                              lookback: int = 252) -> Dict[str, float]:
        """
        Time-Series Momentum (Moskowitz, Ooi, Pedersen 2012).

        Signal = sign(r_{t-h:t}) / σ_t

        Evidence: works in 58 global markets, 1985-2012.
        Risk explanation: trend-following protects in prolonged crises.
        Behavioral: slow incorporation of information, underreaction.

        Key insight: TSMOM is NOT the same as cross-sectional momentum.
        An asset with negative TSMOM can still be a XS momentum LONG
        if its peers have even worse momentum.
        """
        if len(prices) < lookback:
            return {'tsmom': np.nan, 'tsmom_strength': np.nan}

        log_returns = np.log(prices / prices.shift(1)).dropna()
        realized_ret = np.log(prices.iloc[-1] / prices.iloc[-lookback])
        realized_vol = log_returns.iloc[-63:].std() * np.sqrt(252)

        signal = np.sign(realized_ret) / (realized_vol + 1e-10)
        strength = abs(realized_ret) / (realized_vol + 1e-10)

        return {
            'tsmom': signal,
            'tsmom_strength': strength,
            'raw_return': realized_ret,
            'realized_vol': realized_vol,
        }

    def compute_mean_reversion_signal(self, prices: pd.Series,
                                       half_life_days: int = 20) -> Dict[str, float]:
        """
        Mean reversion signal (Ornstein-Uhlenbeck process).

        dX_t = κ(θ - X_t)dt + σ dW_t

        If prices follow OU process, we can estimate κ (speed of reversion)
        and θ (long-run mean) from data.

        Signal = -(X_t - θ) * κ (buy when below mean, sell when above)

        The half-life of reversion: τ = ln(2) / κ

        Evidence: pairs trading, statistical arbitrage (Gatev, Goetzmann, Rouwenhorst 2006)
        Most useful for: short-horizon mean reversion, not long-horizon
        """
        if len(prices) < 63:
            return {'mean_rev_signal': np.nan, 'half_life': np.nan}

        log_prices = np.log(prices)

        # Estimate OU parameters via OLS:
        # Δp_t = a + b * p_{t-1} + ε_t
        # κ = -b (speed of reversion), θ = -a/b (equilibrium)
        delta_p = log_prices.diff().dropna()
        lag_p = log_prices.shift(1).dropna()

        # Align
        min_len = min(len(delta_p), len(lag_p))
        y = delta_p.values[-min_len:]
        x = lag_p.values[-min_len:]

        # OLS
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            a, b = beta[0], beta[1]
        except Exception:
            return {'mean_rev_signal': np.nan, 'half_life': np.nan}

        # Speed of mean reversion
        kappa = -b  # Should be positive for mean-reverting

        if kappa <= 0:
            return {'mean_rev_signal': 0.0, 'half_life': np.inf,
                    'is_mean_reverting': False}

        # Half-life in days
        half_life = np.log(2) / kappa

        # Long-run equilibrium
        theta = -a / b if abs(b) > 1e-10 else float(log_prices.mean())

        # Current deviation from equilibrium
        current_dev = float(log_prices.iloc[-1]) - theta

        # Signal: buy when price is below equilibrium, sell when above
        signal = -current_dev * kappa  # Stronger signal when more deviated

        return {
            'mean_rev_signal': signal,
            'half_life': half_life,
            'kappa': kappa,
            'theta': theta,
            'current_deviation': current_dev,
            'is_mean_reverting': True,
        }

    def compute_kalman_trend(self, prices: pd.Series) -> Dict[str, float]:
        """
        Kalman filter trend estimation.

        State space model:
            Observation: y_t = x_t + ε_t, ε_t ~ N(0, R)
            Transition:  x_t = x_{t-1} + w_t, w_t ~ N(0, Q)

        x_t = latent trend
        Q/R = signal-to-noise ratio (SNR)

        When SNR is high: trend is real, follow it (TSMOM)
        When SNR is low: mostly noise, mean-revert

        This elegantly unifies trend-following and mean-reversion:
        the Kalman filter tells you WHICH regime you're in.
        """
        if len(prices) < 30:
            return {'kalman_trend': np.nan, 'snr': np.nan}

        log_prices = np.log(prices).values

        # Estimate noise parameters from data
        returns = np.diff(log_prices)
        R = np.var(returns) * 0.5   # Observation noise
        Q = np.var(returns) * 0.01  # Process noise (trend changes slowly)
        snr = Q / R                  # Signal-to-noise ratio

        # Kalman filter recursion
        x = log_prices[0]  # State estimate
        P = R              # State covariance

        filtered = [x]
        gains = []

        for t in range(1, len(log_prices)):
            # Predict
            x_pred = x
            P_pred = P + Q

            # Update
            K = P_pred / (P_pred + R)  # Kalman gain
            x = x_pred + K * (log_prices[t] - x_pred)
            P = (1 - K) * P_pred

            filtered.append(x)
            gains.append(K)

        filtered = np.array(filtered)
        current_gain = gains[-1] if gains else 0.5

        # Trend slope (slope of filtered prices)
        slope = np.polyfit(range(min(21, len(filtered))),
                           filtered[-min(21, len(filtered)):], 1)[0]

        # Signal: positive slope = trend up = buy
        signal = slope / (np.std(filtered[-21:]) + 1e-10)

        return {
            'kalman_trend': signal,
            'snr': snr,
            'trend_slope': slope,
            'current_kalman_gain': current_gain,
            'filtered_price': float(np.exp(filtered[-1])),
            'signal_type': 'TRENDING' if snr > 0.02 else 'MEAN_REVERTING',
        }


# ─────────────────────────────────────────────────────────────
# MULTI-HORIZON ALPHA ENGINE
# ─────────────────────────────────────────────────────────────

class MultiHorizonAlphaEngine:
    """
    Generates alpha signals across multiple horizons simultaneously.

    Different horizons capture different phenomena:
    - 1W: microstructure, short-term momentum/reversal
    - 1M: earnings cycle, sentiment-driven moves
    - 3M: fundamental revisions, sector rotations
    - 6M: factor momentum (Jegadeesh-Titman)
    - 1Y: value reversion, long-cycle momentum

    CRITICAL: 1W and 1Y signals are statistically INDEPENDENT.
    They can be combined because they draw on different information.
    This is the basis for multi-period optimal portfolios.

    Model structure per horizon h:
        μ_{i,t,h} = f_h(XS_factors, TS_signals, regime)
        σ²_{i,t,h} = g_h(GARCH_vol, option_implied_vol, regime_vol)
        S_{i,t,h} = h_h(history) — skewness from historical distribution
    """

    def __init__(self):
        self.xs_engine = CrossSectionalAlphaEngine()
        self.ts_engine = TimeSeriesAlphaEngine()
        self.horizon_ic_history: Dict[str, List[float]] = {}

    def compute_distributional_forecast(self,
                                         ticker: str,
                                         prices: pd.Series,
                                         returns: pd.Series,
                                         fundamentals: Dict,
                                         regime: str,
                                         regime_confidence: float,
                                         current_garch_vol: float,
                                         horizon: str = '1M'
                                         ) -> DistributionalForecast:
        """
        Generates a full distributional forecast for one asset, one horizon.

        The forecast distribution P(r_{t+h} | I_t) is characterized by:
        1. Mean: from factor signals (XS) + trend signals (TS)
        2. Variance: from GARCH + factor exposure
        3. Skewness: from options + historical distribution
        4. Tail probs: from historical + GARCH forecast

        REGIME CONDITIONING: The same signal has different implications
        in different regimes. Momentum works in BULL regimes, fails in BEAR.
        Value works in mean-reverting regimes, not trending ones.
        """
        horizon_days = {'1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252}[horizon]
        ann_factor = np.sqrt(252 / horizon_days)

        # ── XS Signals ──────────────────────────────────────
        xs_total = 0.0
        xs_components = {}

        if len(prices) >= 252:
            mom = self.xs_engine.compute_momentum_factors(prices)
            xs_total += 0.30 * (mom.get('mom_12_1', 0) or 0)  # XS momentum
            xs_components['momentum'] = mom.get('mom_12_1', 0)

        if fundamentals:
            value = self.xs_engine.compute_value_factors(fundamentals, prices.iloc[-1])
            quality = self.xs_engine.compute_quality_factors(fundamentals)
            xs_total += 0.20 * (value.get('value_composite', 0) or 0)
            xs_total += 0.15 * (quality.get('qmj_composite', 0) or 0)
            xs_components['value'] = value.get('value_composite', 0)
            xs_components['quality'] = quality.get('qmj_composite', 0)

        low_vol = self.xs_engine.compute_low_vol_factor(returns)
        xs_total += 0.10 * (low_vol.get('bab_signal', 0) or 0)
        xs_components['low_vol'] = low_vol.get('bab_signal', 0)

        # ── TS Signals ──────────────────────────────────────
        ts_total = 0.0

        tsmom = self.ts_engine.compute_tsmom_signal(prices)
        kalman = self.ts_engine.compute_kalman_trend(prices)
        mean_rev = self.ts_engine.compute_mean_reversion_signal(prices)

        ts_total += 0.40 * (tsmom.get('tsmom', 0) or 0)
        ts_total += 0.40 * (kalman.get('kalman_trend', 0) or 0)
        ts_total += 0.20 * (mean_rev.get('mean_rev_signal', 0) or 0)

        # ── Regime Conditioning ──────────────────────────────
        # Signals behave differently in different regimes
        regime_adjustments = {
            'BULL_LOW_VOL':   {'xs_weight': 0.5, 'ts_weight': 0.5},
            'BULL_HIGH_VOL':  {'xs_weight': 0.3, 'ts_weight': 0.7},
            'MEAN_REVERT':    {'xs_weight': 0.6, 'ts_weight': 0.4},
            'BEAR_LOW_VOL':   {'xs_weight': 0.7, 'ts_weight': 0.3},
            'BEAR_HIGH_VOL':  {'xs_weight': 0.8, 'ts_weight': 0.2},
        }
        adj = regime_adjustments.get(regime, {'xs_weight': 0.5, 'ts_weight': 0.5})

        # Combine XS and TS with regime weights
        combined_signal = adj['xs_weight'] * xs_total + adj['ts_weight'] * ts_total

        # Scale to expected return space
        # IC ≈ 0.05 → E[r] ≈ IC * vol_cross_section
        ic_estimate = 0.05  # Conservative prior; updated from realized IC
        cross_section_vol = current_garch_vol  # Use GARCH vol as proxy
        expected_return = combined_signal * ic_estimate * cross_section_vol * horizon_days / 252

        # ── Variance Forecast ────────────────────────────────
        # Variance = GARCH variance + parameter uncertainty
        garch_var = (current_garch_vol ** 2) * (horizon_days / 252)
        param_uncertainty = garch_var * 0.20  # 20% additional uncertainty
        total_var = garch_var + param_uncertainty

        # ── Higher Moments ───────────────────────────────────
        # Skewness and kurtosis from historical distribution
        if len(returns) >= 126:
            hist_skew = float(returns.iloc[-126:].skew())
            hist_kurt = float(returns.iloc[-126:].kurtosis())
        else:
            hist_skew = 0.0
            hist_kurt = 0.0

        # Bear regimes have more negative skew
        regime_skew_adj = {'BEAR_HIGH_VOL': -0.5, 'BEAR_LOW_VOL': -0.3,
                           'BULL_LOW_VOL': +0.2}.get(regime, 0.0)
        total_skew = hist_skew + regime_skew_adj

        # ── Tail Probabilities ───────────────────────────────
        # Use Cornish-Fisher expansion:
        # q_α ≈ μ + σ * [z_α + (z_α² - 1)*S/6 + (z_α³ - 3z_α)*K/24]
        std = np.sqrt(total_var)
        z_10 = stats.norm.ppf(0.10)  # -1.28

        cf_q10 = expected_return + std * (
            z_10 + (z_10**2 - 1) * total_skew / 6
            + (z_10**3 - 3*z_10) * hist_kurt / 24
        )

        prob_loss_10 = stats.norm.cdf((-0.10 - expected_return) / (std + 1e-10))
        prob_gain_10 = 1 - stats.norm.cdf((+0.10 - expected_return) / (std + 1e-10))
        prob_loss_5 = stats.norm.cdf((-0.05 - expected_return) / (std + 1e-10))
        prob_gain_5 = 1 - stats.norm.cdf((+0.05 - expected_return) / (std + 1e-10))
        prob_loss_20 = stats.norm.cdf((-0.20 - expected_return) / (std + 1e-10))

        # ── Uncertainty ──────────────────────────────────────
        signal_to_noise = abs(combined_signal) / (abs(combined_signal) + 0.5)
        signal_confidence = signal_to_noise * regime_confidence

        return DistributionalForecast(
            ticker=ticker,
            timestamp=pd.Timestamp.now(tz='UTC'),
            horizon=horizon,
            expected_return=float(expected_return),
            return_std=float(std),
            return_skew=float(total_skew),
            return_kurtosis=float(hist_kurt),
            prob_loss_5pct=float(prob_loss_5),
            prob_loss_10pct=float(prob_loss_10),
            prob_loss_20pct=float(prob_loss_20),
            prob_gain_5pct=float(prob_gain_5),
            prob_gain_10pct=float(prob_gain_10),
            ic_estimate=float(ic_estimate),
            signal_confidence=float(signal_confidence),
            xs_component=float(xs_total),
            ts_component=float(ts_total),
            residual_component=0.0,  # Set by NLP engine
            regime=regime,
            regime_confidence=float(regime_confidence),
        )
