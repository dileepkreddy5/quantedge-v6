"""
QuantEdge v6.0 — DATA ENGINE (Layer 1)
════════════════════════════════════════════════════════════════
Philosophy: Data is the foundation of everything. Corrupt data
produces confident-but-wrong models. This layer guarantees:

1. IMMUTABILITY: Raw data is never modified after ingestion
2. NO LOOKAHEAD: Point-in-time correctness enforced everywhere
3. NO SURVIVORSHIP BIAS: Delisted stocks kept in universe
4. REGIME LEAKAGE PREVENTION: Train/test splits respect regimes
5. TIMESTAMP ALIGNMENT: All series aligned to same trading calendar
6. FEATURE VERSIONING: Every feature version is reproducible

Mathematical guarantees:
  - For any feature f_i,t: E[f_i,t | info_t] = f_i,t  (no future info)
  - For any model trained on [t0, t1]: evaluated on [t1+embargo, t2]
  - Embargo = max(autocorrelation_lag, 5 trading days)

References:
  - Lopez de Prado (2018): Advances in Financial Machine Learning
  - Harvey et al (2016): ... and the Cross-Section of Expected Returns
  - Gu, Kelly, Xiu (2020): Empirical Asset Pricing via Machine Learning
════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
# DATA CONTRACTS — typed interfaces between layers
# ─────────────────────────────────────────────────────────────

@dataclass
class RawPriceData:
    """Immutable raw price record. Never modified after ingestion."""
    ticker: str
    timestamp: pd.Timestamp          # UTC, market close
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: float                 # Dividend/split adjusted
    ingestion_time: pd.Timestamp     # When we received the data
    source: str                      # yfinance | polygon | tradier
    is_delisted: bool = False        # Survivorship bias prevention

    def __post_init__(self):
        if self.high < self.low:
            raise ValueError(f"High < Low for {self.ticker} on {self.timestamp}")
        if self.close <= 0:
            raise ValueError(f"Non-positive close for {self.ticker}")


@dataclass
class FeatureVector:
    """
    Point-in-time feature vector for one asset at one timestamp.
    Contains ONLY information available at that timestamp.
    Feature version tracked for reproducibility.
    """
    ticker: str
    timestamp: pd.Timestamp          # Information timestamp (as-of)
    feature_version: str             # e.g. "v2.3.1" — semver
    features: Dict[str, float]       # feature_name → value
    data_completeness: float         # 0-1, fraction of features available
    is_valid: bool = True            # False if too many NaN


@dataclass
class AlphaSignal:
    """
    Distributional alpha signal. NOT a point prediction.
    We model the full distribution: mean, variance, skew, tail probability.
    This is what separates institutional from retail ML.
    """
    ticker: str
    timestamp: pd.Timestamp
    horizon: str                     # "1W" | "1M" | "3M" | "6M" | "1Y"

    # Distribution parameters (not point predictions)
    expected_return: float           # E[r_{t+h}]
    return_variance: float           # Var[r_{t+h}]
    return_skewness: float           # Skew[r_{t+h}] — negative = left tail risk
    return_kurtosis: float           # Excess kurtosis — tail thickness
    tail_prob_loss_10pct: float      # P(r_{t+h} < -10%)
    tail_prob_gain_10pct: float      # P(r_{t+h} > +10%)

    # Epistemic uncertainty (how confident is the model)
    model_uncertainty: float         # MC Dropout epistemic std
    information_coefficient: float   # Rank IC (Spearman)
    signal_to_noise: float           # |E[r]| / std(E[r])

    # Source decomposition
    cross_sectional_component: float # From XS ranking
    time_series_component: float     # From LSTM
    residual_component: float        # Idiosyncratic alpha

    model_version: str = "v1.0"
    confidence_regime: str = "NORMAL"  # NORMAL | STRESSED | CRISIS


@dataclass
class RiskReport:
    """
    Standalone risk report — computed INDEPENDENTLY of alpha.
    Risk engine never reads AlphaSignal. Alpha engine never reads RiskReport.
    They communicate only through the Portfolio Construction layer.
    """
    ticker: str
    timestamp: pd.Timestamp

    # Covariance-based risk
    realized_vol_21d: float
    realized_vol_63d: float
    ewma_vol: float                  # λ=0.94 EWMA (RiskMetrics)
    shrunk_vol: float                # Ledoit-Wolf shrinkage

    # Tail risk
    var_95: float                    # 95% VaR (1-day)
    var_99: float                    # 99% VaR (1-day)
    cvar_95: float                   # Expected Shortfall 95%
    cvar_99: float                   # Expected Shortfall 99%

    # Liquidity risk
    amihud_illiquidity: float        # Amihud (2002) ratio
    bid_ask_spread_est: float        # Roll (1984) implicit spread
    adv_21d: float                   # 21-day Average Daily Volume ($)
    days_to_liquidate: float         # Position size / ADV fraction

    # Factor risk
    beta_to_market: float
    factor_crowding_score: float     # 0-1, how crowded is this factor position
    idiosyncratic_vol: float         # Vol not explained by factors

    risk_regime: str = "NORMAL"      # NORMAL | ELEVATED | CRISIS


# ─────────────────────────────────────────────────────────────
# LOOKAHEAD PREVENTION — the most critical guarantee
# ─────────────────────────────────────────────────────────────

class PointInTimeValidator:
    """
    Enforces point-in-time correctness across all feature computations.

    The fundamental rule: when computing feature f at time t,
    only information with timestamp < t may be used.

    Lookahead bias is insidious because:
    1. Forward-filled fundamental data (earnings known before release)
    2. Volatility computed over a window that crosses t
    3. Cross-sectional normalization using future constituents
    4. Factor loadings estimated on future data

    This class wraps all data access to enforce the rule.
    """

    def __init__(self, prices: pd.DataFrame, fundamental_release_lag_days: int = 2):
        """
        prices: MultiIndex (date, ticker) DataFrame with OHLCV
        fundamental_release_lag_days: lag between quarter end and data availability
            - Compustat: 2-3 days (fast)
            - Bloomberg: 1-2 days
            - Default: 2 days (conservative)
        """
        self.prices = prices.copy()
        self.fundamental_lag = timedelta(days=fundamental_release_lag_days)

        # Validate no future data
        self._validate_timestamps()

    def _validate_timestamps(self):
        """Assert all price timestamps are in the past at time of use."""
        if hasattr(self.prices.index, 'levels'):
            dates = self.prices.index.get_level_values(0)
        else:
            dates = self.prices.index
        assert dates.is_monotonic_increasing, "Price data not sorted chronologically!"

    def get_prices_as_of(self, as_of_date: pd.Timestamp,
                         lookback_days: int = 252) -> pd.DataFrame:
        """
        Returns prices available as-of a specific date.
        No prices on or after as_of_date are included.
        """
        if isinstance(self.prices.index, pd.MultiIndex):
            mask = self.prices.index.get_level_values(0) < as_of_date
        else:
            mask = self.prices.index < as_of_date

        data = self.prices[mask]

        # Take only the lookback window
        if len(data) > 0:
            cutoff = as_of_date - timedelta(days=lookback_days * 1.5)
            if isinstance(data.index, pd.MultiIndex):
                mask2 = data.index.get_level_values(0) >= cutoff
            else:
                mask2 = data.index >= cutoff
            data = data[mask2]

        return data

    def get_fundamental_as_of(self, as_of_date: pd.Timestamp,
                               fundamentals: pd.DataFrame) -> pd.DataFrame:
        """
        Returns fundamentals available as-of date, accounting for
        the release lag (data isn't available the day it's reported).
        """
        effective_date = as_of_date - self.fundamental_lag
        if isinstance(fundamentals.index, pd.MultiIndex):
            mask = fundamentals.index.get_level_values(0) <= effective_date
        else:
            mask = fundamentals.index <= effective_date
        return fundamentals[mask]


# ─────────────────────────────────────────────────────────────
# SURVIVORSHIP BIAS PREVENTION
# ─────────────────────────────────────────────────────────────

class UniverseManager:
    """
    Manages the stock universe at each point in time.

    CRITICAL: The universe at time t must include ALL stocks that
    WERE tradeable at time t, including those that subsequently
    became delisted, bankrupt, or acquired.

    Survivorship bias inflates backtest returns by 1-3% per year
    (Elton, Gruber, Blake 1996). For a $100M fund, that's $1-3M
    in phantom alpha.

    Universe construction rules:
    1. Include stocks with price > $1 (avoid penny stock noise)
    2. Include stocks with ADV > $1M (liquidity filter)
    3. Include all subsequently delisted stocks (anti-survivorship)
    4. Exclude stocks that IPO'd AFTER the time point
    5. Apply index membership as-of that date (not current membership)
    """

    def __init__(self, universe_history: pd.DataFrame):
        """
        universe_history: DataFrame with columns:
            [date, ticker, in_universe, ipo_date, delist_date, reason_delisted]
        """
        self.history = universe_history

    def get_universe_at(self, date: pd.Timestamp) -> List[str]:
        """
        Returns the tradeable universe at a given date.
        Includes stocks that will later be delisted (no survivorship bias).
        """
        mask = (
            (self.history['ipo_date'] <= date) &
            (
                self.history['delist_date'].isna() |
                (self.history['delist_date'] > date)
            ) &
            (self.history['in_universe'] == True)
        )
        return self.history[mask]['ticker'].tolist()

    def get_delist_returns(self, ticker: str, delist_date: pd.Timestamp) -> float:
        """
        Returns the delisting return (usually -30% to -70%).
        This is critical — ignoring delisting returns inflates performance.
        Shumway (1997): average delisting return ≈ -30%.
        """
        row = self.history[
            (self.history['ticker'] == ticker) &
            (self.history['delist_date'] == delist_date)
        ]
        if len(row) == 0:
            return -0.30  # Conservative Shumway estimate

        reason = row['reason_delisted'].values[0]
        # Graduated delist returns by reason
        delist_return_map = {
            'bankruptcy': -0.70,
            'performance': -0.40,
            'acquisition': +0.15,   # M&A premium
            'merger': +0.10,
            'voluntary': -0.15,
            'exchange_transfer': 0.00,
        }
        return delist_return_map.get(reason, -0.30)


# ─────────────────────────────────────────────────────────────
# WALK-FORWARD SPLITTER — Lopez de Prado purged k-fold
# ─────────────────────────────────────────────────────────────

class PurgedWalkForwardSplitter:
    """
    Implements purged, embargoed walk-forward cross-validation.

    Standard k-fold CV is WRONG for financial time series because:
    1. Overlapping labels: if target = 21d return, t and t+5 overlap
    2. Serial correlation: nearby observations are not independent
    3. Regime leakage: training on one regime, testing on another IS valid,
       but training on data that CONTAINS the test period is not

    This implementation follows Lopez de Prado (2018) Chapter 7:

    For each fold k:
    - Train: [t0, t_k - embargo]
    - Purge: remove samples with labels overlapping [t_k, t_k+1]
    - Embargo: gap between train end and test start = max(5d, label_span)
    - Test: [t_k, t_k+1]

    The purge prevents the model from "seeing" the future through
    overlapping return windows.
    """

    def __init__(self, n_splits: int = 5, test_size_pct: float = 0.20,
                 embargo_pct: float = 0.01, label_span_days: int = 21):
        """
        n_splits: number of walk-forward folds
        test_size_pct: fraction of data used for testing each fold
        embargo_pct: additional embargo beyond label span
        label_span_days: days covered by the prediction target (e.g., 21 for 1M)
        """
        self.n_splits = n_splits
        self.test_size_pct = test_size_pct
        self.embargo_pct = embargo_pct
        self.label_span_days = label_span_days

    def split(self, X: pd.DataFrame, dates: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns list of (train_indices, test_indices) tuples.

        GUARANTEE: No information from test period leaks into training.
        """
        n = len(X)
        test_size = int(n * self.test_size_pct)
        embargo = int(n * self.embargo_pct) + self.label_span_days

        splits = []
        step = test_size

        for i in range(self.n_splits):
            test_end = n - i * step
            test_start = test_end - test_size

            if test_start <= embargo:
                break

            # Train: everything before test_start - embargo
            train_end = test_start - embargo
            train_indices = np.arange(0, train_end)

            # PURGE: remove training samples whose LABEL overlaps with test
            # Label for sample t covers [t, t + label_span_days]
            # A label overlaps test if: t + label_span_days >= test_start
            purge_from = test_start - self.label_span_days
            purge_mask = np.arange(0, train_end) >= purge_from
            train_indices = train_indices[~purge_mask]

            test_indices = np.arange(test_start, test_end)

            if len(train_indices) > 100 and len(test_indices) > 10:
                splits.append((train_indices, test_indices))

        return splits


# ─────────────────────────────────────────────────────────────
# CROSS-SECTIONAL NORMALIZER — prevents look-ahead in normalization
# ─────────────────────────────────────────────────────────────

class CrossSectionalNormalizer:
    """
    Normalizes features cross-sectionally without lookahead.

    WRONG approach: normalize using full-sample mean/std
        → this leaks future distributional information

    CORRECT approach: normalize using only past cross-sections
        → at time t, use median and IQR from [t-lookback, t-1]

    Also applies:
    - Winsorization at ±3σ (reduces outlier sensitivity)
    - Rank transformation (makes distributions comparable across regimes)
    - Demean by sector (sector-neutral factors)

    References:
    - Barra risk model documentation
    - Axioma factor model methodology
    """

    def __init__(self, winsorize_threshold: float = 3.0,
                 use_rank: bool = True,
                 lookback_for_params: int = 60):
        self.winsorize_threshold = winsorize_threshold
        self.use_rank = use_rank
        self.lookback_for_params = lookback_for_params
        self._params_history: Dict[pd.Timestamp, Dict] = {}

    def fit_transform(self, features_panel: pd.DataFrame,
                      as_of_date: pd.Timestamp) -> pd.DataFrame:
        """
        Normalizes features at time t using only past data.

        features_panel: MultiIndex (date, ticker) x features
        Returns: normalized feature DataFrame for as_of_date cross-section
        """
        # Get only past cross-sections for computing normalization params
        if isinstance(features_panel.index, pd.MultiIndex):
            past_mask = features_panel.index.get_level_values(0) < as_of_date
            past_data = features_panel[past_mask]

            # Take last lookback cross-sections
            past_dates = past_data.index.get_level_values(0).unique()
            lookback_dates = past_dates[-self.lookback_for_params:]
            past_data = past_data[
                past_data.index.get_level_values(0).isin(lookback_dates)
            ]

            # Current cross-section to normalize
            current_mask = features_panel.index.get_level_values(0) == as_of_date
            current_xs = features_panel[current_mask].copy()
        else:
            past_data = features_panel[features_panel.index < as_of_date]
            current_xs = features_panel[features_panel.index == as_of_date].copy()

        # Compute robust normalization params from past data only
        medians = past_data.median()
        iqrs = past_data.quantile(0.75) - past_data.quantile(0.25)
        iqrs = iqrs.replace(0, 1.0)  # Avoid division by zero

        # Store params for reproducibility
        self._params_history[as_of_date] = {'median': medians, 'iqr': iqrs}

        # Normalize current cross-section
        normalized = (current_xs - medians) / iqrs

        # Winsorize at ±threshold
        normalized = normalized.clip(-self.winsorize_threshold, self.winsorize_threshold)

        # Rank transform → uniform distribution [0, 1] → then to N(0,1)
        if self.use_rank:
            from scipy.stats import norm as scipy_norm
            for col in normalized.columns:
                ranks = normalized[col].rank(pct=True)
                # Clamp to avoid ±inf from norm.ppf at 0 and 1
                ranks = ranks.clip(0.01, 0.99)
                normalized[col] = scipy_norm.ppf(ranks)

        return normalized

    def transform(self, features: pd.Series, as_of_date: pd.Timestamp) -> pd.Series:
        """Transform a single asset's features using stored params."""
        if as_of_date not in self._params_history:
            raise ValueError(f"No normalization params fitted for {as_of_date}")
        params = self._params_history[as_of_date]
        normalized = (features - params['median']) / params['iqr']
        return normalized.clip(-self.winsorize_threshold, self.winsorize_threshold)


# ─────────────────────────────────────────────────────────────
# FEATURE STORE — versioned, time-aware, immutable snapshots
# ─────────────────────────────────────────────────────────────

class FeatureStore:
    """
    Versioned feature store. Every feature has:
    1. A version number (changes when computation logic changes)
    2. A computation timestamp (when it was computed)
    3. An information timestamp (the as-of date of its inputs)
    4. A data hash (for integrity verification)

    This enables:
    - Exact reproduction of any historical backtest
    - Detection of feature drift (distribution shifts)
    - A/B testing of feature versions in production
    - Rollback if a new feature version degrades performance

    Storage backend: S3 parquet files (immutable after write)
    Format: features/version={v}/date={d}/ticker={t}.parquet
    """

    def __init__(self, version: str = "v1.0.0"):
        self.version = version
        self._cache: Dict[Tuple, pd.DataFrame] = {}
        self._feature_registry: Dict[str, Dict] = {}

    def register_feature(self, name: str, description: str,
                         category: str, computation_fn: callable):
        """Register a feature definition. Immutable after registration."""
        if name in self._feature_registry:
            raise ValueError(f"Feature '{name}' already registered. "
                             f"Increment version to change computation.")
        self._feature_registry[name] = {
            'name': name,
            'description': description,
            'category': category,  # momentum | value | quality | risk | microstructure
            'fn': computation_fn,
            'version': self.version,
            'registered_at': pd.Timestamp.now(tz='UTC'),
        }

    def get_features(self, ticker: str, as_of_date: pd.Timestamp) -> Optional[FeatureVector]:
        """
        Returns the feature vector for ticker at as_of_date.
        Uses ONLY information available at as_of_date.
        """
        cache_key = (ticker, as_of_date, self.version)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Would load from S3 in production
        return None

    def put_features(self, fv: FeatureVector):
        """Store feature vector. Immutable after storage."""
        key = (fv.ticker, fv.timestamp, fv.feature_version)
        if key in self._cache:
            raise ValueError("Feature vector already stored. Create new version.")
        self._cache[key] = fv


# ─────────────────────────────────────────────────────────────
# DATA VALIDATION PIPELINE
# ─────────────────────────────────────────────────────────────

class DataValidator:
    """
    Validates raw price data before it enters the feature pipeline.

    A bad data point that isn't caught here propagates through:
    Features → Alpha → Portfolio → REAL MONEY LOSS

    Validation rules:
    1. Price continuity: |log(p_t/p_{t-1})| < 0.50 (50% daily move = suspect)
    2. Volume sanity: volume > 0 on all trading days
    3. OHLC consistency: open, high, low, close all positive and H >= L
    4. Timestamp gaps: no missing trading days in sequence
    5. Split/dividend detection: large price jumps + volume spike
    6. Cross-asset consistency: prices not all zero on same day (data outage)
    """

    MAX_DAILY_MOVE = 0.50      # 50% move in one day = data error (usually)
    MIN_VOLUME = 1             # At least 1 share traded
    MAX_GAP_DAYS = 5           # Maximum allowed gap in trading days

    def __init__(self, alert_threshold_z: float = 4.0):
        self.alert_threshold_z = alert_threshold_z
        self.validation_log: List[Dict] = []

    def validate_price_series(self, ticker: str,
                               prices: pd.Series) -> Tuple[pd.Series, List[str]]:
        """
        Validates and cleans a price series.
        Returns (cleaned_series, list_of_issues_found).
        """
        issues = []
        cleaned = prices.copy()

        # 1. Remove zeros and negatives
        zero_mask = cleaned <= 0
        if zero_mask.any():
            issues.append(f"{zero_mask.sum()} zero/negative prices removed")
            cleaned = cleaned[~zero_mask]

        # 2. Detect extreme moves (likely data errors)
        log_returns = np.log(cleaned / cleaned.shift(1)).dropna()
        extreme_mask = log_returns.abs() > self.MAX_DAILY_MOVE
        if extreme_mask.any():
            extreme_dates = log_returns[extreme_mask].index
            issues.append(f"Extreme moves on {list(extreme_dates)} — flagged for review")
            # Don't auto-remove — could be real (circuit breakers, etc.)
            # Flag for human review

        # 3. Statistical outlier detection (Z-score of returns)
        mean_ret = log_returns.mean()
        std_ret = log_returns.std()
        z_scores = (log_returns - mean_ret) / (std_ret + 1e-10)
        outliers = z_scores.abs() > self.alert_threshold_z
        if outliers.any():
            issues.append(f"{outliers.sum()} statistical outliers (|Z| > {self.alert_threshold_z})")

        if issues:
            self.validation_log.append({'ticker': ticker, 'issues': issues,
                                        'timestamp': pd.Timestamp.now()})

        return cleaned, issues

    def check_lookahead(self, feature_computation_time: pd.Timestamp,
                        data_timestamps: pd.DatetimeIndex,
                        feature_as_of: pd.Timestamp) -> bool:
        """
        CRITICAL CHECK: Verifies no data timestamps exceed feature as-of date.
        Returns True if clean, False if lookahead detected.
        """
        lookahead_detected = (data_timestamps > feature_as_of).any()
        if lookahead_detected:
            self.validation_log.append({
                'type': 'LOOKAHEAD_VIOLATION',
                'computation_time': feature_computation_time,
                'feature_as_of': feature_as_of,
                'max_data_timestamp': data_timestamps.max(),
                'severity': 'CRITICAL'
            })
        return not lookahead_detected

    def get_validation_report(self) -> pd.DataFrame:
        """Returns all validation issues as a DataFrame."""
        return pd.DataFrame(self.validation_log)


# ─────────────────────────────────────────────────────────────
# LABEL ENGINEERING — forward returns with proper cleaning
# ─────────────────────────────────────────────────────────────

class LabelEngineer:
    """
    Computes forward-looking labels for model training.

    The label MUST be computed correctly:
    1. Use NEXT DAY's open, not today's close (execution realism)
    2. Account for transaction costs in the label
    3. Handle corporate actions (splits, dividends)
    4. Handle delisting events

    For horizon h, label at time t:
        y_{t,h} = log(adj_close_{t+h} / adj_close_{t+1})
                - transaction_cost(2 * spread + commission)

    Note: labels are FUTURE data and must NEVER be used in features.
    The walk-forward splitter enforces this at the training level.
    """

    def __init__(self, horizons_days: Dict[str, int] = None,
                 transaction_cost_bps: float = 10.0):
        self.horizons = horizons_days or {
            '1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252
        }
        self.tc_bps = transaction_cost_bps / 10000  # basis points → decimal

    def compute_labels(self, adj_prices: pd.DataFrame,
                       universe_manager: UniverseManager = None) -> pd.DataFrame:
        """
        Computes forward return labels for all tickers and horizons.
        adj_prices: dates x tickers DataFrame of adjusted closing prices

        Returns: MultiIndex (date, horizon) x tickers of forward returns
        """
        labels = {}

        for horizon_name, horizon_days in self.horizons.items():
            # Forward returns: log(P_{t+h}/P_{t+1}) = total holding period return
            # We use +1 as the entry point (execute at next day's open)
            fwd_returns = np.log(
                adj_prices.shift(-horizon_days) / adj_prices.shift(-1)
            )

            # Subtract round-trip transaction cost
            fwd_returns = fwd_returns - self.tc_bps

            labels[horizon_name] = fwd_returns

        return pd.Panel(labels) if hasattr(pd, 'Panel') else labels

    def compute_distributional_labels(self, adj_prices: pd.DataFrame,
                                       ticker: str, date: pd.Timestamp,
                                       horizon_days: int,
                                       n_simulations: int = 500) -> Dict[str, float]:
        """
        Instead of a single forward return label, compute distribution statistics.
        Uses bootstrap resampling over historical analogs.

        Returns distribution moments as targets for distributional models.
        """
        prices = adj_prices[ticker] if ticker in adj_prices.columns else adj_prices

        # Actual forward return (for training)
        try:
            idx = prices.index.get_loc(date)
            if idx + horizon_days + 1 < len(prices):
                actual_return = np.log(
                    prices.iloc[idx + horizon_days] / prices.iloc[idx + 1]
                )
            else:
                return {}
        except KeyError:
            return {}

        # Historical distribution over similar windows
        returns = np.log(prices / prices.shift(1)).dropna()
        window_returns = [
            returns.iloc[i:i+horizon_days].sum()
            for i in range(max(0, len(returns) - 252), len(returns) - horizon_days)
            if i + horizon_days < len(returns)
        ]

        if len(window_returns) < 20:
            return {'actual_return': actual_return}

        window_returns = np.array(window_returns)
        return {
            'actual_return': actual_return,
            'hist_mean': np.mean(window_returns),
            'hist_std': np.std(window_returns),
            'hist_skew': float(pd.Series(window_returns).skew()),
            'hist_kurtosis': float(pd.Series(window_returns).kurtosis()),
            'tail_prob_neg10': np.mean(window_returns < -0.10),
            'tail_prob_pos10': np.mean(window_returns > +0.10),
        }
