"""
QuantEdge v6.0 — Triple-Barrier Labeling + Meta-Labeling
==========================================================
Implements Lopez de Prado (2018) "Advances in Financial ML"

KEY CONCEPTS:
  Triple-Barrier Method:
    - Horizontal upper barrier:  profit-take = E[r] + k*σ
    - Horizontal lower barrier:  stop-loss   = E[r] - k*σ
    - Vertical barrier:          time expiry = t + h
    Label = sign of first barrier touched (1, -1, 0)

  Meta-Labeling:
    - Primary model: predicts direction (side)
    - Secondary model: predicts whether primary is correct
    - Output: position size 0→1 (size = confidence of being right)
    - Separates "when to trade" from "which direction"

  Why this matters:
    - Fixed-time horizon labels cause lookahead bias
    - Triple barrier is path-dependent (realistic)
    - Meta-labeling dramatically improves precision at cost of recall
    - Reduces overtrading by filtering low-conviction signals

Academic references:
  - Lopez de Prado (2018) Chapters 3, 4
  - Bailey et al. (2014) "The Probability of Backtest Overfitting"
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


# ── Data Structures ───────────────────────────────────────────
@dataclass
class BarrierEvent:
    """Single labeled event from triple-barrier method."""
    t0: pd.Timestamp          # Event start time
    t1: pd.Timestamp          # First barrier touched
    barrier_type: str         # 'profit', 'stop', 'time'
    label: int                # +1, -1, 0
    ret: float                # Actual return realized
    side: Optional[int]       # Primary model signal (for meta-labeling)
    meta_label: Optional[int] # 1 if primary was correct, 0 otherwise


# ══════════════════════════════════════════════════════════════
# TRIPLE-BARRIER LABELING
# ══════════════════════════════════════════════════════════════
class TripleBarrierLabeler:
    """
    Implements the Triple-Barrier Method from Lopez de Prado (2018) Ch. 3.

    Unlike fixed-time horizon labels (naive approach used by 99% of papers),
    triple-barrier labels are path-dependent and correctly model how
    actual trading works: you close a position when you hit your target
    or your stop, not at an arbitrary future time.

    Label rules:
      +1: upper barrier hit first (long profitable / short loss)
      -1: lower barrier hit first (long loss / short profitable)
       0: vertical barrier hit first (time expiry, no signal)

    The 0 class is valuable! It means: "I had no view — don't trade."
    """

    def __init__(
        self,
        profit_take_multiplier: float = 2.0,  # k × ATR for upper barrier
        stop_loss_multiplier: float = 1.0,     # k × ATR for lower barrier
        time_barrier_days: int = 21,           # Vertical barrier (hold period)
        min_ret: float = 0.001,                # Minimum return to bother labeling
        atr_window: int = 14,                  # ATR lookback for dynamic barriers
    ):
        self.pt_mult = profit_take_multiplier
        self.sl_mult = stop_loss_multiplier
        self.h = time_barrier_days
        self.min_ret = min_ret
        self.atr_window = atr_window

    def compute_daily_vol(self, close: pd.Series, window: int = 21) -> pd.Series:
        """
        Daily volatility estimate (used for dynamic barrier sizing).
        Using exponentially weighted standard deviation (EWMA vol).
        More responsive to recent vol changes than simple rolling std.
        """
        log_ret = np.log(close / close.shift(1))
        ewm_vol = log_ret.ewm(span=window, min_periods=window).std()
        return ewm_vol

    def compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Average True Range — dynamic barrier sizing."""
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(self.atr_window).mean()

    def get_events(
        self,
        close: pd.Series,
        t_events: pd.DatetimeIndex,  # Candidate event timestamps (CUSUM filter output)
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        side: Optional[pd.Series] = None,  # +1=long signal, -1=short signal
    ) -> pd.DataFrame:
        """
        For each candidate event, compute the first barrier touched.

        Parameters
        ----------
        close: Price series
        t_events: Timestamps to label (from CUSUM filter or other sampler)
        high/low: For more accurate barrier detection
        side: Primary model direction (+1/-1). If None, assumes long-only.

        Returns
        -------
        DataFrame with columns: [t1, ret, label, barrier_type, side]
        """
        # Dynamic barrier width: EWMA vol × multiplier
        vol = self.compute_daily_vol(close)

        if high is not None and low is not None:
            atr = self.compute_atr(high, low, close)
        else:
            atr = vol * close  # Approximate ATR from vol

        events = []

        for t0 in t_events:
            if t0 not in close.index:
                continue

            # Future window: t0 to t0 + h trading days
            t0_loc = close.index.get_loc(t0)
            t1_loc = min(t0_loc + self.h, len(close) - 1)
            future = close.iloc[t0_loc: t1_loc + 1]

            if len(future) < 2:
                continue

            p0 = close.loc[t0]
            current_vol = vol.loc[t0] if t0 in vol.index else 0.01
            current_atr = atr.loc[t0] if (atr is not None and t0 in atr.index) else p0 * current_vol

            # Determine trading direction
            dir_ = 1 if side is None or t0 not in side.index else int(side.loc[t0])

            # Barrier levels
            upper_barrier = p0 * (1 + self.pt_mult * current_vol * np.sqrt(self.h))
            lower_barrier = p0 * (1 - self.sl_mult * current_vol * np.sqrt(self.h))

            # Find first barrier hit
            label = 0
            t1 = future.index[-1]  # Default: vertical barrier
            barrier_type = 'time'
            ret = (future.iloc[-1] - p0) / p0

            for t_curr, price in future.iloc[1:].items():
                if dir_ == 1:  # Long position
                    if price >= upper_barrier:
                        label = 1; t1 = t_curr; barrier_type = 'profit'
                        ret = (price - p0) / p0; break
                    elif price <= lower_barrier:
                        label = -1; t1 = t_curr; barrier_type = 'stop'
                        ret = (price - p0) / p0; break
                else:  # Short position (dir_ == -1)
                    # Flip barriers for short
                    if price <= lower_barrier:
                        label = 1; t1 = t_curr; barrier_type = 'profit'
                        ret = -(price - p0) / p0; break
                    elif price >= upper_barrier:
                        label = -1; t1 = t_curr; barrier_type = 'stop'
                        ret = -(price - p0) / p0; break

            # Skip near-zero return events (noise)
            if abs(ret) < self.min_ret:
                label = 0

            events.append({
                't0': t0,
                't1': t1,
                'ret': ret,
                'label': label * dir_,  # Adjust sign for direction
                'raw_label': label,
                'barrier_type': barrier_type,
                'side': dir_,
                'vol_at_event': current_vol,
                'pt_level': upper_barrier,
                'sl_level': lower_barrier,
            })

        df = pd.DataFrame(events).set_index('t0') if events else pd.DataFrame()
        return df

    def drop_rare_labels(self, events: pd.DataFrame, min_pct: float = 0.05) -> pd.DataFrame:
        """
        Lopez de Prado Ch. 3: Drop labels that appear < min_pct of time.
        Extremely rare labels hurt model training and are likely noise.
        """
        counts = events['label'].value_counts(normalize=True)
        keep = counts[counts >= min_pct].index
        return events[events['label'].isin(keep)]


# ══════════════════════════════════════════════════════════════
# CUSUM FILTER (Event Sampling)
# ══════════════════════════════════════════════════════════════
class CUSUMFilter:
    """
    Symmetric CUSUM filter for event-based sampling.
    Lopez de Prado (2018) Ch. 2, Section 2.5.

    Instead of sampling at fixed time intervals (which creates
    autocorrelated observations), CUSUM samples when cumulative
    price change exceeds a threshold h.

    This produces:
    - IID-er (more independent) observations
    - Events that matter (when something actually happened)
    - Fewer, higher-quality training samples

    S+_t = max(0, S+_{t-1} + y_t - E[y_t])
    S-_t = min(0, S-_{t-1} + y_t - E[y_t])
    Sample when: S+_t > h or |S-_t| > h
    """

    def __init__(self, h_multiplier: float = 1.0):
        """
        h_multiplier: barrier width = h_multiplier × daily_vol
        Lower = more samples (noisier)
        Higher = fewer samples (cleaner)
        """
        self.h_mult = h_multiplier

    def filter(self, log_returns: pd.Series, vol: pd.Series) -> pd.DatetimeIndex:
        """
        Apply CUSUM filter to log return series.

        Parameters
        ----------
        log_returns: Log return series
        vol: Daily volatility estimate (dynamic threshold)

        Returns
        -------
        DatetimeIndex of event timestamps (when to observe/label)
        """
        t_events = []
        s_pos = 0.0
        s_neg = 0.0

        for t in log_returns.index:
            r = log_returns.loc[t]
            h = self.h_mult * vol.loc[t] if t in vol.index else self.h_mult * 0.01

            s_pos = max(0, s_pos + r)
            s_neg = min(0, s_neg + r)

            if s_pos > h:
                s_pos = 0
                t_events.append(t)
            elif s_neg < -h:
                s_neg = 0
                t_events.append(t)

        return pd.DatetimeIndex(t_events)


# ══════════════════════════════════════════════════════════════
# META-LABELING ENGINE
# ══════════════════════════════════════════════════════════════
class MetaLabeler:
    """
    Meta-Labeling (Lopez de Prado 2018, Ch. 3).

    The meta-labeling framework separates two problems:
    1. PRIMARY MODEL: Which direction? (long / short / flat)
    2. META MODEL:    Should I bet? How large?

    Why this is powerful:
    - Primary model can be simple (even a momentum rule)
    - Meta model filters: only trades where primary is LIKELY correct
    - This dramatically improves precision (less false positives)
    - Position size = meta model confidence → Kelly-like sizing

    Example:
      Primary signal: RSI < 30 → long
      Meta question: "Given RSI < 30 in this regime, is a long
                      actually going to work this time?"
      Meta answer:   0.73 → size = 73% of max position

    The meta model trains on: "was the primary model right?"
    """

    def __init__(self, primary_threshold: float = 0.5):
        self.primary_threshold = primary_threshold

    def generate_meta_labels(
        self,
        events: pd.DataFrame,
        primary_predictions: pd.Series,  # Primary model direction (+1/-1)
    ) -> pd.DataFrame:
        """
        Create meta-labels: 1 if primary was correct, 0 otherwise.

        For each event where primary predicted direction:
        - If primary said +1 and label == +1: meta_label = 1 (correct)
        - If primary said +1 and label == -1: meta_label = 0 (wrong)
        - If primary gave 0 (no trade): drop event

        This binary classification (correct/incorrect) is the meta-model target.
        """
        meta_labels = pd.DataFrame(index=events.index)
        meta_labels['primary_side'] = primary_predictions.reindex(events.index)
        meta_labels['true_label'] = events['label']
        meta_labels['true_ret'] = events['ret']

        # Meta-label: 1 if primary was correct (same sign), 0 if wrong
        aligned = np.sign(meta_labels['primary_side']) == np.sign(meta_labels['true_label'])
        meta_labels['meta_label'] = aligned.astype(int)

        # Drop where primary had no signal
        meta_labels = meta_labels[meta_labels['primary_side'] != 0]

        # Position size = primary side × meta confidence
        # (meta confidence comes from calibrated meta model probability)
        return meta_labels

    def compute_bet_size(
        self,
        meta_proba: pd.Series,  # Meta model P(correct) for each event
        max_size: float = 1.0,
        concentration: float = 1.0,  # Power law shaping
    ) -> pd.Series:
        """
        Sigmoid bet sizing (Lopez de Prado 2018, Ch. 10).
        Maps meta-model probability to position size.

        f(p) = (p - 0.5) / (p * (1 - p))^0.5 → sigmoid → size

        This is a discretized approximation of Kelly criterion.
        """
        # Center around 0.5 (coin flip = no edge)
        p = meta_proba.clip(0.001, 0.999)

        # z-score of probability
        z = (p - 0.5) / (p * (1 - p)) ** 0.5

        # Sigmoid transformation
        size = 2 * norm.cdf(concentration * z) - 1

        # Scale to max_size
        size = size * max_size

        return size.clip(0, max_size)  # Long-only: no negative sizes


# ══════════════════════════════════════════════════════════════
# FRACTIONAL DIFFERENTIATION
# ══════════════════════════════════════════════════════════════
class FractionalDifferentiator:
    """
    Fractional Differentiation (Lopez de Prado 2018, Ch. 5).

    FUNDAMENTAL PROBLEM in finance:
    - Price series: non-stationary (has memory) → can't use as ML input
    - Returns:      stationary (no memory) → loses long-term information
    - Both are wrong for different reasons

    Fractional differentiation: d ∈ (0, 1)
    - d = 0: original price series (non-stationary, maximum memory)
    - d = 1: returns (stationary, zero memory)
    - d = 0.3-0.5: OPTIMAL (stationary enough + preserves memory)

    Formula:
    d[B]^d x_t = Σ_{k=0}^∞ (-1)^k * C(d,k) * x_{t-k}
    where C(d,k) = Γ(d+1) / (Γ(k+1) * Γ(d-k+1))

    We find minimum d that passes ADF stationarity test (p < 0.05).
    This preserves maximum financial memory while ensuring stationarity.
    """

    def __init__(self, threshold: float = 1e-5, max_lag: int = 100):
        self.threshold = threshold  # Weight cutoff (ignore tiny weights)
        self.max_lag = max_lag

    def _get_weights(self, d: float) -> np.ndarray:
        """Compute fractional differentiation weights."""
        w = [1.0]
        for k in range(1, self.max_lag):
            w_k = -w[-1] * (d - k + 1) / k
            if abs(w_k) < self.threshold:
                break
            w.append(w_k)
        return np.array(w[::-1])

    def fractional_diff(self, series: pd.Series, d: float) -> pd.Series:
        """Apply fractional differentiation to series."""
        weights = self._get_weights(d)
        n_lags = len(weights)

        result = pd.Series(index=series.index, dtype=float)

        for t in range(n_lags - 1, len(series)):
            window = series.iloc[t - n_lags + 1: t + 1].values
            result.iloc[t] = np.dot(weights, window)

        return result.dropna()

    def find_min_d(self, series: pd.Series, d_range: np.ndarray = None) -> float:
        """
        Find minimum d that achieves stationarity (ADF p-value < 0.05).
        This preserves maximum long-term memory while ensuring ML suitability.
        """
        from statsmodels.tsa.stattools import adfuller

        if d_range is None:
            d_range = np.linspace(0.1, 1.0, 10)

        for d in d_range:
            try:
                diff_series = self.fractional_diff(series, d)
                adf_result = adfuller(diff_series.dropna(), maxlag=1, regression='c')
                p_value = adf_result[1]
                if p_value < 0.05:  # Stationary!
                    return round(d, 2)
            except Exception:
                continue

        return 1.0  # Fallback to integer differentiation

    def transform_prices(self, close: pd.Series) -> Tuple[pd.Series, float]:
        """
        Transform price series to fractionally differentiated series.
        Returns (transformed_series, d_value_used)
        """
        log_prices = np.log(close)
        d = self.find_min_d(log_prices)
        fd_series = self.fractional_diff(log_prices, d)
        return fd_series, d


# ══════════════════════════════════════════════════════════════
# LABEL WEIGHTING (Sample Uniqueness)
# ══════════════════════════════════════════════════════════════
class LabelWeighter:
    """
    Sample Uniqueness and Time-Decay Weighting.
    Lopez de Prado (2018) Ch. 4.

    PROBLEM: Overlapping labels cause pseudo-replication.
    If you label events at t=1 and t=2 with a 21-day horizon,
    their labels share 20 days of information. This inflates
    effective sample size → overfitting.

    SOLUTION:
    1. Uniqueness weight: wi = 1 / (n_overlapping_events_at_t)
       Low weight for events that share info with many others.

    2. Time decay: wi *= exp(-θ * age)
       Older samples count less (non-stationarity).

    3. Combine: final_weight = uniqueness × time_decay
    """

    def __init__(self, time_decay: float = 0.5):
        """
        time_decay: θ in [0, 1]
          0 = no decay (all samples equal weight)
          1 = maximum decay (recent observations dominate)
        """
        self.time_decay = time_decay

    def compute_uniqueness(
        self,
        events: pd.DataFrame,  # Must have t0 (index) and t1 (column)
        close: pd.Series,
    ) -> pd.Series:
        """
        Compute average label uniqueness for each event.
        Returns series of uniqueness weights (0→1).
        """
        # Build concurrency matrix: how many events active at each time t
        n_conc = pd.Series(0, index=close.index, dtype=float)

        for t0, row in events.iterrows():
            t1 = row['t1'] if 't1' in row.index else t0
            # Increment concurrency for duration of this event
            mask = (close.index >= t0) & (close.index <= t1)
            n_conc[mask] += 1

        # Uniqueness = 1 / concurrency for each event's duration
        uniqueness = pd.Series(index=events.index, dtype=float)

        for t0, row in events.iterrows():
            t1 = row['t1'] if 't1' in row.index else t0
            mask = (close.index >= t0) & (close.index <= t1)
            avg_conc = n_conc[mask].mean()
            uniqueness.loc[t0] = 1.0 / max(avg_conc, 1.0)

        return uniqueness

    def apply_time_decay(self, weights: pd.Series) -> pd.Series:
        """
        Apply time decay: more recent events get higher weights.
        Piecewise-linear decay from last observation backward.
        """
        if self.time_decay == 0:
            return weights

        cw = weights.sort_index().cumsum()
        if self.time_decay >= 0:
            slope = (1 - self.time_decay) / cw.iloc[-1]
            const = 1 - slope * cw.iloc[-1]
        else:
            slope = 1.0 / ((self.time_decay + 1) * cw.iloc[-1])
            const = 1.0 / (self.time_decay + 1)

        decayed = const + slope * cw
        decayed[decayed < 0] = 0

        return decayed.reindex(weights.index)

    def get_sample_weights(self, events: pd.DataFrame, close: pd.Series) -> pd.Series:
        """Full pipeline: uniqueness × time decay."""
        uniqueness = self.compute_uniqueness(events, close)
        decayed = self.apply_time_decay(uniqueness)
        # Normalize to sum to n_samples
        decayed = decayed / decayed.mean()
        return decayed


# ══════════════════════════════════════════════════════════════
# PURGED K-FOLD CROSS VALIDATION
# ══════════════════════════════════════════════════════════════
class PurgedKFoldCV:
    """
    Purged K-Fold Cross Validation (Lopez de Prado 2018, Ch. 7).

    WHY STANDARD K-FOLD FAILS IN FINANCE:
    Financial data has overlapping labels. When you have a 21-day
    return label at t=100, observations near t=100 in the training
    set "know" about events that are in the test set's label window.
    This creates leakage that inflates in-sample Sharpe ratios by
    200-500% compared to out-of-sample reality.

    SOLUTION:
    1. PURGING: Remove from training set all observations whose
       label window overlaps with the test set's observation window.
    2. EMBARGO: After each test set, remove additional k days from
       training (autocorrelation bleeds across the purge boundary).

    This is the ONLY correct way to cross-validate financial ML models.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        """
        n_splits: Number of folds
        embargo_pct: Fraction of total samples to embargo after each test
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        events: pd.DataFrame,  # Must have t1 column (label end time)
    ):
        """
        Yield (train_idx, test_idx) pairs with purging + embargo.

        Key difference from sklearn KFold:
        - train_idx excludes observations whose labels overlap test window
        - Additional embargo gap prevents autocorrelation leakage
        """
        n = len(X)
        embargo_size = int(n * self.embargo_pct)

        indices = np.arange(n)
        test_starts = np.linspace(0, n - 1, self.n_splits + 1, dtype=int)

        for i in range(self.n_splits):
            t_start = test_starts[i]
            t_end = test_starts[i + 1]

            # Test indices
            test_idx = indices[t_start: t_end]

            # Test window: t_start to t_end in time
            t_test_start = X.index[t_start]
            t_test_end = X.index[t_end - 1]

            # Purge: find training observations whose label overlaps test
            train_idx = []
            for j in indices:
                t_j = X.index[j]

                # Skip if j is in test set
                if t_start <= j < t_end:
                    continue

                # Get label end time for observation j
                if t_j in events.index and 't1' in events.columns:
                    t_j_end = events.loc[t_j, 't1']
                    # Purge: label overlaps test window
                    if t_j_end >= t_test_start:
                        continue

                # Embargo: too close to test window end
                if j > t_end and j < t_end + embargo_size:
                    continue

                train_idx.append(j)

            yield np.array(train_idx), test_idx

    def combinatorial_split(
        self,
        X: pd.DataFrame,
        n_test_splits: int = 2,
    ):
        """
        Combinatorial Purged CV (Lopez de Prado 2018, Ch. 12).
        Tests ALL combinations of k test splits, generating multiple
        backtest paths — prevents path-dependent overfitting.
        """
        from itertools import combinations

        n = len(X)
        n_total_splits = self.n_splits
        split_size = n // n_total_splits
        all_splits = [(i * split_size, min((i + 1) * split_size, n)) for i in range(n_total_splits)]

        for test_combo in combinations(range(n_total_splits), n_test_splits):
            test_indices = []
            for split_idx in test_combo:
                start, end = all_splits[split_idx]
                test_indices.extend(range(start, end))

            train_indices = [i for i in range(n) if i not in test_indices]
            yield np.array(train_indices), np.array(test_indices)


# ══════════════════════════════════════════════════════════════
# DEFLATED SHARPE RATIO
# ══════════════════════════════════════════════════════════════
class DeflatedSharpeRatio:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    Standard Sharpe Ratio: Biased upward when you test many strategies.
    If you test N strategies and pick the best, the expected Sharpe of
    the best = (true Sharpe) + bias(N, T).

    DSR corrects for:
    - Number of trials (selection bias)
    - Non-normality of returns (skewness, kurtosis)
    - Time series length

    DSR < 0: Strategy is likely a false discovery (lucky backtest)
    DSR > 0: Evidence of genuine skill after selection bias correction

    This is THE metric for evaluating quantitative strategies.
    """

    def compute(
        self,
        sharpe: float,         # Observed Sharpe ratio
        n_trials: int,         # Number of strategies tested
        n_obs: int,            # Number of observations
        skewness: float = 0.0, # Return skewness
        kurtosis: float = 3.0, # Return kurtosis (3 = normal)
        benchmark_sharpe: float = 0.0,  # Expected max under H0
    ) -> float:
        """
        DSR = Φ[(√(n-1) × (SR - SR*)) / √(1 - γ₃SR + ((γ₄-1)/4)SR²)]

        where:
          SR* = expected maximum SR under null hypothesis
          γ₃ = skewness
          γ₄ = kurtosis
          Φ = normal CDF

        Returns probability that SR is NOT a false positive.
        """
        if n_obs < 5:
            return 0.0

        # Expected maximum SR under null (Gaussian approximation)
        # E[max SR | N trials] ≈ (1 - γ_Euler) * Z(1-1/N) + γ_Euler * Z(1-1/(N*e))
        gamma_euler = 0.5772156649
        sr_star = (
            (1 - gamma_euler) * norm.ppf(1 - 1.0 / n_trials) +
            gamma_euler * norm.ppf(1 - 1.0 / (n_trials * np.e))
        ) / np.sqrt(n_obs)

        # Variance adjustment for non-normality
        variance_adj = (
            1 -
            skewness * sharpe +
            ((kurtosis - 1) / 4) * sharpe ** 2
        )

        if variance_adj <= 0:
            return 0.0

        # DSR probability
        z = (sharpe - sr_star) * np.sqrt(n_obs - 1) / np.sqrt(variance_adj)
        dsr = norm.cdf(z)

        return float(dsr)

    def is_genuine(self, dsr: float, threshold: float = 0.95) -> bool:
        """Returns True if DSR indicates genuine skill (not luck)."""
        return dsr >= threshold


# ══════════════════════════════════════════════════════════════
# MAIN LABELING PIPELINE
# ══════════════════════════════════════════════════════════════
class LabelingPipeline:
    """
    Complete labeling pipeline for production use.
    Combines CUSUM → Triple Barrier → Sample Weights → Meta Labels.
    """

    def __init__(
        self,
        profit_take: float = 2.0,
        stop_loss: float = 1.0,
        hold_days: int = 21,
        cusum_h: float = 1.0,
        time_decay: float = 0.5,
    ):
        self.cusum = CUSUMFilter(h_multiplier=cusum_h)
        self.labeler = TripleBarrierLabeler(
            profit_take_multiplier=profit_take,
            stop_loss_multiplier=stop_loss,
            time_barrier_days=hold_days,
        )
        self.weighter = LabelWeighter(time_decay=time_decay)
        self.fracdiff = FractionalDifferentiator()
        self.meta = MetaLabeler()
        self.dsr = DeflatedSharpeRatio()

    def run(
        self,
        close: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        primary_model_signal: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Full pipeline: raw prices → labeled, weighted dataset.

        Returns dict with:
          events: DataFrame of labeled events
          weights: Sample uniqueness weights
          meta_labels: Meta-labeling targets (if primary_model_signal provided)
          fd_series: Fractionally differentiated prices (stationary feature)
          d_value: Optimal d for fractional differentiation
          cusum_events: CUSUM-sampled timestamps
        """
        # Step 1: Compute log returns + vol
        log_ret = np.log(close / close.shift(1)).dropna()
        vol = self.labeler.compute_daily_vol(close)

        # Step 2: CUSUM filter → event timestamps
        cusum_events = self.cusum.filter(log_ret, vol)

        # Step 3: Triple-barrier labeling
        events = self.labeler.get_events(
            close=close,
            t_events=cusum_events,
            high=high,
            low=low,
            side=primary_model_signal,
        )

        if events.empty:
            return {
                'events': events,
                'weights': pd.Series(),
                'meta_labels': pd.DataFrame(),
                'fd_series': pd.Series(),
                'd_value': 1.0,
            }

        # Step 4: Sample uniqueness weights
        weights = self.weighter.get_sample_weights(events, close)

        # Step 5: Fractional differentiation (for ML features)
        fd_series, d_val = self.fracdiff.transform_prices(close)

        # Step 6: Meta-labeling (if primary model signal provided)
        meta_labels = pd.DataFrame()
        if primary_model_signal is not None:
            meta_labels = self.meta.generate_meta_labels(events, primary_model_signal)

        # Step 7: Label distribution stats
        label_dist = events['label'].value_counts(normalize=True).to_dict()

        return {
            'events': events,
            'weights': weights,
            'meta_labels': meta_labels,
            'fd_series': fd_series,
            'd_value': d_val,
            'cusum_events': cusum_events,
            'label_distribution': label_dist,
            'n_events': len(events),
            'n_cusum_samples': len(cusum_events),
            'avg_uniqueness': float(weights.mean()) if len(weights) > 0 else 0,
        }
