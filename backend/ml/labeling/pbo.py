"""
QuantEdge v6.0 — Probability of Backtest Overfitting (PBO)
==========================================================
Implements Bailey, Borwein, López de Prado & Zhu (2014):
  "The Probability of Backtest Overfitting"

Bailey DOI: 10.21314/JOR.2016.333

KEY CONCEPT:
  DSR asks: "Is this ONE Sharpe ratio real after selection bias?"
  PBO asks: "Across N strategy configs, what fraction overfit?"

METHOD (Combinatorial Symmetric Cross-Validation, CSCV):
  1. Split N×T return matrix (N strategies × T periods) into S equal slices
  2. Form every combination C(S, S/2) of train/test slice assignments
  3. For each split:
     - Find best strategy (highest Sharpe) on TRAIN slices
     - Measure its rank on TEST slices (median-centered logit)
     - Count how often "best in-sample" ranks below median OOS
  4. PBO = fraction of combinations where in-sample winner loses OOS

INTERPRETATION:
  PBO < 0.10:  strategy family is likely genuine (low overfit risk)
  PBO 0.10-0.50: uncertain, more testing needed
  PBO > 0.50:  strategy family is overfit — best in-sample is usually worst OOS
  PBO == 0.50: coin flip, i.e., no edge to in-sample selection

Reference:
  Bailey et al. (2014), "The Probability of Backtest Overfitting",
  J. of Computational Finance, 20(4): 39-69
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List, Optional, Tuple
from loguru import logger


def logit(p: float, eps: float = 1e-8) -> float:
    """Stable logit function clipping to (eps, 1-eps)."""
    p = float(np.clip(p, eps, 1 - eps))
    return float(np.log(p / (1 - p)))


class PBOCalculator:
    """
    Combinatorial Symmetric Cross-Validation (CSCV) for PBO.

    Usage:
        pbo = PBOCalculator(n_slices=8)  # 8 slices → C(8,4) = 70 combinations
        result = pbo.compute(returns_matrix)
        # returns_matrix: np.ndarray shape (N_strategies, T_periods)
    """

    def __init__(self, n_slices: int = 8, min_strategies: int = 2):
        """
        Args:
            n_slices: Number of equal-sized slices to split timeline into.
                      Must be even. Higher = more combinations but needs longer T.
                      8 is a reasonable default; 16 for longer backtests.
            min_strategies: Minimum N required to compute PBO (needs variation).
        """
        if n_slices % 2 != 0:
            raise ValueError("n_slices must be even")
        self.n_slices = n_slices
        self.min_strategies = min_strategies

    def compute(
        self,
        returns_matrix: np.ndarray,
        min_obs_per_slice: int = 5,
    ) -> Dict:
        """
        Compute PBO from a matrix of strategy returns.

        Args:
            returns_matrix: shape (N_strategies, T_periods)
                            each row = one strategy's returns through time
            min_obs_per_slice: minimum periods per slice to trust sharpe estimate

        Returns:
            {
              'pbo': float,              # P(best IS strategy < median OOS)
              'logit_mean': float,       # Mean of logit-transformed OOS ranks
              'logit_std': float,        # Std
              'n_combinations': int,     # How many splits were evaluated
              'n_strategies': int,
              'n_periods': int,
              'interpretation': str,     # Plain-English summary
              'is_overfit': bool,        # True if PBO > 0.5
            }
        """
        R = np.asarray(returns_matrix, dtype=np.float64)
        if R.ndim != 2:
            raise ValueError(f"returns_matrix must be 2D, got shape {R.shape}")

        N, T = R.shape
        if N < self.min_strategies:
            return self._empty_result(N, T, reason=f"need ≥{self.min_strategies} strategies, got {N}")

        slice_size = T // self.n_slices
        if slice_size < min_obs_per_slice:
            return self._empty_result(N, T, reason=f"slice size {slice_size} < {min_obs_per_slice} min obs")

        # Split into slices, discarding any tail that doesn't fit
        usable_T = slice_size * self.n_slices
        R = R[:, :usable_T]
        slices = R.reshape(N, self.n_slices, slice_size)  # (N, n_slices, slice_size)

        # Sharpe per (strategy, slice). Annualized assuming daily data (252).
        mean_per_slice = slices.mean(axis=2)  # (N, n_slices)
        std_per_slice = slices.std(axis=2) + 1e-10
        sharpe_per_slice = mean_per_slice / std_per_slice * np.sqrt(252)

        # All C(n_slices, n_slices/2) ways to pick train slices; rest = test
        half = self.n_slices // 2
        slice_ids = list(range(self.n_slices))
        combos = list(combinations(slice_ids, half))

        logit_values = []
        for train_slices in combos:
            train_set = set(train_slices)
            test_slices = [s for s in slice_ids if s not in train_set]

            # Mean Sharpe of each strategy on TRAIN
            train_sharpe = sharpe_per_slice[:, list(train_slices)].mean(axis=1)
            # Mean Sharpe of each strategy on TEST
            test_sharpe = sharpe_per_slice[:, test_slices].mean(axis=1)

            # Strategy that looked best in-sample
            winner = int(np.argmax(train_sharpe))

            # Rank of that winner on test: what fraction of strategies it beats OOS
            # (higher is better; 0.5 = median; 1.0 = best OOS; 0 = worst OOS)
            n_beaten = int((test_sharpe[winner] > test_sharpe).sum())
            rank = n_beaten / (N - 1) if N > 1 else 0.5
            # Convert rank to logit to symmetrize distribution for mean/std
            logit_values.append(logit(rank))

        logit_arr = np.asarray(logit_values)
        # PBO = P(logit < 0) = P(winner ranks below median OOS)
        pbo_value = float((logit_arr < 0).mean())

        return {
            'pbo': round(pbo_value, 4),
            'logit_mean': round(float(logit_arr.mean()), 4),
            'logit_std': round(float(logit_arr.std()), 4),
            'n_combinations': len(combos),
            'n_strategies': int(N),
            'n_periods': int(T),
            'n_slices': int(self.n_slices),
            'interpretation': self._interpret(pbo_value),
            'is_overfit': pbo_value > 0.5,
        }

    def _interpret(self, pbo: float) -> str:
        if pbo < 0.10:
            return "LOW: strategy family appears genuine, low overfit risk"
        if pbo < 0.30:
            return "MILD: some overfit risk but acceptable"
        if pbo < 0.50:
            return "MODERATE: noticeable overfit risk, consider more robust validation"
        if pbo < 0.70:
            return "HIGH: likely overfit, best in-sample strategies underperform OOS"
        return "SEVERE: the backtest is dominated by overfit; in-sample winners are OOS losers"

    def _empty_result(self, N: int, T: int, reason: str) -> Dict:
        return {
            'pbo': None,
            'logit_mean': None,
            'logit_std': None,
            'n_combinations': 0,
            'n_strategies': int(N),
            'n_periods': int(T),
            'n_slices': int(self.n_slices),
            'interpretation': f"Skipped: {reason}",
            'is_overfit': None,
        }


def compute_pbo_from_ticker_history(
    ticker_returns: np.ndarray,
    n_strategies: int = 8,
    n_slices: int = 6,
    seed: int = 42,
) -> Dict:
    """
    Shortcut: synthesize N variant strategies from one return series
    by varying holding period and signal threshold. Useful for single-ticker
    PBO where the analyze endpoint needs a PBO estimate on the fly.

    Args:
        ticker_returns: 1D array of daily returns (float), length >= 100
        n_strategies: how many synthetic variants to test
        n_slices: CSCV slices

    Returns:
        Same dict shape as PBOCalculator.compute()
    """
    r = np.asarray(ticker_returns, dtype=np.float64).flatten()
    if len(r) < 100:
        calc = PBOCalculator(n_slices=n_slices)
        return calc._empty_result(n_strategies, len(r), "returns too short for PBO")

    rng = np.random.default_rng(seed)
    # Synthetic strategies: each one takes a random rolling threshold on z-score
    strategies = []
    for i in range(n_strategies):
        thresh = 0.5 + 0.3 * rng.standard_normal()  # random threshold near 0.5
        lookback = int(10 + 20 * rng.random())       # lookback 10-30 days
        # Rolling z-score signal
        s = pd.Series(r)
        z = (s - s.rolling(lookback, min_periods=5).mean()) / (s.rolling(lookback, min_periods=5).std() + 1e-10)
        # Long when z > thresh, short when z < -thresh
        position = np.where(z > thresh, 1, np.where(z < -thresh, -1, 0))
        strategy_returns = position * r
        strategies.append(strategy_returns)

    returns_matrix = np.asarray(strategies)
    calc = PBOCalculator(n_slices=n_slices)
    return calc.compute(returns_matrix)
