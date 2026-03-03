"""
QuantEdge v6.0 — CAPITAL ALLOCATION ENGINE (Layer 6)
════════════════════════════════════════════════════════════════
SEPARATE from portfolio construction. This layer controls the
SCALE of the portfolio, not the composition.

Portfolio construction asks: "What to buy?"
Capital allocation asks: "How much to deploy?"

This distinction is critical:
- A perfect portfolio at 200% leverage kills you in a crisis
- A mediocre portfolio at 30% leverage survives anything
- Capital allocation is the difference between survival and ruin

VOLATILITY TARGETING:
    Target: σ_target = 10% annual
    Scale factor: k_t = σ_target / σ̂_t
    Gross exposure: GE_t = k_t * GE_base

    When realized vol spikes (crisis):
        σ̂_t ↑ → k_t ↓ → GE_t ↓ (automatic de-risking)

    When vol is low (complacency):
        σ̂_t ↓ → k_t ↑ → GE_t ↑ (but capped at max_leverage)

DRAWDOWN GOVERNOR:
    Reduces capital automatically during drawdown:
    If drawdown > -5%:  scale by 0.90
    If drawdown > -10%: scale by 0.70
    If drawdown > -15%: scale by 0.50
    If drawdown > -20%: scale by 0.25

This prevents the ruin trap: losing more trying to recover losses.

REGIME-BASED DE-RISKING:
    CRISIS regime: cap leverage at 0.5x
    ELEVATED regime: cap leverage at 0.75x
    NORMAL regime: full leverage allowed

Mathematical references:
  - Moreira & Muir (2017): Volatility-Managed Portfolios
  - Hurst et al (2017): A Century of Evidence on Trend-Following
  - Kaminski (2014): In Search of Crisis Alpha
════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CapitalAllocationDecision:
    """Output of the capital allocation engine."""
    scale_factor: float              # 0.0 to max_leverage
    gross_exposure_target: float     # e.g. 0.80 = 80% deployed
    net_exposure_limit: float        # Net long exposure limit
    vol_regime: str                  # CALM | NORMAL | ELEVATED | CRISIS
    drawdown_state: str              # NORMAL | CAUTION | WARNING | CRITICAL
    drawdown_current: float          # Current drawdown from high-water mark
    vol_estimated_annual: float      # Current volatility estimate
    recommended_action: str          # FULL_DEPLOY | REDUCE | HALT
    max_new_positions: int           # How many new positions to open
    reason: str                      # Human-readable explanation


class VolatilityTargetingEngine:
    """
    Dynamic volatility targeting with multiple estimators.

    Uses ensemble of volatility estimators for robustness:
    1. EWMA (fast, regime-responsive)
    2. Realized vol (backward-looking, stable)
    3. GARCH forecast (forward-looking)

    The combination reduces estimation error and regime lag.

    Moreira & Muir (2017) showed that simple vol-targeting
    improves Sharpe ratio by ~0.3 across all major factors.
    This is one of the most robust findings in modern finance.
    """

    TARGET_VOL = 0.10       # 10% annual vol target
    MAX_LEVERAGE = 1.5      # Never exceed 1.5x gross
    MIN_LEVERAGE = 0.10     # Never go below 10% deployed

    def __init__(self, target_vol: float = 0.10,
                 ewma_lambda: float = 0.94,
                 vol_floor: float = 0.05):
        self.target_vol = target_vol
        self.ewma_lambda = ewma_lambda
        self.vol_floor = vol_floor   # Minimum vol estimate (prevents leverage > max)
        self._ewma_var: Optional[float] = None

    def update_ewma_vol(self, daily_return: float) -> float:
        """Update EWMA variance estimate with new return."""
        if self._ewma_var is None:
            self._ewma_var = daily_return ** 2
        else:
            self._ewma_var = (self.ewma_lambda * self._ewma_var
                             + (1 - self.ewma_lambda) * daily_return ** 2)
        return np.sqrt(self._ewma_var * 252)

    def compute_vol_ensemble(self, returns: pd.Series) -> Tuple[float, str]:
        """
        Ensemble volatility estimate.

        vol_estimate = w1 * vol_ewma + w2 * vol_21d + w3 * vol_63d

        Returns (annual_vol, vol_regime)
        """
        if len(returns) < 5:
            return 0.15, 'NORMAL'  # Default estimate

        # EWMA vol
        ewma_var = 0.0
        for r in returns.values:
            ewma_var = self.ewma_lambda * ewma_var + (1 - self.ewma_lambda) * r**2
        vol_ewma = np.sqrt(ewma_var * 252)

        # 21-day realized vol
        vol_21d = returns.iloc[-21:].std() * np.sqrt(252) if len(returns) >= 21 else vol_ewma

        # 63-day realized vol (more stable)
        vol_63d = returns.iloc[-63:].std() * np.sqrt(252) if len(returns) >= 63 else vol_21d

        # Ensemble (more weight on recent)
        vol_estimate = 0.50 * vol_ewma + 0.30 * vol_21d + 0.20 * vol_63d

        # Ensure vol floor
        vol_estimate = max(vol_estimate, self.vol_floor)

        # Classify regime
        historical_vol = returns.std() * np.sqrt(252) if len(returns) >= 252 else 0.15
        vol_ratio = vol_estimate / (historical_vol + 1e-10)

        if vol_ratio > 2.5:
            regime = 'CRISIS'
        elif vol_ratio > 1.5:
            regime = 'ELEVATED'
        elif vol_ratio < 0.7:
            regime = 'CALM'
        else:
            regime = 'NORMAL'

        return float(vol_estimate), regime

    def compute_scale_factor(self, vol_estimate: float,
                              regime: str) -> float:
        """
        k_t = σ_target / σ̂_t

        Moreira & Muir (2017):
        "A strategy that targets constant volatility by scaling inversely
        with realized variance earns significantly higher Sharpe ratios."

        Key insight: reducing leverage when vol is high AVOIDS the
        worst drawdowns, which are concentrated in high-vol periods.
        """
        # Base scale from vol targeting
        scale = self.target_vol / vol_estimate

        # Regime overlay (additional de-risking)
        regime_cap = {
            'CALM':     self.MAX_LEVERAGE,
            'NORMAL':   self.MAX_LEVERAGE,
            'ELEVATED': min(self.MAX_LEVERAGE, 0.75),
            'CRISIS':   min(self.MAX_LEVERAGE, 0.50),
        }

        scale = min(scale, regime_cap.get(regime, 1.0))
        scale = max(scale, self.MIN_LEVERAGE)

        return float(scale)


class DrawdownGovernor:
    """
    Automatic de-risking based on portfolio drawdown.

    This is the circuit breaker of the portfolio.
    When drawdown exceeds thresholds, we automatically reduce risk.

    Why? Two reasons:
    1. Psychological: humans make bad decisions under stress.
       Automated de-risking removes emotion.
    2. Mathematical: after large losses, variance of ruin (bankruptcy)
       increases exponentially. Better to reduce size than risk ruin.

    The "Three Strikes" rule:
    - Strike 1 (5% DD): Reduce to 90% allocation
    - Strike 2 (10% DD): Reduce to 70% allocation, stop new positions
    - Strike 3 (15% DD): Reduce to 50% allocation, review all models
    - Critical (20% DD): 25% allocation, halt trading, full review

    Recovery rule: gradually increase allocation as drawdown recovers.
    Don't "all-in" after drawdown — that's how people lose everything.
    """

    DRAWDOWN_THRESHOLDS = {
        -0.05: ('CAUTION',  0.90),   # -5%: CAUTION, 90% capital
        -0.10: ('WARNING',  0.70),   # -10%: WARNING, 70% capital
        -0.15: ('CRITICAL', 0.50),   # -15%: CRITICAL, 50% capital
        -0.20: ('HALT',     0.25),   # -20%: HALT, 25% capital
    }

    def __init__(self):
        self.high_water_mark: float = 1.0
        self.portfolio_value: float = 1.0
        self.drawdown_history: List[float] = []
        self.recovery_rate: float = 0.02  # Increase allocation 2% per day of recovery

    def update(self, portfolio_value: float) -> Tuple[str, float, float]:
        """
        Update drawdown state with new portfolio value.
        Returns (state, current_drawdown, allocation_scale)
        """
        self.portfolio_value = portfolio_value

        # Update high-water mark
        if portfolio_value > self.high_water_mark:
            self.high_water_mark = portfolio_value

        # Current drawdown
        drawdown = (portfolio_value / self.high_water_mark) - 1.0
        self.drawdown_history.append(drawdown)

        # Determine state and allocation
        state = 'NORMAL'
        allocation = 1.0

        for threshold, (s, a) in sorted(self.DRAWDOWN_THRESHOLDS.items()):
            if drawdown <= threshold:
                state = s
                allocation = a

        return state, float(drawdown), float(allocation)

    def recovery_trajectory(self, current_drawdown: float,
                             days_in_recovery: int) -> float:
        """
        Gradual position restoration after drawdown recovery.

        We don't snap back to full allocation immediately.
        Increase by recovery_rate per day until fully restored.
        This prevents "false dawn" re-risking.
        """
        if current_drawdown >= 0:
            # Fully recovered
            return min(1.0, days_in_recovery * self.recovery_rate)

        # Still in drawdown but improving
        recovery_pct = min(1.0, abs(current_drawdown) * 5)
        return 1.0 - recovery_pct


# ─────────────────────────────────────────────────────────────
# MASTER CAPITAL ALLOCATION ENGINE
# ─────────────────────────────────────────────────────────────

class CapitalAllocationEngine:
    """
    Master capital allocation engine.
    Combines volatility targeting + drawdown governor + regime overlay.

    This is the LAST GATE before orders go to market.
    All signals flow through this engine. This is what keeps you alive.
    """

    def __init__(self, target_vol: float = 0.10,
                 max_leverage: float = 1.5,
                 base_gross_exposure: float = 0.95):
        self.vol_engine = VolatilityTargetingEngine(target_vol=target_vol)
        self.dd_governor = DrawdownGovernor()
        self.max_leverage = max_leverage
        self.base_exposure = base_gross_exposure
        self.target_vol = target_vol
        self._decision_history: List[CapitalAllocationDecision] = []

    def compute_allocation(self,
                            portfolio_returns: pd.Series,
                            portfolio_value: float,
                            risk_regime: str,
                            num_alpha_signals: int = 20
                            ) -> CapitalAllocationDecision:
        """
        Computes the capital allocation decision.

        This is the master function that combines all scaling factors.
        Scale factors are MULTIPLICATIVE:
            final_scale = vol_scale * dd_scale * regime_scale

        Each factor independently reduces risk.
        The product ensures all risks are addressed simultaneously.
        """

        # 1. Volatility estimation and targeting
        vol_estimate, vol_regime = self.vol_engine.compute_vol_ensemble(portfolio_returns)
        vol_scale = self.vol_engine.compute_scale_factor(vol_estimate, vol_regime)

        # 2. Drawdown governor
        dd_state, current_dd, dd_scale = self.dd_governor.update(portfolio_value)

        # 3. Regime overlay (from separate regime engine)
        regime_scale = {
            'BULL_LOW_VOL':   1.00,
            'BULL_HIGH_VOL':  0.85,
            'MEAN_REVERT':    0.90,
            'BEAR_LOW_VOL':   0.70,
            'BEAR_HIGH_VOL':  0.50,
            'CRISIS':         0.35,
            'UNKNOWN':        0.80,
        }.get(risk_regime, 0.80)

        # 4. Combine scaling factors
        combined_scale = vol_scale * dd_scale * regime_scale
        gross_exposure = self.base_exposure * combined_scale
        gross_exposure = np.clip(gross_exposure, 0.05, self.max_leverage)

        # 5. Net exposure limit (more conservative in bear/crisis)
        net_limit = gross_exposure * {
            'BULL_LOW_VOL':   1.00,
            'BULL_HIGH_VOL':  0.90,
            'MEAN_REVERT':    0.80,
            'BEAR_LOW_VOL':   0.60,
            'BEAR_HIGH_VOL':  0.40,
            'CRISIS':         0.20,
        }.get(risk_regime, 0.70)

        # 6. Recommended action
        if gross_exposure < 0.20 or dd_state == 'HALT':
            action = 'HALT'
            max_new = 0
        elif gross_exposure < 0.50 or dd_state == 'CRITICAL':
            action = 'REDUCE'
            max_new = 0
        elif dd_state in ('WARNING', 'CAUTION'):
            action = 'REDUCE'
            max_new = 2
        else:
            action = 'FULL_DEPLOY'
            max_new = num_alpha_signals

        # 7. Build reason string
        reason_parts = [
            f"Vol={vol_estimate*100:.1f}%(target={self.target_vol*100:.0f}%)",
            f"scale={vol_scale:.2f})",
            f"DD={current_dd*100:.1f}%(scale={dd_scale:.2f})",
            f"Regime={risk_regime}(scale={regime_scale:.2f})",
            f"Final={gross_exposure:.2f}x",
        ]
        reason = " | ".join(reason_parts)

        decision = CapitalAllocationDecision(
            scale_factor=float(combined_scale),
            gross_exposure_target=float(gross_exposure),
            net_exposure_limit=float(net_limit),
            vol_regime=vol_regime,
            drawdown_state=dd_state,
            drawdown_current=float(current_dd),
            vol_estimated_annual=float(vol_estimate),
            recommended_action=action,
            max_new_positions=max_new,
            reason=reason,
        )

        self._decision_history.append(decision)
        return decision

    def apply_scale_to_weights(self, weights: Dict[str, float],
                                 decision: CapitalAllocationDecision
                                 ) -> Dict[str, float]:
        """
        Applies the capital allocation scale to portfolio weights.

        The optimizer produces fully-invested weights (sum=1).
        We scale them down to the target gross exposure.
        """
        if decision.recommended_action == 'HALT':
            # Move to maximum cash — return empty weights
            return {k: 0.0 for k in weights}

        target_gross = decision.gross_exposure_target
        current_gross = sum(abs(v) for v in weights.values())

        if current_gross < 1e-6:
            return weights

        scale = target_gross / current_gross
        scaled = {k: v * scale for k, v in weights.items()}

        # Enforce net exposure limit
        net = sum(scaled.values())
        if abs(net) > decision.net_exposure_limit:
            net_adj = decision.net_exposure_limit / (abs(net) + 1e-10) * np.sign(net)
            # Scale shorts to bring net within limit
            long_sum = sum(v for v in scaled.values() if v > 0)
            short_sum = sum(v for v in scaled.values() if v < 0)
            if abs(net) > decision.net_exposure_limit:
                adj_factor = decision.net_exposure_limit / abs(net)
                scaled = {k: v * adj_factor for k, v in scaled.items()}

        return scaled

    def get_performance_summary(self) -> Dict[str, float]:
        """Summary statistics for the capital allocation engine."""
        if not self._decision_history:
            return {}

        scales = [d.scale_factor for d in self._decision_history]
        exposures = [d.gross_exposure_target for d in self._decision_history]
        dds = [d.drawdown_current for d in self._decision_history]

        return {
            'avg_scale_factor': np.mean(scales),
            'avg_gross_exposure': np.mean(exposures),
            'min_gross_exposure': np.min(exposures),
            'max_drawdown': np.min(dds),
            'current_drawdown': dds[-1] if dds else 0.0,
            'pct_time_halted': np.mean([d.recommended_action == 'HALT'
                                        for d in self._decision_history]),
            'pct_time_full_deploy': np.mean([d.recommended_action == 'FULL_DEPLOY'
                                             for d in self._decision_history]),
        }
