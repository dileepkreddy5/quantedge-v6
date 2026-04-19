"""
QuantEdge v6.0 — Position Sizer
================================
Takes ranked tickers and produces actual dollar-weighted position sizes.

Four sizing methods:
  1. Equal weight         — baseline, equal dollar per position
  2. Inverse volatility   — size ∝ 1/vol, favors stable names
  3. Equal Risk Contrib   — each position contributes equally to portfolio risk
  4. Hierarchical Risk Parity — López de Prado (2016), cluster-aware

Then applies:
  - Volatility targeting (scales total exposure to hit target vol)
  - Regime multiplier (from regime_overlay)
  - Max position size cap (no single name >10% default)
  - Min position size floor (no dust positions)

Output: dollar amounts and share counts per ticker, ready to place orders.

References:
  Markowitz (1952) — Modern Portfolio Theory
  Maillard, Roncalli, Teïletche (2010) — ERC portfolios
  López de Prado (2016) — Building Diversified Portfolios That Outperform
  Moreira & Muir (2017) — Volatility-Managed Portfolios
"""

import os
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


POLYGON_BASE = "https://api.polygon.io"


@dataclass
class Position:
    """A single position's sizing output."""
    ticker: str
    weight: float                  # fraction of portfolio (0-1)
    dollars: float                 # dollar allocation
    shares: int                    # integer share count
    current_price: float
    volatility_annual: float       # annualized vol used in sizing
    risk_contribution: float       # fraction of total portfolio vol
    rank: int                      # from the scanner


@dataclass
class PortfolioAllocation:
    """Complete portfolio sizing result."""
    total_capital: float
    deployed_capital: float        # after regime multiplier
    cash_reserve: float            # capital - deployed
    method: str                    # which sizing method was used
    regime: str
    regime_multiplier: float
    target_volatility: float
    estimated_portfolio_vol: float
    positions: List[Position]
    warnings: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════
async def fetch_returns_matrix(
    tickers: List[str],
    api_key: str,
    session: aiohttp.ClientSession,
    days: int = 252,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Fetch aligned daily returns for all tickers over trailing N days.
    Returns (returns_df, current_prices_dict).
    """
    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 60)

    async def _fetch_one(ticker: str):
        url = (f"{POLYGON_BASE}/v2/aggs/ticker/{ticker.upper()}"
               f"/range/1/day/{start.isoformat()}/{end.isoformat()}")
        params = {"adjusted": "true", "sort": "asc", "limit": 500, "apiKey": api_key}
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return ticker, None
                data = await resp.json()
        except Exception:
            return ticker, None
        results = data.get("results", [])
        if len(results) < 50:
            return ticker, None
        df = pd.DataFrame(results)
        df["datetime"] = pd.to_datetime(df["t"], unit="ms")
        df = df[["datetime", "c"]].set_index("datetime").rename(columns={"c": ticker})
        return ticker, df

    tasks = [_fetch_one(t) for t in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    dfs = []
    prices = {}
    for ticker, df in results:
        if df is not None:
            dfs.append(df)
            prices[ticker] = float(df[ticker].iloc[-1])

    if not dfs:
        return pd.DataFrame(), prices

    # Align all price series
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.join(d, how="outer")
    merged = merged.ffill().dropna(how="all").tail(days)
    returns = merged.pct_change().dropna()
    return returns, prices


# ══════════════════════════════════════════════════════════════
# SIZING METHODS
# ══════════════════════════════════════════════════════════════
def size_equal_weight(tickers: List[str]) -> Dict[str, float]:
    """Simple 1/N weighting."""
    if not tickers:
        return {}
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}


def size_inverse_volatility(returns: pd.DataFrame) -> Dict[str, float]:
    """
    Weight proportional to 1/vol.
    Stable names get bigger positions, volatile names get smaller.
    """
    if returns.empty:
        return {}
    vols = returns.std(axis=0) * np.sqrt(252)  # annualized
    vols = vols[vols > 0]  # drop zero-vol
    if vols.empty:
        return {}
    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()
    return weights.to_dict()


def size_equal_risk_contribution(
    returns: pd.DataFrame,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> Dict[str, float]:
    """
    Equal Risk Contribution (ERC): each position contributes equally to
    portfolio volatility. Iterative solver.

    Reference: Maillard, Roncalli, Teïletche (2010)
    """
    if returns.empty:
        return {}
    tickers = returns.columns.tolist()
    n = len(tickers)
    if n == 1:
        return {tickers[0]: 1.0}

    cov = returns.cov().values * 252  # annualized cov matrix
    # Regularize to avoid singularity
    cov = cov + np.eye(n) * 1e-8

    # Start with inverse-vol weights
    vols = np.sqrt(np.diag(cov))
    w = (1.0 / vols) / (1.0 / vols).sum()

    for iteration in range(max_iter):
        portfolio_vol = np.sqrt(w @ cov @ w)
        marginal_contrib = cov @ w / portfolio_vol
        risk_contrib = w * marginal_contrib
        target = portfolio_vol / n

        # Adjust weights toward equal risk contribution
        adjustment = target / risk_contrib
        w_new = w * adjustment
        w_new = w_new / w_new.sum()

        if np.max(np.abs(w_new - w)) < tol:
            break
        w = w_new

    return dict(zip(tickers, w.tolist()))


def size_hierarchical_risk_parity(returns: pd.DataFrame) -> Dict[str, float]:
    """
    Hierarchical Risk Parity (López de Prado 2016).
    Clusters correlated assets, allocates between clusters, then within.

    Uses your existing HRP implementation in portfolio_engine.py if available.
    Falls back to inverse-vol if HRP unavailable.
    """
    try:
        from ml.portfolio.portfolio_engine import HierarchicalRiskParity
    except ImportError:
        logger.warning("HRP not available — falling back to inverse-vol")
        return size_inverse_volatility(returns)

    if returns.empty:
        return {}
    tickers = returns.columns.tolist()
    if len(tickers) < 3:
        # HRP needs clustering; fall back for tiny portfolios
        return size_inverse_volatility(returns)

    try:
        hrp = HierarchicalRiskParity(linkage_method='single')
        cov = returns.cov().values * 252
        corr = returns.corr().values
        weights = hrp.compute_weights(cov, corr, asset_names=tickers)
        if isinstance(weights, dict):
            return weights
        # Could be ndarray depending on impl
        return dict(zip(tickers, weights))
    except Exception as e:
        logger.warning(f"HRP compute failed, falling back to inverse-vol: {e}")
        return size_inverse_volatility(returns)


# ══════════════════════════════════════════════════════════════
# VOL TARGETING + CAPS
# ══════════════════════════════════════════════════════════════
def estimate_portfolio_vol(
    weights: Dict[str, float],
    returns: pd.DataFrame,
) -> float:
    """Portfolio vol given weights and return history. Annualized."""
    if not weights or returns.empty:
        return 0.0
    tickers = [t for t in weights if t in returns.columns]
    if not tickers:
        return 0.0
    w = np.array([weights[t] for t in tickers])
    cov = returns[tickers].cov().values * 252
    var = float(w @ cov @ w)
    return float(np.sqrt(max(var, 0)))


def apply_volatility_target(
    weights: Dict[str, float],
    returns: pd.DataFrame,
    target_vol: float = 0.12,
    max_leverage: float = 1.0,
) -> Tuple[Dict[str, float], float]:
    """
    Scale weights so portfolio hits target annualized vol.
    Returns (scaled_weights, scaling_factor).
    Capped at max_leverage (default 1.0 = no leverage).
    """
    current_vol = estimate_portfolio_vol(weights, returns)
    if current_vol <= 0:
        return weights, 1.0
    scale = min(target_vol / current_vol, max_leverage)
    return {t: w * scale for t, w in weights.items()}, scale


def apply_position_caps(
    weights: Dict[str, float],
    max_position: float = 0.10,
    min_position: float = 0.01,
) -> Dict[str, float]:
    """
    Cap individual positions. Redistribute excess to uncapped positions.
    Drop positions below min_position (dust).
    """
    if not weights:
        return weights

    # Drop dust
    weights = {t: w for t, w in weights.items() if w >= min_position}
    if not weights:
        return {}

    # Iteratively apply cap
    for _ in range(10):
        capped = {t: min(w, max_position) for t, w in weights.items()}
        total = sum(capped.values())
        if total == 0:
            break

        # Redistribute excess to uncapped positions
        excess = 1.0 - total
        if excess <= 0.001:
            # Renormalize and done
            return {t: w / total for t, w in capped.items()}

        uncapped = {t: w for t, w in capped.items() if w < max_position - 1e-6}
        if not uncapped:
            # Everyone is capped — just normalize
            return {t: w / total for t, w in capped.items()}
        uncapped_total = sum(uncapped.values())
        if uncapped_total == 0:
            return {t: w / total for t, w in capped.items()}
        for t in uncapped:
            capped[t] += excess * (uncapped[t] / uncapped_total)
        weights = capped

    total = sum(weights.values())
    return {t: w / total for t, w in weights.items()} if total > 0 else weights


# ══════════════════════════════════════════════════════════════
# MAIN POSITION SIZER
# ══════════════════════════════════════════════════════════════
class PositionSizer:
    """
    Top-level entry. Takes ranked ticker list + capital, returns
    full portfolio allocation with risk-parity sizing and regime adjustment.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")

    async def size(
        self,
        ranked_tickers: List[str],
        total_capital: float,
        method: str = "hrp",
        target_vol: float = 0.12,
        max_position: float = 0.10,
        min_position: float = 0.01,
        regime_multiplier: float = 1.0,
        regime_label: str = "normal",
    ) -> PortfolioAllocation:
        """
        Size a portfolio.

        Parameters:
          ranked_tickers: list of tickers in ranked order (best first)
          total_capital: dollars available
          method: 'equal' | 'inverse_vol' | 'erc' | 'hrp'
          target_vol: annualized volatility target (12% is typical)
          max_position: max fraction of portfolio per name
          min_position: min fraction (below this = dust, dropped)
          regime_multiplier: from regime_overlay (0.5-1.1)
          regime_label: for output metadata
        """
        warnings = []
        if not ranked_tickers:
            return PortfolioAllocation(
                total_capital=total_capital,
                deployed_capital=0, cash_reserve=total_capital,
                method=method, regime=regime_label,
                regime_multiplier=regime_multiplier,
                target_volatility=target_vol,
                estimated_portfolio_vol=0,
                positions=[],
                warnings=["No tickers provided"],
            )

        # Cap selection list — typically 15-20 names for diversification
        selected = ranked_tickers[:20]

        # Fetch return data
        async with aiohttp.ClientSession() as session:
            returns, prices = await fetch_returns_matrix(
                selected, self.api_key, session, days=252
            )

        if returns.empty:
            warnings.append("Insufficient return data — falling back to equal weight")
            weights = size_equal_weight(selected)
        else:
            # Drop tickers where we don't have return data
            available = [t for t in selected if t in returns.columns]
            if len(available) < len(selected):
                dropped = set(selected) - set(available)
                warnings.append(f"Dropped (no price data): {sorted(dropped)}")
            selected = available

            # Compute base weights
            method_lower = method.lower()
            if method_lower == "equal":
                weights = size_equal_weight(selected)
            elif method_lower in ("inverse_vol", "ivp"):
                weights = size_inverse_volatility(returns[selected])
            elif method_lower == "erc":
                weights = size_equal_risk_contribution(returns[selected])
            elif method_lower == "hrp":
                weights = size_hierarchical_risk_parity(returns[selected])
            else:
                warnings.append(f"Unknown method '{method}' — using HRP")
                weights = size_hierarchical_risk_parity(returns[selected])

        # Apply position caps
        weights = apply_position_caps(weights, max_position=max_position, min_position=min_position)

        # Apply vol targeting
        vol_target_effective = target_vol
        scaling = 1.0
        if not returns.empty:
            weights, scaling = apply_volatility_target(
                weights, returns, target_vol=vol_target_effective, max_leverage=1.0
            )

        # Apply regime multiplier to total exposure
        weights = {t: w * regime_multiplier for t, w in weights.items()}

        # Compute dollar allocations and share counts
        positions: List[Position] = []
        deployed = 0.0
        portfolio_vol = estimate_portfolio_vol(weights, returns) if not returns.empty else 0.0

        for rank_idx, ticker in enumerate(ranked_tickers, 1):
            if ticker not in weights or weights[ticker] <= 0:
                continue
            w = weights[ticker]
            dollars = w * total_capital
            price = prices.get(ticker, 0)
            shares = int(dollars / price) if price > 0 else 0
            actual_dollars = shares * price
            vol = float(returns[ticker].std() * np.sqrt(252)) if ticker in returns.columns else 0.0

            # Risk contribution: this position's share of total portfolio risk
            # Rough: (w_i * vol_i) / sum(w_j * vol_j) — not exact ERC but close
            if portfolio_vol > 0:
                risk_contrib = (w * vol) / portfolio_vol if portfolio_vol > 0 else 0.0
            else:
                risk_contrib = 0.0

            positions.append(Position(
                ticker=ticker,
                weight=round(w, 4),
                dollars=round(actual_dollars, 2),
                shares=shares,
                current_price=round(price, 2),
                volatility_annual=round(vol, 4),
                risk_contribution=round(risk_contrib, 4),
                rank=rank_idx,
            ))
            deployed += actual_dollars

        positions.sort(key=lambda p: p.rank)

        return PortfolioAllocation(
            total_capital=total_capital,
            deployed_capital=round(deployed, 2),
            cash_reserve=round(total_capital - deployed, 2),
            method=method,
            regime=regime_label,
            regime_multiplier=regime_multiplier,
            target_volatility=target_vol,
            estimated_portfolio_vol=round(portfolio_vol, 4),
            positions=positions,
            warnings=warnings,
        )


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════
async def _test():
    import sys
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set"); sys.exit(1)

    # Test portfolio: top semiconductor cluster from the scan
    ranked = ["LRCX", "AMD", "ADI", "AMAT", "KLAC", "NVDA", "AAPL", "GOOGL",
              "MSFT", "META", "AVGO", "CAT", "CVX", "COP", "ORCL"]

    print(f"\nSizing {len(ranked)} tickers with $100,000 capital...")
    print(f"Method: HRP | Target vol: 12% | Regime: NORMAL (1.0x)")

    sizer = PositionSizer(api_key=api_key)
    alloc = await sizer.size(
        ranked_tickers=ranked,
        total_capital=100_000,
        method="hrp",
        target_vol=0.12,
        regime_multiplier=1.0,
        regime_label="normal",
    )

    print(f"\n{'='*80}")
    print(f"  PORTFOLIO ALLOCATION — {alloc.method.upper()}")
    print(f"{'='*80}")
    print(f"  Total capital:       ${alloc.total_capital:,.2f}")
    print(f"  Deployed:            ${alloc.deployed_capital:,.2f}  ({alloc.deployed_capital/alloc.total_capital*100:.1f}%)")
    print(f"  Cash reserve:        ${alloc.cash_reserve:,.2f}")
    print(f"  Target vol:          {alloc.target_volatility*100:.1f}%")
    print(f"  Estimated port vol:  {alloc.estimated_portfolio_vol*100:.1f}%")
    print(f"  Regime:              {alloc.regime} ({alloc.regime_multiplier}x)")

    if alloc.warnings:
        print(f"\n  Warnings:")
        for w in alloc.warnings:
            print(f"    - {w}")

    print(f"\n  {'Rank':<5}{'Ticker':<8}{'Weight':>8}{'Shares':>8}{'Price':>9}{'Dollars':>11}{'AnnVol':>8}{'RiskC':>7}")
    print(f"  {'-'*5}{'-'*8}{'-'*8}{'-'*8}{'-'*9}{'-'*11}{'-'*8}{'-'*7}")
    for p in alloc.positions:
        print(f"  {p.rank:<5}{p.ticker:<8}{p.weight*100:>7.2f}%{p.shares:>8}"
              f"${p.current_price:>7.2f}${p.dollars:>10,.0f}"
              f"{p.volatility_annual*100:>7.1f}%{p.risk_contribution*100:>6.1f}%")

    # Sanity check
    total_weight = sum(p.weight for p in alloc.positions)
    print(f"\n  Total weight: {total_weight*100:.1f}% (should be <=100% before cash)")
    print(f"  Positions: {len(alloc.positions)}")


if __name__ == "__main__":
    asyncio.run(_test())
