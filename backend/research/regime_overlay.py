"""
QuantEdge v6.0 — Regime Overlay
================================
Market-wide risk regime detector. Applies a multiplier to all ranking scores
based on current market stress, protecting against market-wide drawdowns.

Signals:
  1. VIX level (fear gauge)
     - VIXY ETF used as proxy since Polygon doesn't serve $VIX directly on Starter
     - SPY volatility fallback if VIXY unavailable
  2. Market breadth (% of universe above 200-day MA)
     - Computed from actual universe scan data

Regime classification:
  CALM      VIX < 15 and breadth > 0.70  → multiplier 1.10
  NORMAL    VIX 15-20 or breadth 0.50-0.70 → multiplier 1.00
  ELEVATED  VIX 20-30 or breadth 0.30-0.50 → multiplier 0.80
  PANIC     VIX > 30 or breadth < 0.30     → multiplier 0.50

Reference: Whaley (2000) on VIX, Campbell & Vuolteenaho (2004) on breadth
"""

import os
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger


POLYGON_BASE = "https://api.polygon.io"


@dataclass
class RegimeState:
    regime: str                # "calm" / "normal" / "elevated" / "panic"
    multiplier: float          # applied to composite scores
    vix_level: Optional[float] = None
    spy_vol_20d: Optional[float] = None
    breadth_pct_above_200ma: Optional[float] = None
    reasoning: str = ""
    timestamp: Optional[str] = None


async def fetch_spy_volatility(session: aiohttp.ClientSession, api_key: str) -> Optional[float]:
    """
    Fallback volatility proxy: 20-day realized vol of SPY.
    When VIX data is unavailable, this approximates market stress.
    """
    end = datetime.utcnow().date()
    start = end - timedelta(days=90)

    url = (f"{POLYGON_BASE}/v2/aggs/ticker/SPY"
           f"/range/1/day/{start.isoformat()}/{end.isoformat()}")
    params = {"adjusted": "true", "sort": "asc", "limit": 90, "apiKey": api_key}

    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
    except Exception:
        return None

    results = data.get("results", [])
    if len(results) < 25:
        return None

    closes = [r["c"] for r in results]
    returns = np.diff(np.log(closes))
    vol_20d = float(np.std(returns[-20:]) * np.sqrt(252) * 100)  # annualized, as percent
    return round(vol_20d, 2)


async def fetch_vixy_level(session: aiohttp.ClientSession, api_key: str) -> Optional[float]:
    """
    VIXY ETF tracks short-term VIX futures. Not exactly VIX but correlates highly.
    Returns the current VIXY price — used as a proxy VIX signal.

    NOTE: Polygon's /v2/last/trade and /v1/last_quote are premium tier only.
    We use daily aggregates and take the most recent close.
    """
    end = datetime.utcnow().date()
    start = end - timedelta(days=5)

    url = (f"{POLYGON_BASE}/v2/aggs/ticker/VIXY"
           f"/range/1/day/{start.isoformat()}/{end.isoformat()}")
    params = {"adjusted": "true", "sort": "desc", "limit": 5, "apiKey": api_key}

    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
    except Exception:
        return None

    results = data.get("results", [])
    if not results:
        return None
    return float(results[0].get("c", 0))


def classify_regime(
    vix: Optional[float],
    spy_vol: Optional[float],
    breadth: Optional[float],
) -> RegimeState:
    """
    Combine VIX + SPY realized vol + breadth into a single regime label.

    Heuristic: use SPY realized vol as our primary stress signal since VIXY
    doesn't track pure VIX. SPY annualized vol maps roughly to VIX levels.
    """
    # Primary stress signal: SPY realized vol (approximates VIX in magnitude)
    stress = spy_vol if spy_vol is not None else 18.0  # default to "normal"

    # Classify based on stress thresholds (these track VIX bucket thresholds)
    if stress < 12:
        vol_regime = "calm"
    elif stress < 20:
        vol_regime = "normal"
    elif stress < 32:
        vol_regime = "elevated"
    else:
        vol_regime = "panic"

    # Breadth overlay: if breadth collapses, escalate regime one step
    if breadth is not None:
        if breadth < 0.30:
            # Serious structural weakness — escalate to panic
            vol_regime = "panic"
        elif breadth < 0.50 and vol_regime in ("calm", "normal"):
            # Bad breadth with normal vol = hidden weakness
            vol_regime = "elevated"

    multipliers = {
        "calm": 1.10,
        "normal": 1.00,
        "elevated": 0.80,
        "panic": 0.50,
    }

    reasoning_parts = []
    if vix is not None:
        reasoning_parts.append(f"VIXY={vix:.2f}")
    if spy_vol is not None:
        reasoning_parts.append(f"SPY vol 20d={spy_vol:.1f}%")
    if breadth is not None:
        reasoning_parts.append(f"breadth={breadth*100:.0f}% above 200MA")

    return RegimeState(
        regime=vol_regime,
        multiplier=multipliers[vol_regime],
        vix_level=vix,
        spy_vol_20d=spy_vol,
        breadth_pct_above_200ma=breadth,
        reasoning=" | ".join(reasoning_parts),
        timestamp=datetime.utcnow().isoformat(),
    )


def compute_breadth_from_scores(raw_scores: Dict[str, Dict]) -> Optional[float]:
    """
    Compute market breadth from the scanner's raw factor data.
    % of universe with pct_above_ma200 > 0.

    Called AFTER a scan completes; requires scanner output.
    """
    if not raw_scores:
        return None
    above = 0
    total = 0
    for ticker, data in raw_scores.items():
        pct = data.get("metrics", {}).get("pct_above_ma200")
        if pct is not None:
            total += 1
            if pct > 0:
                above += 1
    if total == 0:
        return None
    return round(above / total, 3)


class RegimeDetector:
    """
    Standalone regime detector. Run once before a scan to get the multiplier.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")

    async def detect(self, breadth_hint: Optional[float] = None) -> RegimeState:
        """
        Fetch VIX/vol data from Polygon, combine with breadth_hint if provided.
        breadth_hint comes from a recent scan's raw_scores.
        """
        if not self.api_key:
            return RegimeState(regime="normal", multiplier=1.0, reasoning="No API key")

        async with aiohttp.ClientSession() as session:
            vix, spy_vol = await asyncio.gather(
                fetch_vixy_level(session, self.api_key),
                fetch_spy_volatility(session, self.api_key),
                return_exceptions=False,
            )

        return classify_regime(vix, spy_vol, breadth_hint)


# Apply regime to scanner output
def apply_regime_to_rankings(
    rankings: Dict[str, List[Dict]],
    regime: RegimeState,
) -> Dict[str, List[Dict]]:
    """
    Multiply composite scores by regime multiplier.
    Re-rank after adjustment (order may change at boundaries but mostly preserves).
    Adds regime metadata to each entry.
    """
    out = {}
    mult = regime.multiplier
    for horizon, entries in rankings.items():
        adjusted = []
        for r in entries:
            base = r["composite_score"]
            adj = round(base * mult, 2)
            new_r = dict(r)
            new_r["composite_score_raw"] = base
            new_r["composite_score"] = adj
            new_r["regime_multiplier"] = mult
            adjusted.append(new_r)
        # Re-rank (order can shift slightly if ties resolve differently)
        adjusted.sort(key=lambda x: x["composite_score"], reverse=True)
        for i, r in enumerate(adjusted, 1):
            r["rank"] = i
        out[horizon] = adjusted
    return out


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════
async def _test():
    import sys
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set"); sys.exit(1)

    print("Detecting market regime...")
    detector = RegimeDetector(api_key=api_key)
    state = await detector.detect()

    print(f"\n{'='*60}")
    print(f"  MARKET REGIME STATE")
    print(f"{'='*60}")
    print(f"  Regime:         {state.regime.upper()}")
    print(f"  Multiplier:     {state.multiplier}x")
    print(f"  VIXY level:     {state.vix_level}")
    print(f"  SPY 20d vol:    {state.spy_vol_20d}%")
    print(f"  Breadth:        {state.breadth_pct_above_200ma}")
    print(f"  Reasoning:      {state.reasoning}")
    print(f"  Timestamp:      {state.timestamp}")

    print(f"\n  INTERPRETATION:")
    if state.regime == "calm":
        print("    Market is calm. Deploy confidently. Slight risk-on bias.")
    elif state.regime == "normal":
        print("    Normal market conditions. Standard position sizing.")
    elif state.regime == "elevated":
        print("    Elevated stress. Reduce position sizes. Focus on highest quality.")
    else:
        print("    PANIC REGIME. Pause new deployments. Consider defensive posture.")


if __name__ == "__main__":
    asyncio.run(_test())
