"""
QuantEdge v6.0 — Universe Scanner
==================================
Runs the FactorEngine across the full universe concurrently,
applies composite scoring for three horizons, and ranks.

Output per scan:
  - ranked list for short_term  (1-3 month horizon)
  - ranked list for medium_term (6-12 month horizon)
  - ranked list for long_term   (2-5 year horizon)

Each composite is a weighted sum of the 4 factor scores. Weights differ
per horizon based on academic evidence about which factors predict at
which time scales.

HORIZON WEIGHTS (defaults; tunable):
  short_term:  momentum 40%  accumulation 25%  trend 20%  quality 15%
  medium_term: quality 30%   momentum 25%      value 25%  trend 20%
  long_term:   quality 50%   value 25%         trend 15%  momentum 10%

Note: "value" is embedded in Quality (we use sector-relative valuation inside
quality_engine.py). No separate value factor.
"""

import os
import time
import asyncio
import aiohttp
from typing import List, Optional, Dict
from dataclasses import dataclass, asdict
from loguru import logger

from research.universe import UniverseBuilder, UniverseEntry
from research.factor_engine import FactorEngine, FactorScores


# Horizon weights — mapped to the 4 factors we actually compute
HORIZON_WEIGHTS: Dict[str, Dict[str, float]] = {
    "short_term": {
        "momentum":     0.40,
        "accumulation": 0.30,
        "trend":        0.20,
        "quality":      0.10,
    },
    "medium_term": {
        "quality":      0.35,
        "momentum":     0.25,
        "trend":        0.20,
        "accumulation": 0.20,
    },
    "long_term": {
        "quality":      0.55,
        "trend":        0.20,
        "momentum":     0.15,
        "accumulation": 0.10,
    },
}


@dataclass
class RankedTicker:
    """A ticker ranked within a horizon."""
    ticker: str
    composite_score: float
    rank: int
    quality: float
    momentum: float
    accumulation: float
    trend: float
    metrics: Dict[str, float]


def compute_composite(fs: FactorScores, horizon: str) -> float:
    """Weighted sum of factor scores for the given horizon."""
    w = HORIZON_WEIGHTS.get(horizon, HORIZON_WEIGHTS["medium_term"])
    score = (
        w["quality"]      * fs.quality +
        w["momentum"]     * fs.momentum +
        w["accumulation"] * fs.accumulation +
        w["trend"]        * fs.trend
    )
    return round(score, 2)


class UniverseScanner:
    """
    Runs factor engine across the universe with bounded concurrency.
    Produces ranked lists for all three horizons.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        concurrency: int = 8,
        skip_quality: bool = False,
    ):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        self.concurrency = concurrency
        self.skip_quality = skip_quality
        self.factor_engine = FactorEngine(api_key=self.api_key)
        self.universe_builder = UniverseBuilder(api_key=self.api_key)

    async def _score_ticker(
        self,
        ticker: str,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
    ) -> Optional[FactorScores]:
        async with semaphore:
            try:
                return await self.factor_engine.score(
                    ticker, session=session, skip_quality=self.skip_quality
                )
            except Exception as e:
                logger.debug(f"Ticker {ticker} failed: {e}")
                return None

    async def scan(
        self,
        max_tickers: Optional[int] = None,
        tickers: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run full scan.

        If `tickers` is provided, scan only those (useful for tests).
        Otherwise build the universe and scan all of it, optionally capped
        at `max_tickers`.

        Returns:
          {
            "scan_timestamp": unix ts,
            "duration_seconds": float,
            "universe_size": int,
            "tickers_scored": int,
            "tickers_failed": int,
            "rankings": {
              "short_term":  [RankedTicker, ...],
              "medium_term": [...],
              "long_term":   [...],
            },
            "raw_scores": {ticker: FactorScores.asdict, ...},
          }
        """
        t0 = time.time()

        if tickers is None:
            universe = await self.universe_builder.build()
            if max_tickers:
                universe = universe[:max_tickers]
            tickers = [u.ticker for u in universe]

        logger.info(f"Scanning {len(tickers)} tickers with concurrency={self.concurrency}")

        sem = asyncio.Semaphore(self.concurrency)
        connector = aiohttp.TCPConnector(limit=50)
        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [self._score_ticker(t, session, sem) for t in tickers]
            results = await asyncio.gather(*tasks, return_exceptions=False)

        scored: List[FactorScores] = []
        failed = 0
        for r in results:
            if r is None or r.data_quality in ("insufficient", "error"):
                failed += 1
            else:
                scored.append(r)

        # Compute composites and rank for each horizon
        rankings = {}
        for horizon in ("short_term", "medium_term", "long_term"):
            ranked = []
            for fs in scored:
                composite = compute_composite(fs, horizon)
                ranked.append({
                    "ticker": fs.ticker,
                    "composite_score": composite,
                    "quality": fs.quality,
                    "momentum": fs.momentum,
                    "accumulation": fs.accumulation,
                    "trend": fs.trend,
                    "metrics": fs.metrics,
                })
            ranked.sort(key=lambda x: x["composite_score"], reverse=True)
            for i, r in enumerate(ranked, 1):
                r["rank"] = i
            rankings[horizon] = ranked

        duration = time.time() - t0
        logger.info(
            f"Scan complete: {len(scored)} scored, {failed} failed, "
            f"{duration:.1f}s"
        )

        return {
            "scan_timestamp": time.time(),
            "duration_seconds": round(duration, 1),
            "universe_size": len(tickers),
            "tickers_scored": len(scored),
            "tickers_failed": failed,
            "rankings": rankings,
            "raw_scores": {fs.ticker: {
                "quality": fs.quality,
                "momentum": fs.momentum,
                "accumulation": fs.accumulation,
                "trend": fs.trend,
                "metrics": fs.metrics,
            } for fs in scored},
        }


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════
async def _test_small():
    """Scan a small basket — validates scanner logic without full universe."""
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set"); return

    test_basket = [
        "AAPL", "MSFT", "GOOGL", "NVDA", "META", "TSLA", "AVGO", "AMD",
        "JPM", "V", "MA", "WMT", "KO", "PEP", "PG", "JNJ",
        "XOM", "CVX", "CAT", "DE", "BA", "LMT", "F", "GM",
        "NFLX", "DIS", "ORCL", "CRM", "ADBE", "INTC", "IBM", "CSCO",
    ]

    # Skip quality for speed test (quality fetch dominates runtime)
    print(f"Quick scan of {len(test_basket)} tickers (price-only, no quality)...")
    scanner = UniverseScanner(api_key=api_key, concurrency=8, skip_quality=True)
    result = await scanner.scan(tickers=test_basket)

    print(f"\n{'='*80}")
    print(f"  SCAN SUMMARY")
    print(f"{'='*80}")
    print(f"  Duration:       {result['duration_seconds']}s")
    print(f"  Universe:       {result['universe_size']}")
    print(f"  Scored:         {result['tickers_scored']}")
    print(f"  Failed:         {result['tickers_failed']}")

    for horizon in ("short_term", "medium_term", "long_term"):
        print(f"\n{'─'*80}")
        print(f"  TOP 10 — {horizon.upper().replace('_', ' ')}")
        print(f"{'─'*80}")
        print(f"  {'Rank':<5}{'Ticker':<8}{'Score':>7} {'Qual':>6}{'Mom':>6}{'Acc':>6}{'Trnd':>6}")
        print(f"  {'-'*5}{'-'*8}{'-'*7} {'-'*6}{'-'*6}{'-'*6}{'-'*6}")
        for r in result["rankings"][horizon][:10]:
            print(f"  {r['rank']:<5}{r['ticker']:<8}{r['composite_score']:>7.1f} "
                  f"{r['quality']:>6.1f}{r['momentum']:>6.1f}"
                  f"{r['accumulation']:>6.1f}{r['trend']:>6.1f}")


async def _test_full(max_tickers: int = 50, skip_quality: bool = True):
    """Scan the first N tickers of the real universe."""
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set"); return

    print(f"Full-universe scan (max {max_tickers} tickers, skip_quality={skip_quality})...")
    scanner = UniverseScanner(api_key=api_key, concurrency=8, skip_quality=skip_quality)
    result = await scanner.scan(max_tickers=max_tickers)

    print(f"\n{'='*80}")
    print(f"  UNIVERSE SCAN — {max_tickers} tickers")
    print(f"{'='*80}")
    print(f"  Duration:       {result['duration_seconds']}s")
    print(f"  Scored:         {result['tickers_scored']}")
    print(f"  Failed:         {result['tickers_failed']}")

    for horizon in ("short_term", "medium_term", "long_term"):
        print(f"\n{'─'*80}")
        print(f"  TOP 20 — {horizon.upper().replace('_', ' ')}")
        print(f"{'─'*80}")
        print(f"  {'Rank':<5}{'Ticker':<8}{'Score':>7} {'Qual':>6}{'Mom':>6}{'Acc':>6}{'Trnd':>6}")
        for r in result["rankings"][horizon][:20]:
            print(f"  {r['rank']:<5}{r['ticker']:<8}{r['composite_score']:>7.1f} "
                  f"{r['quality']:>6.1f}{r['momentum']:>6.1f}"
                  f"{r['accumulation']:>6.1f}{r['trend']:>6.1f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "small":
            asyncio.run(_test_small())
        elif sys.argv[1] == "full":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
            asyncio.run(_test_full(max_tickers=n))
        else:
            asyncio.run(_test_small())
    else:
        asyncio.run(_test_small())
