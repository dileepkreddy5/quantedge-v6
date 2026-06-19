"""
QuantEdge v6.0 — Ascent Scan Job
=================================
The scheduled job that powers the Ascent Radar board:
  1. Build universe (name/sector/market_cap)
  2. Run UniverseScanner -> raw factor metrics
  3. Enrich with dist_from_52w_high
  4. Compute ascent score per ticker
  5. Rank top N and persist a timestamped snapshot
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, List

import aiohttp
from loguru import logger

from research.universe import UniverseBuilder
from research.universe_scanner import UniverseScanner
from research.factor_engine import fetch_price_history
from research.ascent import compute_ascent, rank_ascent, AscentScore
from services.ascent_store import AscentStore


ASCENT_UNIVERSE_SIZE = 500
ASCENT_TOP_N = 25


async def _enrich_52w_high(ticker: str, metrics: Dict, session: aiohttp.ClientSession,
                           api_key: str) -> None:
    try:
        df = await fetch_price_history(ticker, api_key=api_key, session=session)
        if df is None or len(df) < 60:
            return
        close = df["close"]
        high_52w = float(close.tail(252).max())
        cur = float(close.iloc[-1])
        if high_52w > 0:
            metrics["dist_from_52w_high"] = max(0.0, (high_52w - cur) / high_52w)
    except Exception:
        return


class AscentScanJob:
    def __init__(self, store: AscentStore, api_key: str):
        self.store = store
        self.api_key = api_key

    async def run(self, universe_size: int = ASCENT_UNIVERSE_SIZE,
                  top_n: int = ASCENT_TOP_N) -> Dict:
        t0 = time.time()
        logger.info(f"🛰️  Ascent scan starting (universe={universe_size})")

        builder = UniverseBuilder(api_key=self.api_key)
        entries = await builder.build(max_tickers=universe_size)
        meta = {e.ticker: e for e in entries}
        tickers = [e.ticker for e in entries]

        scanner = UniverseScanner(api_key=self.api_key)
        scan = await scanner.scan(tickers=tickers)
        raw = scan.get("raw_scores", {})

        scores: List[AscentScore] = []
        async with aiohttp.ClientSession() as session:
            for tk, payload in raw.items():
                metrics = dict(payload.get("metrics", {}))
                await _enrich_52w_high(tk, metrics, session, self.api_key)
                e = meta.get(tk)
                scores.append(compute_ascent(
                    ticker=tk,
                    name=getattr(e, "name", "") if e else "",
                    sector=getattr(e, "sector", "") if e else "",
                    market_cap=getattr(e, "market_cap", None) if e else None,
                    metrics=metrics,
                ))

        ranked = rank_ascent(scores, top_n=top_n)
        scan_time = datetime.now(timezone.utc)
        await self.store.save_snapshot(scan_time, ranked)

        dur = time.time() - t0
        logger.info(f"🛰️  Ascent scan complete: {len(ranked)} ranked, {dur:.1f}s")
        return {"scan_time": scan_time.isoformat(), "ranked": len(ranked),
                "duration_seconds": round(dur, 1)}
