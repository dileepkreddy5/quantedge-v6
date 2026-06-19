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

import asyncio
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
MIN_MARKET_CAP = 2_000_000_000  # mid-cap floor: drop small/micro/nano (too volatile)
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


# Lightweight per-ticker details: name, sector, market_cap (one cheap call).
_POLYGON_BASE = "https://api.polygon.io"

async def _fetch_details(ticker: str, session: aiohttp.ClientSession, api_key: str) -> Dict:
    out = {"name": "", "sector": "", "market_cap": None}
    try:
        url = f"{_POLYGON_BASE}/v3/reference/tickers/{ticker.upper()}?apiKey={api_key}"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return out
            data = await resp.json()
        res = data.get("results", {})
        out["name"] = res.get("name", "") or ""
        out["sector"] = res.get("sic_description", "") or ""
        mc = res.get("market_cap")
        if not mc:
            shares = res.get("weighted_shares_outstanding") or res.get("share_class_shares_outstanding")
            # market_cap may be absent on Starter; leave None if no shares
            mc = None
        out["market_cap"] = mc
    except Exception:
        pass
    return out


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
        sem = asyncio.Semaphore(12)  # bound concurrency for the detail+price calls
        async with aiohttp.ClientSession() as session:

            async def _process(tk, payload):
                async with sem:
                    metrics = dict(payload.get("metrics", {}))
                    await _enrich_52w_high(tk, metrics, session, self.api_key)
                    e = meta.get(tk)
                    details = await _fetch_details(tk, session, self.api_key)
                    _name = details["name"] or (getattr(e, "name", "") if e else "")
                    _sector = details["sector"] or (getattr(e, "sector", "") if e else "")
                    _mcap = details["market_cap"] if details["market_cap"] is not None else (getattr(e, "market_cap", None) if e else None)
                    return compute_ascent(
                        ticker=tk, name=_name, sector=_sector,
                        market_cap=_mcap, metrics=metrics,
                    )

            tasks = [_process(tk, payload) for tk, payload in raw.items()]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            scores = [r for r in results if isinstance(r, AscentScore)]

        # Mid-cap floor: keep mid / large / mega only. Names with unknown cap are
        # dropped here too (can't confirm they clear the floor) — keeps the board
        # focused on the tradeable, climbable universe.
        before = len(scores)
        scores = [s for s in scores if s.market_cap is not None and s.market_cap >= MIN_MARKET_CAP]
        logger.info(f"Mid-cap floor: {len(scores)}/{before} tickers >= ${MIN_MARKET_CAP/1e9:.0f}B")

        ranked = rank_ascent(scores, top_n=top_n)
        scan_time = datetime.now(timezone.utc)
        await self.store.save_snapshot(scan_time, ranked)

        dur = time.time() - t0
        logger.info(f"🛰️  Ascent scan complete: {len(ranked)} ranked, {dur:.1f}s")
        return {"scan_time": scan_time.isoformat(), "ranked": len(ranked),
                "duration_seconds": round(dur, 1)}
