"""
QuantEdge v6.0 — Peer Stats Scan Job
=====================================
Scans the full universe (like Ascent) but stores EVERY scored name with its
SIC + factor metrics, so peer-percentile comparison has a real population.
Runs once daily (peer rankings move slowly).
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
from services.peer_store import PeerStore
from services.ascent_scan_job import _fetch_details
from quantedge.fundamentals.edgar_bulk import company_facts_from_bulk
from quantedge.fundamentals.peer_fundamentals import compute_peer_fundamentals
from quantedge.fundamentals.universe_full import ticker_cik_map

PEER_UNIVERSE_SIZE = 3000


class PeerScanJob:
    def __init__(self, store: PeerStore, api_key: str):
        self.store = store
        self.api_key = api_key

    async def run(self, universe_size: int = PEER_UNIVERSE_SIZE) -> Dict:
        t0 = time.time()
        logger.info(f"🧬 Peer stats scan starting (universe={universe_size})")

        # Name, SIC and market cap already live in the universe table, refreshed from
        # Polygon's reference data. Reading them locally avoids one detail call per
        # ticker, which is what previously kept this capped at 500 names.
        meta, tickers = {}, []
        pool = getattr(self, "pool", None) or getattr(self.store, "pool", None)
        if pool is not None:
            try:
                async with pool.acquire() as conn:
                    urows = await conn.fetch(
                        "SELECT ticker, name, sic, sic_code, market_cap FROM universe "
                        "WHERE sic IS NOT NULL AND active "
                        "ORDER BY market_cap DESC NULLS LAST LIMIT $1", universe_size)
                from services.peer_store import is_shell
                for r in urows:
                    if is_shell(r["sic"]):
                        continue
                    tickers.append(r["ticker"])
                    meta[r["ticker"]] = {"name": r["name"], "sic": r["sic"],
                                         "sic_code": r["sic_code"],
                                         "market_cap": r["market_cap"]}
                logger.info(f"universe table supplied {len(tickers)} classified tickers")
            except Exception as e:
                logger.warning(f"universe table unavailable ({e}), falling back to builder")

        if not tickers:
            builder = UniverseBuilder(api_key=self.api_key)
            entries = await builder.build(max_tickers=universe_size)
            meta = {e.ticker: {"name": getattr(e, "name", ""), "sic": getattr(e, "sector", ""),
                               "market_cap": getattr(e, "market_cap", None)} for e in entries}
            tickers = [e.ticker for e in entries]

        # ticker->CIK map for reading fundamentals from the local bulk file (no API calls)
        try:
            cik_map = ticker_cik_map()
        except Exception as e:
            logger.warning(f"CIK map failed, fundamentals will be skipped: {e}")
            cik_map = {}

        scanner = UniverseScanner(api_key=self.api_key)
        scan = await scanner.scan(tickers=tickers)
        raw = scan.get("raw_scores", {})

        rows: List[Dict] = []
        sem = asyncio.Semaphore(12)
        async with aiohttp.ClientSession() as session:

            async def _process(tk, payload):
                async with sem:
                    metrics = dict(payload.get("metrics", {}))
                    e = meta.get(tk) or {}
                    # Only reach for details when the universe table lacks them.
                    details = e if e.get("sic") else await _fetch_details(tk, session, self.api_key)
                    # enrich with fundamental factors from the local bulk companyfacts file
                    _cik = cik_map.get(tk)
                    _mcap_for_fund = details.get("market_cap")
                    if _cik:
                        try:
                            _facts = company_facts_from_bulk(_cik)
                            if _facts:
                                _fund = compute_peer_fundamentals(_facts, market_cap=_mcap_for_fund)
                                metrics.update(_fund)  # merge fundamental factors alongside technical
                        except Exception as _fe:
                            logger.debug(f"fundamentals failed for {tk}: {_fe}")
                    _name = details.get("name") or e.get("name") or ""
                    _sic = details.get("sic") or details.get("sector") or e.get("sic") or ""
                    _mcap = details.get("market_cap") if details.get("market_cap") is not None else e.get("market_cap")
                    return {
                        "ticker": tk,
                        "name": _name,
                        "sic": _sic,
                        "sic_code": e.get("sic_code"),
                        "market_cap": _mcap,
                        "factors": metrics,
                    }

            results = await asyncio.gather(
                *[_process(tk, p) for tk, p in raw.items()],
                return_exceptions=True,
            )
            rows = [r for r in results if isinstance(r, dict)]

        scan_time = datetime.now(timezone.utc)
        await self.store.save_snapshot(scan_time, rows)
        dur = round(time.time() - t0, 1)
        logger.info(f"🧬 Peer stats scan complete: {len(rows)} stored, {dur}s")
        return {"scan_time": scan_time.isoformat(), "stored": len(rows), "duration_seconds": dur}
