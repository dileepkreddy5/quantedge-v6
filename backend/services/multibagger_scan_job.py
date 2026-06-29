"""Nightly multibagger scan — refreshes fundamentals, then scans live universe.

Sequence each night:
  1. Re-download EDGAR companyfacts.zip (fresh fundamentals — companies that
     filed since yesterday are now included).
  2. Run the bulk scan: pulls the LIVE company list + LIVE prices from Polygon
     (so market caps and tier membership are always current), reads fundamentals
     from the just-refreshed local zip, scores, writes the artifact.
  3. Enrich the surviving top-300 with live price ladder + volume.

Live data (caps, listings, prices) is always fetched fresh; only the
quarterly-filing fundamentals come from the nightly-refreshed local file.
"""
from __future__ import annotations
import asyncio
from loguru import logger


class MultibaggerScanJob:
    def __init__(self, top_universe=1000, display=100, rank_pool=4000):
        self.top_universe = top_universe
        self.display = display
        self.rank_pool = rank_pool

    async def run(self):
        logger.info("🔍 Nightly multibagger scan starting…")
        try:
            from quantedge.fundamentals.edgar_bulk import download_bulk
            from quantedge.fundamentals.bulk_scan import run_bulk_scan
            from quantedge.fundamentals.enrich_artifact import enrich
            # 1. fresh fundamentals
            await asyncio.to_thread(download_bulk, True)   # force refresh
            logger.info("✅ EDGAR bulk facts refreshed")
            # 2. scan live universe against fresh fundamentals
            await asyncio.to_thread(run_bulk_scan, self.top_universe, self.display, self.rank_pool, 12)
            logger.info("✅ Universe scanned, artifact written")
            # 3. enrich top names with live price/volume
            await asyncio.to_thread(enrich)
            logger.info("✅ Multibagger scan complete — top names enriched")
        except Exception as e:
            logger.error(f"Multibagger scan failed: {e}")
