"""Nightly rebound scan — refreshes the discounted-quality shortlist.

Runs the rebound scan over the live universe against current fundamentals:
finds stocks down >=35% from their 3y high that pass the health + no-falling-
knife gates, scores and stages them, writes the artifact the /rebound/*
endpoints serve. The artifact includes each name's prior_high so the live API
can compute recovery-to-high progress on every request.

Runs at 02:30 ET, after the multibagger scan (02:00) which refreshes the EDGAR
bulk file this reuses.
"""
from __future__ import annotations
import asyncio
import os
from loguru import logger


class ReboundScanJob:
    def __init__(self, price_db=None, insider_db=None, out_path=None,
                 sample=None, workers=8):
        self.price_db = price_db or os.environ.get("PRICE_DB", "/app/data/price_store.db")
        self.insider_db = insider_db or os.environ.get(
            "INSIDER_DB", "/app/data/insider_store.db")
        self.out_path = out_path or os.environ.get(
            "REBOUND_ARTIFACT", "/app/data/rebound_artifact.json")
        self.sample = sample
        self.workers = workers

    async def run(self):
        logger.info("🔍 Nightly rebound scan starting…")
        try:
            from quantedge.fundamentals.rebound.scan import run_scan
            artifact = await asyncio.to_thread(
                run_scan, self.price_db, self.insider_db, None,
                self.sample, False, self.workers, None)
            await asyncio.to_thread(self._write, artifact)
            n = artifact.get("total_passed", 0)
            logger.info(f"✅ Rebound scan complete — {n} names, artifact at {self.out_path}")
        except Exception as e:
            logger.warning(f"Rebound scan failed: {e}")

    def _write(self, artifact):
        import json
        tmp = self.out_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(artifact, fh, default=str)
        os.replace(tmp, self.out_path)   # atomic — readers never see a partial file
