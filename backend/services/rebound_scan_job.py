"""Nightly rebound scan — 02:30 ET.

Rewritten: the original imported quantedge.fundamentals.rebound.scan with a
sqlite/insider-DB signature from an external research environment; that
module never existed in this repo, so the job failed on its import line
every night and logged one warning that container restarts erased. Now
pool-based against the v1 engine, and failure is a loud logger.error.
"""
import asyncio, json, os
from loguru import logger


class ReboundScanJob:
    def __init__(self, pool=None):
        self.pool = pool
        self.out_path = os.environ.get(
            "REBOUND_ARTIFACT", "/app/data/rebound_artifact.json")

    async def run(self):
        logger.info("🔍 Nightly rebound scan starting…")
        try:
            if self.pool is None:
                raise RuntimeError("ReboundScanJob needs a DB pool")
            from quantedge.fundamentals.rebound.scan import run_scan
            artifact = await run_scan(self.pool)
            tmp = self.out_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(artifact, fh, default=str)
            os.replace(tmp, self.out_path)
            logger.info(f"✅ Rebound scan complete — {artifact['n_passed_gates']} "
                        f"names, artifact at {self.out_path}")
        except Exception as e:
            logger.error(f"❌ Rebound scan FAILED (artifact stale): {e}")
