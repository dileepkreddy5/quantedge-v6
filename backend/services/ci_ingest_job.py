"""Nightly Company Intelligence ingest — 05:00 ET, after EDGAR's overnight
posting window. Append-only; failure is a loud error."""
from loguru import logger

class CIIngestJob:
    def __init__(self, pool): self.pool = pool
    async def run(self):
        logger.info("🔎 CI ingest starting…")
        try:
            from quantedge.intel.edgar_adapter import ingest
            stats = await ingest(self.pool)
            logger.info(f"✅ CI ingest complete: {stats}")
        except Exception as e:
            logger.error(f"❌ CI ingest FAILED: {e}")
