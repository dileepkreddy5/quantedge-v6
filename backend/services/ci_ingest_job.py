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
            logger.info(f"✅ CI EDGAR ingest complete: {stats}")
        except Exception as e:
            logger.error(f"❌ CI EDGAR ingest FAILED: {e}")
        try:
            # Capital allocation reads the bulk zip the 02:00 ET multibagger job
            # refreshed; runs after it by schedule (05:00 ET).
            from quantedge.intel.xbrl_capital_adapter import ingest_capital
            stats = await ingest_capital(self.pool)
            logger.info(f"✅ CI capital-allocation ingest complete: {stats}")
        except Exception as e:
            logger.error(f"❌ CI capital-allocation ingest FAILED: {e}")
        try:
            from quantedge.intel.f13_adapter import ingest_13f
            stats = await ingest_13f(self.pool)
            logger.info(f"✅ CI 13F ingest complete: {stats}")
        except Exception as e:
            logger.error(f"❌ CI 13F ingest FAILED: {e}")
        try:
            from quantedge.intel.attention_adapter import ingest_attention
            stats = await ingest_attention(self.pool)
            logger.info(f"✅ CI attention ingest complete: {stats}")
        except Exception as e:
            logger.error(f"❌ CI attention ingest FAILED: {e}")
        try:
            from quantedge.intel.officer_change import classify_502
            logger.info(f"✅ CI 5.02 officer changes classified: {await classify_502(self.pool, days=10)}")
        except Exception as e:
            logger.error(f"❌ CI 5.02 classification FAILED: {e}")
