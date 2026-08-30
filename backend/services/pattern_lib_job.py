"""Nightly pattern rebuild — 18:45 ET, after bars sync.

Rebuilds the analog window libraries, then the formations scan, then the
conditions/evolution scan, so all three Pattern Lab artifacts share the
market's last session. Each stage is independently fenced: a failure logs
and moves on rather than leaving the later artifacts stale silently.
Total cost ~25 min of CPU on a closed market.
"""
from loguru import logger
from core.artifact_paths import artifact_write_path


class PatternLibJob:
    def __init__(self, pool):
        self.pool = pool

    async def run(self):
        logger.info("🧬 Pattern nightly rebuild starting…")
        try:
            from quantedge.patterns.library import build_library
            stats = await build_library(self.pool, "/app/models/patterns")
            logger.info(f"✅ Pattern library rebuilt: { {k: v['windows'] for k, v in stats.items()} }")
            try:
                from routers.patterns_router import _LIB
                _LIB._cache.clear()
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Pattern library rebuild failed: {e}")
        try:
            from quantedge.patterns.formations import scan_formations
            counts = await scan_formations(self.pool, str(artifact_write_path("formations_scan.json")))
            logger.info(f"✅ Formations rescanned: {sum(counts.values())} occurrences")
        except Exception as e:
            logger.error(f"Formations scan failed: {e}")
        try:
            from quantedge.patterns.conditions import scan_conditions
            res = await scan_conditions(self.pool, str(artifact_write_path("conditions_scan.json")))
            logger.info(f"✅ Conditions rescanned: {res}")
        except Exception as e:
            logger.error(f"Conditions scan failed: {e}")
        logger.info("🧬 Pattern nightly rebuild complete")
