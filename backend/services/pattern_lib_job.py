"""Nightly pattern-library rebuild.

Rebuilds the analog window libraries after the bars sync lands, so the
query trajectory and the library share the same last session. Without
this the library freezes at build date — the artifact-staleness disease
(multibagger, relationships) in a new organ.
"""
from loguru import logger


class PatternLibJob:
    def __init__(self, pool):
        self.pool = pool

    async def run(self):
        logger.info("🧬 Pattern library rebuild starting…")
        try:
            from quantedge.patterns.library import build_library
            stats = await build_library(self.pool, "/app/models/patterns")
            logger.info(f"✅ Pattern library rebuilt: {stats}")
            # Invalidate the in-process cache so queries load the new files.
            try:
                from routers.patterns_router import _LIB
                _LIB._cache.clear()
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Pattern library rebuild failed: {e}")
