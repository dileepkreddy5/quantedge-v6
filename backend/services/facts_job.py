from loguru import logger
class FactsJob:
    """Nightly facts sheet: price facts then business facts, for every company."""
    def __init__(self, pool): self.pool = pool
    async def run(self):
        try:
            from quantedge.facts.price_facts import build_price_facts
            from quantedge.facts.business_facts import build_business_facts
            p = await build_price_facts(self.pool); b = await build_business_facts(self.pool)
            logger.info(f"✅ Facts sheet: {p['companies']} companies priced, {b['with_quarters']} with quarters")
        except Exception as e:
            logger.error(f"❌ Facts sheet FAILED: {e}")
