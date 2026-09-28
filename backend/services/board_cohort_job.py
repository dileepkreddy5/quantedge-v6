from loguru import logger
class BoardCohortJob:
    def __init__(self, pool): self.pool = pool
    async def run(self):
        try:
            from services.board_cohorts import snapshot, fill_outcomes
            logger.info(f"✅ Board cohorts: {await snapshot(self.pool)} · {await fill_outcomes(self.pool)}")
        except Exception as e:
            logger.error(f"❌ Board cohort job FAILED: {e}")
