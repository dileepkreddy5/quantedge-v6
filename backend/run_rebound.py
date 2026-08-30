"""Manual rebound scan runner — in the repo so it survives rebuilds."""
import asyncio, asyncpg, os
async def main():
    pool = await asyncpg.create_pool('postgresql://quantedge:'
        + os.environ.get('POSTGRES_PASSWORD','') + '@postgres:5432/quantedge',
        min_size=2, max_size=4)
    from services.rebound_scan_job import ReboundScanJob
    await ReboundScanJob(pool).run()
    await pool.close()
asyncio.run(main())
