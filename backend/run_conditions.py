"""Manual conditions scan runner — in the repo so it survives image rebuilds."""
import asyncio, asyncpg, os
async def main():
    pool = await asyncpg.create_pool('postgresql://quantedge:'
        + os.environ.get('POSTGRES_PASSWORD','') + '@postgres:5432/quantedge',
        min_size=2, max_size=4)
    from quantedge.patterns.conditions import scan_conditions
    from core.artifact_paths import artifact_write_path
    print("DONE:", await scan_conditions(pool, str(artifact_write_path("conditions_scan.json"))), flush=True)
    await pool.close()
asyncio.run(main())
