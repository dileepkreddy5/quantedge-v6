"""Manual pattern-library build runner (the nightly PatternLibJob does the
same via the scheduler). Lives in the repo so it survives image rebuilds —
its /tmp-copied predecessor vanished on every `docker compose build`."""
import asyncio, asyncpg, os

async def main():
    pool = await asyncpg.create_pool(
        'postgresql://quantedge:' + os.environ.get('POSTGRES_PASSWORD', '')
        + '@postgres:5432/quantedge', min_size=2, max_size=4)
    from quantedge.patterns.library import build_library
    stats = await build_library(pool, "/app/models/patterns")
    print("DONE:", stats, flush=True)
    await pool.close()

asyncio.run(main())
