import asyncio, asyncpg, os, sys
async def main():
    pool = await asyncpg.create_pool('postgresql://quantedge:'+os.environ.get('POSTGRES_PASSWORD','')+'@postgres:5432/quantedge', min_size=2, max_size=4)
    from quantedge.intel.xbrl_capital_adapter import ingest_capital
    print("DONE:", await ingest_capital(pool, limit=int(sys.argv[1]) if len(sys.argv) > 1 else 700), flush=True)
    await pool.close()
asyncio.run(main())
