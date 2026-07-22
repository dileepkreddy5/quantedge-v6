"""
QuantEdge v6.0 — Daily Bars Store
==================================
Every US-listed stock's daily bars, held locally.

The platform previously worked off a 486-name snapshot, which meant Exxon had no
peer group, 130 companies were unclassified, and correlations compared against a
fifth of the market. One Polygon grouped-daily call returns every ticker's bar for
a single session, so keeping the full universe current costs one request per day.

Backfill is per-ticker (Polygon has no bulk history endpoint); the nightly top-up
is a single call.
"""
from __future__ import annotations

import asyncio, os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import asyncpg, httpx
from loguru import logger

POLY = os.environ.get("POLYGON_API_KEY", "")

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS daily_bars (
    ticker      TEXT        NOT NULL,
    d           DATE        NOT NULL,
    o           DOUBLE PRECISION,
    h           DOUBLE PRECISION,
    l           DOUBLE PRECISION,
    c           DOUBLE PRECISION NOT NULL,
    v           BIGINT,
    PRIMARY KEY (ticker, d)
);
CREATE INDEX IF NOT EXISTS idx_bars_date   ON daily_bars (d);
CREATE INDEX IF NOT EXISTS idx_bars_ticker ON daily_bars (ticker, d DESC);

CREATE TABLE IF NOT EXISTS universe (
    ticker       TEXT PRIMARY KEY,
    name         TEXT,
    sic          TEXT,
    sic_code     TEXT,
    cik          TEXT,
    exchange     TEXT,
    type         TEXT,
    active       BOOLEAN DEFAULT TRUE,
    market_cap   DOUBLE PRECISION,
    shares_out   DOUBLE PRECISION,
    last_seen    DATE,
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_universe_sic ON universe (sic);
"""


class BarsStore:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def ensure_tables(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(CREATE_SQL)
        logger.info("daily_bars + universe tables verified")

    # ── Universe ────────────────────────────────────────────────
    async def refresh_universe(self) -> int:
        """Pull every active US common stock from Polygon's reference endpoint."""
        rows, url = [], (
            "https://api.polygon.io/v3/reference/tickers?market=stocks&active=true"
            f"&type=CS&limit=1000&apiKey={POLY}")
        async with httpx.AsyncClient(timeout=40) as client:
            while url:
                r = await client.get(url)
                if r.status_code != 200:
                    break
                j = r.json() or {}
                for t in j.get("results", []):
                    rows.append((t.get("ticker"), t.get("name"), t.get("sic_description"),
                                 str(t.get("cik") or "") or None, t.get("primary_exchange"),
                                 t.get("type"), True, date.today()))
                nxt = j.get("next_url")
                url = f"{nxt}&apiKey={POLY}" if nxt else None
                await asyncio.sleep(0.05)
        if not rows:
            return 0
        async with self.pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO universe (ticker,name,sic,cik,exchange,type,active,last_seen)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                ON CONFLICT (ticker) DO UPDATE SET
                  name=EXCLUDED.name, sic=EXCLUDED.sic, cik=EXCLUDED.cik,
                  exchange=EXCLUDED.exchange, active=TRUE,
                  last_seen=EXCLUDED.last_seen, updated_at=NOW()
            """, rows)
        logger.info(f"universe refreshed: {len(rows)} tickers")
        return len(rows)

    async def enrich_universe(self, concurrency: int = 10, limit: int = 0) -> Dict:
        """SIC and share count come from the per-ticker detail endpoint, not the
        list endpoint. Without SIC there is no industry classification, which is
        what leaves companies in an 'Other' bucket with no comparable peers."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT ticker FROM universe WHERE sic IS NULL AND active ORDER BY ticker"
                + (f" LIMIT {int(limit)}" if limit else ""))
        tickers = [r["ticker"] for r in rows]
        if not tickers:
            return {"pending": 0}
        sem = asyncio.Semaphore(concurrency)
        got, miss = [], 0

        async with httpx.AsyncClient(timeout=25) as client:
            async def one(tk: str):
                nonlocal miss
                async with sem:
                    try:
                        r = await client.get(
                            f"https://api.polygon.io/v3/reference/tickers/{tk}?apiKey={POLY}")
                        if r.status_code != 200:
                            miss += 1; return
                        d = (r.json() or {}).get("results") or {}
                        sic = d.get("sic_description")
                        if not sic:
                            miss += 1; return
                        got.append((sic, str(d.get("sic_code") or "") or None,
                                    d.get("weighted_shares_outstanding"),
                                    d.get("market_cap"), tk))
                    except Exception:
                        miss += 1

            for i in range(0, len(tickers), 250):
                await asyncio.gather(*[one(t) for t in tickers[i:i + 250]])
                if got:
                    async with self.pool.acquire() as conn:
                        await conn.executemany(
                            "UPDATE universe SET sic=$1, sic_code=$2, shares_out=$3,"
                            " market_cap=$4, updated_at=NOW() WHERE ticker=$5", got)
                    got.clear()
                logger.info(f"enriched {min(i+250, len(tickers))}/{len(tickers)} · {miss} without sic")
        return {"processed": len(tickers), "no_sic": miss}

    async def enrich_from_edgar(self) -> Dict:
        """Polygon has no SIC for roughly a fifth of the universe, XOM included.
        EDGAR's bulk company file is already on disk for the Multibagger scan and
        carries SIC for every filer, so it fills the gap without another API."""
        # SIC comes from the SEC submissions endpoint rather than companyfacts.zip,
        # which carries financial facts but not classification.
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT ticker, cik FROM universe WHERE sic_code IS NULL AND cik IS NOT NULL AND active")
        if not rows:
            return {"pending": 0}

        # submissions.zip carries SIC; companyfacts does not. Fall back to the
        # per-company submissions endpoint, which is free and unthrottled at this rate.
        import httpx as _hx
        got, miss = [], 0
        headers = {"User-Agent": "QuantEdge research contact@quantedge.local"}
        async with _hx.AsyncClient(timeout=20, headers=headers) as client:
            # SEC permits ~10 req/s. A semaphore alone does not bound the rate, so
            # concurrency is kept low and each worker sleeps a full interval.
            sem = asyncio.Semaphore(3)

            async def one(tk: str, cik: str):
                nonlocal miss
                async with sem:
                    try:
                        c10 = str(cik).zfill(10)
                        r = await client.get(f"https://data.sec.gov/submissions/CIK{c10}.json")
                        if r.status_code == 429:
                            await asyncio.sleep(3)
                            r = await client.get(f"https://data.sec.gov/submissions/CIK{c10}.json")
                        if r.status_code != 200:
                            miss += 1; return
                        j = r.json() or {}
                        sic, desc = j.get("sic"), j.get("sicDescription")
                        if not sic:
                            miss += 1; return
                        got.append((desc or "", str(sic), tk))
                    except Exception:
                        miss += 1
                    await asyncio.sleep(0.35)   # 3 workers x 0.35s ≈ 8.5 req/s

            batch = [(r["ticker"], r["cik"]) for r in rows]
            for i in range(0, len(batch), 150):
                await asyncio.gather(*[one(t, c) for t, c in batch[i:i + 150]])
                if got:
                    async with self.pool.acquire() as conn:
                        await conn.executemany(
                            "UPDATE universe SET sic=$1, sic_code=$2, updated_at=NOW() WHERE ticker=$3", got)
                    got.clear()
                logger.info(f"edgar sic {min(i+150, len(batch))}/{len(batch)} · {miss} missing")
        return {"processed": len(batch), "missing": miss}

    # ── Bars ────────────────────────────────────────────────────
    async def upsert_bars(self, ticker: str, bars: List[Dict]) -> int:
        if not bars:
            return 0
        recs = [(ticker, datetime.utcfromtimestamp(b["t"] / 1000).date(),
                 b.get("o"), b.get("h"), b.get("l"), b["c"], int(b.get("v") or 0))
                for b in bars if b.get("c")]
        async with self.pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO daily_bars (ticker,d,o,h,l,c,v) VALUES ($1,$2,$3,$4,$5,$6,$7)
                ON CONFLICT (ticker,d) DO UPDATE SET c=EXCLUDED.c, v=EXCLUDED.v
            """, recs)
        return len(recs)

    async def backfill(self, tickers: List[str], years: int = 2, concurrency: int = 8) -> Dict:
        """One-off historical load. Polygon offers no bulk history, so this is
        per-ticker and slow; it only needs to run once."""
        end, start = date.today(), date.today() - timedelta(days=int(years * 365.25))
        s, e = start.isoformat(), end.isoformat()
        sem = asyncio.Semaphore(concurrency)
        done = {"ok": 0, "empty": 0, "rows": 0}

        async with httpx.AsyncClient(timeout=30) as client:
            async def one(tk: str):
                async with sem:
                    try:
                        u = (f"https://api.polygon.io/v2/aggs/ticker/{tk}/range/1/day/{s}/{e}"
                             f"?adjusted=true&sort=asc&limit=50000&apiKey={POLY}")
                        r = await client.get(u)
                        if r.status_code != 200:
                            done["empty"] += 1; return
                        res = (r.json() or {}).get("results") or []
                        if not res:
                            done["empty"] += 1; return
                        n = await self.upsert_bars(tk, res)
                        done["ok"] += 1; done["rows"] += n
                    except Exception:
                        done["empty"] += 1

            for i in range(0, len(tickers), 200):
                await asyncio.gather(*[one(t) for t in tickers[i:i + 200]])
                logger.info(f"backfill {min(i+200, len(tickers))}/{len(tickers)} · "
                            f"{done['ok']} ok · {done['rows']} rows")
        return done

    async def sync_day(self, d: Optional[date] = None) -> int:
        """Nightly top-up: one call returns every ticker's bar for a session."""
        target = d or (date.today() - timedelta(days=1))
        for back in range(0, 5):
            day = target - timedelta(days=back)
            if day.weekday() >= 5:
                continue
            url = (f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/"
                   f"{day.isoformat()}?adjusted=true&apiKey={POLY}")
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.get(url)
            if r.status_code != 200:
                continue
            res = (r.json() or {}).get("results") or []
            if not res:
                continue
            recs = [(b["T"], day, b.get("o"), b.get("h"), b.get("l"), b["c"], int(b.get("v") or 0))
                    for b in res if b.get("c") and b.get("T")]
            async with self.pool.acquire() as conn:
                await conn.executemany("""
                    INSERT INTO daily_bars (ticker,d,o,h,l,c,v) VALUES ($1,$2,$3,$4,$5,$6,$7)
                    ON CONFLICT (ticker,d) DO UPDATE SET c=EXCLUDED.c, v=EXCLUDED.v
                """, recs)
            logger.info(f"synced {len(recs)} bars for {day}")
            return len(recs)
        return 0

    async def stats(self) -> Dict:
        async with self.pool.acquire() as conn:
            return {
                "tickers": await conn.fetchval("SELECT count(DISTINCT ticker) FROM daily_bars"),
                "rows": await conn.fetchval("SELECT count(*) FROM daily_bars"),
                "first": str(await conn.fetchval("SELECT min(d) FROM daily_bars") or ""),
                "last": str(await conn.fetchval("SELECT max(d) FROM daily_bars") or ""),
                "universe": await conn.fetchval("SELECT count(*) FROM universe"),
            }
