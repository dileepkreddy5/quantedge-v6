"""
QuantEdge v6.0 — Ascent Radar Store
====================================
Persists every Ascent scan as a timestamped snapshot in PostgreSQL so the
board has MEMORY: rank/score now vs 3 days, 1 week, 1 month ago + first-seen.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import asyncpg
from loguru import logger


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ascent_snapshots (
    id            BIGSERIAL PRIMARY KEY,
    scan_time     TIMESTAMPTZ NOT NULL,
    ticker        TEXT        NOT NULL,
    name          TEXT,
    sector        TEXT,
    rank          INTEGER     NOT NULL,
    ascent_score  REAL        NOT NULL,
    strength_score REAL,
    volume_score  REAL,
    tier_score    REAL,
    high_score    REAL,
    tier          TEXT,
    market_cap    DOUBLE PRECISION,
    flags         JSONB,
    metrics       JSONB
);
CREATE INDEX IF NOT EXISTS idx_ascent_scan_time ON ascent_snapshots (scan_time DESC);
CREATE INDEX IF NOT EXISTS idx_ascent_ticker    ON ascent_snapshots (ticker, scan_time DESC);

CREATE TABLE IF NOT EXISTS ascent_first_seen (
    ticker      TEXT PRIMARY KEY,
    first_seen  TIMESTAMPTZ NOT NULL
);
"""


class AscentStore:
    def __init__(self, db_pool: asyncpg.Pool):
        self.pool = db_pool

    async def ensure_tables(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(CREATE_SQL)
        logger.info("✅ Ascent tables verified/created")

    async def save_snapshot(self, scan_time: datetime, ranked: List[Dict]) -> int:
        if not ranked:
            return 0
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for r in ranked:
                    await conn.execute(
                        """
                        INSERT INTO ascent_snapshots
                        (scan_time, ticker, name, sector, rank, ascent_score,
                         strength_score, volume_score, tier_score, high_score,
                         tier, market_cap, flags, metrics)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                        """,
                        scan_time, r["ticker"], r.get("name", ""), r.get("sector", ""),
                        r["rank"], r["ascent_score"],
                        r.get("strength_score"), r.get("volume_score"),
                        r.get("tier_score"), r.get("high_score"),
                        r.get("tier"), r.get("market_cap"),
                        json.dumps(r.get("flags", [])),
                        json.dumps(r.get("metrics", {})),
                    )
                    await conn.execute(
                        """
                        INSERT INTO ascent_first_seen (ticker, first_seen)
                        VALUES ($1, $2)
                        ON CONFLICT (ticker) DO NOTHING
                        """,
                        r["ticker"], scan_time,
                    )
        logger.info(f"Ascent snapshot saved: {len(ranked)} tickers @ {scan_time.isoformat()}")
        return len(ranked)

    async def _rank_at(self, conn, ticker: str, target: datetime) -> Optional[Dict]:
        row = await conn.fetchrow(
            """
            SELECT rank, ascent_score, scan_time
            FROM ascent_snapshots
            WHERE ticker = $1 AND scan_time <= $2
            ORDER BY scan_time DESC
            LIMIT 1
            """,
            ticker, target,
        )
        if row:
            return {"rank": row["rank"], "ascent_score": float(row["ascent_score"])}
        return None

    async def get_latest_board(self, top_n: int = 25) -> Dict:
        async with self.pool.acquire() as conn:
            latest_time = await conn.fetchval("SELECT MAX(scan_time) FROM ascent_snapshots")
            if latest_time is None:
                return {"scan_time": None, "rows": [], "history_available": False}

            rows = await conn.fetch(
                """
                SELECT ticker, name, sector, rank, ascent_score,
                       strength_score, volume_score, tier_score, high_score,
                       tier, market_cap, flags, metrics
                FROM ascent_snapshots
                WHERE scan_time = $1
                ORDER BY rank ASC
                LIMIT $2
                """,
                latest_time, top_n,
            )

            now = latest_time
            t_3d = now - timedelta(days=3)
            t_1w = now - timedelta(days=7)
            t_1m = now - timedelta(days=30)

            oldest = await conn.fetchval("SELECT MIN(scan_time) FROM ascent_snapshots")
            history_available = oldest is not None and (now - oldest) > timedelta(days=2)

            out_rows = []
            for r in rows:
                tk = r["ticker"]
                d3 = await self._rank_at(conn, tk, t_3d)
                w1 = await self._rank_at(conn, tk, t_1w)
                m1 = await self._rank_at(conn, tk, t_1m)
                first_seen = await conn.fetchval(
                    "SELECT first_seen FROM ascent_first_seen WHERE ticker = $1", tk
                )
                is_new = first_seen is not None and first_seen >= latest_time

                def delta(prev):
                    if not prev:
                        return None
                    return {
                        "rank_change": prev["rank"] - r["rank"],
                        "score_change": round(float(r["ascent_score"]) - prev["ascent_score"], 1),
                    }

                out_rows.append({
                    "rank": r["rank"], "ticker": tk, "name": r["name"], "sector": r["sector"],
                    "ascent_score": float(r["ascent_score"]),
                    "strength_score": r["strength_score"], "volume_score": r["volume_score"],
                    "tier_score": r["tier_score"], "high_score": r["high_score"],
                    "tier": r["tier"], "market_cap": r["market_cap"],
                    "flags": json.loads(r["flags"]) if r["flags"] else [],
                    "delta_3d": delta(d3), "delta_1w": delta(w1), "delta_1m": delta(m1),
                    "first_seen": first_seen.isoformat() if first_seen else None,
                    "is_new": is_new,
                })

            return {"scan_time": latest_time.isoformat(), "rows": out_rows,
                    "history_available": history_available}

    async def prune(self, keep_days: int = 120) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM ascent_snapshots WHERE scan_time < $1", cutoff)
        return 0
