"""
QuantEdge v6.0 — Peer Stats Store
==================================
Stores the FULL scored universe (not just the Ascent top-25) grouped by a
cleaned sector bucket, so per-factor percentile ranks among true peers are
meaningful. Refreshed daily.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

import asyncpg
from loguru import logger

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS peer_stats (
    id          BIGSERIAL PRIMARY KEY,
    scan_time   TIMESTAMPTZ NOT NULL,
    ticker      TEXT NOT NULL,
    name        TEXT,
    sic         TEXT,
    bucket      TEXT,
    market_cap  DOUBLE PRECISION,
    factors     JSONB
);
CREATE INDEX IF NOT EXISTS idx_peer_scan   ON peer_stats (scan_time DESC);
CREATE INDEX IF NOT EXISTS idx_peer_bucket ON peer_stats (bucket, scan_time DESC);
CREATE INDEX IF NOT EXISTS idx_peer_ticker ON peer_stats (ticker, scan_time DESC);
"""

# Map raw SIC descriptions -> broad, comparable sector buckets.
# SIC is coarse/inconsistent (e.g. a chip-materials firm tagged "PLASTICS"),
# so we bucket by keyword to make peers actually comparable.
_BUCKET_RULES = [
    ("Technology",    ["SEMICONDUCTOR", "COMPUTER", "SOFTWARE", "ELECTRONIC", "INSTRUMENT",
                        "PERIPHERAL", "DATA PROCESS", "COMMUNICATIONS EQUIP", "PLASTICS PRODUCTS",
                        "SPECIAL INDUSTRY MACHINERY", "PHOTOGRAPHIC"]),
    ("Healthcare",    ["PHARMACEUTICAL", "BIOLOGICAL", "MEDICAL", "HEALTH", "SURGICAL", "DENTAL",
                        "DIAGNOSTIC", "HOSPITAL"]),
    ("Financials",    ["BANK", "INSURANCE", "FINANCE", "SECURITY BROKER", "INVESTMENT", "CREDIT",
                        "REAL ESTATE INVESTMENT"]),
    ("Consumer",      ["RETAIL", "APPAREL", "FOOD", "BEVERAGE", "RESTAURANT", "EATING",
                        "HOUSEHOLD", "FOOTWEAR", "TOYS", "JEWELRY", "PERSONAL"]),
    ("Industrials",   ["MACHINERY", "AIRCRAFT", "INDUSTRIAL", "STEEL", "METAL", "CONSTRUCTION",
                        "ENGINES", "MOTOR VEHICLE", "TRANSPORT", "RAILROAD", "AEROSPACE"]),
    ("Energy",        ["PETROLEUM", "OIL", "GAS", "ENERGY", "COAL", "DRILLING"]),
    ("Communications",["TELEVISION", "CABLE", "BROADCAST", "TELEPHONE", "PUBLISHING", "ADVERTISING",
                        "MOTION PICTURE", "TELECOM"]),
    ("Materials",     ["CHEMICAL", "MINING", "PAPER", "FOREST", "AGRICULTURAL", "FERTILIZER",
                        "GOLD", "COPPER"]),
    ("Utilities",     ["ELECTRIC SERVICES", "UTILITY", "WATER SUPPLY", "GAS DISTRIBUTION"]),
]


def bucket_for(sic: Optional[str]) -> str:
    if not sic:
        return "Other"
    u = sic.upper()
    for bucket, keys in _BUCKET_RULES:
        if any(k in u for k in keys):
            return bucket
    return "Other"


class PeerStore:
    def __init__(self, db_pool: asyncpg.Pool):
        self.pool = db_pool

    async def ensure_tables(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(CREATE_SQL)
        logger.info("✅ Peer stats table verified/created")

    async def save_snapshot(self, scan_time: datetime, rows: List[Dict]) -> int:
        if not rows:
            return 0
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for r in rows:
                    await conn.execute(
                        """INSERT INTO peer_stats
                           (scan_time, ticker, name, sic, bucket, market_cap, factors)
                           VALUES ($1,$2,$3,$4,$5,$6,$7)""",
                        scan_time, r["ticker"], r.get("name", ""), r.get("sic", ""),
                        bucket_for(r.get("sic", "")), r.get("market_cap"),
                        json.dumps(r.get("factors", {})),
                    )
                # keep only the 3 most recent scan_times to bound table growth
                await conn.execute("""
                    DELETE FROM peer_stats WHERE scan_time NOT IN (
                        SELECT DISTINCT scan_time FROM peer_stats ORDER BY scan_time DESC LIMIT 3
                    )""")
        logger.info(f"Peer stats snapshot saved: {len(rows)} tickers @ {scan_time.isoformat()}")
        return len(rows)

    async def get_peers(self, ticker: str) -> Dict:
        """Return the ticker's bucket peers from the latest snapshot + the ticker's row."""
        ticker = ticker.upper().strip()
        async with self.pool.acquire() as conn:
            latest = await conn.fetchval("SELECT max(scan_time) FROM peer_stats")
            if not latest:
                return {"available": False}
            me = await conn.fetchrow(
                "SELECT * FROM peer_stats WHERE ticker=$1 AND scan_time=$2", ticker, latest)
            if not me:
                return {"available": False, "reason": "ticker not in universe"}
            peers = await conn.fetch(
                "SELECT * FROM peer_stats WHERE bucket=$1 AND scan_time=$2 ORDER BY ticker",
                me["bucket"], latest)
        return {
            "available": True,
            "bucket": me["bucket"],
            "scan_time": latest.isoformat(),
            "me": dict(me),
            "peers": [dict(p) for p in peers],
        }
