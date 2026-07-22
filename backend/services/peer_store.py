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


# Narrower groupings used when enough companies share a real industry.
# Nine broad buckets put carmakers with railroads; these restore comparability.
_SUB_RULES = [
    ("Automotive",        ["MOTOR VEHICLE", "AUTOMOTIVE", "TRUCK", "AUTO PARTS", "CAR BODIES"]),
    ("Airlines",          ["AIR TRANSPORT", "AIRLINE", "AIR COURIER"]),
    ("Rail & Freight",    ["RAILROAD", "TRUCKING", "FREIGHT", "MARINE", "SHIPPING", "COURIER"]),
    ("Aerospace/Defense", ["AIRCRAFT", "AEROSPACE", "GUIDED MISSILE", "ORDNANCE", "DEFENSE"]),
    ("Metals & Steel",    ["STEEL", "METAL", "ALUMINUM", "IRON", "FOUNDRIES"]),
    ("Machinery",         ["MACHINERY", "ENGINES", "TURBINE", "FARM EQUIPMENT", "CONSTRUCTION MACHINERY"]),
    ("Semiconductors",    ["SEMICONDUCTOR", "ELECTRONIC COMPONENT"]),
    ("Software",          ["SOFTWARE", "PREPACKAGED", "DATA PROCESS", "COMPUTER PROGRAMMING"]),
    ("Hardware",          ["COMPUTER", "PERIPHERAL", "STORAGE DEVICE", "ELECTRONIC COMPUTERS"]),
    ("Comms Equipment",   ["COMMUNICATIONS EQUIP", "TELEPHONE APPARATUS", "RADIO", "BROADCAST EQUIP"]),
    ("Biotech",           ["BIOLOGICAL", "BIOTECH"]),
    ("Pharma",            ["PHARMACEUTICAL", "MEDICINAL"]),
    ("Medical Devices",   ["SURGICAL", "MEDICAL INSTRUMENT", "DENTAL", "ORTHOPEDIC", "DIAGNOSTIC"]),
    ("Healthcare Svcs",   ["HOSPITAL", "HEALTH SERVICES", "NURSING", "MANAGED CARE"]),
    ("Banks",             ["BANK", "SAVINGS INSTITUTION", "CREDIT UNION"]),
    ("Insurance",         ["INSURANCE", "SURETY", "TITLE INSUR"]),
    ("Capital Markets",   ["SECURITY BROKER", "INVESTMENT ADVICE", "INVESTMENT OFFICE", "ASSET MANAGE"]),
    ("REITs",             ["REAL ESTATE INVESTMENT"]),
    ("Retail",            ["RETAIL", "DEPARTMENT STORE", "GROCERY", "VARIETY STORE"]),
    ("Restaurants",       ["EATING", "RESTAURANT"]),
    ("Apparel & Luxury",  ["APPAREL", "FOOTWEAR", "JEWELRY", "LEATHER"]),
    ("Food & Beverage",   ["FOOD", "BEVERAGE", "DAIRY", "BAKERY", "SUGAR", "BREWERIES"]),
    ("Household Goods",   ["HOUSEHOLD", "PERSONAL", "SOAP", "COSMETIC", "FURNITURE"]),
    ("Oil & Gas E&P",     ["CRUDE PETROLEUM", "OIL AND GAS", "DRILLING", "OIL ROYALTY"]),
    ("Refining & Midstream",["PETROLEUM REFINING", "PIPELINE", "NATURAL GAS TRANSMISSION"]),
    ("Chemicals",         ["CHEMICAL", "FERTILIZER", "PLASTICS MATERIALS", "PAINT"]),
    ("Mining",            ["MINING", "GOLD", "COPPER", "COAL", "QUARRYING"]),
    ("Paper & Packaging", ["PAPER", "FOREST", "CONTAINER", "PACKAGING"]),
    ("Media",             ["TELEVISION", "MOTION PICTURE", "PUBLISHING", "BROADCAST", "CABLE"]),
    ("Telecom",           ["TELEPHONE COMMUNICATIONS", "TELECOM", "WIRELESS"]),
    ("Utilities-Electric",["ELECTRIC SERVICES", "ELECTRIC & OTHER"]),
    ("Utilities-Gas/Water",["GAS DISTRIBUTION", "WATER SUPPLY", "NATURAL GAS DISTRIB"]),
    ("Construction",      ["CONSTRUCTION", "HOMEBUILD", "GENERAL BUILDING"]),
    ("Advertising",       ["ADVERTISING", "MARKETING"]),
    # Industries present in the full universe but missing from the original list.
    ("Oilfield Services",   ["OIL & GAS FIELD SERVICES", "OIL AND GAS FIELD SERV"]),
    ("Real Estate",         ["REAL ESTATE", "OPERATIVE BUILDERS", "LAND SUBDIVID"]),
    ("Hotels & Leisure",    ["HOTELS", "MOTELS", "AMUSEMENT", "RECREATION", "GAMBLING", "RACING"]),
    ("Diagnostics & Labs",  ["MEDICAL LABORATORIES", "ELECTROMEDICAL", "IN VITRO", "CLINICAL LAB"]),
    ("Instruments",         ["LABORATORY ANALYTICAL", "INDUSTRIAL INSTRUMENTS", "MEASURING & CONTROL",
                             "MEAS & TESTING", "SEARCH, DETECTION", "OPTICAL INSTRUMENT"]),
    ("Business Services",   ["MANAGEMENT CONSULTING", "HELP SUPPLY", "BUSINESS SERVICES",
                             "COMPUTER PROGRAMMING", "PREPACKAGED SOFTWARE"]),
    ("Education",           ["EDUCATIONAL SERVICES"]),
    ("Specialty Finance",   ["FINANCE SERVICES", "FINANCE LESSORS", "PERSONAL CREDIT",
                             "MORTGAGE BANKERS", "PATENT OWNERS"]),
    ("Electronics Mfg",     ["PRINTED CIRCUIT BOARDS", "ELECTRONIC CONNECTORS", "ELECTRICAL INDUSTRIAL"]),
    ("Consumer Products",   ["SPORTING & ATHLETIC", "TOYS", "GAMES"]),
    ("Transport Services",  ["TRANSPORTATION SERVICES", "ARRANGEMENT OF TRANSPORT"]),
    ("Telecom Services",    ["COMMUNICATIONS SERVICES"]),
]

# Shell companies with no operations. They have a SIC but no business to compare.
_EXCLUDE_SIC = ("BLANK CHECK",)


def is_shell(sic: Optional[str]) -> bool:
    """SPACs and blank-check vehicles hold cash and nothing else."""
    return bool(sic) and any(k in sic.upper() for k in _EXCLUDE_SIC)


def sub_bucket_for(sic: Optional[str]) -> Optional[str]:
    """The narrow industry, where one is identifiable. Falls back to None so the
    caller can use the broad bucket when a peer group would be too small."""
    if not sic:
        return None
    u = sic.upper()
    for name, keys in _SUB_RULES:
        if any(k in u for k in keys):
            return name
    return None


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

        # Nine broad buckets put carmakers alongside railroads. Narrow to the real
        # industry when there are enough companies for percentiles to mean anything.
        # The scanned universe is a few hundred names, so industry groups are small.
        # Eight is the floor at which a percentile still carries information.
        MIN_GROUP = 8
        my_sub = sub_bucket_for(me["sic"])
        group_label, group_kind = me["bucket"], "sector"
        if my_sub:
            narrowed = [p for p in peers if sub_bucket_for(p["sic"]) == my_sub]
            if len(narrowed) >= MIN_GROUP:
                peers = narrowed
                group_label, group_kind = my_sub, "industry"

        # "Other" is not a peer group — it is everything the classifier could not
        # place, including companies with no SIC at all. Ranking against it is noise.
        if group_kind == "sector" and group_label == "Other":
            return {"available": False,
                    "reason": "no comparable peer group — this company is not classified in the scanned universe"}

        return {
            "available": True,
            "bucket": group_label,
            "group_kind": group_kind,
            "broad_sector": me["bucket"],
            "scan_time": latest.isoformat(),
            "me": dict(me),
            "peers": [dict(p) for p in peers],
        }
