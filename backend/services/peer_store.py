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


# SIC is hierarchical: 4 digits is the specific industry, 3 the group, 2 the
# major sector. Grouping on the code rather than on keyword rules gives real
# granularity — Meta and Alphabet share 7370 while Datadog and Snowflake sit at
# 7372, a distinction no amount of keyword matching recovers.
SIC_MIN_GROUP = 8


def sic_levels(code: Optional[str]) -> List[str]:
    """Progressively broader keys for a SIC code, most specific first."""
    if not code:
        return []
    c = str(code).strip().zfill(4)[:4]
    if not c.isdigit():
        return []
    return [c, c[:3], c[:2]]


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
                           (scan_time, ticker, name, sic, sic_code, bucket, market_cap, factors)
                           VALUES ($1,$2,$3,$4,$5,$6,$7,$8)""",
                        scan_time, r["ticker"], r.get("name", ""), r.get("sic", ""),
                        r.get("sic_code"), bucket_for(r.get("sic", "")), r.get("market_cap"),
                        json.dumps(r.get("factors", {})),
                    )
                # keep only the 3 most recent scan_times to bound table growth
                await conn.execute("""
                    DELETE FROM peer_stats WHERE scan_time NOT IN (
                        SELECT DISTINCT scan_time FROM peer_stats ORDER BY scan_time DESC LIMIT 3
                    )""")
        logger.info(f"Peer stats snapshot saved: {len(rows)} tickers @ {scan_time.isoformat()}")
        return len(rows)

    async def get_peers(self, ticker: str, live_sic: str = None,
                        live_sic_code: str = None, live_name: str = None,
                        live_market_cap: float = None) -> Dict:
        """Return the ticker's bucket peers from the latest snapshot + the ticker's row.

        When the ticker is not in the last scan (e.g. a megacap the hardcoded
        507-name universe omits), we still classify it live from the SIC the
        analyze pipeline already fetched and return its real peer group. The
        target's own factor percentiles are unavailable in that case — it was
        never scored — so the response carries classified_live=True and the
        router renders peers without a self-rank rather than a dead tab."""
        ticker = ticker.upper().strip()
        classified_live = False
        async with self.pool.acquire() as conn:
            latest = await conn.fetchval("SELECT max(scan_time) FROM peer_stats")
            if not latest:
                return {"available": False}
            me = await conn.fetchrow(
                "SELECT * FROM peer_stats WHERE ticker=$1 AND scan_time=$2", ticker, latest)
            if not me:
                # Fallback: classify from live analyze data and borrow the bucket.
                if not (live_sic or live_sic_code):
                    return {"available": False, "reason": "ticker not in universe"}
                # Classify from whatever we have: description via bucket_for,
                # else the SIC major group (first 2 digits) as a coarse bucket.
                _bucket = bucket_for(live_sic or "") if live_sic else None
                if (not _bucket or _bucket == "Other") and live_sic_code:
                    _major = str(live_sic_code)[:2]
                    _SIC_MAJOR = {"73":"Technology","35":"Technology","36":"Technology",
                                  "38":"Technology","28":"Healthcare","80":"Healthcare",
                                  "60":"Financials","61":"Financials","62":"Financials",
                                  "63":"Financials","64":"Financials","67":"Financials",
                                  "37":"Industrials","33":"Industrials","34":"Industrials",
                                  "13":"Energy","29":"Energy","48":"Communications",
                                  "27":"Communications","59":"Consumer","58":"Consumer",
                                  "56":"Consumer","53":"Consumer","20":"Consumer","54":"Consumer"}
                    _bucket = _SIC_MAJOR.get(_major)
                if not _bucket or _bucket == "Other":
                    return {"available": False, "reason": "ticker not in universe"}
                me = {
                    "ticker": ticker, "name": live_name, "sic": live_sic,
                    "sic_code": live_sic_code, "bucket": _bucket,
                    "market_cap": live_market_cap, "factors": {},
                }
                classified_live = True
            peers = await conn.fetch(
                "SELECT * FROM peer_stats WHERE bucket=$1 AND scan_time=$2 ORDER BY ticker",
                me["bucket"], latest)
            peers = [dict(p) for p in peers]
            if classified_live:
                # The target is not in peer_stats; add its live row so downstream
                # narrowing counts it, but it carries no factors to self-rank.
                peers = [p for p in peers if (p.get("ticker") or "").upper() != ticker] + [me]

        # Nine broad buckets put carmakers alongside railroads. Narrow to the real
        # industry when there are enough companies for percentiles to mean anything.
        # Walk down the SIC hierarchy: the exact 4-digit industry first, then the
        # 3-digit group, then the 2-digit sector. Keyword buckets put Meta with
        # Snowflake; SIC 7370 puts it with Alphabet, which is the real comparison.
        # Five companies that genuinely make computers is a better comparison than
        # thirty that merely share a leading digit, so the exact-industry threshold
        # is lower than the one for broader fallbacks.
        MIN_EXACT, MIN_GROUP = 4, 8
        group_label, group_kind = me["bucket"], "sector"
        my_code = (me.get("sic_code") or "")
        if my_code:
            for depth, kind, floor in ((4, "industry", MIN_EXACT), (3, "group", MIN_GROUP), (2, "sector-sic", MIN_GROUP)):
                key = my_code[:depth]
                narrowed = [p for p in peers
                            if (p.get("sic_code") or "")[:depth] == key]
                if len(narrowed) >= floor:
                    peers = narrowed
                    # Name the group after what the company actually does.
                    group_label = (me.get("sic") or me["bucket"]).title()
                    if depth < 4:
                        group_label = sub_bucket_for(me["sic"]) or group_label
                    group_kind = kind
                    break
            else:
                my_sub = sub_bucket_for(me["sic"])
                if my_sub:
                    narrowed = [p for p in peers if sub_bucket_for(p["sic"]) == my_sub]
                    if len(narrowed) >= MIN_GROUP:
                        peers = narrowed
                        group_label, group_kind = my_sub, "industry"
        else:
            my_sub = sub_bucket_for(me["sic"])
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
            "classified_live": classified_live,
            "bucket": group_label,
            "group_kind": group_kind,
            "broad_sector": me["bucket"],
            "scan_time": latest.isoformat(),
            "me": dict(me),
            "peers": [dict(p) if not isinstance(p, dict) else p for p in peers],
        }
