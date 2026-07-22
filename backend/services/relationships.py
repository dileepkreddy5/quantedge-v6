"""
QuantEdge v6.0 — Disclosed Business Relationships
==================================================
Suppliers, customers and competitors read from what companies actually file.

Correlation cannot answer this: asking which stocks move with Tesla returns NVE
Corp and IonQ, because co-movement measures shared volatility, not commerce.
Real relationships are disclosed in 10-K narrative sections — Tesla's filing
names Panasonic and CATL; Apple's names its suppliers. This module reads those
sentences and keeps the evidence alongside each link.

Where a filing says "one customer accounted for 22% of revenue" without naming
the customer, the relationship is genuinely unavailable and is not guessed.
"""
from __future__ import annotations

import asyncio, re
from typing import Dict, List, Optional, Tuple

import asyncpg, httpx
from loguru import logger

UA = {"User-Agent": "QuantEdge Research contact@dileepkapu.com"}

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS relationships (
    id           SERIAL PRIMARY KEY,
    src_ticker   TEXT NOT NULL,
    dst_name     TEXT NOT NULL,
    dst_ticker   TEXT,
    kind         TEXT NOT NULL,     -- SUPPLIER | CUSTOMER | COMPETITOR | PARTNER
    evidence     TEXT,              -- the sentence it came from
    filing_date  DATE,
    accession    TEXT,
    confidence   REAL DEFAULT 0.5,
    updated_at   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (src_ticker, dst_name, kind)
);
CREATE INDEX IF NOT EXISTS idx_rel_src ON relationships (src_ticker);
CREATE INDEX IF NOT EXISTS idx_rel_dst ON relationships (dst_ticker);
"""

# Language that establishes direction. A company "purchases from" a supplier and
# "sells to" a customer; conflating the two inverts the entire graph.
_PATTERNS: List[Tuple[str, str]] = [
    ("SUPPLIER",   r"(?:purchase[sd]?|source[sd]?|procure[sd]?|buy|obtain[s]?|supplied by|"
                   r"rely on|depend[s]? on|agreement with)\s+(?:[\w\s,]{0,40}?)\s*from\s+"),
    ("SUPPLIER",   r"(?:our|key|primary|principal|sole)\s+suppliers?\s+(?:include[s]?|are|is)\s+"),
    ("CUSTOMER",   r"(?:sell[s]?|supply|provide[s]?|deliver[s]?|revenue from)\s+"
                   r"(?:[\w\s,]{0,40}?)\s*to\s+"),
    ("CUSTOMER",   r"(?:our|largest|principal|significant)\s+customers?\s+(?:include[s]?|are|is)\s+"),
    ("COMPETITOR", r"(?:compete[s]?\s+(?:with|against)|competitors?\s+(?:include[s]?|are|is)|"
                   r"competition\s+from)\s+"),
    ("PARTNER",    r"(?:partnership|joint venture|collaborat\w+|alliance)\s+with\s+"),
]

# Corporate suffixes to strip when matching a mention back to a ticker.
_SUFFIX = re.compile(
    r"\b(inc|corp|corporation|company|co|ltd|limited|plc|llc|lp|holdings?|group|"
    r"technologies|technology|systems|solutions|industries|international|"
    r"n\.?v|s\.?a|a\.?g|s\.?e|kk|ab|as|oyj)\b\.?", re.I)


def _norm(name: str) -> str:
    n = _SUFFIX.sub("", name.lower())
    return re.sub(r"[^a-z0-9 ]", "", n).strip()


class RelationshipExtractor:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self._name_index: Dict[str, str] = {}

    async def ensure_tables(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(CREATE_SQL)
        logger.info("relationships table verified")

    async def _load_name_index(self) -> None:
        """Map normalised company names back to tickers so a mention of
        'Panasonic Holdings Corporation' resolves to its listing."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT ticker, name FROM universe WHERE name IS NOT NULL AND active")
        idx = {}
        for r in rows:
            key = _norm(r["name"])
            if len(key) >= 4 and key not in idx:
                idx[key] = r["ticker"]
                first = key.split()[0]
                if len(first) >= 5:
                    idx.setdefault(first, r["ticker"])
        self._name_index = idx
        logger.info(f"name index built: {len(idx)} entries")

    def _resolve(self, mention: str) -> Optional[str]:
        k = _norm(mention)
        if not k:
            return None
        if k in self._name_index:
            return self._name_index[k]
        first = k.split()[0] if k.split() else ""
        return self._name_index.get(first) if len(first) >= 5 else None

    async def _latest_10k(self, client: httpx.AsyncClient, cik: str) -> Optional[Tuple[str, str, str]]:
        c10 = str(cik).zfill(10)
        r = await client.get(f"https://data.sec.gov/submissions/CIK{c10}.json")
        if r.status_code != 200:
            return None
        rec = (r.json().get("filings") or {}).get("recent") or {}
        for i, form in enumerate(rec.get("form", [])):
            if form == "10-K":
                return (rec["accessionNumber"][i].replace("-", ""),
                        rec["primaryDocument"][i], rec["filingDate"][i])
        return None

    @staticmethod
    def _business_section(text: str) -> str:
        """Item 1 and 1A carry the supplier, customer and competitor language.
        A 10-K runs 200+ pages; scanning all of it produces noise from the
        financial statements and exhibits."""
        t = re.sub(r"<[^>]+>", " ", text)
        t = re.sub(r"&#?\w+;", " ", t)
        t = re.sub(r"\s+", " ", t)
        m = re.search(r"item\s*1\b.{0,40}business", t, re.I)
        n = re.search(r"item\s*2\b.{0,40}propert", t, re.I)
        if m:
            return t[m.start(): n.start() if n and n.start() > m.start() else m.start() + 250_000]
        return t[:250_000]

    def _extract(self, section: str) -> List[Dict]:
        out, seen = [], set()
        # Company mentions are capitalised multi-word phrases; require at least
        # one token to look like a proper noun rather than sentence-initial caps.
        cand = re.compile(r"\b([A-Z][a-zA-Z&.\-]+(?:\s+[A-Z][a-zA-Z&.\-]+){0,3})\b")
        for kind, pat in _PATTERNS:
            for m in re.finditer(pat, section, re.I):
                tail = section[m.end(): m.end() + 200]
                for cm in cand.finditer(tail[:120]):
                    name = cm.group(1).strip(" .,;")
                    if len(name) < 4 or name.lower() in ("the", "our", "we", "united states"):
                        continue
                    key = (kind, _norm(name))
                    if not key[1] or key in seen:
                        continue
                    seen.add(key)
                    s = max(0, m.start() - 120)
                    out.append({"kind": kind, "name": name,
                                "evidence": section[s: m.end() + 160].strip()})
                    break
        return out

    async def extract_for(self, ticker: str, cik: str) -> Dict:
        if not self._name_index:
            await self._load_name_index()
        async with httpx.AsyncClient(timeout=45, headers=UA, follow_redirects=True) as client:
            meta = await self._latest_10k(client, cik)
            if not meta:
                return {"ticker": ticker, "found": 0, "reason": "no 10-K"}
            acc, doc, fdate = meta
            await asyncio.sleep(0.4)
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
            r = await client.get(url)
            if r.status_code != 200:
                return {"ticker": ticker, "found": 0, "reason": f"filing fetch {r.status_code}"}
            rels = self._extract(self._business_section(r.text))

        rows = []
        for rel in rels:
            dst = self._resolve(rel["name"])
            if dst == ticker:
                continue
            rows.append((ticker, rel["name"], dst, rel["kind"],
                         rel["evidence"][:600], fdate, acc,
                         0.85 if dst else 0.45))
        if rows:
            async with self.pool.acquire() as conn:
                await conn.executemany("""
                    INSERT INTO relationships
                      (src_ticker,dst_name,dst_ticker,kind,evidence,filing_date,accession,confidence)
                    VALUES ($1,$2,$3,$4,$5,$6::date,$7,$8)
                    ON CONFLICT (src_ticker,dst_name,kind) DO UPDATE SET
                      dst_ticker=EXCLUDED.dst_ticker, evidence=EXCLUDED.evidence,
                      filing_date=EXCLUDED.filing_date, updated_at=NOW()
                """, rows)
        return {"ticker": ticker, "found": len(rows), "filing_date": fdate,
                "resolved": sum(1 for r in rows if r[2])}
