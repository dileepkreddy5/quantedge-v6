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
# Suppliers name their customers because concentration is a reportable risk;
# buyers have no such obligation. Skyworks lists Apple, Cisco and Ericsson by
# name; Apple's own filing names nobody. So the graph is read from the supply
# side and the edges reversed to answer "who benefits when this company grows".
_PATTERNS: List[Tuple[str, str]] = [
    ("CUSTOMER_OF", r"(?:our\s+)?(?:key|significant|principal|largest|major|primary|top)?\s*"
                    r"customers?\s+(?:include[sd]?|are|comprise[sd]?|consist(?:s|ed)? of)\s*:?\s+"),
    ("CUSTOMER_OF", r"(?:we\s+)?(?:sell|supply|provide|ship)\s+(?:our\s+)?(?:products?|solutions?|"
                    r"services?)\s+to\s+(?:companies\s+such\s+as\s+|customers\s+including\s+)?"),
    ("SUPPLIER_OF", r"(?:our\s+)?(?:key|principal|primary|main|major)?\s*suppliers?\s+"
                    r"(?:include[sd]?|are)\s*:?\s+"),
    ("COMPETITOR",  r"(?:our\s+)?(?:primary|principal|main|key)?\s*competitors?\s+"
                    r"(?:include[sd]?|are)\s*:?\s+"),
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
        # Matching on the first word alone maps "Sierra Wireless" to Sierra Bancorp
        # and "Texas Instruments" to a bank in Texas. Single-word keys are only
        # safe when exactly one company in the universe carries that word.
        idx, first_counts, first_map = {}, {}, {}
        for r in rows:
            key = _norm(r["name"])
            if len(key) >= 4 and key not in idx:
                idx[key] = r["ticker"]
            parts = key.split()
            if parts and len(parts[0]) >= 5:
                first_counts[parts[0]] = first_counts.get(parts[0], 0) + 1
                first_map.setdefault(parts[0], r["ticker"])
        for w, n in first_counts.items():
            if n == 1 and w not in idx:
                idx[w] = first_map[w]
        self._name_index = idx
        logger.info(f"name index built: {len(idx)} entries")

    def _resolve(self, mention: str) -> Optional[str]:
        k = _norm(mention)
        if not k:
            return None
        if k in self._name_index:
            return self._name_index[k]
        # Try progressively shorter prefixes before giving up, so "Apple Inc"
        # resolves while an ambiguous single word stays unresolved.
        parts = k.split()
        for n in range(len(parts), 0, -1):
            cand = " ".join(parts[:n])
            if len(cand) >= 5 and cand in self._name_index:
                return self._name_index[cand]
        # "Cisco Systems" against a universe entry of "Cisco", or the reverse:
        # allow a match when one name is a prefix of the other and long enough
        # to be unambiguous.
        if len(k) >= 6:
            hits = [t for nm, t in self._name_index.items()
                    if len(nm) >= 5 and (nm.startswith(k) or k.startswith(nm))]
            if len(set(hits)) == 1:
                return hits[0]
        return None

    async def _latest_10k(self, client: httpx.AsyncClient, cik: str) -> Optional[Tuple[str, str, str]]:
        c10 = str(cik).zfill(10)
        r = await client.get(f"https://data.sec.gov/submissions/CIK{c10}.json")
        if r.status_code == 429:
            # SEC throttles for minutes at a time. Backing off once beats silently
            # recording the ticker as having no filing and never revisiting it.
            await asyncio.sleep(10)
            r = await client.get(f"https://data.sec.gov/submissions/CIK{c10}.json")
        if r.status_code != 200:
            logger.warning(f"submissions CIK{c10}: HTTP {r.status_code}")
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
        # The first "Item 1 ... Business" match is the table of contents, which
        # sits a few lines from "Item 2 ... Properties" and yields a section of
        # about a hundred characters. The real body is the last such match.
        starts = [x.start() for x in re.finditer(r"item\s*1\b.{0,60}?business", t, re.I)]
        ends = [x.start() for x in re.finditer(r"item\s*2\b.{0,60}?propert", t, re.I)]
        best, best_len = None, 0
        for s in starts:
            e = next((x for x in ends if x > s + 2000), None)
            span = (e - s) if e else (len(t) - s)
            if span > best_len:
                best, best_len = (s, e), span
        if best and best_len > 5000:
            s, e = best
            return t[s: e if e else s + 400_000]
        return t[:400_000]

    def _extract(self, section: str) -> List[Dict]:
        """These phrases are followed by a comma-separated run of company names,
        so the whole list is taken rather than the first match."""
        out, seen = [], set()
        name_re = re.compile(r"[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,3}")
        # Geographies and industry shorthand are not companies. A supplier writing
        # "customers include OEMs in Europe and Asia" is describing a market, and
        # storing that as a relationship would be noise dressed as data.
        stop = {"the", "our", "we", "united states", "company", "inc", "and", "other",
                "these", "such", "including", "certain", "various", "many", "some",
                "oems", "odms", "oem", "odm", "europe", "asia", "america", "canada",
                "germany", "japan", "china", "india", "australia", "france", "korea",
                "mexico", "brazil", "africa", "u s", "us", "uk", "emea", "apac",
                "north america", "south america", "latin america", "middle east",
                "medicaid", "medicare", "government", "military", "federal",
                "aerospace", "defense", "automotive", "industrial", "commercial",
                "customers", "clients", "distributors", "retailers", "dealers"}
        for kind, pat in _PATTERNS:
            for m in re.finditer(pat, section, re.I):
                tail = section[m.end(): m.end() + 420]
                # the list ends at a sentence boundary or a trailing clause
                cut = re.search(r"\.\s+[A-Z]|\band\s+others\b|\betc\b|\bamong\s+others\b", tail)
                run = tail[: cut.start()] if cut else tail[:260]
                for part in re.split(r",|\band\b|;", run):
                    part = part.strip(" .,;:()")
                    if len(part) < 3:
                        continue
                    nm = name_re.match(part)
                    if not nm:
                        continue
                    name = nm.group(0).strip(" .,;")
                    if len(name) < 3 or name.lower() in stop:
                        continue
                    key = (kind, _norm(name))
                    if not key[1] or key in seen:
                        continue
                    seen.add(key)
                    # Start at the sentence boundary rather than a fixed offset,
                    # so the quote reads as a sentence instead of a fragment.
                    lo = max(0, m.start() - 260)
                    window = section[lo: m.end() + 240]
                    dot = window.rfind(". ", 0, m.start() - lo)
                    ev = window[dot + 2:] if dot > 0 else window
                    out.append({"kind": kind, "name": name, "evidence": ev.strip()})
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
            # asyncpg binds a real date object; the ::date cast does not coerce a string.
            from datetime import datetime as _dt
            try:
                _fd = _dt.strptime(fdate, "%Y-%m-%d").date()
            except Exception:
                _fd = None
            rows.append((ticker, rel["name"], dst, rel["kind"],
                         rel["evidence"][:600], _fd, acc,
                         0.85 if dst else 0.45))
        if rows:
            async with self.pool.acquire() as conn:
                await conn.executemany("""
                    INSERT INTO relationships
                      (src_ticker,dst_name,dst_ticker,kind,evidence,filing_date,accession,confidence)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                    ON CONFLICT (src_ticker,dst_name,kind) DO UPDATE SET
                      dst_ticker=EXCLUDED.dst_ticker, evidence=EXCLUDED.evidence,
                      filing_date=EXCLUDED.filing_date, updated_at=NOW()
                """, rows)
        return {"ticker": ticker, "found": len(rows), "filing_date": fdate,
                "resolved": sum(1 for r in rows if r[2])}
