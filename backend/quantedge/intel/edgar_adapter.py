"""EDGAR adapter — the first Company Intelligence source.

Reads each company's submissions index (data.sec.gov/submissions), which
carries acceptanceDateTime — the exact moment the filing became public —
and items for 8-Ks. Stores every new filing as immutable raw evidence,
classifies 8-K items deterministically into events, and parses Form 4 XML
into DERIVED transaction rows with structured citations.

Write discipline: INSERT ... ON CONFLICT DO NOTHING only. No UPDATE, no
DELETE, anywhere in this module.
Rate discipline: SEC allows ~10 req/s; 3 workers sleeping 0.35s ≈ 8.5/s.
"""
from __future__ import annotations
import asyncio, hashlib, json
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta, timezone
import httpx
from loguru import logger
from .schema import CREATE_SQL, ITEM_RULES, IGNORE_ITEMS

HEADERS = {"User-Agent": "QuantEdge research contact@quantedge.local"}
FORM4_LOOKBACK_DAYS = 120
EIGHTK_LOOKBACK_DAYS = 365
FORM4_MAX_PER_COMPANY_PER_RUN = 12
EXTRACTOR_FORM4 = "form4-xml-v1"


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _parse_form4(xml_text: str) -> dict | None:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    def t(el, path):
        n = el.find(path); return (n.text or "").strip() if n is not None and n.text else None
    owners = []
    for o in root.findall("reportingOwner"):
        rel = o.find("reportingOwnerRelationship")
        owners.append({"name": t(o, "reportingOwnerId/rptOwnerName"),
                       "is_director": t(rel, "isDirector") in ("1", "true") if rel is not None else False,
                       "is_officer": t(rel, "isOfficer") in ("1", "true") if rel is not None else False,
                       "officer_title": t(rel, "officerTitle") if rel is not None else None})
    txns = []
    for tr in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = t(tr, "transactionCoding/transactionCode")
        ad = t(tr, "transactionAmounts/transactionAcquiredDisposedCode/value")
        try:
            shares = float(t(tr, "transactionAmounts/transactionShares/value") or 0)
            price = float(t(tr, "transactionAmounts/transactionPricePerShare/value") or 0)
        except ValueError:
            shares, price = 0.0, 0.0
        txns.append({"date": t(tr, "transactionDate/value"), "code": code,
                     "acquired_disposed": ad, "shares": shares, "price": price,
                     "value": round(shares * price, 2),
                     "open_market": code in ("P", "S")})
    return {"owners": owners, "transactions": txns}


async def ensure_tables(pool):
    async with pool.acquire() as c:
        await c.execute(CREATE_SQL)


async def _company_universe(pool, limit: int):
    return await pool.fetch("""
        SELECT u.ticker, u.cik, u.market_cap
        FROM universe u
        JOIN (SELECT ticker FROM daily_bars GROUP BY ticker HAVING count(*) >= 500) b USING (ticker)
        WHERE u.cik IS NOT NULL AND u.active
        ORDER BY u.market_cap DESC NULLS LAST LIMIT $1""", limit)


async def ingest(pool, limit: int = 700, concurrency: int = 3) -> dict:
    await ensure_tables(pool)
    companies = await _company_universe(pool, limit)
    logger.info(f"[ci/edgar] ingesting {len(companies)} companies")
    sem = asyncio.Semaphore(concurrency)
    stats = {"companies": len(companies), "raw_new": 0, "events_new": 0,
             "form4_parsed": 0, "http_fail": 0}
    now = datetime.now(timezone.utc)
    cut8k = (now - timedelta(days=EIGHTK_LOOKBACK_DAYS)).date()
    cutf4 = (now - timedelta(days=FORM4_LOOKBACK_DAYS)).date()

    async with httpx.AsyncClient(timeout=25, headers=HEADERS) as client:
        async def one(tk: str, cik: str):
            async with sem:
                try:
                    c10 = str(cik).zfill(10)
                    r = await client.get(f"https://data.sec.gov/submissions/CIK{c10}.json")
                    if r.status_code == 429:
                        await asyncio.sleep(3)
                        r = await client.get(f"https://data.sec.gov/submissions/CIK{c10}.json")
                    if r.status_code != 200:
                        stats["http_fail"] += 1; return
                    rec = (r.json() or {}).get("filings", {}).get("recent", {})
                    n = len(rec.get("accessionNumber", []))
                    f4_done = 0
                    for i in range(n):
                        form = rec["form"][i]
                        fdate = date.fromisoformat(rec["filingDate"][i])
                        if form == "8-K" and fdate < cut8k: continue
                        if form == "4" and (fdate < cutf4 or f4_done >= FORM4_MAX_PER_COMPANY_PER_RUN): continue
                        if form not in ("8-K", "4"): continue
                        acc = rec["accessionNumber"][i]
                        acc_dt = rec.get("acceptanceDateTime", [None] * n)[i]
                        filed_at = (datetime.fromisoformat(acc_dt.replace("Z", "+00:00"))
                                    if acc_dt else datetime.combine(fdate, datetime.min.time(), timezone.utc))
                        payload = {"accession": acc, "form": form, "filingDate": rec["filingDate"][i],
                                   "reportDate": rec.get("reportDate", [None] * n)[i] or None,
                                   "acceptanceDateTime": acc_dt, "items": rec.get("items", [""] * n)[i],
                                   "primaryDocument": rec.get("primaryDocument", [""] * n)[i]}
                        # Form 4: fetch and parse the raw XML into the payload too.
                        if form == "4":
                            pdoc = payload["primaryDocument"] or ""
                            raw_name = pdoc.split("/")[-1]
                            url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                                   f"{acc.replace('-', '')}/{raw_name}")
                            await asyncio.sleep(0.35)
                            rx = await client.get(url)
                            if rx.status_code == 200:
                                parsed = _parse_form4(rx.text)
                                if parsed: payload["form4"] = parsed; stats["form4_parsed"] += 1
                            f4_done += 1
                        ch = _hash(payload)
                        async with pool.acquire() as conn:
                            ev_id = await conn.fetchval("""
                                INSERT INTO ci_raw_evidence (source_type, source_id, form_type, cik, filed_at, raw_payload, content_hash)
                                VALUES ('SEC', $1, $2, $3, $4, $5::jsonb, $6)
                                ON CONFLICT (source_type, source_id, content_hash) DO NOTHING RETURNING id""",
                                acc, form, str(cik), filed_at, json.dumps(payload), ch)
                            if ev_id is None:
                                continue          # already have this exact evidence
                            stats["raw_new"] += 1
                            ev_date = date.fromisoformat(payload["reportDate"]) if payload["reportDate"] else fdate
                            if form == "8-K":
                                items = [x.strip() for x in (payload["items"] or "").split(",") if x.strip()]
                                for it in items:
                                    if it in IGNORE_ITEMS: continue
                                    etype, sig, title = ITEM_RULES.get(it, ("unclassified_item", "UNKNOWN", f"8-K item {it}"))
                                    eid = await conn.fetchval("""
                                        INSERT INTO ci_events (company_id, ticker, event_date, available_at, evidence_id, item_code, event_type, significance, title)
                                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                                        ON CONFLICT (evidence_id, item_code) DO NOTHING RETURNING id""",
                                        str(cik), tk, ev_date, filed_at, ev_id, it, etype, sig, title)
                                    if eid: stats["events_new"] += 1
                            else:
                                p4 = payload.get("form4") or {}
                                who = ", ".join(o["name"] for o in p4.get("owners", []) if o.get("name")) or "insider"
                                eid = await conn.fetchval("""
                                    INSERT INTO ci_events (company_id, ticker, event_date, available_at, evidence_id, item_code, event_type, significance, title)
                                    VALUES ($1,$2,$3,$4,$5,'FORM4','insider_transaction','RELEVANT',$6)
                                    ON CONFLICT (evidence_id, item_code) DO NOTHING RETURNING id""",
                                    str(cik), tk, ev_date, filed_at, ev_id, f"Form 4 filed by {who}")
                                if eid:
                                    stats["events_new"] += 1
                                    for k, tr in enumerate(p4.get("transactions", [])):
                                        await conn.execute("""
                                            INSERT INTO ci_derived (event_id, layer, extractor_version, field, value, citation, confidence, status)
                                            VALUES ($1,'DERIVED',$2,'transaction',$3::jsonb,$4::jsonb,'high','observed')""",
                                            eid, EXTRACTOR_FORM4, json.dumps({**tr, "owners": p4.get("owners", [])}),
                                            json.dumps({"accession": acc, "document": "4",
                                                        "location": f"nonDerivativeTable/transaction[{k}]"}))
                except Exception as e:
                    stats["http_fail"] += 1
                    logger.debug(f"[ci/edgar] {tk}: {type(e).__name__}: {e}")
                await asyncio.sleep(0.35)

        batch = [(r["ticker"], r["cik"]) for r in companies]
        for i in range(0, len(batch), 60):
            await asyncio.gather(*[one(t, c) for t, c in batch[i:i + 60]])
            logger.info(f"[ci/edgar] {min(i+60, len(batch))}/{len(batch)} · "
                        f"raw+{stats['raw_new']} events+{stats['events_new']} f4 {stats['form4_parsed']} fail {stats['http_fail']}")
    return stats
