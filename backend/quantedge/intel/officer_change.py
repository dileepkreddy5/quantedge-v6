"""8-K item 5.02 reader: which officer changes are warning signs?

5.02 covers directors joining, planned retirements, and abrupt exits alike. Only an
abrupt departure of the CEO, CFO or chief/principal accounting officer is treated as
serious. Reads the filing's Item 5.02 text once per event; stores the verdict with a
citation (append-only, extractor-versioned). Paced under SEC fair-access limits."""
from __future__ import annotations
import asyncio, html, json, re
import httpx
from loguru import logger

EXTRACTOR = "officer-change-v2"
# v1 matched a role word and a leaving word anywhere in 5,000 characters: 107 of 278
# filings came out "abrupt", mostly appointment announcements that mention a predecessor.
# v2 requires both in ONE sentence, and treats appointments/retirements as planned.
ROLE = re.compile(r"(?<!interim )(?<!acting )(chief executive officer|chief financial officer|principal financial officer|chief accounting officer|principal accounting officer)", re.I)
LEAVE = re.compile(r"\b(resigned|resigns|will resign|has resigned|tendered (?:his|her|their) resignation|was terminated|were terminated|"
                   r"terminated (?:his|her|the) employment|departed|will depart|stepped down|will step down|steps down|was removed|separated from)\b", re.I)
APPOINT = re.compile(r"\b(appoint|named|elected|hired|promoted|succeed)", re.I)
PLANNED = re.compile(r"(retire|retirement|succession plan|planned transition|will continue to serve|remain with the company|to pursue other)", re.I)
IMMED = re.compile(r"(effective immediately|with immediate effect)", re.I)


def classify(text: str) -> dict:
    t = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    t = re.sub(r"\s+", " ", t)
    i = t.lower().find("5.02")
    sec = t[i:i + 6000] if i >= 0 else t[:6000]
    sents = re.split(r"(?<=[.;!?])\s+", sec)
    verdict, role, immed = "routine_change", None, False
    for k, sn in enumerate(sents):
        r = ROLE.search(sn)
        if not r or not LEAVE.search(sn): continue
        role = r.group(1).lower()
        if APPOINT.search(sn) or PLANNED.search(sn) or (k + 1 < len(sents) and PLANNED.search(sents[k + 1])):
            if verdict == "routine_change": verdict = "planned_exec_transition"
            continue
        verdict = "abrupt_exec_exit"
        immed = bool(IMMED.search(sn) or (k + 1 < len(sents) and IMMED.search(sents[k + 1])))
        break
    return {"class": verdict, "role": role, "immediate": immed}


async def classify_502(pool, days: int = 120) -> dict:
    from quantedge.fundamentals.edgar_bulk import UA
    rows = await pool.fetch("""
        SELECT e.id, e.company_id cik, r.raw_payload->>'accession' acc, r.raw_payload->>'primaryDocument' doc
        FROM ci_events e JOIN ci_raw_evidence r ON r.id = e.evidence_id
        WHERE e.item_code = '5.02' AND e.available_at > NOW() - ($1 || ' days')::interval
          AND NOT EXISTS (SELECT 1 FROM ci_derived d WHERE d.event_id = e.id AND d.extractor_version = $2)""", str(days), EXTRACTOR)
    stats = {"todo": len(rows), "abrupt": 0, "planned": 0, "routine": 0, "failed": 0}
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": UA}) as cx:
        for r in rows:
            if not (r["acc"] and r["doc"]): stats["failed"] += 1; continue
            url = f"https://www.sec.gov/Archives/edgar/data/{int(r['cik'])}/{r['acc'].replace('-', '')}/{r['doc']}"
            try:
                resp = await cx.get(url)
                if resp.status_code in (403, 429): await asyncio.sleep(15); resp = await cx.get(url)
                if resp.status_code != 200: stats["failed"] += 1; await asyncio.sleep(0.4); continue
                v = classify(resp.text)
            except Exception:
                stats["failed"] += 1; await asyncio.sleep(0.4); continue
            stats[{"abrupt_exec_exit": "abrupt", "planned_exec_transition": "planned"}.get(v["class"], "routine")] += 1
            await pool.execute("""INSERT INTO ci_derived (event_id, layer, extractor_version, field, value, citation, confidence, status)
                                  VALUES ($1,'DERIVED',$2,'officer_change_class',$3::jsonb,$4::jsonb,'medium','observed')""",
                               r["id"], EXTRACTOR, json.dumps(v),
                               json.dumps({"accession": r["acc"], "document": r["doc"], "location": "Item 5.02", "url": url}))
            await asyncio.sleep(0.4)                      # ~2.5 requests/second
    logger.info(f"[ci/5.02] {stats}")
    return stats
