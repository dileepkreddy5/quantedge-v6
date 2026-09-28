"""Breakthrough reader: the press releases companies attach to 8-K filings.

Reads the exhibit 99 press release (else the main document) of 8-Ks carrying items
8.01, 7.01, 1.01 or 2.02, finds specific recognisable phrases sentence by sentence,
and records each finding with its type, direction, the sentence itself and a citation.
Append-only, extractor-versioned, paced under SEC fair-access limits."""
from __future__ import annotations
import asyncio, html, json, re
import httpx
from loguru import logger

EXTRACTOR = "press-release-v3"
RULES = [
    # an approval must be an EVENT ("received approval", "FDA approved X"), never the adjective "FDA-approved"
    ("fda_approval", "positive", r"\b(receiv(?:ed|es)|granted|obtain(?:ed|s)|announc(?:ed|es) (?:the )?(?:FDA )?)\b[^.]{0,80}\b(approval|clearance|510\(k\) clearance)\b[^.]{0,80}\b(FDA|Food and Drug Administration)\b|\b(receiv(?:ed|es)|granted|obtain(?:ed|s))\b[^.]{0,40}\b(FDA|Food and Drug Administration)\b[^.]{0,40}\b(approval|clearance)\b|\b(FDA|Food and Drug Administration)\b (?:has )?(approved|cleared|granted (?:approval|clearance))\b"),
    ("breakthrough_designation", "positive", r"\b(receiv(?:ed|es)|granted)\b[^.]{0,60}Breakthrough (Therapy|Device) Designation"),
    ("trial_positive", "positive", r"\b(met|achieved) (its|the|all) (primary|co-primary) endpoints?\b|\bpositive (topline|top-line|interim|final) (results|data)\b|\bstatistically significant (improvement|reduction|benefit)"),
    ("trial_negative", "negative", r"\b(did not|failed to) (meet|achieve) (its|the) primary endpoint|\bdiscontinu\w+ (the|its) (trial|study|program)\b"),
    ("coverage", "positive", r"\b(Medicare|CMS|MolDX)\b[^.]{0,80}\b(coverage|reimbursement)\b|\bcoverage (decision|expansion|determination)\b|\blocal coverage determination\b"),
    # "guidance"/"outlook" must sit right next to the raise, not anywhere in a long sentence
    ("guidance_raise", "positive", r"\b(rais(?:es|ed|ing)|increas(?:es|ed|ing)|lift(?:s|ed|ing))\b(?: its| our| the)?(?: full[- ]year| fiscal(?: year)?| annual| 20\d\d| \d?Q\d?(?: 20\d\d)?)?[^.]{0,45}\b(guidance|outlook)\b"),
    ("guidance_cut", "negative", r"\b(lower(?:s|ed|ing)|reduc(?:es|ed|ing)|cut(?:s|ting)?|withdr(?:aws|ew|awing))\b(?: its| our| the)?(?: full[- ]year| fiscal(?: year)?| annual| 20\d\d)?[^.]{0,25}\b(guidance|outlook)\b"),
    ("major_contract", "positive", r"\b(award(?:ed|s)?|selected|signed|won)\b[^.]{0,80}\b(contract|agreement|order)\b[^.]{0,80}\$\s?\d[\d,.]*\s?(million|billion)"),
    # "record" must be a claim in lowercase ("delivered record revenue"), not a name ("Carbon Record")
    ("record_results", "positive", r"(?-i:\brecord) (?:quarterly |annual |full[- ]year |fiscal )?(?:revenue|sales|net sales)\b"),
    ("product_launch", "positive", r"\b(launch(?:es|ed)|introduc(?:es|ed)|unveil(?:s|ed))\b (?:its |the |a |an )?(?:new|first|next-generation)\b[^.]{0,60}\b(product|platform|system|device|drug|therapy|model|service|chip|vehicle)\b"),
    ("acquisition", "neutral", r"\bdefinitive agreement to acquire\b|\bto be acquired by\b|\bmerger agreement\b"),
]
NEGATED = re.compile(r"\b(no|not|never|currently no|if approved|potential(?:ly)?|seek(?:ing|s)? (?:FDA )?approval|pending|expects? to|plans? to|anticipat\w+|may|could|would)\b", re.I)
INTENT = re.compile(r"\b(look(?:s|ing)? forward to|expects? to|plans? to|intends? to|hopes? to|aims? to|will)\b[^.]{0,30}\b(rais|increas|lift|lower|reduc|cut)", re.I)
MONTHS = {m: i for i, m in enumerate(["january","february","march","april","may","june","july","august","september","october","november","december"], 1)}
_RX = [(k, d, re.compile(p, re.I)) for k, d, p in RULES]


def classify(text: str, filed=None) -> list[dict]:
    t = html.unescape(re.sub(r"<[^>]+>", " ", text or "")); t = re.sub(r"\s+", " ", t)[:60000]
    # Abbreviations end in a period but don't end a sentence: v1 lost "awarded a contract by
    # the U.S. Army valued at $480 million" because "U.S." split the sentence in two.
    for a, b in (("U.S.", "US"), ("U.K.", "UK"), ("Inc.", "Inc"), ("Corp.", "Corp"), ("Co.", "Co"), ("Ltd.", "Ltd"),
                 ("No.", "No"), ("approx.", "approx"), ("Mr.", "Mr"), ("Ms.", "Ms"), ("Dr.", "Dr"), ("vs.", "vs"), ("St.", "St")):
        t = t.replace(a, b)
    found = {}
    for sent in re.split(r"(?<=[.!?])\s+", t):
        if len(sent) < 25 or len(sent) > 900: continue
        letters = [ch for ch in sent if ch.isalpha()]
        if letters and sum(ch.isupper() for ch in letters) / len(letters) > 0.5: continue     # slide/heading text
        if filed is not None:                                                                  # an old event restated
            m = re.search(r"\b(?:in|on) (January|February|March|April|May|June|July|August|September|October|November|December)(?: \d{1,2},)? (20\d\d)\b", sent, re.I)
            if m:
                import datetime as _d
                when = _d.date(int(m.group(2)), MONTHS[m.group(1).lower()], 28)
                if (filed - when).days > 45: continue
        for k, d, rx in _RX:
            if k in ("fda_approval", "breakthrough_designation", "trial_positive", "coverage", "product_launch") and NEGATED.search(sent) \
               and not re.search(r"\b(received|announced|granted|met|achieved)\b", sent, re.I):
                continue
            if k in ("guidance_raise", "guidance_cut") and INTENT.search(sent): continue   # an intention, not a change
            if k not in found and rx.search(sent):
                found[k] = {"type": k, "direction": d, "sentence": sent.strip()[:240]}
    return list(found.values())


async def read_press_releases(pool, days: int = 60) -> dict:
    from quantedge.fundamentals.edgar_bulk import UA
    rows = await pool.fetch("""
        SELECT min(e.id) event_id, min(e.ticker) ticker, min(e.company_id) cik, r.raw_payload->>'accession' acc,
               r.raw_payload->>'primaryDocument' doc, min(e.available_at) available_at
        FROM ci_events e JOIN ci_raw_evidence r ON r.id = e.evidence_id
        WHERE r.form_type = '8-K' AND e.item_code IN ('8.01','7.01','1.01','2.02')
          AND e.available_at > NOW() - ($1 || ' days')::interval
          AND NOT EXISTS (SELECT 1 FROM ci_derived d JOIN ci_events e2 ON e2.id = d.event_id
                          WHERE e2.evidence_id = r.id AND d.extractor_version = $2)
        GROUP BY r.id, r.raw_payload ORDER BY min(e.available_at) DESC""", str(days), EXTRACTOR)
    stats = {"filings": len(rows), "with_findings": 0, "findings": 0, "failed": 0}
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": UA}) as cx:
        async def get(url):
            r = await cx.get(url)
            if r.status_code in (403, 429): await asyncio.sleep(15); r = await cx.get(url)
            await asyncio.sleep(0.4)
            return r
        for row in rows:
            try:
                base = f"https://www.sec.gov/Archives/edgar/data/{int(row['cik'])}/{row['acc'].replace('-', '')}"
                idx = await get(base + "/index.json")
                names = [i["name"] for i in (idx.json().get("directory", {}).get("item", []) if idx.status_code == 200 else [])]
                ex = next((n for n in names if re.search(r"(ex|dex)-?99", n, re.I) and n.lower().endswith((".htm", ".html", ".txt"))), None)
                doc = ex or row["doc"]; url = f"{base}/{doc}"
                resp = await get(url)
                if resp.status_code != 200: stats["failed"] += 1; continue
                found = classify(resp.text, row["available_at"].date())
            except Exception:
                stats["failed"] += 1; continue
            await pool.execute("""INSERT INTO ci_derived (event_id, layer, extractor_version, field, value, citation, confidence, status)
                                  VALUES ($1,'DERIVED',$2,'press_release',$3::jsonb,$4::jsonb,'medium','observed')""",
                               row["event_id"], EXTRACTOR, json.dumps({"findings": found}),
                               json.dumps({"accession": row["acc"], "document": doc, "url": url}))
            if found: stats["with_findings"] += 1; stats["findings"] += len(found)
    logger.info(f"[ci/press] {stats}")
    return stats
