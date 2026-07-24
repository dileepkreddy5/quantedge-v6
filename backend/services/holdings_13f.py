"""13F institutional holdings — quarterly bulk load from the SEC.

Every manager over $100M files a 13F within 45 days of quarter end, listing each
position by CUSIP and share count. The data is therefore always 45 days stale by
regulation — that is true of every source, Bloomberg included — so the as-of date
is served alongside it rather than hidden.

The file is ~99MB zipped, 396MB as INFOTABLE.tsv, and a new one appears four
times a year. The loader checks weekly and skips work when the latest quarter is
already loaded, so the cost is one download per quarter rather than per request.
"""
from __future__ import annotations
import io, re, zipfile, asyncio
import httpx
from loguru import logger
from quantedge.scoring.edgar_xbrl import UA

INDEX = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
BASE = "https://www.sec.gov"

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS holdings_13f (
    quarter      TEXT NOT NULL,
    cusip        TEXT NOT NULL,
    issuer       TEXT,
    manager      TEXT NOT NULL,
    shares       BIGINT,
    value_usd    BIGINT,
    PRIMARY KEY (quarter, cusip, manager)
);
CREATE INDEX IF NOT EXISTS idx_13f_cusip ON holdings_13f (cusip, quarter);
CREATE INDEX IF NOT EXISTS idx_13f_issuer ON holdings_13f (lower(issuer), quarter);

CREATE TABLE IF NOT EXISTS holdings_13f_meta (
    quarter    TEXT PRIMARY KEY,
    loaded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    n_rows     BIGINT,
    n_managers INT
);
"""


async def latest_dataset_url(client: httpx.AsyncClient) -> tuple[str, str] | None:
    """Return (absolute_url, quarter_label) for the most recent 13F dataset."""
    r = await client.get(INDEX, headers={"User-Agent": UA}, timeout=60)
    if r.status_code != 200:
        logger.warning(f"13F index returned HTTP {r.status_code}")
        return None
    links = re.findall(r'href="([^"]*form-13f-data-sets/[^"]*\.zip)"', r.text, re.I)
    if not links:
        return None
    # Filenames are date ranges: 01mar2026-31may2026_form13f.zip. The listing is
    # newest-first, so the first entry is current.
    path = links[0]
    label = path.rsplit("/", 1)[-1].replace("_form13f.zip", "")
    return (BASE + path if path.startswith("/") else path), label


async def load_quarter(pool, force: bool = False) -> dict:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_SQL)

    async with httpx.AsyncClient(timeout=600, follow_redirects=True) as client:
        found = await latest_dataset_url(client)
        if not found:
            return {"loaded": False, "reason": "no dataset link found"}
        url, quarter = found

        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT n_rows, loaded_at FROM holdings_13f_meta WHERE quarter=$1", quarter)
        if existing and not force:
            return {"loaded": False, "quarter": quarter, "reason": "already loaded",
                    "n_rows": existing["n_rows"], "loaded_at": str(existing["loaded_at"])}

        logger.info(f"13F: downloading {quarter}")
        r = await client.get(url, headers={"User-Agent": UA})
        if r.status_code != 200:
            logger.error(f"13F download HTTP {r.status_code} for {quarter}")
            return {"loaded": False, "quarter": quarter, "reason": f"HTTP {r.status_code}"}

    z = zipfile.ZipFile(io.BytesIO(r.content))

    # accession -> manager name
    managers: dict[str, str] = {}
    with z.open("COVERPAGE.tsv") as fh:
        cols = fh.readline().decode("utf-8", "replace").rstrip("\r\n").split("\t")
        try:
            i_acc, i_name = cols.index("ACCESSION_NUMBER"), cols.index("FILINGMANAGER_NAME")
        except ValueError:
            i_acc, i_name = 0, 1
        for line in fh:
            p = line.decode("utf-8", "replace").rstrip("\r\n").split("\t")
            if len(p) > max(i_acc, i_name):
                managers[p[i_acc]] = p[i_name]

    rows: list[tuple] = []
    seen: set[tuple] = set()
    with z.open("INFOTABLE.tsv") as fh:
        cols = fh.readline().decode("utf-8", "replace").rstrip("\r\n").split("\t")
        ix = {c: i for i, c in enumerate(cols)}
        for line in fh:
            p = line.decode("utf-8", "replace").rstrip("\r\n").split("\t")
            if len(p) < len(cols):
                continue
            # Puts and calls are not share ownership.
            if p[ix.get("PUTCALL", 0)] if "PUTCALL" in ix else "":
                continue
            if (p[ix["SSHPRNAMTTYPE"]] or "").strip().upper() != "SH":
                continue
            mgr = managers.get(p[ix["ACCESSION_NUMBER"]])
            cusip = (p[ix["CUSIP"]] or "").strip().upper()
            if not mgr or not cusip:
                continue
            key = (quarter, cusip, mgr)
            if key in seen:      # a manager can report one issuer across several lines
                continue
            seen.add(key)
            try:
                shares = int(float(p[ix["SSHPRNAMT"]] or 0))
                value = int(float(p[ix["VALUE"]] or 0))
            except ValueError:
                continue
            rows.append((quarter, cusip, (p[ix["NAMEOFISSUER"]] or "").strip(), mgr, shares, value))

    logger.info(f"13F: parsed {len(rows):,} positions from {len(managers):,} managers")

    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM holdings_13f WHERE quarter=$1", quarter)
        CHUNK = 50_000
        for i in range(0, len(rows), CHUNK):
            await conn.copy_records_to_table(
                "holdings_13f",
                records=rows[i:i + CHUNK],
                columns=["quarter", "cusip", "issuer", "manager", "shares", "value_usd"])
        await conn.execute(
            "INSERT INTO holdings_13f_meta (quarter, n_rows, n_managers) VALUES ($1,$2,$3) "
            "ON CONFLICT (quarter) DO UPDATE SET n_rows=EXCLUDED.n_rows, "
            "n_managers=EXCLUDED.n_managers, loaded_at=now()",
            quarter, len(rows), len(managers))

    return {"loaded": True, "quarter": quarter, "n_rows": len(rows), "n_managers": len(managers)}


async def ownership_for(pool, ticker: str, company_name: str | None,
                        shares_outstanding: float | None) -> dict:
    """Institutional ownership for one company, from the latest loaded quarter.

    13F reports positions by CUSIP and issuer name, not ticker. Rather than
    depend on an external CUSIP map, resolve by issuer name against the filings
    themselves and then sanity-check: the summed institutional position cannot
    exceed shares outstanding, and a match yielding more than that is the wrong
    CUSIP rather than a remarkable ownership structure.
    """
    if not company_name:
        return {"available": False, "reason": "no company name to match on"}

    # "Apple Inc." -> "APPLE"; issuer names in 13F are inconsistent about suffixes.
    stem = re.sub(r"[^A-Z ]", " ", company_name.upper())
    stem = re.sub(r"\b(INC|CORP|CORPORATION|CO|COMPANY|LTD|PLC|HOLDINGS|GROUP|CLASS [A-C]|COM)\b", " ", stem)
    stem = " ".join(stem.split())
    if len(stem) < 3:
        return {"available": False, "reason": "company name too short to match"}

    async with pool.acquire() as conn:
        q = await conn.fetchval("SELECT max(quarter) FROM holdings_13f_meta")
        if not q:
            return {"available": False, "reason": "no 13F quarter loaded yet"}

        # Candidate CUSIPs whose issuer name starts with the stem, largest first.
        cands = await conn.fetch(
            """SELECT cusip, max(issuer) AS issuer, count(*) AS n_managers,
                      sum(shares)::bigint AS total_shares
               FROM holdings_13f
               WHERE quarter=$1 AND lower(issuer) LIKE lower($2)
               GROUP BY cusip ORDER BY total_shares DESC LIMIT 5""",
            q, stem.split()[0] + "%")
        if not cands:
            return {"available": False, "quarter": q,
                    "reason": f"no 13F issuer matching '{stem}'"}

        pick = None
        for c in cands:
            if shares_outstanding and c["total_shares"] > shares_outstanding * 1.05:
                continue      # more shares than exist — wrong CUSIP
            pick = c
            break
        if pick is None:
            return {"available": False, "quarter": q,
                    "reason": "matched issuer holds more shares than outstanding"}

        # Large managers file through subsidiaries — BlackRock reports under
        # iShares, BlackRock Fund Advisors and BlackRock Institutional Trust
        # separately, Vanguard under Capital Management, Portfolio Management and
        # Fiduciary Trust. Ungrouped, the top-holder list shows fragments and
        # understates every major holder. Group by the family name.
        raw = await conn.fetch(
            """SELECT manager, sum(shares)::bigint AS shares, sum(value_usd)::bigint AS value_usd
               FROM holdings_13f WHERE quarter=$1 AND cusip=$2
               GROUP BY manager""", q, pick["cusip"])
        n_managers_here = len(raw)

    FAMILIES = [
        ("Vanguard", "VANGUARD"), ("BlackRock", "BLACKROCK"), ("BlackRock", "ISHARES"),
        ("State Street", "STATE STREET"), ("Fidelity", "FMR "), ("Fidelity", "FIDELITY"),
        ("Geode Capital", "GEODE"), ("Charles Schwab", "SCHWAB"),
        ("JPMorgan", "JPMORGAN"), ("JPMorgan", "J.P. MORGAN"), ("JPMorgan", "JP MORGAN"),
        ("Morgan Stanley", "MORGAN STANLEY"), ("Goldman Sachs", "GOLDMAN"),
        ("Bank of America", "BANK OF AMERICA"), ("Wells Fargo", "WELLS FARGO"),
        ("Northern Trust", "NORTHERN TRUST"), ("Invesco", "INVESCO"),
        ("T. Rowe Price", "T. ROWE"), ("T. Rowe Price", "T ROWE"),
        ("Capital Group", "CAPITAL RESEARCH"), ("Capital Group", "CAPITAL WORLD"),
        ("Amundi", "AMUNDI"), ("UBS", "UBS "), ("Deutsche Bank", "DEUTSCHE"),
        ("Norges Bank", "NORGES"), ("Legal & General", "LEGAL & GENERAL"),
        ("Franklin Resources", "FRANKLIN"), ("Dimensional", "DIMENSIONAL"),
        ("Nuveen", "NUVEEN"), ("Nuveen", "TEACHERS INSURANCE"),
    ]

    def family(name: str) -> str:
        up = (name or "").upper()
        for label, needle in FAMILIES:
            if needle in up:
                return label
        # Otherwise strip the entity suffix so "X Advisors LLC" and "X LP" merge.
        base = re.sub(r"[.,]", " ", name or "")
        base = re.sub(r"\b(LLC|L\.?P|INC|CORP|CO|LTD|PLC|GROUP|HOLDINGS|ADVISORS?|"
                      r"ADVISERS?|MANAGEMENT|MGMT|CAPITAL|PARTNERS|TRUST|BANK|"
                      r"INVESTMENTS?|ASSET|INTERNATIONAL|GLOBAL|COMPANY)\b", " ",
                      base, flags=re.I)
        base = " ".join(base.split())
        return base.title() if base else (name or "unknown")

    grouped: dict[str, dict] = {}
    for h in raw:
        k = family(h["manager"])
        g = grouped.setdefault(k, {"shares": 0, "value_usd": 0, "n_entities": 0})
        g["shares"] += int(h["shares"] or 0)
        g["value_usd"] += int(h["value_usd"] or 0)
        g["n_entities"] += 1
    holders = sorted(
        ({"manager": k, **v} for k, v in grouped.items()),
        key=lambda x: -x["shares"])[:15]

    total = int(pick["total_shares"])
    inst_pct = (total / shares_outstanding * 100) if shares_outstanding else None
    top = [{"manager": h["manager"],
            "shares": int(h["shares"]),
            "value_usd": int(h["value_usd"]),
            "n_entities": h.get("n_entities", 1),
            "pct_of_company": round(h["shares"] / shares_outstanding * 100, 2) if shares_outstanding else None}
           for h in holders]

    return {
        "available": True,
        "quarter": q,
        "cusip": pick["cusip"],
        "issuer": pick["issuer"],
        "n_managers": n_managers_here,   # filers holding THIS company, not the dataset total
        "institutional_shares": total,
        "shares_outstanding": int(shares_outstanding) if shares_outstanding else None,
        "institutional_pct": round(inst_pct, 1) if inst_pct is not None else None,
        "other_pct": round(100 - inst_pct, 1) if inst_pct is not None else None,
        "top_holders": top,
        "note": ("13F positions as reported for the quarter ending in this period and filed "
                 "up to 45 days afterwards. Every source carries the same lag; it is a "
                 "regulatory filing deadline, not a data limitation. 'Other' is everything "
                 "not held by a 13F filer: retail, insiders, and managers under the $100M "
                 "reporting threshold."),
    }
