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
import datetime as _dt
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

# Managers whose positions are a decision rather than an index weight. An index
# fund holds every large company by construction; these hold what they chose.
NOTABLE = [
    ("Berkshire Hathaway", "BERKSHIRE"), ("Pershing Square", "PERSHING SQUARE"),
    ("Bridgewater", "BRIDGEWATER"), ("Baupost", "BAUPOST"),
    ("Tiger Global", "TIGER GLOBAL"), ("Appaloosa", "APPALOOSA"),
    ("Third Point", "THIRD POINT"), ("Elliott", "ELLIOTT"),
    ("Renaissance Technologies", "RENAISSANCE TECH"), ("Citadel", "CITADEL"),
    ("Millennium", "MILLENNIUM MANAGEMENT"), ("Point72", "POINT72"),
    ("Lone Pine", "LONE PINE"), ("Viking Global", "VIKING GLOBAL"),
    ("Coatue", "COATUE"), ("Greenlight", "GREENLIGHT CAPITAL"),
    ("Icahn", "ICAHN"), ("ValueAct", "VALUEACT"), ("Starboard", "STARBOARD VALUE"),
    ("Duquesne", "DUQUESNE"), ("Soros", "SOROS FUND"), ("Two Sigma", "TWO SIGMA"),
    ("AQR", "AQR CAPITAL"), ("Marshall Wace", "MARSHALL WACE"),
]


def _family(name: str) -> str:
    """Group a filing entity into its parent. BlackRock files through iShares and
    several trusts, Vanguard through ten entities; ungrouped, every large holder
    appears as a set of fragments and its real stake is invisible."""
    up = (name or "").upper()
    for label, needle in FAMILIES:
        if needle in up:
            return label
    base = re.sub(r"[.,]", " ", name or "")
    base = re.sub(r"\b(LLC|L\.?P|INC|CORP|CO|LTD|PLC|GROUP|HOLDINGS|ADVISORS?|"
                  r"ADVISERS?|MANAGEMENT|MGMT|CAPITAL|PARTNERS|TRUST|BANK|"
                  r"INVESTMENTS?|ASSET|INTERNATIONAL|GLOBAL|COMPANY)\b", " ",
                  base, flags=re.I)
    base = " ".join(base.split())
    return base.title() if base else (name or "unknown")


def _notable(name: str) -> str | None:
    up = (name or "").upper()
    for label, needle in NOTABLE:
        if needle in up:
            return label
    return None


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
                "SELECT n_rows, loaded_at FROM holdings_13f_meta WHERE quarter=$1", "file:" + quarter)
        if existing and not force:
            return {"loaded": False, "quarter": quarter, "reason": "already loaded",
                    "n_rows": existing["n_rows"], "loaded_at": str(existing["loaded_at"])}

        logger.info(f"13F: downloading {quarter}")
        r = await client.get(url, headers={"User-Agent": UA})
        if r.status_code != 200:
            logger.error(f"13F download HTTP {r.status_code} for {quarter}")
            return {"loaded": False, "quarter": quarter, "reason": f"HTTP {r.status_code}"}

    z = zipfile.ZipFile(io.BytesIO(r.content))

    # Older archives nest the files inside a folder; newer ones do not.
    def member(name: str) -> str:
        for n in z.namelist():
            if n.rsplit("/", 1)[-1] == name:
                return n
        raise KeyError(f"{name} not in archive")

    # accession -> reported period, and which accessions actually carry holdings.
    # The dataset filename is a FILING-DATE range, not a position date: the first
    # row of one file is a 31-MAR-2026 filing reporting positions as at
    # 30-SEP-2025. Keying on the filename mixed periods, so a manager who filed
    # late appeared to open an enormous new position. PERIODOFREPORT is the
    # as-of date and is what the table is keyed on.
    # 13F-NT is a notice that nothing is reportable; only 13F-HR carries holdings.
    periods: dict[str, str] = {}
    with z.open(member("SUBMISSION.tsv")) as fh:
        cols = fh.readline().decode("utf-8", "replace").rstrip("\r\n").split("\t")
        ix = {c: i for i, c in enumerate(cols)}
        for line in fh:
            p = line.decode("utf-8", "replace").rstrip("\r\n").split("\t")
            if len(p) < len(cols):
                continue
            if not p[ix["SUBMISSIONTYPE"]].startswith("13F-HR"):
                continue
            raw_period = (p[ix["PERIODOFREPORT"]] or "").strip()   # 31-MAR-2026
            try:
                d = _dt.datetime.strptime(raw_period, "%d-%b-%Y").date()
            except ValueError:
                continue
            periods[p[ix["ACCESSION_NUMBER"]]] = d.isoformat()

    # accession -> manager name
    managers: dict[str, str] = {}
    with z.open(member("COVERPAGE.tsv")) as fh:
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
    with z.open(member("INFOTABLE.tsv")) as fh:
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
            period = periods.get(p[ix["ACCESSION_NUMBER"]])
            if not period:
                continue          # 13F-NT, an amendment, or an unparseable date
            key = (period, cusip, mgr)
            if key in seen:      # a manager can report one issuer across several lines
                continue
            seen.add(key)
            try:
                shares = int(float(p[ix["SSHPRNAMT"]] or 0))
                value = int(float(p[ix["VALUE"]] or 0))
            except ValueError:
                continue
            rows.append((period, cusip, (p[ix["NAMEOFISSUER"]] or "").strip(), mgr, shares, value))

    logger.info(f"13F: parsed {len(rows):,} positions from {len(managers):,} managers")

    touched = sorted({r[0] for r in rows})
    async with pool.acquire() as conn:
        for per in touched:
            await conn.execute("DELETE FROM holdings_13f WHERE quarter=$1", per)
        CHUNK = 50_000
        for i in range(0, len(rows), CHUNK):
            await conn.copy_records_to_table(
                "holdings_13f",
                records=rows[i:i + CHUNK],
                columns=["quarter", "cusip", "issuer", "manager", "shares", "value_usd"])
        for per in touched:
            n = sum(1 for r in rows if r[0] == per)
            m = len({r[3] for r in rows if r[0] == per})
            await conn.execute(
                "INSERT INTO holdings_13f_meta (quarter, n_rows, n_managers) VALUES ($1,$2,$3) "
                "ON CONFLICT (quarter) DO UPDATE SET n_rows=EXCLUDED.n_rows, "
                "n_managers=EXCLUDED.n_managers, loaded_at=now()", per, n, m)
        # Record the source file so the weekly check can skip it next time.
        await conn.execute(
            "INSERT INTO holdings_13f_meta (quarter, n_rows, n_managers) VALUES ($1,$2,$3) "
            "ON CONFLICT (quarter) DO UPDATE SET n_rows=EXCLUDED.n_rows, loaded_at=now()",
            "file:" + quarter, len(rows), len(managers))

    return {"loaded": True, "source_file": quarter, "periods": touched,
            "n_rows": len(rows), "n_managers": len(managers)}


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
        q = await conn.fetchval("SELECT max(quarter) FROM holdings_13f_meta WHERE quarter NOT LIKE 'file:%'")
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

    family = _family

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



# flow_for (quarter-over-quarter institutional change) was removed rather than
# shipped. Managers reorganise which subsidiary files: between 2025-09-30 and
# 2026-03-31 VANGUARD GROUP INC stopped filing and five Vanguard entities began,
# so a comparison of filer positions reported +5,474% for a holding where
# nothing had moved. Filer-level names are not stable across quarters and no
# grouping heuristic can separate a restructured filing from real accumulation.
# A correct version needs position-level continuity the 13F data does not carry.


async def notable_holders(pool, cusip: str, shares_outstanding: float | None = None) -> dict:
    """Discretionary managers holding this company, with conviction and direction.

    Index funds hold every large company by construction, so their presence says
    nothing. A manager who chose the position is different, and three things make
    that choice legible: how much of the company they hold, how much of THEIR OWN
    book it represents, and whether they added or cut it last quarter. A name
    that is 8% of Berkshire's portfolio is a statement; the same dollar value
    inside Vanguard's index is arithmetic.
    """
    async with pool.acquire() as conn:
        qs = [r["quarter"] for r in await conn.fetch(
            "SELECT quarter FROM holdings_13f_meta WHERE quarter NOT LIKE 'file:%' ORDER BY quarter DESC LIMIT 8")]
    if not qs:
        return {"available": False, "reason": "no 13F quarter loaded"}

    ordered = sorted(qs, reverse=True)
    now_q = ordered[0]

    async with pool.acquire() as conn:
        held = await conn.fetch(
            """SELECT manager, sum(shares)::bigint sh, sum(value_usd)::bigint val
               FROM holdings_13f WHERE quarter=$1 AND cusip=$2 GROUP BY manager""",
            now_q, cusip)
        # Each notable manager's total book, so the position can be expressed as
        # a share of what they actually run.
        books = {}
        names = {r["manager"] for r in held if _notable(r["manager"])}
        if names:
            for r in await conn.fetch(
                """SELECT manager, sum(value_usd)::bigint total FROM holdings_13f
                   WHERE quarter=$1 AND manager = ANY($2::text[]) GROUP BY manager""",
                now_q, list(names)):
                books[r["manager"]] = int(r["total"] or 0)

    out = {}
    for r in held:
        label = _notable(r["manager"])
        if not label:
            continue
        e = out.setdefault(label, {"manager": label, "shares": 0, "value_usd": 0,
                                   "book_usd": 0})
        e["shares"] += int(r["sh"] or 0)
        e["value_usd"] += int(r["val"] or 0)
        e["book_usd"] += books.get(r["manager"], 0)

    rows = []
    for e in out.values():
        rows.append({
            "manager": e["manager"],
            "shares": e["shares"],
            "value_usd": e["value_usd"],
            "pct_of_company": round(e["shares"] / shares_outstanding * 100, 3) if shares_outstanding else None,
            "pct_of_their_book": round(e["value_usd"] / e["book_usd"] * 100, 2) if e["book_usd"] else None,
            # No quarter-over-quarter action: see the note above flow_for.
        })
    rows.sort(key=lambda x: -(x["value_usd"] or 0))

    return {"available": True, "quarter": now_q, 
            "n_notable": len(rows), "holders": rows[:12]}


async def concentration_for(pool, cusip: str, shares_outstanding: float | None) -> dict:
    """Ownership concentration measured from 13F filings.

    These figures previously came from Schedule 13D/G, which is filed only when a
    holder crosses 5%. For Microsoft nobody has, so five signals were blank; for
    Apple two filings existed and the numbers described those two rather than the
    ownership structure. 13F covers every manager over $100M, so the same
    questions — how many holders, how concentrated, how large is the largest —
    can be answered from thousands of positions instead of a handful.
    """
    async with pool.acquire() as conn:
        q = await conn.fetchval(
            "SELECT max(quarter) FROM holdings_13f_meta "
            "WHERE quarter NOT LIKE 'file:%' AND n_managers >= 2000")
        if not q:
            return {"available": False, "reason": "no covered quarter loaded"}
        rows = await conn.fetch(
            """SELECT manager, sum(shares)::bigint sh FROM holdings_13f
               WHERE quarter=$1 AND cusip=$2 GROUP BY manager""", q, cusip)
    if not rows:
        return {"available": False, "reason": "no 13F positions for this CUSIP"}

    grouped: dict[str, int] = {}
    for r in rows:
        k = _family(r["manager"])
        grouped[k] = grouped.get(k, 0) + int(r["sh"] or 0)
    sizes = sorted(grouped.values(), reverse=True)
    total = sum(sizes)
    if not total:
        return {"available": False, "reason": "positions sum to zero"}

    def pct_of_co(n):
        return (n / shares_outstanding * 100) if shares_outstanding else None

    # Herfindahl over institutional holders: 0 = spread across many, 1 = one holder.
    hhi = sum((s / total) ** 2 for s in sizes)
    above5 = sum(1 for s in sizes if shares_outstanding and s / shares_outstanding > 0.05)

    return {
        "available": True, "quarter": q,
        "holder_families": len(sizes),
        "top_holder_pct": round(pct_of_co(sizes[0]), 2) if pct_of_co(sizes[0]) is not None else None,
        "top3_pct": round(pct_of_co(sum(sizes[:3])), 2) if pct_of_co(sum(sizes[:3])) is not None else None,
        "top10_pct": round(pct_of_co(sum(sizes[:10])), 2) if pct_of_co(sum(sizes[:10])) is not None else None,
        "avg_holder_pct": round(pct_of_co(total / len(sizes)), 4) if pct_of_co(total / len(sizes)) is not None else None,
        "n_above_5pct": above5,
        "hhi": round(hhi, 4),
        "concentration_note": (
            f"{len(sizes):,} institutional holders; the largest holds "
            f"{round(pct_of_co(sizes[0]),1) if shares_outstanding else '—'}% of the company "
            f"and the top ten hold {round(pct_of_co(sum(sizes[:10])),1) if shares_outstanding else '—'}%."),
    }
