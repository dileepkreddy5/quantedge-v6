"""Company Intelligence — capital allocation from the XBRL bulk on disk.

The question: what is management actually spending on, quarter by quarter,
before the payoff shows in the income statement? Six cash-flow tag families
read from companyfacts.zip (already on the edgar_data volume), every value
traceable to a filing accession.

PIT: available_at = the 'filed' date the SEC attaches to each fact, at day
granularity (acceptance time lives in the submissions index, which is not
joined here — disclosed, not hidden). event_date = period end.

Q4 honesty: cash-flow items in a 10-K are annual, so Q4 has no quarterly
frame. Where all of CYyyyy and CYyyyyQ1..Q3 exist, Q4 = annual - 9M and is
written as DERIVED with method='annual_minus_9m'. Otherwise Q4 is absent.

Significance is a stated threshold, not a judgment: a quarter where CapEx
or R&D exceeds 1.4x its own trailing four-quarter average is RELEVANT.
"""
from __future__ import annotations
import asyncio, hashlib, json, re
from datetime import datetime, date, timezone
from loguru import logger
from quantedge.fundamentals.edgar_bulk import company_facts_from_bulk

TAGS = {
    "capex":      ("PaymentsToAcquirePropertyPlantAndEquipment", "Capital expenditures"),
    "rnd":        ("ResearchAndDevelopmentExpense",              "R&D expense"),
    "buybacks":   ("PaymentsForRepurchaseOfCommonStock",         "Share repurchases"),
    "debt_issued":("ProceedsFromIssuanceOfLongTermDebt",         "Long-term debt issued"),
    "debt_repaid":("RepaymentsOfLongTermDebt",                   "Long-term debt repaid"),
    "dividends":  ("PaymentsOfDividends",                        "Dividends paid"),
    "sbc":        ("ShareBasedCompensation",                     "Stock-based compensation"),
}
SPIKE = 1.4
EXTRACTOR = "xbrl-capital-v1"
_Q = re.compile(r"^CY(\d{4})Q([1-4])$"); _A = re.compile(r"^CY(\d{4})$")


def _rows(facts: dict, tag: str) -> list[dict]:
    """One row per PERIOD, anchored to the EARLIEST filing that reported it.

    companyfacts repeats each period in every later filing's comparatives and
    hangs SEC's 'frame' label on the latest one; taking that row would stamp
    old quarters with new filing dates — a 13-month phantom lag on the first
    test run. First-report is the honest available_at."""
    try:
        units = facts["facts"]["us-gaap"][tag]["units"]
    except KeyError:
        return []
    by_period: dict[tuple, dict] = {}
    frame_of: dict[tuple, str] = {}
    for r in units.get("USD", []):
        if r.get("form") not in ("10-Q", "10-K") or r.get("val") is None or not r.get("start"):
            continue
        key = (r["start"], r["end"])
        if r.get("frame"):
            frame_of[key] = r["frame"]
        cur = by_period.get(key)
        if cur is None or r["filed"] < cur["filed"]:
            by_period[key] = {"val": float(r["val"]), "end": r["end"], "start": r["start"],
                              "filed": r["filed"], "accn": r["accn"], "form": r["form"]}
    out = []
    for key, r in by_period.items():
        fr = frame_of.get(key)
        if not fr:
            continue                     # no SEC frame → ambiguous duration; skip
        out.append({**r, "frame": fr})
    return out


def _quarterly_series(rows: list[dict]) -> dict[str, dict]:
    """frame -> row, with Q4 derived from annual - 9M where possible."""
    q = {r["frame"]: r for r in rows if _Q.match(r["frame"])}
    ann = {r["frame"]: r for r in rows if _A.match(r["frame"])}
    for fr, r in ann.items():
        y = fr[2:]
        qs = [q.get(f"CY{y}Q{i}") for i in (1, 2, 3)]
        if all(qs) and f"CY{y}Q4" not in q:
            q[f"CY{y}Q4"] = {**r, "frame": f"CY{y}Q4",
                             "val": r["val"] - sum(x["val"] for x in qs),
                             "derived_method": "annual_minus_9m"}
    return q


def _sha(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


async def ingest_capital(pool, limit: int | None = None) -> dict:
    companies = await pool.fetch("""
        SELECT u.ticker, u.cik FROM universe u
        JOIN (SELECT ticker FROM daily_bars GROUP BY ticker HAVING count(*) >= 500) b USING (ticker)
        WHERE u.cik IS NOT NULL AND u.active
        ORDER BY u.market_cap DESC NULLS LAST LIMIT $1""", limit if limit else 100000)
    stats = {"companies": len(companies), "no_facts": 0, "filings_new": 0, "events_new": 0,
             "derived_new": 0, "relevant": 0}
    for co in companies:
        tk, cik = co["ticker"], str(co["cik"])
        facts = await asyncio.to_thread(company_facts_from_bulk, cik)
        if not facts:
            stats["no_facts"] += 1; continue
        series = {k: _quarterly_series(_rows(facts, tag)) for k, (tag, _) in TAGS.items()}
        # Group every quarterly value by the filing (accession) that carried it.
        by_accn: dict[str, dict] = {}
        for k, ser in series.items():
            for fr, r in ser.items():
                a = by_accn.setdefault(r["accn"], {"form": r["form"], "filed": r["filed"],
                                                   "end": r["end"], "values": {}})
                a["values"].setdefault(k, {})[fr] = {"val": r["val"], "end": r["end"],
                                                     **({"derived_method": r["derived_method"]} if "derived_method" in r else {})}
        async with pool.acquire() as conn:
            known = {row["source_id"] for row in await conn.fetch(
                "SELECT source_id FROM ci_raw_evidence WHERE cik=$1 AND source_type='SEC_XBRL'", cik)}
            for accn, a in by_accn.items():
                if accn in known:
                    continue
                payload = {"accession": accn, "form": a["form"], "filed": a["filed"], "values": a["values"]}
                filed_at = datetime.combine(date.fromisoformat(a["filed"]), datetime.min.time(), timezone.utc)
                ev_id = await conn.fetchval("""
                    INSERT INTO ci_raw_evidence (source_type, source_id, form_type, cik, filed_at, raw_payload, content_hash)
                    VALUES ('SEC_XBRL', $1, $2, $3, $4, $5::jsonb, $6)
                    ON CONFLICT (source_type, source_id, content_hash) DO NOTHING RETURNING id""",
                    accn, a["form"], cik, filed_at, json.dumps(payload), _sha(payload))
                if ev_id is None:
                    continue
                stats["filings_new"] += 1
                # Significance: latest quarter in this filing vs trailing 4 for capex/rnd.
                sig, flags = "ROUTINE", {}
                for k in ("capex", "rnd"):
                    frames = sorted(a["values"].get(k, {}).keys())
                    if not frames: continue
                    latest = frames[-1]
                    allq = sorted(f for f in series[k] if _Q.match(f) and f < latest)[-4:]
                    if len(allq) == 4:
                        avg = sum(series[k][f]["val"] for f in allq) / 4
                        cur = a["values"][k][latest]["val"]
                        if avg > 0:
                            flags[f"{k}_vs_trailing4"] = round(cur / avg, 2)
                            if cur > SPIKE * avg: sig = "RELEVANT"
                _pref = [k for k in ("capex", "rnd", "buybacks", "dividends", "debt_issued", "debt_repaid", "sbc")
                         if k in a["values"]][:3]
                title = f"{a['form']} cash-flow: " + ", ".join(
                    f"{TAGS[k][1]} ${max(a['values'][k].values(), key=lambda x: x['end'])['val']/1e6:,.0f}M"
                    for k in _pref)
                eid = await conn.fetchval("""
                    INSERT INTO ci_events (company_id, ticker, event_date, available_at, evidence_id, item_code, event_type, significance, title)
                    VALUES ($1,$2,$3,$4,$5,'XBRL','capital_allocation',$6,$7)
                    ON CONFLICT (evidence_id, item_code) DO NOTHING RETURNING id""",
                    cik, tk, date.fromisoformat(a["end"]), filed_at, ev_id, sig, title[:300])
                if not eid: continue
                stats["events_new"] += 1
                if sig == "RELEVANT": stats["relevant"] += 1
                for k, frames in a["values"].items():
                    for fr, v in frames.items():
                        await conn.execute("""
                            INSERT INTO ci_derived (event_id, layer, extractor_version, field, value, citation, confidence, status)
                            VALUES ($1,'DERIVED',$2,$3,$4::jsonb,$5::jsonb,$6,'observed')""",
                            eid, EXTRACTOR, k,
                            json.dumps({"frame": fr, "value_usd": v["val"], "period_end": v["end"],
                                        **({"method": v["derived_method"]} if "derived_method" in v else {})}),
                            json.dumps({"accession": accn, "document": a["form"], "location": f"us-gaap:{TAGS[k][0]}",
                                        "frame": fr}),
                            "medium" if "derived_method" in v else "high")
                        stats["derived_new"] += 1
                for f, ratio in flags.items():
                    await conn.execute("""
                        INSERT INTO ci_derived (event_id, layer, extractor_version, field, value, citation, confidence, status)
                        VALUES ($1,'DERIVED',$2,$3,$4::jsonb,$5::jsonb,'high','observed')""",
                        eid, EXTRACTOR, f, json.dumps({"ratio": ratio, "threshold": SPIKE}),
                        json.dumps({"accession": accn, "document": a["form"], "location": "computed vs trailing 4 quarterly frames"}))
    logger.info(f"[ci/xbrl] {stats}")
    return stats
