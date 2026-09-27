"""Company Intelligence — institutional ownership from the 13F bulk dataset.

holdings_13f is SEC's aggregated 13F data: positions by (quarter, CUSIP,
manager). No ticker, no per-filing date. Two honest consequences:

  Entity resolution: ticker -> CUSIP by normalized-name match against the
  quarter's issuers (same method the ownership tab uses). Every match is
  recorded in ci_entity_map with match_method='alias' (prefix) or 'exact'
  (normalized equality) and MEDIUM/HIGH confidence — inspectable, revocable.

  Point-in-time: as_of = quarter end (event_date); available_at = quarter
  end + 45 days, the 13F filing deadline by which every position in the
  set was public. Conservative by design: no backtest can see a position
  before the last filer disclosed it. Stated in every event's citation.

One institutional_snapshot event per (ticker, quarter): manager count,
total shares/value, and the quarter-over-quarter change — new managers,
exits, net shares — all DERIVED with citations to the dataset quarter.
"""
from __future__ import annotations
import hashlib, json, re
from datetime import date, datetime, timedelta, timezone
from loguru import logger

DEADLINE_DAYS = 45
EXTRACTOR = "13f-bulk-v1"
_SUFFIX = re.compile(r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings?|group|the|class [abc]|common stock|cl [abc])\b\.?", re.I)


def _norm(name: str) -> str:
    n = _SUFFIX.sub(" ", (name or "").lower())
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def _sha(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


async def ingest_13f(pool, limit: int | None = None) -> dict:
    stats = {"companies": 0, "matched": 0, "exact": 0, "alias": 0, "unmatched": 0,
             "events_new": 0, "derived_new": 0}
    async with pool.acquire() as c:
        await c.execute("CREATE INDEX IF NOT EXISTS idx_13f_q_cusip ON holdings_13f (quarter, cusip)")
        quarters = [r["quarter"] for r in await c.fetch(
            "SELECT quarter FROM holdings_13f_meta WHERE quarter NOT LIKE 'file:%' "
            "AND n_managers >= 2000 ORDER BY quarter DESC LIMIT 2")]
        # Partial loads (a handful of managers) made Apple look like 1 holder and
        # every QoQ change look like thousands of 'new' managers. Complete quarters only.
        if not quarters:
            return {"error": "no 13F quarter loaded"}
        q_new, q_old = quarters[0], (quarters[1] if len(quarters) > 1 else None)
        companies = await c.fetch("""
            SELECT u.ticker, u.cik, u.name FROM universe u
            JOIN (SELECT ticker FROM daily_bars GROUP BY ticker HAVING count(*) >= 500) b USING (ticker)
            WHERE u.cik IS NOT NULL AND u.active AND u.name IS NOT NULL
            ORDER BY u.market_cap DESC NULLS LAST""" + (f" LIMIT {int(limit)}" if limit else ""))
        # Issuer directory for the latest quarter, once: ~10k issuers, not 5k scans.
        issuers = await c.fetch("""
            SELECT cusip, max(issuer) AS issuer, sum(shares) AS total_shares
            FROM holdings_13f WHERE quarter=$1 GROUP BY cusip""", q_new)
    stats["companies"] = len(companies)
    by_norm: dict[str, list] = {}
    for r in issuers:
        by_norm.setdefault(_norm(r["issuer"]), []).append(r)

    # One matcher, not two: reuse the OWNERSHIP tab's ownership_for (stem
    # LIKE-prefix, largest holding wins), which resolves the messy issuer
    # names 13F actually uses ("AppleComputerInc", "Amazoncom Inc"). A
    # second, stricter matcher here picked Apple Hospitality REIT for AAPL.
    from services.holdings_13f import ownership_for
    async def resolve(tk: str, name: str):
        r = await ownership_for(pool, tk, name, None)
        if not r.get("available") or not r.get("cusip"):
            return None
        return {"cusip": r["cusip"], "issuer": r.get("issuer") or ""}, "alias", "MEDIUM"

    # The entity map doubles as a cache: a company resolved once keeps its
    # CUSIP until a row is superseded or rejected. Without this, every night
    # re-ran ~4,000 unindexable LIKE scans for answers that change quarterly.
    async with pool.acquire() as c:
        known = {r["ticker"]: ({"cusip": r["external_id"], "issuer": r["external_name"]},
                               r["match_method"], r["match_confidence"])
                 for r in await c.fetch("""SELECT ticker, external_id, external_name, match_method, match_confidence
                                           FROM ci_entity_map WHERE source_type='SEC_13F' AND status='active'""")}
    stats["cached"] = 0
    now = datetime.now(timezone.utc)
    for co in companies:
        tk, cik = co["ticker"], str(co["cik"])
        if tk in known:
            hit = known[tk]; stats["cached"] += 1
        else:
            hit = await resolve(tk, co["name"])
        if not hit:
            stats["unmatched"] += 1; continue
        best, method, conf = hit
        stats["matched"] += 1; stats[method] += 1
        cusip, issuer = best["cusip"], best["issuer"]
        async with pool.acquire() as c:
            await c.execute("""
                INSERT INTO ci_entity_map (company_id, ticker, entity_type, source_type, external_id,
                                           external_name, match_method, match_confidence, verified_at)
                SELECT $1,$2,'cusip','SEC_13F',$3,$4,$5,$6,NULL
                WHERE NOT EXISTS (SELECT 1 FROM ci_entity_map
                                  WHERE company_id=$1 AND source_type='SEC_13F' AND external_id=$3 AND status='active')""",
                cik, tk, cusip, issuer, method, conf)
            # Snapshot + QoQ change in one query via full outer join on manager.
            snap = await c.fetchrow("""
                WITH n AS (SELECT manager, shares, value_usd FROM holdings_13f WHERE quarter=$1 AND cusip=$3),
                     o AS (SELECT manager, shares FROM holdings_13f WHERE quarter=$2 AND cusip=$3)
                SELECT count(n.manager) AS managers, sum(n.shares) AS shares, sum(n.value_usd) AS value_usd,
                       count(*) FILTER (WHERE n.manager IS NOT NULL AND o.manager IS NULL) AS new_managers,
                       count(*) FILTER (WHERE n.manager IS NULL AND o.manager IS NOT NULL) AS exited_managers,
                       sum(coalesce(n.shares,0)) - sum(coalesce(o.shares,0)) AS net_share_change,
                       (SELECT count(*) FROM o) AS prev_managers
                FROM n FULL OUTER JOIN o ON o.manager = n.manager""", q_new, q_old or "", cusip)
            if not snap or not snap["managers"]:
                continue
            q_end = date.fromisoformat(q_new)
            avail = datetime.combine(q_end + timedelta(days=DEADLINE_DAYS), datetime.min.time(), timezone.utc)
            payload = {"quarter": q_new, "prev_quarter": q_old, "cusip": cusip, "issuer": issuer,
                       "managers": snap["managers"], "shares": int(snap["shares"] or 0),
                       "value_usd": int(snap["value_usd"] or 0),
                       "new_managers": snap["new_managers"], "exited_managers": snap["exited_managers"],
                       "net_share_change": int(snap["net_share_change"] or 0),
                       "prev_managers": snap["prev_managers"]}
            ev_id = await c.fetchval("""
                INSERT INTO ci_raw_evidence (source_type, source_id, form_type, cik, filed_at, raw_payload, content_hash)
                VALUES ('SEC_13F', $1, '13F-AGG', $2, $3, $4::jsonb, $5)
                ON CONFLICT (source_type, source_id, content_hash) DO NOTHING RETURNING id""",
                f"13F:{q_new}:{cusip}", cik, avail, json.dumps(payload), _sha(payload))
            if ev_id is None:
                continue
            chg = payload["net_share_change"]; pm = payload["prev_managers"] or 0
            sig = "RELEVANT" if (pm and (payload["new_managers"] + payload["exited_managers"]) > 0.15 * pm) else "ROUTINE"
            title = (f"13F {q_new}: {payload['managers']:,} managers, {payload['shares']/1e6:,.1f}M sh"
                     + (f" ({'+' if chg >= 0 else ''}{chg/1e6:,.1f}M vs {q_old} · +{payload['new_managers']} new / -{payload['exited_managers']} exited)" if q_old else ""))
            eid = await c.fetchval("""
                INSERT INTO ci_events (company_id, ticker, event_date, available_at, evidence_id, item_code, event_type, significance, title)
                VALUES ($1,$2,$3,$4,$5,'13F','institutional_snapshot',$6,$7)
                ON CONFLICT (evidence_id, item_code) DO NOTHING RETURNING id""",
                cik, tk, q_end, avail, ev_id, sig, title[:300])
            if not eid:
                continue
            stats["events_new"] += 1
            cite = {"dataset": "SEC 13F bulk", "quarter": q_new, "cusip": cusip,
                    "pit_rule": f"available_at = quarter end + {DEADLINE_DAYS}d filing deadline (conservative)",
                    "entity_match": {"method": method, "confidence": conf, "issuer": issuer}}
            for field in ("managers", "shares", "value_usd", "new_managers", "exited_managers", "net_share_change"):
                await c.execute("""
                    INSERT INTO ci_derived (event_id, layer, extractor_version, field, value, citation, confidence, status)
                    VALUES ($1,'DERIVED',$2,$3,$4::jsonb,$5::jsonb,$6,'observed')""",
                    eid, EXTRACTOR, field, json.dumps({"value": payload[field], "as_of": q_new}),
                    json.dumps(cite), "high" if method == "exact" else "medium")
                stats["derived_new"] += 1
    logger.info(f"[ci/13f] {stats}")
    return stats
