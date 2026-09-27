"""Company Intelligence — market attention from the Polygon news feed.

One pass per day over ALL news (not one call per ticker): each article
carries a tickers[] array, so counting mentions gives the whole universe's
attention series from a few paginated calls. Complete UTC days only; rows
append-only. A5 of the frozen contract: attention is SECONDARY evidence,
descriptive. It exists for one comparison — fundamental-event activity vs
attention — which is a research trigger, never a score.

attention_spike event: 7-day article count >= 3x the company's own trailing
90-day weekly average, with at least 10 articles (so a jump from 1 to 3
isn't a 'spike'). available_at = end of the day the threshold was crossed.
"""
from __future__ import annotations
import asyncio, hashlib, json, os
from datetime import date, datetime, timedelta, timezone
import httpx
from loguru import logger

POLY = os.environ.get("POLYGON_API_KEY", "")
SPIKE_X, SPIKE_MIN = 3.0, 10

SQL = """
CREATE TABLE IF NOT EXISTS ci_attention_daily (
    ticker        TEXT NOT NULL,
    d             DATE NOT NULL,
    n_articles    INT  NOT NULL,
    n_publishers  INT  NOT NULL,
    retrieved_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ticker, d)
);
CREATE INDEX IF NOT EXISTS idx_ci_att_d ON ci_attention_daily (d);
"""


async def _day(client, d: date) -> dict[str, dict]:
    counts: dict[str, dict] = {}
    url = (f"https://api.polygon.io/v2/reference/news?published_utc.gte={d.isoformat()}"
           f"&published_utc.lt={(d + timedelta(days=1)).isoformat()}&limit=1000&order=asc&sort=published_utc&apiKey={POLY}")
    pages = 0
    while url and pages < 40:
        r = await client.get(url)
        if r.status_code == 429:
            await asyncio.sleep(15); r = await client.get(url)
        if r.status_code != 200:
            raise RuntimeError(f"news {d}: HTTP {r.status_code}")
        j = r.json() or {}
        for a in j.get("results", []):
            pub = (a.get("publisher") or {}).get("name") or "?"
            for tk in a.get("tickers") or []:
                c = counts.setdefault(tk, {"n": 0, "pubs": set()})
                c["n"] += 1; c["pubs"].add(pub)
        nxt = j.get("next_url"); url = f"{nxt}&apiKey={POLY}" if nxt else None
        pages += 1
        await asyncio.sleep(0.25)
    return counts


async def ingest_attention(pool, backfill_days: int = 100) -> dict:
    async with pool.acquire() as c:
        await c.execute(SQL)
        universe = {r["ticker"] for r in await c.fetch("SELECT ticker FROM universe WHERE active AND cik IS NOT NULL")}
        have = {r["d"] for r in await c.fetch("SELECT DISTINCT d FROM ci_attention_daily")}
    end = datetime.now(timezone.utc).date() - timedelta(days=1)     # complete days only
    days = [end - timedelta(days=i) for i in range(backfill_days) if (end - timedelta(days=i)) not in have]
    stats = {"days_new": 0, "rows_new": 0, "spikes_new": 0}
    async with httpx.AsyncClient(timeout=40) as client:
        for d in sorted(days):
            counts = await _day(client, d)
            rows = [(tk, d, v["n"], len(v["pubs"])) for tk, v in counts.items() if tk in universe]
            async with pool.acquire() as c:
                await c.executemany("""INSERT INTO ci_attention_daily (ticker, d, n_articles, n_publishers)
                                       VALUES ($1,$2,$3,$4) ON CONFLICT (ticker, d) DO NOTHING""", rows)
            stats["days_new"] += 1; stats["rows_new"] += len(rows)
    # Spike detection on the most recent complete day, vs each company's own history.
    # Refuses to run until the series covers the full baseline window: with no
    # history, prior90 = 0 and every covered company looked like a spike.
    async with pool.acquire() as c:
        oldest = await c.fetchval("SELECT min(d) FROM ci_attention_daily")
        if oldest is None or oldest > end - timedelta(days=96):
            logger.info(f"[ci/attention] spike detection skipped: history starts {oldest}, need 97 days")
            logger.info(f"[ci/attention] {stats}")
            return stats
        spikes = await c.fetch("""
            WITH w AS (SELECT ticker,
                         sum(n_articles) FILTER (WHERE d > $1::date - 7)  AS last7,
                         sum(n_articles) FILTER (WHERE d <= $1::date - 7 AND d > $1::date - 97) AS prior90
                       FROM ci_attention_daily WHERE d > $1::date - 97 GROUP BY ticker)
            SELECT w.ticker, u.cik, w.last7, w.prior90 FROM w JOIN universe u USING (ticker)
            WHERE w.prior90 > 0 AND w.last7 >= $2 AND w.last7 >= $3 * w.prior90 / 90.0 * 7""",
            end, SPIKE_MIN, SPIKE_X)
        avail = datetime.combine(end + timedelta(days=1), datetime.min.time(), timezone.utc)
        for s in spikes:
            base_wk = round((s["prior90"] or 0) / 90 * 7, 1)
            payload = {"date": end.isoformat(), "last7": s["last7"], "prior90": s["prior90"],
                       "baseline_weekly": base_wk, "rule": f">= {SPIKE_X}x own 90d weekly avg and >= {SPIKE_MIN} articles"}
            h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            ev = await c.fetchval("""INSERT INTO ci_raw_evidence (source_type, source_id, form_type, cik, filed_at, raw_payload, content_hash)
                                     VALUES ('POLYGON_NEWS',$1,'NEWS-AGG',$2,$3,$4::jsonb,$5)
                                     ON CONFLICT (source_type, source_id, content_hash) DO NOTHING RETURNING id""",
                                  f"att:{s['ticker']}:{end}", str(s["cik"]), avail, json.dumps(payload), h)
            if ev is None: continue
            eid = await c.fetchval("""INSERT INTO ci_events (company_id, ticker, event_date, available_at, evidence_id, item_code, event_type, significance, title)
                                      VALUES ($1,$2,$3,$4,$5,'NEWS','attention_spike','RELEVANT',$6)
                                      ON CONFLICT (evidence_id, item_code) DO NOTHING RETURNING id""",
                                   str(s["cik"]), s["ticker"], end, avail, ev,
                                   f"Attention spike: {s['last7']} articles in 7d vs {base_wk}/wk baseline")
            if eid: stats["spikes_new"] += 1
    logger.info(f"[ci/attention] {stats}")
    return stats
