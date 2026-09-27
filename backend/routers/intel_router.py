"""Company Intelligence — read endpoints over the three-layer CI store.

Read-only. Every event carries its three timestamps; evidence_tier is
mechanical from source_type (SEC = PRIMARY). Families with no adapter yet
are reported as NO_SOURCE — a distinct state from 'nothing found'.
"""
from __future__ import annotations
import json
from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter()
TIER = {"SEC": "PRIMARY"}
FAMILIES = {   # evidence families → adapter status; UI renders NO_SOURCE honestly
    "sec_filings": "active", "insider_transactions": "active",
    "institutional_13f": "planned_session_2", "market_attention": "planned_session_2",
    "patents": "planned_session_3b", "research_papers": "planned_session_3b",
    "customers": "no_source", "government_contracts": "no_source",
    "capacity_utilization": "no_source", "job_postings": "no_source",
    "earnings_transcripts": "no_source",
}


@router.get("/intel/{ticker}/timeline")
async def intel_timeline(ticker: str, request: Request,
                         days: int = Query(365, ge=7, le=1500),
                         significance: str | None = Query(None)):
    pool = getattr(request.app.state, "db", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="database not connected")
    tk = ticker.upper().strip()
    rows = await pool.fetch("""
        SELECT e.id, e.company_id, e.event_date, e.available_at, e.retrieved_at,
               e.item_code, e.event_type, e.significance, e.title,
               r.source_type, r.source_id, r.form_type, r.filed_at
        FROM ci_events e JOIN ci_raw_evidence r ON r.id = e.evidence_id
        WHERE e.ticker = $1 AND e.available_at > NOW() - ($2 || ' days')::interval
          AND ($3::text IS NULL OR e.significance = $3)
        ORDER BY e.available_at DESC LIMIT 400""", tk, str(days), significance)
    if not rows:
        covered = await pool.fetchval("SELECT count(*) FROM ci_events WHERE ticker=$1", tk)
        return {"ticker": tk, "events": [], "covered": bool(covered),
                "note": ("no events in window" if covered else
                         "ticker not yet in the CI ingest universe (top-700 by market cap)"),
                "families": FAMILIES}
    ids = [r["id"] for r in rows]
    derived = await pool.fetch("""
        SELECT event_id, layer, extractor_version, field, value, citation, confidence, status
        FROM ci_derived WHERE event_id = ANY($1::bigint[])""", ids)
    by_event: dict[int, list] = {}
    for d in derived:
        by_event.setdefault(d["event_id"], []).append({
            "layer": d["layer"], "extractor": d["extractor_version"], "field": d["field"],
            "value": json.loads(d["value"]) if isinstance(d["value"], str) else d["value"],
            "citation": json.loads(d["citation"]) if isinstance(d["citation"], str) else d["citation"],
            "confidence": d["confidence"], "status": d["status"]})
    events = []
    for r in rows:
        ev = {"id": r["id"], "event_date": r["event_date"].isoformat() if r["event_date"] else None,
              "available_at": r["available_at"].isoformat(), "retrieved_at": r["retrieved_at"].isoformat(),
              "lag_days": ((r["available_at"].date() - r["event_date"]).days if r["event_date"] else None),
              "item_code": r["item_code"], "event_type": r["event_type"],
              "significance": r["significance"], "title": r["title"],
              "evidence_tier": TIER.get(r["source_type"], "SECONDARY"),
              "source": {"type": r["source_type"], "accession": r["source_id"], "form": r["form_type"],
                         "url": f"https://www.sec.gov/Archives/edgar/data/{int(r['company_id'])}/{r['source_id'].replace('-', '')}/"},
              "derived": by_event.get(r["id"], [])}
        events.append(ev)
    sig_counts = await pool.fetch("""
        SELECT significance, count(*) n FROM ci_events
        WHERE ticker=$1 AND available_at > NOW() - ($2 || ' days')::interval GROUP BY 1""", tk, str(days))
    # Insider net over the window, from DERIVED rows — open-market only.
    ins = await pool.fetch("""
        SELECT d.value FROM ci_derived d JOIN ci_events e ON e.id = d.event_id
        WHERE e.ticker=$1 AND d.field='transaction' AND e.available_at > NOW() - ($2 || ' days')::interval""",
        tk, str(days))
    buy = sell = 0.0; nb = ns = 0
    for r in ins:
        v = json.loads(r["value"]) if isinstance(r["value"], str) else r["value"]
        if not v.get("open_market"): continue
        if v.get("code") == "P": buy += v.get("value", 0); nb += 1
        elif v.get("code") == "S": sell += v.get("value", 0); ns += 1
    return {"ticker": tk, "window_days": days, "events": events,
            "counts": {r["significance"]: r["n"] for r in sig_counts},
            "insider_open_market": {"buys": nb, "buy_value": round(buy), "sells": ns,
                                    "sell_value": round(sell), "net_value": round(buy - sell),
                                    "note": "open-market P/S codes only; grants/exercises excluded. Form 4 ingest is capped at 12 most recent filings per company per run (120d lookback) — heavy filers are undercounted until Session 2 lifts the cap."},
            "families": FAMILIES,
            "pit_note": "available_at is the SEC acceptance timestamp — the moment the information became public. event_date is the economic date. Downstream engines read available_at."}
