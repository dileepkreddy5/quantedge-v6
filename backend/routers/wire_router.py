"""The Wire — what QuantEdge detected in the last N hours, across every filer,
as a timestamped feed. Union of Company Intelligence events (with their public
timestamps), attention spikes, insider clusters, patterns that completed on
the last session, and new listings. Every line carries its evidence tier."""
from __future__ import annotations
import json
from fastapi import APIRouter, Query, Request
from core.artifact_paths import artifact_read_path

router = APIRouter()


@router.get("/wire")
async def wire(request: Request, hours: int = Query(24, ge=1, le=168), limit: int = Query(80, ge=10, le=300)):
    pool = request.app.state.db
    items = []
    ev = await pool.fetch("""
        SELECT e.ticker, e.event_type, e.significance, e.title, e.available_at, e.item_code, r.source_type
        FROM ci_events e JOIN ci_raw_evidence r ON r.id = e.evidence_id
        WHERE e.available_at > NOW() - ($1 || ' hours')::interval
          AND (e.significance = 'MATERIAL' OR e.event_type IN ('attention_spike','institutional_snapshot'))
        ORDER BY e.available_at DESC LIMIT 200""", str(hours))
    TIER = {"SEC": "PRIMARY", "SEC_XBRL": "PRIMARY", "SEC_13F": "PRIMARY", "POLYGON_NEWS": "SECONDARY"}
    for r in ev:
        items.append({"ts": r["available_at"].isoformat(), "ticker": r["ticker"], "kind": r["event_type"],
                      "tier": TIER.get(r["source_type"], "SECONDARY"), "title": r["title"], "significance": r["significance"]})
    ins = await pool.fetch("""
        SELECT e.ticker, e.available_at, d.value FROM ci_derived d JOIN ci_events e ON e.id = d.event_id
        WHERE d.field='transaction' AND e.available_at > NOW() - ($1 || ' hours')::interval""", str(hours))
    agg = {}
    for r in ins:
        v = json.loads(r["value"]) if isinstance(r["value"], str) else r["value"]
        if not v.get("open_market"): continue
        a = agg.setdefault(r["ticker"], {"buy": 0.0, "sell": 0.0, "ts": r["available_at"]})
        a["buy" if v.get("code") == "P" else "sell"] += v.get("value", 0); a["ts"] = max(a["ts"], r["available_at"])
    for tk, a in agg.items():
        if a["buy"] >= 1e6 or a["sell"] >= 5e6:
            side = "buying" if a["buy"] >= a["sell"] else "selling"; amt = max(a["buy"], a["sell"])
            items.append({"ts": a["ts"].isoformat(), "ticker": tk, "kind": "insider_cluster", "tier": "PRIMARY",
                          "title": f"Insider open-market {side}: ${amt/1e6:.1f}M", "significance": "RELEVANT"})
    fp = artifact_read_path("fired_last_night.json")
    if fp:
        f = json.loads(fp.read_text())
        for x in f.get("fired", [])[:60]:
            o = x.get("odds_21d")
            items.append({"ts": f"{f['session']}T21:00:00+00:00", "ticker": x["ticker"], "kind": f"pattern_{x['family']}",
                          "tier": "DERIVED", "significance": "RELEVANT",
                          "title": f"{x['pattern'].replace('_',' ')} completed ({x['direction']})" + (f" — {o['positive_pct']}% up at 21d, n={o['n']:,}" if o else " — odds not yet measured")})
    new = await pool.fetch("""SELECT ticker, min(d) f FROM daily_bars GROUP BY ticker HAVING min(d) > CURRENT_DATE - 7 ORDER BY f DESC LIMIT 10""")
    for r in new:
        items.append({"ts": f"{r['f']}T21:00:00+00:00", "ticker": r["ticker"], "kind": "new_listing", "tier": "PRIMARY",
                      "title": "First session in QuantEdge's price universe", "significance": "ROUTINE"})
    items.sort(key=lambda x: x["ts"], reverse=True)
    return {"hours": hours, "n": len(items), "items": items[:limit],
            "note": "Timestamps are when each fact became public (SEC acceptance) or the session it was measured; DERIVED = computed by QuantEdge from primary data."}


@router.get("/boards/track_record")
async def boards_track_record(request: Request):
    from services.board_cohorts import track_record
    return await track_record(request.app.state.db)
