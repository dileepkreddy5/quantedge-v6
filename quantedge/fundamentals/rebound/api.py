"""Rebound tracker — live API (your thesis as a running product).

Serves the discounted-quality shortlist (down hard + strong financials) and
tracks each pick's progress back toward its prior high. This is the buy-low-
good-financials-wait-for-recovery idea, live on the site.

Endpoints (mounted under /api/v6):
  GET  /rebound/list                -> current shortlist by tier, with the
                                       recovery target and progress for each
  GET  /rebound/pick/{ticker}       -> one name's full thesis + recovery track
  GET  /rebound/tracker             -> everything being tracked, sorted by
                                       progress-to-recovery

Honesty is built into the payload: every response carries a plain-language
disclaimer that this is a research shortlist, not investment advice, and that
the recovery thesis has not been validated across a bear market on the current
data window. No fabricated numbers — recovery progress is computed live from
the price store, and prior-high targets from real history.
"""
from __future__ import annotations
import json, os
from datetime import date, timedelta
from typing import Dict, List, Optional

DISCLAIMER = ("Research shortlist, not investment advice. These are companies "
              "that are down significantly from prior highs but show strong "
              "financials. The recovery thesis has not been validated across a "
              "full bear-market cycle on the available data window. Do your own "
              "research.")


def _load_artifact(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path))
    except Exception:
        return None


def recovery_progress(entry_price: float, current_price: float,
                      prior_high: float) -> Dict:
    """How far back toward the prior high has it come since entry? Real math,
    no fabrication. progress_pct = fraction of the entry->high gap closed."""
    if prior_high <= entry_price or entry_price <= 0:
        return {"progress_pct": None, "reached_high": current_price >= prior_high}
    gap = prior_high - entry_price
    closed = current_price - entry_price
    return {
        "progress_pct": round(max(0.0, min(1.0, closed / gap)) * 100, 1),
        "reached_high": current_price >= prior_high,
        "upside_to_high_pct": round((prior_high / current_price - 1) * 100, 1)
                              if current_price > 0 else None,
    }


def build_list_payload(artifact: Dict, price_lookup=None) -> Dict:
    """Shape the scan artifact into the list response, adding live recovery
    progress per name when a price_lookup is provided."""
    tiers = artifact.get("tiers", {})
    out_tiers = {}
    for tier_name, rows in tiers.items():
        out_rows = []
        for r in rows:
            row = {
                "ticker": r["ticker"], "name": r.get("name"),
                "score": r["score"], "stage": r["stage"], "tier": r["tier"],
                "drawdown_from_high_pct": round(r["drawdown"] * 100, 1)
                    if r.get("drawdown") is not None else None,
                "thesis": r.get("thesis"),
                "entry_price": r.get("price"),
            }
            if price_lookup and r.get("prior_high") and r.get("price"):
                cur = price_lookup(r["ticker"])
                if cur:
                    row["current_price"] = cur
                    row["recovery"] = recovery_progress(
                        r["price"], cur, r["prior_high"])
            out_rows.append(row)
        out_tiers[tier_name] = out_rows
    return {
        "as_of": artifact.get("as_of"),
        "generated": artifact.get("generated"),
        "tiers": out_tiers,
        "stage_counts": artifact.get("stage_counts"),
        "total_passed": artifact.get("total_passed"),
        "disclaimer": DISCLAIMER,
    }


def register_routes(router, artifact_path: str, price_store=None):
    """Attach rebound endpoints to an existing FastAPI APIRouter."""
    def _price_lookup(ticker):
        if not price_store:
            return None
        try:
            bars = price_store.series(ticker, date.today() - timedelta(days=10),
                                      date.today())
            return bars[-1][1] if bars else None
        except Exception:
            return None

    @router.get("/rebound/list")
    def rebound_list():
        art = _load_artifact(artifact_path)
        if not art:
            return {"error": "no scan available yet", "disclaimer": DISCLAIMER}
        return build_list_payload(art, _price_lookup)

    @router.get("/rebound/pick/{ticker}")
    def rebound_pick(ticker: str):
        art = _load_artifact(artifact_path)
        if not art:
            return {"error": "no scan available yet", "disclaimer": DISCLAIMER}
        ticker = ticker.upper()
        for rows in art.get("tiers", {}).values():
            for r in rows:
                if r["ticker"] == ticker:
                    payload = build_list_payload({"tiers": {"x": [r]}}, _price_lookup)
                    return {"pick": payload["tiers"]["x"][0], "disclaimer": DISCLAIMER}
        return {"error": f"{ticker} not in current shortlist", "disclaimer": DISCLAIMER}

    @router.get("/rebound/tracker")
    def rebound_tracker():
        art = _load_artifact(artifact_path)
        if not art:
            return {"error": "no scan available yet", "disclaimer": DISCLAIMER}
        payload = build_list_payload(art, _price_lookup)
        allrows = [r for rows in payload["tiers"].values() for r in rows]
        tracked = [r for r in allrows if r.get("recovery")]
        tracked.sort(key=lambda r: r["recovery"].get("progress_pct") or -1, reverse=True)
        return {"tracked": tracked, "count": len(tracked), "disclaimer": DISCLAIMER}

    return router
