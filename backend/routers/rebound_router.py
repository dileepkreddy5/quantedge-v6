"""
QuantEdge v6.0 — Rebound Tracker Router
========================================
Exposes the discounted-quality rebound shortlist and live recovery tracking —
the "buy stocks that are down but have strong financials, track them back to
their prior high" thesis, as a live product.

Endpoints:
  GET  /rebound/list            — current shortlist by tier (small/mid/large)
  GET  /rebound/tracker         — all names sorted by recovery-to-high progress
  GET  /rebound/pick/{ticker}   — one name's thesis + live recovery track

Serves the pre-computed nightly artifact (fast). Recovery progress toward each
name's prior high is computed live from the price store when available.
Every response carries an honest disclaimer.
"""
import json
import os
from datetime import date, timedelta
from typing import Optional
from fastapi import APIRouter
from loguru import logger

router = APIRouter()

ARTIFACT_PATH = os.environ.get(
    "REBOUND_ARTIFACT", "/app/data/rebound_artifact.json")

DISCLAIMER = ("Research shortlist, not investment advice. These are companies "
              "down significantly from prior highs but showing strong "
              "financials. The recovery thesis has not been validated across a "
              "full bear-market cycle on the available data window. Do your own "
              "research.")


def _load() -> Optional[dict]:
    if not os.path.exists(ARTIFACT_PATH):
        return None
    try:
        with open(ARTIFACT_PATH) as fh:
            return json.load(fh)
    except Exception as e:
        logger.warning(f"rebound artifact load failed: {e}")
        return None


def _recovery(entry: float, current: float, high: float) -> dict:
    if high <= entry or entry <= 0:
        return {"progress_pct": None, "reached_high": current >= high}
    return {
        "progress_pct": round(max(0.0, min(1.0, (current - entry) / (high - entry))) * 100, 1),
        "reached_high": current >= high,
        "upside_to_high_pct": round((high / current - 1) * 100, 1) if current > 0 else None,
    }


def _price_lookup(ticker: str):
    try:
        from quantedge.data.price_store import PriceStore
        store = PriceStore(os.environ.get("PRICE_DB", "/app/data/price_store.db"))
        bars = store.series(ticker, date.today() - timedelta(days=10), date.today())
        store.close()
        return bars[-1][1] if bars else None
    except Exception:
        return None


def _shape(artifact: dict, live_prices: bool = True) -> dict:
    out_tiers = {}
    for tier_name, rows in artifact.get("tiers", {}).items():
        shaped = []
        for r in rows:
            row = {
                "ticker": r["ticker"], "name": r.get("name", r["ticker"]),
                "score": r["score"], "stage": r["stage"], "tier": r["tier"],
                "drawdown_from_high_pct": round(r["drawdown"] * 100, 1)
                    if r.get("drawdown") is not None else None,
                "thesis": r.get("thesis"), "entry_price": r.get("price"),
                "prior_high": r.get("prior_high"),
            }
            if live_prices and r.get("prior_high") and r.get("price"):
                cur = _price_lookup(r["ticker"])
                if cur:
                    row["current_price"] = cur
                    row["recovery"] = _recovery(r["price"], cur, r["prior_high"])
            shaped.append(row)
        out_tiers[tier_name] = shaped
    return {
        "as_of": artifact.get("as_of"), "generated": artifact.get("generated"),
        "tiers": out_tiers, "stage_counts": artifact.get("stage_counts"),
        "total_passed": artifact.get("n_passed_gates", artifact.get("total_passed")),
        "n_universe": artifact.get("n_universe"),
        "n_prefilter": artifact.get("n_prefilter"),
        "disclaimer": DISCLAIMER,
    }


@router.get("/rebound/list")
async def rebound_list():
    art = _load()
    if not art:
        return {"error": "no scan available yet", "disclaimer": DISCLAIMER}
    return _shape(art)


@router.get("/rebound/tracker")
async def rebound_tracker():
    art = _load()
    if not art:
        return {"error": "no scan available yet", "disclaimer": DISCLAIMER}
    shaped = _shape(art)
    rows = [r for t in shaped["tiers"].values() for r in t if r.get("recovery")]
    rows.sort(key=lambda r: r["recovery"].get("progress_pct") or -1, reverse=True)
    return {"tracked": rows, "count": len(rows), "disclaimer": DISCLAIMER}


@router.get("/rebound/pick/{ticker}")
async def rebound_pick(ticker: str):
    art = _load()
    if not art:
        return {"error": "no scan available yet", "disclaimer": DISCLAIMER}
    ticker = ticker.upper()
    for rows in art.get("tiers", {}).values():
        for r in rows:
            if r["ticker"] == ticker:
                shaped = _shape({"tiers": {"x": [r]}})
                return {"pick": shaped["tiers"]["x"][0], "disclaimer": DISCLAIMER}
    return {"error": f"{ticker} not in current shortlist", "disclaimer": DISCLAIMER}
