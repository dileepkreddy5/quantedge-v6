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


def _latest_closes() -> dict:
    """All tickers' most-recent close in ONE query, keyed by ticker.

    Uses raw sqlite3 (stdlib) against the price store rather than importing the
    quantedge package — the backend image does not ship that package, so the
    import previously failed and silently disabled recovery. Schema: table
    `bars(t TEXT ticker, d TEXT date, c REAL close)`. Reads the store's actual
    last trading day (not date.today()). Returns {} on any failure — callers
    then show recovery as n/a rather than a fabricated value."""
    import sqlite3
    db = os.environ.get("PRICE_DB", "/app/data/price_store.db")
    if not os.path.exists(db):
        logger.warning(f"price store not found at {db}")
        return {}
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        last = con.execute("SELECT MAX(d) FROM bars").fetchone()
        if not last or not last[0]:
            con.close(); return {}
        rows = con.execute("SELECT t, c FROM bars WHERE d = ?", (last[0],)).fetchall()
        con.close()
        return {t: c for t, c in rows}
    except Exception as e:
        logger.warning(f"latest closes lookup failed: {e}")
        return {}


# Historical recovery-to-prior-high base rates by drawdown depth, measured on
# our own 5y data (recovery backtest, unit 005, healthy cohort). These are REAL
# measured frequencies — the honest odds, not a promise. Small sample; stated as
# such. Journal: 2026-07_recovery_result.md.
_RECOVERY_BASE_RATE = {
    "35-50": {"rate_pct": 13.6, "median_days": 243, "n": 66},
    "50-70": {"rate_pct": 7.1,  "median_days": None, "n": 42},
    "70+":   {"rate_pct": 0.0,  "median_days": None, "n": 15},
}

def _dd_bucket(dd_frac: float) -> str:
    if dd_frac < 0.50: return "35-50"
    if dd_frac < 0.70: return "50-70"
    return "70+"

def _insights(r: dict) -> dict:
    """Real, computed insights per name — no fabrication, all from fields the
    scan already produced."""
    out = {}
    dd = r.get("drawdown")
    price = r.get("price")
    high = r.get("prior_high")
    if price and high and price > 0:
        out["required_return_to_high_pct"] = round((high / price - 1) * 100, 1)
    if dd is not None:
        b = _RECOVERY_BASE_RATE.get(_dd_bucket(dd))
        if b:
            out["historical_recovery"] = {
                "drawdown_bucket": _dd_bucket(dd),
                "recovered_within_1y_pct": b["rate_pct"],
                "median_days_when_recovered": b["median_days"],
                "sample_size": b["n"],
                "note": "measured on 5y data, healthy cohort, small sample — odds not a promise",
            }
    dsl = r.get("days_since_low")
    if dsl is not None:
        out["days_since_low"] = dsl
        out["off_the_lows"] = dsl > 20
    up_share = r.get("up_day_share_1m")
    vol_ratio = r.get("vol_1m_ratio")
    if up_share is not None and vol_ratio is not None:
        out["accumulation_signal"] = bool(up_share >= 0.55 and vol_ratio <= 1.1)
        out["up_day_volume_share_pct"] = round(up_share * 100, 1)
    out["analysis_url"] = f"/dashboard?ticker={r['ticker']}"
    return out


def _shape(artifact: dict, live_prices: bool = True) -> dict:
    closes = _latest_closes() if live_prices else {}
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
                "insights": _insights(r),
            }
            if live_prices and r.get("prior_high") and r.get("price"):
                cur = closes.get(r["ticker"])
                if cur:
                    row["current_price"] = round(cur, 2)
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
