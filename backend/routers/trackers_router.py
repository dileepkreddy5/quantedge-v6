"""Tracker endpoints over the nightly facts sheet (company_facts)."""
from __future__ import annotations
import json
from datetime import timedelta
from fastapi import APIRouter, Query, Request, HTTPException
from core.artifact_paths import artifact_read_path

router = APIRouter()
BUCKETS = ((-0.50, "50"), (-0.30, "30"), (-0.20, "20"))


def _bucket(pct):
    for lim, k in BUCKETS:
        if pct <= lim: return k
    return None


@router.get("/trackers/on-sale")
async def on_sale(request: Request, tier: str = Query("large"), min_drop: float = Query(0.20, ge=0.1, le=0.9),
                  quality: str = Query("strong"), sort: str = Query("size")):
    """Great companies on sale: primary listings, not SPACs, at least min_drop below
    their 5-year high, in ONE size tier. Business health, why it fell, filings around
    the drop, estimated next results, and the history for its depth bucket."""
    if tier not in ("large", "mid", "small"): raise HTTPException(status_code=422, detail="tier must be large|mid|small")
    pool = request.app.state.db
    as_of = await pool.fetchval("SELECT max(as_of) FROM company_facts")
    rows = await pool.fetch("""
        SELECT * FROM company_facts WHERE as_of=$1 AND tier=$2 AND primary_listing AND NOT is_spac AND data_suspect IS NULL
          AND pct_below_high <= $3 ORDER BY market_cap DESC NULLS LAST""", as_of, tier, -min_drop)
    tks = [r["ticker"] for r in rows]
    ev = await pool.fetch("""SELECT ticker, event_date, available_at, item_code, title, significance FROM ci_events
        WHERE ticker = ANY($1) AND event_type NOT IN ('insider_transaction','attention_spike','institutional_snapshot','capital_allocation')
          AND event_date > CURRENT_DATE - 1900 ORDER BY available_at""", tks)
    by = {}
    for e in ev: by.setdefault(e["ticker"], []).append(e)
    bp = artifact_read_path("great_on_sale_base.json"); base = json.loads(bp.read_text()) if bp else {}
    out = []; counts = {"strong": 0, "weakening": 0, "unknown": 0}
    for r in rows:
        f = json.loads(r["fundamentals"]) if isinstance(r["fundamentals"], str) else (r["fundamentals"] or {})
        # "Great" = healthy business, operating profit, profits backed by cash.
        om, sy = f.get("op_margin") or 0, f.get("sales_yoy") or 0
        # a real profit engine: 10%+ operating margin, or fast growth (15%+) with some profit
        strong = bool(f.get("available") and f.get("healthy") and f.get("cash_backed") and (om >= 0.10 or (sy >= 0.15 and om > 0)))
        grade = "strong" if strong else ("weakening" if f.get("available") else "unknown")
        counts[grade] += 1
        if quality == "strong" and not strong: continue
        evs = by.get(r["ticker"], [])
        lo_end = r["low_date"] + timedelta(days=30)
        around = [e for e in evs if r["high_date"] <= e["event_date"] <= lo_end and (e["significance"] in ("MATERIAL", "RELEVANT") or e["item_code"] == "2.02")]
        results = [e for e in evs if e["item_code"] == "2.02"]
        nxt = (results[-1]["event_date"] + timedelta(days=91)) if results else None
        health = ("healthy" if f.get("healthy") else "weakening") if f.get("available") else "unknown"
        out.append({"ticker": r["ticker"], "name": r["name"], "sector": r["sector"], "market_cap": r["market_cap"], "price": r["price"],
                    "returns": {"1d": r["ret_1d"], "1w": r["ret_1w"], "1m": r["ret_1m"], "3m": r["ret_3m"], "6m": r["ret_6m"], "1y": r["ret_1y"]}, "vol_ratio_20_60": r["vol_ratio_20_60"],
                    "high_5y": r["high_5y"], "high_date": str(r["high_date"]), "pct_below_high": r["pct_below_high"],
                    "months_since_high": round(r["sessions_since_high"] / 21, 1), "low_date": str(r["low_date"]),
                    "pct_off_low": r["pct_off_low"], "stage": r["stage"], "drop_cause": r["drop_cause"],
                    "mkt_move_since_high": r["mkt_move_since_high"], "sector_move_since_high": r["sector_move_since_high"],
                    "health": health, "quality": grade, "history_note": r["history_note"], "peer_group": r["peer_group"], "sales_yoy": f.get("sales_yoy"), "op_margin": f.get("op_margin"),
                    "op_margin_change_1y": f.get("op_margin_change_1y"), "cash_backed": f.get("cash_backed"),
                    "sales_quarters": [q["sales"] for q in f.get("quarters", [])][-8:],
                    "around_the_drop": [{"date": str(e["event_date"]), "title": e["title"], "item": e["item_code"]} for e in around[-4:]],
                    "last_results": str(results[-1]["event_date"]) if results else None,
                    "next_results_est": str(nxt) if nxt else None,
                    "history": (base.get("buckets") or {}).get(_bucket(r["pct_below_high"]))})
    def biz(c): return (c["op_margin"] or 0) + 0.5 * max(-0.5, min(1.0, c["sales_yoy"] or 0)) + (0.05 if c["cash_backed"] else 0)
    if sort == "discount": out.sort(key=lambda c: c["pct_below_high"])
    elif sort == "business": out.sort(key=biz, reverse=True)
    else: out.sort(key=lambda c: c["market_cap"] or 0, reverse=True)
    return {"as_of": str(as_of), "tier": tier, "n": len(out), "quality_filter": quality, "sort": sort, "counts": counts, "companies": out,
            "history": base, "high_note": "“High” means the highest closing price in the last 5 years (our data starts September 2021).",
            "note": "Next results dates are estimates: the last reported results date plus about three months."}



def _f(r):
    return json.loads(r["fundamentals"]) if isinstance(r["fundamentals"], str) else (r["fundamentals"] or {})


@router.get("/trackers/quiet-climbers")
async def quiet_climbers(request: Request, tier: str = Query("large")):
    """Rising steadily while few are watching: up over 1y, 6m and 3m, beat the market
    in 15+ of the last 26 weeks. Score = steadiness x 6-month rise, divided by news
    attention relative to the tier's median (so 'quiet' is judged among peers)."""
    if tier not in ("large", "mid", "small"): raise HTTPException(status_code=422, detail="tier must be large|mid|small")
    pool = request.app.state.db
    as_of = await pool.fetchval("SELECT max(as_of) FROM company_facts")
    med = await pool.fetchval("""SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY news_180d) FROM company_facts
                                 WHERE as_of=$1 AND tier=$2 AND primary_listing AND NOT is_spac AND data_suspect IS NULL""", as_of, tier) or 1
    rows = await pool.fetch("""SELECT * FROM company_facts WHERE as_of=$1 AND tier=$2 AND primary_listing AND NOT is_spac AND data_suspect IS NULL
        AND ret_1y > 0.2 AND ret_6m > 0 AND ret_3m > 0 AND weeks_beat_mkt_26 >= 15
        AND (1 + ret_1y) >= (1 + ret_6m)""", as_of, tier)     # first half of the year also up: a climb, not a V
    out = []
    for r in rows:
        rel_att = (r["news_180d"] + 5) / (med + 5)
        if rel_att > 1.25: continue                               # "quiet" = no more coverage than peers (with slack)
        score = (r["weeks_beat_mkt_26"] / 26) * r["ret_6m"] / rel_att
        f = _f(r)
        out.append({"ticker": r["ticker"], "name": r["name"], "sector": r["sector"], "market_cap": r["market_cap"], "price": r["price"],
                    "returns": {"1d": r["ret_1d"], "1w": r["ret_1w"], "1m": r["ret_1m"], "3m": r["ret_3m"], "6m": r["ret_6m"], "1y": r["ret_1y"]},
                    "weeks_beat_mkt_26": r["weeks_beat_mkt_26"], "up_weeks_26": r["up_weeks_26"],
                    "news_180d": r["news_180d"], "attention_vs_peers": round(rel_att, 2),
                    "quiet": rel_att <= 1.0, "pct_below_high": r["pct_below_high"], "vol_ratio_20_60": r["vol_ratio_20_60"],
                    "sales_yoy": f.get("sales_yoy"), "score": score})
    out.sort(key=lambda c: c["score"], reverse=True)
    return {"as_of": str(as_of), "tier": tier, "n": len(out), "tier_median_news_180d": med, "companies": out,
            "note": "Attention = news articles in 180 days (Polygon's feed), compared with the median company in the same size tier."}


@router.get("/trackers/getting-better")
async def getting_better(request: Request, tier: str = Query("large")):
    """Results improving quarter after quarter, from SEC filings: sales growth speeding
    up 2+ quarters in a row, or speeding up with operating margin widening; profits
    backed by cash; latest quarter recent (not stale)."""
    if tier not in ("large", "mid", "small"): raise HTTPException(status_code=422, detail="tier must be large|mid|small")
    pool = request.app.state.db
    as_of = await pool.fetchval("SELECT max(as_of) FROM company_facts")
    rows = await pool.fetch("""SELECT * FROM company_facts WHERE as_of=$1 AND tier=$2 AND primary_listing AND NOT is_spac AND data_suspect IS NULL
        AND fundamentals IS NOT NULL""", as_of, tier)
    out = []
    for r in rows:
        f = _f(r)
        if not f.get("available") or f.get("stale") or (f.get("sales_yoy") or 0) <= 0: continue
        streak, omc = f.get("acceleration_streak") or 0, f.get("op_margin_change_1y")
        if not (streak >= 2 and (omc or 0) > 0 and (f.get("sales_yoy") or 0) >= 0.05): continue
        if not f.get("cash_backed"): continue
        qs = f.get("quarters", [])
        # margin change capped: a year-ago write-off made Hasbro look like +90 points
        score = streak + 10 * min(0.15, max(0, omc or 0)) + min(1.0, f.get("sales_yoy") or 0)
        om_then = qs[-5]["op_margin"] if len(qs) >= 5 else None
        out.append({"ticker": r["ticker"], "name": r["name"], "sector": r["sector"], "market_cap": r["market_cap"], "price": r["price"],
                    "returns": {"1d": r["ret_1d"], "1w": r["ret_1w"], "1m": r["ret_1m"], "3m": r["ret_3m"], "6m": r["ret_6m"], "1y": r["ret_1y"]}, "vol_ratio_20_60": r["vol_ratio_20_60"],
                    "sales_yoy": f.get("sales_yoy"), "sales_yoy_prev": f.get("sales_yoy_prev"), "acceleration_streak": streak,
                    "gross_margin": f.get("gross_margin"), "gross_margin_change_1y": f.get("gross_margin_change_1y"),
                    "op_margin": f.get("op_margin"), "op_margin_year_ago": om_then, "op_margin_change_1y": omc,
                    "one_off_suspected": bool(om_then is not None and om_then < -0.25), "cash_backed": f.get("cash_backed"),
                    "quarters": [{"end": q["end"], "sales": q["sales"], "sales_yoy": q["sales_yoy"], "op_margin": q["op_margin"]} for q in qs],
                    "last_quarter": qs[-1]["end"] if qs else None, "ret_6m": r["ret_6m"], "pct_below_high": r["pct_below_high"], "score": score})
    out.sort(key=lambda c: c["score"], reverse=True)
    return {"as_of": str(as_of), "tier": tier, "n": len(out), "companies": out,
            "note": "From SEC 10-Q/10-K filings, each quarter dated to its first filing."}
