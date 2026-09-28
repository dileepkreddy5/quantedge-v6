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


FIN = ("Financials", "Real Estate")
LAG = {"large": 0.10, "mid": 0.12, "small": 0.15}
SELL_MIN = {"large": 10e6, "mid": 5e6, "small": 1e6}
ITEM_TXT = {"4.02": "Said its past financial statements can no longer be relied on (SEC 8-K item 4.02)",
            "4.01": "Changed its auditor (SEC 8-K item 4.01)",
            "2.06": "Recorded an impairment — wrote down the value of assets (SEC 8-K item 2.06)"}


@router.get("/trackers/warning-signs")
async def warning_signs(request: Request, tier: str = Query("large"), healthy_only: bool = Query(True)):
    """Good companies showing early cracks: serious SEC events and business, insider and
    price signs from the last 30 days, each with its date and evidence."""
    if tier not in ("large", "mid", "small"): raise HTTPException(status_code=422, detail="tier must be large|mid|small")
    pool = request.app.state.db
    as_of = await pool.fetchval("SELECT max(as_of) FROM company_facts")
    rows = await pool.fetch("""SELECT * FROM company_facts WHERE as_of=$1 AND tier=$2 AND primary_listing AND NOT is_spac
                               AND data_suspect IS NULL""", as_of, tier)
    by_tk = {r["ticker"]: r for r in rows}; tks = list(by_tk); ciks = {r["cik"]: r["ticker"] for r in rows if r["cik"]}
    signs: dict[str, list] = {t: [] for t in tks}
    def add(t, sev, date, src, text, note=None):
        signs[t].append({"severity": sev, "date": str(date)[:10], "source": src, "text": text, "note": note})
    for e in await pool.fetch("""SELECT ticker, item_code, available_at FROM ci_events WHERE ticker = ANY($1)
                                 AND item_code IN ('4.02','4.01','2.06') AND available_at > NOW() - INTERVAL '30 days'""", tks):
        add(e["ticker"], "serious", e["available_at"], f"SEC 8-K · ITEM {e['item_code']}", ITEM_TXT[e["item_code"]])
    for e in await pool.fetch("""SELECT e.ticker, e.available_at, d.value, d.citation FROM ci_derived d JOIN ci_events e ON e.id = d.event_id
                                 WHERE e.ticker = ANY($1) AND d.extractor_version = 'officer-change-v2'
                                   AND e.available_at > NOW() - INTERVAL '30 days'""", tks):
        v = json.loads(e["value"]) if isinstance(e["value"], str) else e["value"]
        if v.get("class") == "abrupt_exec_exit":
            # serious only when effective immediately; a departure with notice is a watch sign
            add(e["ticker"], "serious" if v.get("immediate") else "watch", e["available_at"], "SEC 8-K · ITEM 5.02",
                f"The {v.get('role') or 'a senior officer'} is leaving{', effective immediately' if v.get('immediate') else ''}.",
                "A sudden exit of a finance or top executive has often come before restatements or guidance cuts.")
    for e in await pool.fetch("""SELECT cik, form_type, filed_at FROM ci_raw_evidence WHERE form_type IN ('NT 10-Q','NT 10-K')
                                 AND cik = ANY($1) AND filed_at > NOW() - INTERVAL '30 days'""", list(ciks)):
        add(ciks[e["cik"]], "serious", e["filed_at"], f"SEC {e['form_type']}",
            f"Told the SEC it can't file its {'quarterly' if 'Q' in e['form_type'] else 'annual'} report on time.")
    ins: dict[str, dict] = {}
    for e in await pool.fetch("""SELECT cik, filed_at, raw_payload->'form4' f FROM ci_raw_evidence WHERE form_type='4'
                                 AND cik = ANY($1) AND filed_at > NOW() - INTERVAL '30 days'""", list(ciks)):
        f = json.loads(e["f"]) if isinstance(e["f"], str) else (e["f"] or {})
        t = ciks[e["cik"]]; a = ins.setdefault(t, {"sellers": set(), "sold": 0.0, "bought": 0.0, "planned": 0, "unknown": 0, "first": e["filed_at"], "last": e["filed_at"]})
        who = ((f.get("owners") or [{}])[0].get("name")) or "?"
        for tx in f.get("transactions", []):
            if not tx.get("open_market"): continue
            if tx.get("code") == "P": a["bought"] += tx.get("value") or 0; continue
            if tx.get("planned_10b5_1") is True: a["planned"] += 1; continue
            if tx.get("planned_10b5_1") is None: a["unknown"] += 1
            a["sellers"].add(who); a["sold"] += tx.get("value") or 0
            a["first"] = min(a["first"], e["filed_at"]); a["last"] = max(a["last"], e["filed_at"])
    for t, a in ins.items():
        if len(a["sellers"]) >= 3 and a["sold"] >= SELL_MIN[tier] and a["bought"] == 0:
            add(t, "watch", a["last"], "FORM 4 · INSIDERS",
                f"{len(a['sellers'])} insiders sold ${a['sold'] / 1e6:,.1f}M on the open market in 30 days — none bought.",
                "Some sales may be pre-planned; the filings before this month don't record whether." if a["unknown"] else None)
    healthy_then = {}
    for t, r in by_tk.items():
        f = _f(r); qs = f.get("quarters") or []
        fin = r["sector"] in FIN
        healthy_then[t] = (len(qs) >= 3 and (qs[-3].get("sales_yoy") or 0) > 0 and (fin or (qs[-3].get("op_margin") or 0) > 0)) if f.get("available") else None
        if f.get("available") and not f.get("stale") and qs:
            filed = qs[-1].get("filed"); recent = filed and (as_of - __import__("datetime").date.fromisoformat(filed)).days <= 45
            y = [q.get("sales_yoy") for q in qs[-3:]]
            if recent and len(y) == 3 and None not in y and y[1] > y[0] and y[2] <= y[1] - 0.05:
                add(t, "watch", filed, "10-Q · BUSINESS", f"Sales growth slowed to {y[2]*100:+.0f}% from {y[1]*100:+.0f}% the quarter before, after speeding up.",
                    "The reverse of “getting better.”")
            if recent and not fin and (f.get("op_margin_change_1y") or 0) <= -0.04 and f.get("op_margin") is not None:
                add(t, "watch", filed, "10-Q · BUSINESS", f"Operating margin shrank to {f['op_margin']*100:.0f}% from {(f['op_margin']-f['op_margin_change_1y'])*100:.0f}% a year ago.")
            ni, oc = f.get("net_income_ttm"), f.get("op_cash_flow_ttm")
            if recent and not fin and ni and oc is not None and ni > 0 and oc < 0.6 * ni:
                add(t, "watch", filed, "10-Q · BUSINESS", f"Profits ran ahead of the cash they produced: net income ${ni/1e9:,.2f}B over the last year, operating cash flow ${oc/1e9:,.2f}B.",
                    "Earnings not backed by cash is a classic early warning in accounting research.")
        if r["cross_200d_date"] and (r["cross_200d_vol"] or 0) >= 1.5:
            add(t, "watch", r["cross_200d_date"], "PRICE & VOLUME", f"Broke below its 200-day average after a steady uptrend, on {r['cross_200d_vol']:.1f}× normal volume.")
        if r["ret_1m"] is not None and r["peer_ret_1m"] is not None and r["ret_1m"] - r["peer_ret_1m"] <= -LAG[tier]:
            add(t, "watch", as_of, "PRICE · VS PEERS", f"Lagged its industry by {(r['peer_ret_1m'] - r['ret_1m'])*100:.0f} points over the last month ({r['ret_1m']*100:+.0f}% vs peers {r['peer_ret_1m']*100:+.0f}%).")
    # one bad month against peers is noise on its own: it only supports other evidence
    for t in tks:
        if signs[t] and all(x["source"] == "PRICE · VS PEERS" for x in signs[t]): signs[t] = []
    flagged = [t for t in tks if signs[t]]
    first = {t: min(s["date"] for s in signs[t]) for t in flagged}
    px = {}
    if flagged:
        for p in await pool.fetch("""SELECT DISTINCT ON (ticker) ticker, c FROM daily_bars WHERE ticker = ANY($1) AND d >= $2::date
                                     ORDER BY ticker, d""", flagged, __import__("datetime").date.fromisoformat(min(first.values()))):
            px[p["ticker"]] = p["c"]
    out = []
    for t in flagged:
        if healthy_only and healthy_then.get(t) is not True: continue
        r = by_tk[t]; ss = sorted(signs[t], key=lambda s: s["date"], reverse=True)
        ser = sum(s["severity"] == "serious" for s in ss)
        out.append({"ticker": t, "name": r["name"], "sector": r["sector"], "market_cap": r["market_cap"], "price": r["price"],
                    "returns": {"1d": r["ret_1d"], "1w": r["ret_1w"], "1m": r["ret_1m"], "3m": r["ret_3m"], "6m": r["ret_6m"], "1y": r["ret_1y"]},
                    "vol_ratio_20_60": r["vol_ratio_20_60"], "signs": ss, "serious": ser, "watch": len(ss) - ser,
                    "first_sign": first[t], "days_since_first": (as_of - __import__("datetime").date.fromisoformat(first[t])).days,
                    "healthy_6m_ago": healthy_then.get(t)})
    out.sort(key=lambda c: (-c["serious"], -len(c["signs"]), -(c["market_cap"] or 0)))
    return {"as_of": str(as_of), "tier": tier, "n": len(out), "healthy_only": healthy_only, "companies": out,
            "note": "Signs from the last 30 days. Serious: SEC items 4.02, 4.01, 2.06, abrupt CEO/CFO exits, late-filing notices. "
                    "Financials and real estate skip the margin and cash tests. Signs are reasons to look closer, not predictions."}
