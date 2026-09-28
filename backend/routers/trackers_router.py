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
                    "history": ((base.get("by_tier") or {}).get(tier) or base.get("buckets") or {}).get(_bucket(r["pct_below_high"]))})
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


NEXT_TIER = {"mid": ("large", 10e9), "small": ("mid", 2e9)}


@router.get("/trackers/rising-stars")
async def rising_stars(request: Request, tier: str = Query("mid")):
    """Companies on track to move up a size tier: sustained fast sales growth, margins
    widening as they grow, the price confirming it, and (where 13F data exists) more funds
    arriving. Distance to the next tier is shown as arithmetic on the recent pace."""
    import math
    if tier not in NEXT_TIER: raise HTTPException(status_code=422, detail="tier must be mid|small")
    nxt, thr = NEXT_TIER[tier]
    pool = request.app.state.db
    as_of = await pool.fetchval("SELECT max(as_of) FROM company_facts")
    rows = await pool.fetch("""SELECT * FROM company_facts WHERE as_of=$1 AND tier=$2 AND primary_listing AND NOT is_spac
        AND data_suspect IS NULL AND fundamentals IS NOT NULL AND ret_1y > 0 AND weeks_beat_mkt_26 >= 13""", as_of, tier)
    inst = {}
    for x in await pool.fetch("""SELECT DISTINCT ON (e.ticker, d.field) e.ticker, d.field, d.value FROM ci_derived d
            JOIN ci_events e ON e.id = d.event_id WHERE e.event_type = 'institutional_snapshot'
            AND d.field IN ('managers','new_managers','exited_managers') ORDER BY e.ticker, d.field, e.event_date DESC"""):
        v = json.loads(x["value"]) if isinstance(x["value"], str) else x["value"]
        inst.setdefault(x["ticker"], {})[x["field"]] = v.get("value")
    out = []
    for r in rows:
        f = _f(r)
        if not f.get("available") or f.get("stale"): continue
        qs = f.get("quarters") or []; ys = [q.get("sales_yoy") for q in qs[-4:] if q.get("sales_yoy") is not None]
        sy = f.get("sales_yoy") or 0
        if sy < 0.20 or len(ys) < 4 or sum(y >= 0.15 for y in ys) < 3: continue
        if (f.get("op_margin_change_1y") or 0) < 0 or (f.get("gross_margin_change_1y") or 0) < -0.01: continue
        # Real businesses only: meaningful sales now and a year ago, and near profitable today.
        # Without this, drug companies going from $0.1M to $20M read as "+22,000% growth".
        MIN_Q = 50e6 if tier == "mid" else 15e6
        if len(qs) < 5 or (qs[-1].get("sales") or 0) < MIN_Q or (qs[-5].get("sales") or 0) < 0.5 * MIN_Q: continue
        if (f.get("op_margin") or -1) < -0.10 or (qs[-5].get("op_margin") or 0) < -1.0: continue
        mc = r["market_cap"] or 0
        months = round(12 * math.log(thr / mc) / math.log(1 + sy), 0) if mc and mc < thr and sy > 0 else None
        i = inst.get(r["ticker"], {})
        funds_arriving = (i.get("new_managers") or 0) > (i.get("exited_managers") or 0) if i else None
        score = min(1.5, sy) + 5 * min(0.1, f.get("op_margin_change_1y") or 0) + (r["weeks_beat_mkt_26"] / 26) + (0.2 if funds_arriving else 0)
        out.append({"ticker": r["ticker"], "name": r["name"], "sector": r["sector"], "market_cap": mc, "price": r["price"],
                    "returns": {"1d": r["ret_1d"], "1w": r["ret_1w"], "1m": r["ret_1m"], "3m": r["ret_3m"], "6m": r["ret_6m"], "1y": r["ret_1y"]},
                    "vol_ratio_20_60": r["vol_ratio_20_60"], "sales_yoy": sy, "sales_yoy_4q": ys,
                    "op_margin": f.get("op_margin"), "op_margin_year_ago": (qs[-5]["op_margin"] if len(qs) >= 5 else None),
                    "gross_margin_change_1y": f.get("gross_margin_change_1y"), "cash_backed": f.get("cash_backed"),
                    "sales_quarters": [q["sales"] for q in qs], "weeks_beat_mkt_26": r["weeks_beat_mkt_26"],
                    "funds": i or None, "funds_arriving": funds_arriving,
                    "next_tier": nxt, "next_tier_threshold": thr, "months_to_next_tier_at_sales_pace": months, "score": score})
    out.sort(key=lambda c: c["score"], reverse=True)
    return {"as_of": str(as_of), "tier": tier, "next_tier": nxt, "n": len(out), "companies": out,
            "note": ("Months to the next tier = if the company's market value grew at the same rate as its sales over the last year. "
                     "Arithmetic on the recent pace, not a forecast.")}


MIN_Q_SALES = {"large": 250e6, "mid": 50e6, "small": 15e6}


@router.get("/trackers/growth-leaders")
async def growth_leaders(request: Request, tier: str = Query("large")):
    """Companies growing sales 15%+ on a real sales base, checked against four stages:
    improving (growth speeding up), scaling (sustained, margins widening), price confirming
    (beating the market), under the radar (less coverage than peers). Ranked by growth,
    stages passed and margin improvement. 'Early' = sales growing faster than the price."""
    import math
    if tier not in ("large", "mid", "small"): raise HTTPException(status_code=422, detail="tier must be large|mid|small")
    pool = request.app.state.db
    as_of = await pool.fetchval("SELECT max(as_of) FROM company_facts")
    med = await pool.fetchval("""SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY news_180d) FROM company_facts
                                 WHERE as_of=$1 AND tier=$2 AND primary_listing AND NOT is_spac""", as_of, tier) or 1
    rows = await pool.fetch("""SELECT * FROM company_facts WHERE as_of=$1 AND tier=$2 AND primary_listing AND NOT is_spac
        AND data_suspect IS NULL AND fundamentals IS NOT NULL""", as_of, tier)
    inst = {}
    for x in await pool.fetch("""SELECT DISTINCT ON (e.ticker, d.field) e.ticker, d.field, d.value FROM ci_derived d
            JOIN ci_events e ON e.id = d.event_id WHERE e.event_type = 'institutional_snapshot'
            AND d.field IN ('managers','new_managers','exited_managers') ORDER BY e.ticker, d.field, e.event_date DESC"""):
        v = json.loads(x["value"]) if isinstance(x["value"], str) else x["value"]
        inst.setdefault(x["ticker"], {})[x["field"]] = v.get("value")
    nxt = {"mid": ("large", 10e9), "small": ("mid", 2e9)}.get(tier)
    out = []
    for r in rows:
        f = _f(r)
        if not f.get("available") or f.get("stale"): continue
        qs = f.get("quarters") or []
        if len(qs) < 5 or (qs[-1].get("sales") or 0) < MIN_Q_SALES[tier] or (qs[-5].get("sales") or 0) < 0.5 * MIN_Q_SALES[tier]: continue
        sy = f.get("sales_yoy") or 0
        if sy < 0.15: continue
        ys = [q.get("sales_yoy") for q in qs[-4:] if q.get("sales_yoy") is not None]
        omc = f.get("op_margin_change_1y"); om = f.get("op_margin")
        rel_att = (r["news_180d"] + 5) / (med + 5)
        stages = {
            "improving": bool((f.get("acceleration_streak") or 0) >= 2 or f.get("accelerating")),
            "scaling": bool(len(ys) >= 4 and sum(y >= 0.15 for y in ys) >= 3 and (omc or 0) >= 0 and (om or -1) > -0.10),
            "confirming": bool((r["weeks_beat_mkt_26"] or 0) >= 15 and (r["ret_6m"] or 0) > 0 and (r["ret_1y"] or 0) > 0),
            "under_radar": rel_att <= 1.0,
        }
        n_st = sum(stages.values())
        score = 2 * min(1.0, sy) + 0.35 * n_st + 3 * max(-0.1, min(0.1, omc or 0)) + (0.15 if f.get("cash_backed") else 0)
        early = bool(stages["scaling"] and r["ret_1y"] is not None and r["ret_1y"] < sy)
        i = inst.get(r["ticker"], {})
        mc = r["market_cap"] or 0
        months = (round(12 * math.log(nxt[1] / mc) / math.log(1 + sy)) if nxt and mc and mc < nxt[1] else None)
        out.append({"ticker": r["ticker"], "name": r["name"], "sector": r["sector"], "market_cap": mc, "price": r["price"],
                    "returns": {"1d": r["ret_1d"], "1w": r["ret_1w"], "1m": r["ret_1m"], "3m": r["ret_3m"], "6m": r["ret_6m"], "1y": r["ret_1y"]},
                    "vol_ratio_20_60": r["vol_ratio_20_60"], "sales_yoy": sy, "sales_yoy_4q": ys, "acceleration_streak": f.get("acceleration_streak"),
                    "op_margin": om, "op_margin_year_ago": (qs[-5].get("op_margin") if len(qs) >= 5 else None), "cash_backed": f.get("cash_backed"),
                    "sales_quarters": [q["sales"] for q in qs], "weeks_beat_mkt_26": r["weeks_beat_mkt_26"],
                    "news_180d": r["news_180d"], "attention_vs_peers": round(rel_att, 2),
                    "stages": stages, "stages_passed": n_st, "early": early, "pct_below_high": r["pct_below_high"],
                    "funds": i or None, "funds_arriving": ((i.get("new_managers") or 0) > (i.get("exited_managers") or 0)) if i else None,
                    "next_tier": nxt[0] if nxt else None, "months_to_next_tier_at_sales_pace": months, "score": round(score, 3)})
    out.sort(key=lambda c: c["score"], reverse=True)
    return {"as_of": str(as_of), "tier": tier, "n": len(out), "companies": out,
            "counts": {"all_four": sum(c["stages_passed"] == 4 for c in out), "early": sum(c["early"] for c in out),
                       "under_radar": sum(c["stages"]["under_radar"] for c in out)},
            "note": ("Sales growth from SEC filings, each quarter dated to its first filing. 'Early' = sales grew faster than the share "
                     "price over the last year. Months to the next tier = arithmetic on the sales pace, not a forecast.")}


SHOCK = {"large": 0.06, "mid": 0.08, "small": 0.10}


async def _earnings_reactions(pool, tickers, tier):
    """Latest results (8-K 2.02) within 130 days: move from the close before to the second
    session after, volume vs the prior 20 sessions, and whether the gain held 10 sessions on."""
    ev = await pool.fetch("""SELECT DISTINCT ON (ticker) ticker, event_date FROM ci_events WHERE item_code='2.02'
        AND ticker = ANY($1) AND event_date > CURRENT_DATE - 130 ORDER BY ticker, event_date DESC""", tickers)
    if not ev: return {}
    bars = {}
    for b in await pool.fetch("""SELECT ticker, d, c, v FROM daily_bars WHERE ticker = ANY($1) AND d > CURRENT_DATE - 175 ORDER BY ticker, d""",
                              [e["ticker"] for e in ev]):
        bars.setdefault(b["ticker"], []).append(b)
    out = {}
    for e in ev:
        bs = bars.get(e["ticker"], []); ds = [b["d"] for b in bs]
        pre_i = max([i for i, d in enumerate(ds) if d < e["event_date"]], default=None)
        if pre_i is None or pre_i + 2 >= len(bs) or pre_i < 20: continue
        post_i = pre_i + 2
        pre, post = float(bs[pre_i]["c"]), float(bs[post_i]["c"])
        gap = post / pre - 1
        base_v = sum(float(b["v"] or 0) for b in bs[pre_i - 20:pre_i]) / 20 or 1
        vr = max(float(bs[pre_i + 1]["v"] or 0), float(bs[post_i]["v"] or 0)) / base_v
        held = (post_i + 10 < len(bs)) and float(bs[post_i + 10]["c"]) >= post * 0.97 and float(bs[-1]["c"]) >= pre * (1 + gap / 2)
        out[e["ticker"]] = {"date": str(e["event_date"]), "move": gap, "volume_x": vr, "held": held,
                            "shock_up": gap >= SHOCK[tier] and vr >= 2 and held}
    return out


@router.get("/trackers/worth-a-look")
async def worth_a_look(request: Request, tier: str = Query("large")):
    """A shortlist that reads every verified tracker together: discounted quality, early growth
    not yet priced, and breakthroughs (big, held reactions to results) — excluding serious warning
    signs and anything already priced in — each with its case, next catalyst, history and risks.
    Does NOT use ML forecasts (none validated on recent data)."""
    if tier not in ("large", "mid", "small"): raise HTTPException(status_code=422, detail="tier must be large|mid|small")
    pool = request.app.state.db
    sale = await on_sale(request, tier=tier, min_drop=0.20, quality="strong", sort="size")
    grow = await growth_leaders(request, tier=tier)
    warn = await warning_signs(request, tier=tier, healthy_only=False)
    W = {c["ticker"]: c for c in warn["companies"]}
    S = {c["ticker"]: c for c in sale["companies"]}; G = {c["ticker"]: c for c in grow["companies"]}
    tks = list(set(S) | set(G))
    react = await _earnings_reactions(pool, tks, tier)
    fp = artifact_read_path("fired_last_night.json"); fired = {}
    if fp:
        for x in json.loads(fp.read_text()).get("fired", []):
            if x.get("direction") == "bullish" and x.get("odds_21d"): fired[x["ticker"]] = x
    base = (((sale.get("history") or {}).get("by_tier") or {}).get(tier) or (sale.get("history") or {}).get("buckets") or {})
    picks = []
    for t in tks:
        s, g, w, rx = S.get(t), G.get(t), W.get(t), react.get(t)
        if w and w["serious"] > 0: continue
        c = s or g
        r1, sy = (c["returns"] or {}).get("1y"), (g or {}).get("sales_yoy", s.get("sales_yoy") if s else None)
        if r1 is not None and r1 > 2.0 and (sy is None or r1 > 2 * max(sy, 0.01)): continue       # already priced in
        score, case, kinds = 0.0, [], []
        old_high = False
        if s and s["stage"] in ("basing", "turning", "recovering"):
            depth = -s["pct_below_high"]
            old_high = (__import__("datetime").date.today() - __import__("datetime").date.fromisoformat(s["high_date"])).days > 3 * 365
            score += min(0.5, depth) * 2 * (0.5 if old_high else 1.0) + (0.3 if s["stage"] in ("turning", "recovering") else 0) \
                     + (0.2 if s["drop_cause"] in ("market", "industry") else 0) + (0.2 if (s.get("sales_yoy") or 0) > 0.05 else 0)
            res = [e for e in (s.get("around_the_drop") or []) if e.get("item") == "2.02"]
            why = {"market": "a market-wide sell-off", "industry": "an industry-wide sell-off"}.get(s["drop_cause"]) or \
                  (f"a drop around its {res[-1]['date']} results" if res else "company-specific selling (no SEC filing explains it — check the news)")
            case.append(f"{depth*100:.0f}% below its 5-year high after {why}; the business is still growing sales {(s.get('sales_yoy') or 0)*100:+.0f}% with a {(s.get('op_margin') or 0)*100:.0f}% operating margin, and the price is {({'basing':'going sideways','turning':'turning up','recovering':'recovering'})[s['stage']]}.")
            kinds.append("discount")
        gap_pts = (g["sales_yoy"] - r1) if (g and r1 is not None) else None
        if g and g["early"] and (g["stages"]["improving"] or g["stages"]["scaling"]) and gap_pts is not None and gap_pts >= 0.15:
            score += 1.0 + 0.2 * g["stages_passed"]
            case.append(f"Sales growing faster than the price: {g['sales_yoy']*100:+.0f}% vs {r1*100:+.0f}% over the last year" + (f", with growth speeding up for {g['acceleration_streak']} quarters" if (g.get('acceleration_streak') or 0) >= 2 else "") + ".")
            kinds.append("early growth")
        elif g and g["stages_passed"] >= 3 and r1 is not None and r1 < 1.5 * g["sales_yoy"]:
            score += 0.6
            case.append(f"Growth leader passing {g['stages_passed']} of 4 stages: sales {g['sales_yoy']*100:+.0f}%, price {r1*100:+.0f}% in a year.")
            kinds.append("growth")
        if rx and rx["shock_up"]:
            score += 0.8
            case.append(f"Jumped {rx['move']*100:+.0f}% on its {rx['date']} results, on {rx['volume_x']:.1f}× normal volume — and held the gain.")
            kinds.append("breakthrough")
        if not kinds: continue
        risks = []
        if old_high: risks.append(f"Its high was set {'during the 2021 boom' if s['high_date'][:4] == '2021' else 'in ' + s['high_date'][:4]}; prices then may not return.")
        if w:
            score -= 0.3 * w["watch"]; risks += [x["text"] for x in w["signs"][:2]]
        if t in fired and (fired[t]["odds_21d"].get("positive_pct") or 0) >= 55:
            score += 0.2
            case.append(f"Just completed a {fired[t]['pattern'].replace('_',' ')} chart pattern — historically up {fired[t]['odds_21d']['positive_pct']}% of the time over the next month (n={fired[t]['odds_21d']['n']:,}).")
        vol1y = abs((c["returns"] or {}).get("1m") or 0)
        if vol1y > 0.25: risks.append(f"Volatile: moved {vol1y*100:.0f}% in the last month alone.")
        hist = None
        if s:
            bk = "50" if s["pct_below_high"] <= -0.5 else "30" if s["pct_below_high"] <= -0.3 else "20"
            if base.get(bk): hist = f"Of {tier} US companies that fell {bk}%+ from a high, {base[bk]['recovered_pct']:.0f} in 100 got back — typically in ~{round(base[bk]['median_sessions_to_recover']/21)} months."
        picks.append({"ticker": t, "name": c["name"], "sector": c["sector"], "market_cap": c["market_cap"], "price": c.get("price"),
                      "returns": c["returns"], "vol_ratio_20_60": c.get("vol_ratio_20_60"), "kinds": kinds, "case": case[:3],
                      "risks": risks[:3] or ["No warning signs in the last 30 days."], "history": hist,
                      "next_results_est": (s or {}).get("next_results_est"), "last_results_reaction": rx,
                      "pct_below_high": c.get("pct_below_high"), "sales_yoy": sy, "stages": (g or {}).get("stages"),
                      "sales_quarters": (g or {}).get("sales_quarters") or (s or {}).get("sales_quarters"), "score": round(score, 3)})
    picks.sort(key=lambda p: p["score"], reverse=True)
    picks = picks[:30]
    need = [p["ticker"] for p in picks if not p["next_results_est"]]
    if need:
        import datetime as _dt
        as_of_d = await pool.fetchval("SELECT max(as_of) FROM company_facts")
        for r in await pool.fetch("SELECT ticker, fundamentals FROM company_facts WHERE as_of=$1 AND ticker = ANY($2)", as_of_d, need):
            q = (_f(r).get("quarters") or [])
            if not q: continue
            est = _dt.date.fromisoformat(q[-1]["end"]) + _dt.timedelta(days=91 + 35)
            while est < _dt.date.today(): est += _dt.timedelta(days=91)
            for p in picks:
                if p["ticker"] == r["ticker"]: p["next_results_est"] = str(est); p["next_results_basis"] = "estimated from its last reported quarter"
    for i, p in enumerate(picks): p["top5"] = i < 8          # featured (field name kept for the page)
    return {"as_of": sale["as_of"], "tier": tier, "n": len(picks), "companies": picks,
            "note": ("Candidates for your research, not recommendations. Built from verified sources only — SEC filings, prices, "
                     "measured pattern odds, warning signs. ML forecasts are not used while none validate on recent data.")}



MOVER_MIN_DV = {"large": 0, "mid": 5e6, "small": 1e6}


@router.get("/trackers/movers")
async def movers(request: Request, tier: str = Query("large"), period: str = Query("1d"), limit: int = Query(25, ge=5, le=50)):
    """Biggest gainers and losers in one size tier over a period. 1d uses the live (delayed)
    quote; longer periods use the last close. Illiquid names are left out."""
    if tier not in ("large", "mid", "small"): raise HTTPException(status_code=422, detail="tier must be large|mid|small")
    col = {"1w": "ret_1w", "2w": "ret_2w", "1m": "ret_1m", "3m": "ret_3m", "6m": "ret_6m", "1y": "ret_1y"}.get(period)
    if period != "1d" and not col: raise HTTPException(status_code=422, detail="period must be 1d|1w|2w|1m|3m|6m|1y")
    pool = request.app.state.db
    as_of = await pool.fetchval("SELECT max(as_of) FROM company_facts")
    rows = await pool.fetch("""SELECT * FROM company_facts WHERE as_of=$1 AND tier=$2 AND primary_listing AND NOT is_spac
                               AND data_suspect IS NULL AND coalesce(dollar_vol_20,0) >= $3""", as_of, tier, MOVER_MIN_DV[tier])
    by = {r["ticker"]: r for r in rows}
    quote_time = None
    if period == "1d":
        from routers.home_router import _snap
        snap = await _snap(list(by))
        move = {t: (v["chg_pct"] / 100.0) for t, v in snap.items() if v.get("chg_pct") is not None and v.get("price")}
        upd = max([v.get("updated_ns") or 0 for v in snap.values()] or [0])
        quote_time = __import__("datetime").datetime.fromtimestamp(upd / 1e9, tz=__import__("datetime").timezone.utc).isoformat() if upd else None
        vol_today = {t: v.get("volume") or 0 for t, v in snap.items()}
    else:
        move = {t: r[col] for t, r in by.items() if r[col] is not None}
        vol_today = {}
    res = {t: e["event_date"] for e in await pool.fetch("""SELECT DISTINCT ON (ticker) ticker, event_date FROM ci_events
        WHERE item_code='2.02' AND ticker = ANY($1) AND event_date > CURRENT_DATE - 7 ORDER BY ticker, event_date DESC""", list(by))}
    def card(t, m):
        r = by[t]; tags = []
        if r["at_52w_high"]: tags.append("new 52-week high")
        if r["at_52w_low"]: tags.append("new 52-week low")
        if t in res: tags.append(f"results {res[t]}")
        if (r["vol_ratio_20_60"] or 0) >= 1.8: tags.append(f"volume {r['vol_ratio_20_60']:.1f}× its usual")
        return {"ticker": t, "name": r["name"], "sector": r["sector"], "market_cap": r["market_cap"], "price": r["price"], "move": m,
                "returns": {"1d": r["ret_1d"], "1w": r["ret_1w"], "2w": r["ret_2w"], "1m": r["ret_1m"], "3m": r["ret_3m"], "6m": r["ret_6m"], "1y": r["ret_1y"]},
                "vol_ratio_20_60": r["vol_ratio_20_60"], "tags": tags}
    ranked = sorted(move.items(), key=lambda kv: kv[1])
    return {"as_of": str(as_of), "tier": tier, "period": period, "quote_time": quote_time, "universe": len(move),
            "gainers": [card(t, m) for t, m in reversed(ranked[-limit:]) if m > 0],
            "losers": [card(t, m) for t, m in ranked[:limit] if m < 0],
            "note": ("1-day moves use the live quote (15-minute delayed); longer periods use the last close. "
                     "Companies trading under a minimum dollar volume are left out.")}
