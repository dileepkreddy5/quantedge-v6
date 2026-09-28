"""Stock Summary: a plain-English summary from verified sources only (nightly facts sheet,
SEC filings, tracker membership, breakthroughs, the panel's validation report), and daily
prices for the Summary chart. No ML forecasts are used in the wording."""
from __future__ import annotations
import datetime as dt, json
from fastapi import APIRouter, HTTPException, Request

router = APIRouter()
STAGE = {"near_high": "trading near its high", "falling": "still falling", "basing": "going sideways after its fall",
         "turning": "starting to turn up", "recovering": "recovering"}


def _pct(x, d=0): return f"{x*100:+.{d}f}%"


@router.get("/prices/{ticker}")
async def prices(ticker: str, request: Request):
    tk = ticker.upper().strip(); pool = request.app.state.db
    rows = await pool.fetch("SELECT d, o, h, l, c, v FROM daily_bars WHERE ticker=$1 ORDER BY d", tk)
    if not rows: raise HTTPException(status_code=404, detail=f"no price history for {tk}")
    er = await pool.fetch("SELECT DISTINCT event_date FROM ci_events WHERE ticker=$1 AND item_code='2.02' ORDER BY event_date", tk)
    return {"ticker": tk, "bars": [{"d": str(r["d"]), "o": float(r["o"] or r["c"]), "h": float(r["h"] or r["c"]), "l": float(r["l"] or r["c"]),
                                    "c": float(r["c"]), "v": float(r["v"] or 0)} for r in rows],
            "earnings": [str(r["event_date"]) for r in er],
            "note": "Daily closes; history begins September 2021. Earnings dates are the company's results filings (8-K item 2.02)."}


@router.get("/summary/{ticker}")
async def summary(ticker: str, request: Request):
    tk = ticker.upper().strip(); pool = request.app.state.db
    r = await pool.fetchrow("SELECT * FROM company_facts WHERE ticker=$1 ORDER BY as_of DESC LIMIT 1", tk)
    if not r: raise HTTPException(status_code=404, detail=f"{tk} is not in QuantEdge's company universe")
    f = json.loads(r["fundamentals"]) if isinstance(r["fundamentals"], str) else (r["fundamentals"] or {})
    S = []
    # 1. price
    below = r["pct_below_high"]; hd = r["high_date"]
    where = STAGE.get(r["stage"], "")
    s1 = (f"At ${r['price']:,.2f}, it is {abs(below)*100:.0f}% below its 5-year high from {hd:%b %Y}" if below < -0.02
          else f"At ${r['price']:,.2f}, it is at or near its 5-year high") + (f" and {where}." if where and r["stage"] != "near_high" else ".")
    if r["ret_1y"] is not None:
        s1 += f" Over the past year it is {_pct(r['ret_1y'])}" + (f", beating the market in {r['weeks_beat_mkt_26']} of the last 26 weeks." if r["weeks_beat_mkt_26"] is not None else ".")
    S.append({"topic": "price", "text": s1})
    # 2. business
    if f.get("available") and not f.get("stale"):
        fin = r["sector"] in ("Financials", "Real Estate")
        s2 = f"Sales grew {_pct(f['sales_yoy'])} from a year ago" if f.get("sales_yoy") is not None else "Sales growth isn't available"
        streak = f.get("acceleration_streak") or 0
        if f.get("sales_yoy") is not None:
            if streak >= 2: s2 += f", speeding up for {min(streak, 7)}{'+' if streak >= 7 else ''} quarters in a row"
            elif f.get("sales_yoy_prev") is not None and f["sales_yoy"] < f["sales_yoy_prev"] - 0.05: s2 += f", slower than {_pct(f['sales_yoy_prev'])} the quarter before"
        if not fin and f.get("op_margin") is not None:
            ch = f.get("op_margin_change_1y")
            s2 += f"; operating margin is {f['op_margin']*100:.0f}%" + (f" ({'wider' if ch > 0 else 'narrower'} than a year ago)" if ch is not None and abs(ch) >= 0.01 else "")
            s2 += "; profits are backed by cash." if f.get("cash_backed") else "; profits are running ahead of cash flow."
        else: s2 += "."
        S.append({"topic": "business", "text": s2})
    elif f.get("stale"):
        S.append({"topic": "business", "text": "Its latest SEC financial report is more than six months old, so business trends aren't summarised."})
    # 3. insiders + filings
    ins = await pool.fetch("""SELECT raw_payload->'form4' f FROM ci_raw_evidence WHERE form_type='4' AND cik=$1 AND filed_at > NOW() - INTERVAL '90 days'""", r["cik"])
    buy = sell = 0.0
    for x in ins:
        fx = json.loads(x["f"]) if isinstance(x["f"], str) else (x["f"] or {})
        for t in fx.get("transactions", []):
            if not t.get("open_market"): continue
            if t.get("code") == "P": buy += t.get("value") or 0
            else: sell += t.get("value") or 0
    mat = await pool.fetchval("""SELECT count(*) FROM ci_events WHERE ticker=$1 AND significance='MATERIAL' AND available_at > NOW() - INTERVAL '90 days'""", tk)
    s3 = ("Insiders haven't traded on the open market in the last 90 days" if buy + sell == 0 else
          f"Insiders {'bought' if buy > sell else 'sold'} a net ${abs(buy - sell)/1e6:,.1f}M on the open market in the last 90 days")
    s3 += f", and it filed {mat} material SEC report{'s' if mat != 1 else ''} in that time." if mat else ", with no material SEC filings in that time."
    S.append({"topic": "filings", "text": s3})
    # 4. where it shows up (trackers, warnings, breakthroughs)
    shows = []; warns = []
    try:
        from routers.trackers_router import membership, _press_findings
        m = (await membership(request, tier=r["tier"])).get("members", {}).get(tk, []) if r["tier"] in ("large", "mid", "small") else []
        names = {"on-sale": "Great companies on sale", "quiet": "Quiet climbers", "better": "Getting better", "rising": "Rising stars"}
        shows = [names[x] for x in m if x in names]; warns = ["Warning signs"] if "warn" in m else []
        bt = [x for x in (await _press_findings(pool, [tk], days=90)).get(tk, []) if x["direction"] == "positive" and x["type"] != "record_results"]
    except Exception:
        bt = []
    if shows or warns or bt:
        s4 = (f"It's on QuantEdge's {', '.join(shows)} tracker{'s' if len(shows) > 1 else ''}" if shows else "It isn't on any growth or value tracker")
        if warns: s4 += ", and it has warning signs worth reading"
        if bt: s4 += f"; its latest breakthrough: {bt[0]['label'].lower()} on {bt[0]['date']}"
        S.append({"topic": "signals", "text": s4 + "."})
    # 5. models
    try:
        rep = json.load(open("/app/models/panel/training_report.json"))
        ok = [v.get("horizon_label") for v in rep.get("horizons", {}).values() if v.get("reliable")]
        S.append({"topic": "models", "text": (f"QuantEdge's forecasts hold up on recent data only at the {', '.join(ok)} horizon{'s' if len(ok) > 1 else ''}." if ok else
                                              "None of QuantEdge's forecasts currently hold up on recent data, so they aren't used in this summary.")})
    except Exception:
        pass
    last_q = (f.get("quarters") or [{}])[-1].get("end") if f.get("available") else None
    nxt = None
    if last_q:
        n_ = dt.date.fromisoformat(last_q) + dt.timedelta(days=126)
        while n_ < dt.date.today(): n_ += dt.timedelta(days=91)
        nxt = str(n_)
    return {"ticker": tk, "name": r["name"], "as_of": str(r["as_of"]), "sentences": S, "trackers": shows, "has_warnings": bool(warns),
            "breakthroughs": bt[:3], "next_results_est": nxt, "tier": r["tier"], "sector": r["sector"], "history_note": r["history_note"],
            "note": "Written from SEC filings, prices and QuantEdge's nightly facts. Not advice."}
