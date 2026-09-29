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
    facts = None
    if f.get("available"):
        qs = f.get("quarters") or []; last4 = qs[-4:]
        rev = sum(q["sales"] for q in last4) if len(last4) == 4 else None
        ni = f.get("net_income_ttm")
        def _ttm_margin(key):
            if len(last4) != 4 or any(q.get(key) is None or not q.get("sales") for q in last4): return None
            return sum(q[key] * q["sales"] for q in last4) / sum(q["sales"] for q in last4)
        facts = {"revenue_ttm": rev, "ni_ttm": ni, "gross_margin_ttm": _ttm_margin("gross_margin"), "op_margin_ttm": _ttm_margin("op_margin"),
                 "sales_yoy": f.get("sales_yoy"), "gross_margin": f.get("gross_margin"),
                 "op_margin": f.get("op_margin"), "net_margin_ttm": (ni / rev) if (ni is not None and rev) else None,
                 "last_quarter": qs[-1]["end"] if qs else None, "source": "SEC filings"}
    profile = None
    rd = getattr(request.app.state, "redis", None); pk = f"profile:{tk}"
    try:
        hit = await rd.get(pk) if rd is not None else None
        if hit: profile = json.loads(hit)
    except Exception: pass
    if profile is None:
        try:
            import os, httpx
            async with httpx.AsyncClient(timeout=10) as cx:
                res = (await cx.get(f"https://api.polygon.io/v3/reference/tickers/{tk}", params={"apiKey": os.environ.get("POLYGON_API_KEY", "")})).json().get("results") or {}
            profile = {"description": res.get("description"), "website": res.get("homepage_url"), "employees": res.get("total_employees"),
                       "listed": res.get("list_date"), "industry": (res.get("sic_description") or "").capitalize() or None, "source": "Polygon company profile"}
            if rd is not None and profile.get("description"): await rd.setex(pk, 7 * 86400, json.dumps(profile))
        except Exception:
            profile = None
    return {"ticker": tk, "name": r["name"], "as_of": str(r["as_of"]), "sentences": S, "facts": facts, "profile": profile,
            "insiders_90d": {"bought": buy, "sold": sell}, "material_filings_90d": mat, "trackers": shows, "has_warnings": bool(warns),
            "breakthroughs": bt[:3], "next_results_est": nxt, "tier": r["tier"], "sector": r["sector"], "history_note": r["history_note"],
            "note": "Written from SEC filings, prices and QuantEdge's nightly facts. Not advice."}


@router.get("/segments/{ticker}")
async def segments(ticker: str, request: Request):
    """Revenue by product/service, business segment and region from the latest 10-K."""
    tk = ticker.upper().strip(); pool = request.app.state.db
    cik = await pool.fetchval("SELECT cik FROM company_facts WHERE ticker=$1 ORDER BY as_of DESC LIMIT 1", tk)
    if not cik: raise HTTPException(status_code=404, detail=f"{tk} is not in QuantEdge's company universe")
    from quantedge.fundamentals.edgar_bulk import UA
    from quantedge.fundamentals.segments import get_segments
    try:
        d = await get_segments(pool, cik, tk, UA)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"could not read the 10-K: {type(e).__name__}")
    if not d: return {"ticker": tk, "available": False, "note": "This company's latest 10-K doesn't break revenue down by product, segment or region in a readable form."}
    return {"ticker": tk, "available": True, **d}


@router.get("/price-stats/{ticker}")
async def price_stats(ticker: str, request: Request):
    """One source for price statistics everywhere on the page (header, Summary, Price & patterns),
    computed from stored daily closes, with the window stated for every number."""
    import numpy as np
    tk = ticker.upper().strip(); pool = request.app.state.db
    rows = await pool.fetch("SELECT d, c FROM daily_bars WHERE ticker=$1 AND c > 0 ORDER BY d", tk)
    if len(rows) < 30: raise HTTPException(status_code=404, detail=f"not enough price history for {tk}")
    spy = {r["d"]: float(r["c"]) for r in await pool.fetch("SELECT d, c FROM daily_bars WHERE ticker='SPY' AND c > 0 ORDER BY d")}
    d = [r["d"] for r in rows]; c = np.array([float(r["c"]) for r in rows]); n = len(c)
    lr = np.diff(np.log(c))
    vol = lambda k: float(np.std(lr[-k:], ddof=1) * np.sqrt(252)) if len(lr) >= k else None
    def ret(k): return float(c[-1] / c[-1 - k] - 1) if n > k else None
    def spy_ret(k):
        if n <= k or d[-1] not in spy or d[-1 - k] not in spy: return None
        return spy[d[-1]] / spy[d[-1 - k]] - 1
    def dd(arr): return float(np.min(arr / np.maximum.accumulate(arr) - 1)) if len(arr) else None
    beta = None; corr = None
    common = [i for i in range(max(1, n - 252), n) if d[i] in spy and d[i - 1] in spy]
    if len(common) >= 120:
        a = np.array([c[i] / c[i - 1] - 1 for i in common]); b = np.array([spy[d[i]] / spy[d[i - 1]] - 1 for i in common])
        beta = float(np.cov(a, b)[0, 1] / np.var(b, ddof=1)) if np.var(b) > 0 else None
        corr = float(np.corrcoef(a, b)[0, 1]) if np.var(b) > 0 and np.var(a) > 0 else None
    sma = lambda k: float(c[-k:].mean()) if n >= k else None
    s50, s200 = sma(50), sma(200); w = c[-252:]
    out = {"ticker": tk, "as_of": str(d[-1]), "history_start": str(d[0]), "price": float(c[-1]),
           "vol_1m": vol(21), "vol_1y": vol(252), "daily_move_typical": (vol(252) / np.sqrt(252)) if vol(252) else None,
           "beta_1y": beta, "corr_1y": corr, "max_drawdown_1y": dd(w), "max_drawdown_all": dd(c),
           "sma50": s50, "sma200": s200, "above_50d": bool(c[-1] > s50) if s50 else None, "above_200d": bool(c[-1] > s200) if s200 else None,
           "high_52w": float(w.max()), "low_52w": float(w.min()), "pct_from_52w_high": float(c[-1] / w.max() - 1),
           "returns": {}}
    for lab, k in (("1m", 21), ("3m", 63), ("6m", 126), ("1y", 252)):
        r_, s_ = ret(k), spy_ret(k)
        out["returns"][lab] = {"stock": r_, "sp500": s_, "vs_sp500": (r_ - s_) if (r_ is not None and s_ is not None) else None}
    # plain Python types only (numpy values can't be sent as JSON), and no NaN/inf
    import math
    def clean(o):
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.floating, float)): return float(o) if math.isfinite(float(o)) else None
        if isinstance(o, (np.integer,)): return int(o)
        return o
    return clean(out)


async def real_peers(pool, tk: str, n: int = 8):
    """One definition of a company's peers, used everywhere: same 4-digit SIC industry if it has
    6+ members, else the 3-digit group, else the sector — then the n closest in market value.
    (The stored group list was read back alphabetically, so Apple's "peers" were six A-tickers.)"""
    import math
    me = await pool.fetchrow("SELECT * FROM company_facts WHERE ticker=$1 ORDER BY as_of DESC LIMIT 1", tk)
    if not me: return None, [], None
    sic = str(await pool.fetchval("SELECT sic_code FROM universe WHERE ticker=$1", tk) or "")
    cand = await pool.fetch("""SELECT f.*, u.sic_code FROM company_facts f JOIN universe u ON u.ticker = f.ticker
        WHERE f.as_of = $1 AND f.primary_listing AND NOT f.is_spac AND f.data_suspect IS NULL AND f.ticker <> $2
          AND f.tier IN ('large','mid','small') AND f.fundamentals IS NOT NULL""", me["as_of"], tk)
    group, members = None, []
    for label, pref in (("same industry", sic[:4]), ("same industry group", sic[:3])):
        if len(pref) < 3: continue
        mm = [c for c in cand if str(c["sic_code"] or "").startswith(pref)]
        if len(mm) >= 6 or (label == "same industry group" and len(mm) >= 3): group, members = f"{label} (SIC {pref})", mm; break
    if not members:
        members = [c for c in cand if c["sector"] == me["sector"]]; group = f"same sector ({me['sector']})"
    mc0 = me["market_cap"] or 1
    members = sorted(members, key=lambda c: abs(math.log((c["market_cap"] or 1) / mc0)))[:n]
    return group, members, me


@router.get("/valuation-view/{ticker}")
async def valuation_view(ticker: str, request: Request):
    """Valuation that says what it assumes: implied growth vs actual, P/E and P/S against the
    company's own history (point-in-time: each quarter counted from its filing date), real
    peers (same detailed industry, closest in size), and each valuation method on its own line."""
    import math, numpy as np
    tk = ticker.upper().strip(); pool = request.app.state.db
    me = await pool.fetchrow("SELECT * FROM company_facts WHERE ticker=$1 ORDER BY as_of DESC LIMIT 1", tk)
    if not me: raise HTTPException(status_code=404, detail=f"{tk} is not in QuantEdge's company universe")
    fme = json.loads(me["fundamentals"]) if isinstance(me["fundamentals"], str) else (me["fundamentals"] or {})
    # ---- own history: P/E and P/S every trading day, point-in-time ----
    import os
    from ml.fundamentals.quality_engine import fetch_quarterly_financials
    pq = await fetch_quarterly_financials(tk, os.environ.get("POLYGON_API_KEY", ""), limit=24)
    qs = []
    for q in pq or []:
        pe_ = str(getattr(q, "period_end", "") or "")[:10]
        if not pe_: continue
        fd = str(getattr(q, "filing_date", "") or "")[:10] or str(dt.date.fromisoformat(pe_) + dt.timedelta(days=45))
        qs.append({"end": pe_, "avail": fd, "eps": getattr(q, "eps_diluted", None), "rev": getattr(q, "revenue", None), "sh": getattr(q, "diluted_shares", None)})
    qs.sort(key=lambda x: x["end"])
    # Split-adjust the filings: prices are split-adjusted, reported EPS and share counts are not
    # (Nvidia's 10-for-1 in June 2024 made its pre-split P/E look ten times too low). A jump in
    # share count by ~2x, 3x, 4x, 5x, 10x or 20x between quarters is a split; adjust earlier quarters.
    for i in range(1, len(qs)):
        a, b = qs[i - 1]["sh"], qs[i]["sh"]
        if not a or not b: continue
        r = b / a; k = None
        for cand in (2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50):
            if abs(r / cand - 1) < 0.08: k = cand; break
            if abs((1 / r) / cand - 1) < 0.08: k = 1 / cand; break
        if k:
            for j in range(i):
                if qs[j]["eps"] is not None: qs[j]["eps"] /= k
                if qs[j]["sh"]: qs[j]["sh"] *= k
    bars = await pool.fetch("SELECT d, c FROM daily_bars WHERE ticker=$1 AND c > 0 ORDER BY d", tk)
    hist = {"pe": [], "ps": []}
    for b in bars:
        known = [q for q in qs if q["avail"] <= str(b["d"])][-4:]
        if len(known) < 4: continue
        eps = [q["eps"] for q in known]; rev = [q["rev"] for q in known]; sh = known[-1]["sh"]
        if None not in eps and sum(eps) > 0:
            v_ = float(b["c"]) / sum(eps)
            if 0 < v_ < 1000: hist["pe"].append(v_)
        if None not in rev and sh and sum(rev) > 0 and sh > 0:
            v_ = float(b["c"]) / (sum(rev) / sh)
            if 0.01 < v_ < 500: hist["ps"].append(v_)          # impossible values are data errors, not history
    def band(xs):
        if len(xs) < 60: return None
        a = np.array(xs); med0 = np.median(a); now = a[-1]
        a = a[(a > med0 / 8) & (a < med0 * 8)]          # 8x away from its own median is bad data, not history
        if len(a) < 60: return None
        return {"now": float(now), "median": float(np.median(a)), "low": float(np.percentile(a, 5)), "high": float(np.percentile(a, 95)),
                "percentile_now": float((a < now).mean() * 100), "days": int(len(a))}
    own = {"pe": band(hist["pe"]), "ps": band(hist["ps"]), "since": str(bars[0]["d"]) if bars else None}
    # ---- real peers: the shared definition ----
    group, members, _ = await real_peers(pool, tk)
    def row(c, is_self=False):
        f = json.loads(c["fundamentals"]) if isinstance(c["fundamentals"], str) else (c["fundamentals"] or {})
        q4 = (f.get("quarters") or [])[-4:]; rev = sum(q["sales"] for q in q4) if len(q4) == 4 else None
        ni = f.get("net_income_ttm"); mc = c["market_cap"]
        pe_ = (mc / ni) if (mc and ni and ni > 0) else None
        return {"ticker": c["ticker"], "name": c["name"], "self": is_self, "market_cap": mc,
                "pe": pe_ if (pe_ is None or pe_ <= 200) else None, "pe_note": ("n/m — very thin profits" if (pe_ and pe_ > 200) else "loss-making" if (ni is not None and ni <= 0) else "no earnings data" if ni is None else None),
                "loss_making": bool(ni is not None and ni <= 0),
                "ps": (mc / rev) if (mc and rev) else None, "sales_growth": f.get("sales_yoy"), "op_margin": f.get("op_margin"),
                "ret_1y": c["ret_1y"]}
    peers = [row(me, True)] + [row(c) for c in members]
    def med(k): xs = [p[k] for p in peers[1:] if p[k] is not None]; return float(np.median(xs)) if xs else None
    # ---- valuation methods, each on its own line ----
    try:
        from routers.valuation_router import get_valuation
        v = await get_valuation(tk, request, None)
    except Exception:
        v = None
    vd = (v or {}).get("data", v) if isinstance(v, dict) else {}
    km = (vd or {}).get("key_metrics") or {}
    price = me["price"]
    methods = []
    for key, name, what in (("dcf_bear", "DCF · bear case", "cash flows grow slowly, then fade"),
                            ("dcf_base", "DCF · base case", f"discounted at {((vd or {}).get('wacc_used') or 0)*100:.1f}% a year, growth fading to ~3%"),
                            ("dcf_bull", "DCF · bull case", "cash flows keep growing faster for longer"),
                            ("epv_per_share", "Earnings power value", "today's earnings, no growth at all"),
                            ("graham_number", "Graham number", "Benjamin Graham's rule of thumb from earnings and book value"),
                            ("residual_income_value", "Residual income value", "book value plus future profits above the cost of equity")):
        val = km.get(key)
        ok = val is not None and val > 0
        outlier = ok and price and (val > 3 * price or val < 0.25 * price)
        methods.append({"method": name, "assumes": what, "value": val if ok else None,
                        "vs_price": ((val / price - 1) if (ok and price) else None), "outlier": bool(outlier),
                        "note": ("outlier — this method's assumptions don't fit this company (e.g. tiny book value after buybacks)" if outlier
                                 else None if (val is None or ok) else "not meaningful — negative (loss-making or negative equity)")})
    implied = km.get("reverse_dcf_implied_growth")
    return {"ticker": tk, "price": price, "as_of": str(me["as_of"]), "loss_making": bool(fme.get("net_income_ttm") is not None and fme["net_income_ttm"] <= 0),
            "implied_growth": implied, "actual_sales_growth": fme.get("sales_yoy"),
            "own_history": own, "peers": {"group": group, "rows": peers, "median": {k: med(k) for k in ("pe", "ps", "sales_growth", "op_margin", "ret_1y")}},
            "methods": methods,
            "note": "History uses earnings as known on each date (quarters counted from their filing date). Peers: same detailed industry, closest in size, from SEC-based facts. Valuation methods depend heavily on their assumptions."}



@router.get("/analysts/{ticker}")
async def analysts(ticker: str, request: Request):
    """Analyst recommendation counts month by month (Finnhub). Firm-by-firm ratings and price
    targets need a paid feed and are not shown."""
    import os, httpx
    tk = ticker.upper().strip()
    rd = getattr(request.app.state, "redis", None); ck = f"analysts:{tk}"
    try:
        hit = await rd.get(ck) if rd is not None else None
        if hit: return json.loads(hit)
    except Exception: pass
    key = os.environ.get("FINNHUB_API_KEY") or os.environ.get("FINNHUB_KEY") or ""
    if not key:
        try:
            from core.config import settings
            key = getattr(settings, "FINNHUB_API_KEY", "") or ""
        except Exception: key = ""
    if not key: raise HTTPException(status_code=503, detail="analyst data source not configured")
    async with httpx.AsyncClient(timeout=15) as cx:
        rows = (await cx.get("https://finnhub.io/api/v1/stock/recommendation", params={"symbol": tk, "token": key})).json() or []
    months = sorted([{"period": r.get("period"), "strong_buy": r.get("strongBuy", 0), "buy": r.get("buy", 0), "hold": r.get("hold", 0),
                      "sell": r.get("sell", 0), "strong_sell": r.get("strongSell", 0)} for r in rows if r.get("period")], key=lambda x: x["period"])[-12:]
    def bshare(m):
        n = m["strong_buy"] + m["buy"] + m["hold"] + m["sell"] + m["strong_sell"]
        return ((m["strong_buy"] + m["buy"]) / n, n) if n else (None, 0)
    out = {"ticker": tk, "months": months, "source": "Finnhub recommendation trends", "available": bool(months)}
    if months:
        now_s, now_n = bshare(months[-1]); then = months[-7] if len(months) >= 7 else months[0]; then_s, then_n = bshare(then)
        out.update({"latest": months[-1], "buy_share_now": now_s, "analysts_now": now_n, "buy_share_then": then_s, "then_period": then["period"]})
    try:
        if rd is not None: await rd.setex(ck, 12 * 3600, json.dumps(out))
    except Exception: pass
    return out


_STOP = set("the a an and or of to in on for with at by from as is are was were be it its this that stock stocks shares share inc corp company co ltd".split())
def _toks(t):
    import re as _re
    return {w for w in _re.findall(r"[a-z0-9]+", (t or "").lower()) if len(w) > 3 and w not in _STOP}


EVENT_RULES = [   # (kind, weight, pattern) — transparent rules; the old classifier labelled everything "commentary"
    ("results", 5.0, r"\b(earnings|results|quarter(ly)?|q[1-4]\b|fiscal|revenue|sales|profits?|eps|beats?|miss(es|ed)?|guidance|outlook|forecasts?|reports?|posts?|deliveries)\b"),
    ("leadership", 4.0, r"\b(ceo|cfo|chief executive|chief financial|steps? down|resign\w*|appoint\w*|names? .{0,30}(ceo|cfo|chief)|successor|hands? off|retire\w*)\b"),
    ("deal", 4.0, r"\b(acquir\w*|acquisition|merger|merge|buyout|takeover|to buy|stake in|divest\w*|spin[- ]?off)\b"),
    ("capital", 3.5, r"\b(offering|share sale|dilut\w*|buyback|repurchase|dividend|convertible|notes due|raises \$|priced)\b"),
    ("legal", 3.5, r"\b(lawsuit|sues|sued|antitrust|probe|investigat\w*|doj|ftc|sec charges|fine[ds]?|settle\w*|recall\w*|ban(s|ned)?|tariffs?|regulator\w*)\b"),
    ("product", 3.0, r"\b(launch\w*|unveil\w*|introduc\w*|fda|approv\w*|clearance|trial|contract|partnership|agreement|orders?)\b"),
    ("analyst", 2.0, r"\b(upgrade\w*|downgrade\w*|price target|initiat\w*|overweight|underweight|outperform|underperform)\b"),
]
_COMMENTARY = r"(\?\s*$|^\s*(why|is|should|can|will|what|how|here'?s|\d+ )\b|buy the dip|should investors|is it time|best stock|top \d|millionaire|fantastic|no[- ]brainer|screaming|forever|could soar|poised to|what (you|investors) need to know)"
_MARKET = r"\b(stock market today|dow jones|s&p 500|nasdaq composite|wall street (today|closes|opens)|market wrap)\b"
_PUB_W = {"Reuters": 3, "Bloomberg": 3, "The Wall Street Journal": 3, "Financial Times": 3, "Associated Press": 3, "CNBC": 2, "Barron's": 2,
          "MarketWatch": 2, "Business Wire": 2.5, "PR Newswire": 2.5, "GlobeNewswire": 2.5, "Benzinga": 1, "The Motley Fool": 0.5,
          "Zacks Investment Research": 0.5, "InvestorPlace": 0.5, "24/7 Wall St.": 0.5, "Seeking Alpha": 0.8}


@router.get("/news-view/{ticker}")
async def news_view(ticker: str, request: Request):
    """News as events: articles about the same event are merged (type + date), ranked by what
    happened, how widely it was covered and how the stock moved — commentary last. Tone labels are
    Polygon's automated reading of each article (the vendor's AI), shown as such."""
    import os, re, math, httpx, datetime as _d
    from zoneinfo import ZoneInfo
    tk = ticker.upper().strip(); pool = request.app.state.db
    since = (_d.date.today() - _d.timedelta(days=90)).isoformat()
    async with httpx.AsyncClient(timeout=30) as cx:
        res = (await cx.get("https://api.polygon.io/v2/reference/news", params={"ticker": tk, "published_utc.gte": since, "limit": 1000,
                                                                              "order": "desc", "apiKey": os.environ.get("POLYGON_API_KEY", "")})).json().get("results", [])
    bars = await pool.fetch("SELECT d, c FROM daily_bars WHERE ticker=$1 AND d >= $2::date - 5 ORDER BY d", tk, _d.date.fromisoformat(since))
    bd = [b["d"] for b in bars]; bc = [float(b["c"]) for b in bars]
    def tday(ts):
        try: t = _d.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(ZoneInfo("America/New_York"))
        except Exception: return None
        eff = t.date() + (_d.timedelta(days=1) if t.hour >= 16 else _d.timedelta(0))
        return next((i for i, x in enumerate(bd) if x >= eff), None)
    def mv(i): return (bc[i] / bc[i - 1] - 1) if (i is not None and 0 < i < len(bc)) else None
    rules = [(k, w, re.compile(p, re.I)) for k, w, p in EVENT_RULES]
    # Relevance: Polygon tags an article with every ticker it mentions (Apple's feed carried Berkshire's
    # succession, Nvidia's a 6G market report). A story is the company's news only if the headline names it.
    nm = (await pool.fetchval("SELECT name FROM company_facts WHERE ticker=$1 ORDER BY as_of DESC LIMIT 1", tk)) or ""
    base = re.sub(r"(,?\s+(inc|corp|corporation|co|company|ltd|plc|holdings?|group|class [a-z]|common stock|ordinary shares|n\.?v\.?|s\.?a\.?)\b\.?)+.*$", "", nm, flags=re.I).strip()
    aliases = {tk, base} | ({base.split()[0]} if base and len(base.split()[0]) > 3 else set())
    aliases |= {"GOOGL": {"Google", "Alphabet"}, "GOOG": {"Google", "Alphabet"}, "META": {"Meta", "Facebook"}, "BRK.B": {"Berkshire"}}.get(tk, set())
    about_rx = re.compile(r"\b(" + "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True) if a) + r")\b", re.I)
    com, mkt = re.compile(_COMMENTARY, re.I), re.compile(_MARKET, re.I)
    arts = []
    for r in res:
        ins = next((x for x in (r.get("insights") or []) if x.get("ticker") == tk), {})
        title = (r.get("title") or "").strip(); pub = (r.get("publisher") or {}).get("name") or ""
        kind, w = None, 0.0
        for k, wt, rx in rules:
            if rx.search(title): kind, w = k, wt; break
        commentary = bool(com.search(title)); market = bool(mkt.search(title))
        if not about_rx.search(title): continue          # mentions only — not this company's news
        i = tday(r.get("published_utc") or "")
        arts.append({"title": title, "url": r.get("article_url"), "publisher": pub, "published": (r.get("published_utc") or "")[:16].replace("T", " "),
                     "i": i, "kind": ("market" if market else kind or ("commentary" if commentary else "other")), "w": w,
                     "commentary": commentary or market, "tone": ins.get("sentiment"), "tone_reason": ins.get("sentiment_reasoning"),
                     "pub_w": _PUB_W.get(pub, 1.0), "_t": _toks(title)})
    # 1) events: same (non-commentary) type within 2 trading days
    events = []
    for a in sorted([x for x in arts if not x["commentary"] and x["kind"] not in ("other",) and x["i"] is not None], key=lambda x: x["i"]):
        e = next((e for e in events if e["kind"] == a["kind"] and abs(e["i"] - a["i"]) <= 2), None)
        if e: e["members"].append(a)
        else: events.append({"kind": a["kind"], "w": a["w"], "i": a["i"], "members": [a]})
    # 2) commentary and untyped articles join an event they clearly discuss (same days, shared words), else stand alone
    loose = []
    for a in arts:
        if not a["commentary"] and a["kind"] not in ("other",): continue
        home = None
        if a["i"] is not None:
            for e in events:
                if abs(e["i"] - a["i"]) <= 1 and any(len(a["_t"] & m["_t"]) >= 2 for m in e["members"]): home = e; break
        if home: home["members"].append(a)
        else: loose.append(a)
    out = []
    for e in events:
        ms = e["members"]; lead = max(ms, key=lambda m: (not m["commentary"], m["pub_w"]))
        m0, m1 = mv(e["i"]), mv(e["i"] + 1 if e["i"] is not None else None)
        outlets = sorted({m["publisher"] for m in ms})
        big = max(abs(m0 or 0), abs(m1 or 0))
        score = e["w"] + math.log2(len(ms) + 1) + min(3.0, big * 30) + (lead["pub_w"] - 1) * 0.3
        out.append({"type": e["kind"], "headline": lead["title"], "url": lead["url"], "source": lead["publisher"], "published": lead["published"],
                    "event_day": str(bd[e["i"]]) if e["i"] is not None and e["i"] < len(bd) else None, "move_event_day": m0, "move_next_day": m1,
                    "n_articles": len(ms), "outlets": outlets, "score": round(score, 2), "commentary": False,
                    "tone": lead["tone"], "tone_reason": lead["tone_reason"],
                    "also": [{"title": m["title"], "url": m["url"], "source": m["publisher"]} for m in sorted(ms, key=lambda m: -m["pub_w"])[:6] if m is not lead]})
    for a in loose:
        m0 = mv(a["i"])
        out.append({"type": a["kind"], "headline": a["title"], "url": a["url"], "source": a["publisher"], "published": a["published"],
                    "event_day": str(bd[a["i"]]) if a["i"] is not None and a["i"] < len(bd) else None, "move_event_day": m0, "move_next_day": None,
                    "n_articles": 1, "outlets": [a["publisher"]], "score": round(0.5 + (a["pub_w"] - 1) * 0.2, 2), "commentary": True,
                    "tone": a["tone"], "tone_reason": a["tone_reason"], "also": []})
    # ---- what happened: the company's own 8-K filings (authoritative, complete, dated) ----
    ITEM = {"2.02": ("results", 5.0, "Released quarterly results"), "5.02": ("leadership", 3.0, "Leadership change"),
            "1.01": ("agreement", 3.5, "Signed a material agreement"), "1.02": ("agreement", 2.5, "Ended a material agreement"),
            "2.01": ("deal", 4.0, "Completed an acquisition or sale of assets"), "2.03": ("capital", 3.0, "Took on a material debt obligation"),
            "3.02": ("capital", 3.5, "Sold shares not registered publicly"), "2.05": ("restructuring", 4.0, "Announced restructuring or exit costs"),
            "2.06": ("impairment", 4.5, "Recorded an impairment"), "4.01": ("auditor", 4.5, "Changed its auditor"),
            "4.02": ("restatement", 5.0, "Said past financial statements can't be relied on"), "8.01": ("announcement", 2.5, "Made another material announcement"),
            "7.01": ("disclosure", 1.5, "Published an investor presentation or disclosure"), "5.07": ("vote", 0.5, "Reported shareholder vote results"),
            "5.03": ("governance", 0.8, "Changed its bylaws or charter")}
    fil = await pool.fetch("""SELECT e.id, e.item_code, e.available_at, e.company_id cik, r.raw_payload->>'accession' acc, r.raw_payload->>'primaryDocument' doc
        FROM ci_events e JOIN ci_raw_evidence r ON r.id = e.evidence_id
        WHERE e.ticker = $1 AND r.form_type = '8-K' AND e.available_at >= $2::date ORDER BY e.available_at""", tk, _d.date.fromisoformat(since))
    ids = [f_["id"] for f_ in fil]
    der = await pool.fetch("""SELECT event_id, extractor_version, value FROM ci_derived WHERE event_id = ANY($1)
        AND extractor_version IN ('press-release-v3','press-release-v4','officer-change-v2')""", ids) if ids else []
    dmap = {}
    for x in der:
        v_ = json.loads(x["value"]) if isinstance(x["value"], str) else x["value"]; dmap.setdefault(x["event_id"], {})[x["extractor_version"]] = v_
    by_acc = {}
    for f_ in fil:
        g = by_acc.setdefault(f_["acc"], {"items": [], "ids": [], "at": f_["available_at"], "cik": f_["cik"], "doc": f_["doc"]})
        g["items"].append(f_["item_code"]); g["ids"].append(f_["id"])
    filings_out = []
    for acc, g in by_acc.items():
        known = [i_ for i_ in g["items"] if i_ in ITEM]
        if not known or set(known) <= {"5.07", "5.03", "7.01"} and len(known) == len(g["items"]) and "7.01" not in known: pass
        if not known: continue
        lead = max(known, key=lambda i_: ITEM[i_][1]); kind, w, what = ITEM[lead]
        find = []; officer = None
        for i_ in g["ids"]:
            dd = dmap.get(i_, {})
            find += (dd.get("press-release-v4") or dd.get("press-release-v3") or {}).get("findings", [])
            officer = officer or dd.get("officer-change-v2")
        if lead == "5.02" and officer:
            cls = officer.get("class")
            if cls == "abrupt_exec_exit": what, w = f"The {officer.get('role') or 'a senior officer'} is leaving" + (" (effective immediately)" if officer.get("immediate") else ""), 5.0
            elif cls == "planned_exec_transition": what, w = "Planned leadership transition", 3.0
            else: what, w = "Director or officer change", 1.0
        pos = [x for x in find if x.get("direction") == "positive"]; neg = [x for x in find if x.get("direction") == "negative"]
        key = (neg or pos or find or [None])[0]
        if key: w += 1.0
        at = g["at"]; i = tday(at.isoformat() if hasattr(at, "isoformat") else str(at))
        m0, m1 = mv(i), mv(i + 1 if i is not None else None)
        cov = [a for a in arts if a["i"] is not None and i is not None and i <= a["i"] <= i + 2]
        cov_top = sorted(cov, key=lambda a: (-(not a["commentary"]), -a["pub_w"]))[:3]
        score = w + math.log2(len(cov) + 1) * 0.6 + min(3.0, max(abs(m0 or 0), abs(m1 or 0)) * 30)
        filings_out.append({"type": kind, "what": what, "items": sorted(set(g["items"])), "filed": str(at)[:16],
                            "event_day": str(bd[i]) if i is not None and i < len(bd) else None, "move_event_day": m0, "move_next_day": m1,
                            "key_sentence": (key or {}).get("sentence"), "key_label": (key or {}).get("type"),
                            "filing_url": f"https://www.sec.gov/Archives/edgar/data/{int(g['cik'])}/{acc.replace('-', '')}/{g['doc']}" if g.get("cik") and g.get("doc") else None,
                            "coverage_articles": len(cov), "coverage_headlines": [{"title": a["title"], "url": a["url"], "source": a["publisher"], "commentary": a["commentary"]} for a in cov_top],
                            "score": round(score, 2)})
    ranked = sorted([o for o in out if not o["commentary"]], key=lambda o: -o["score"])
    top = ranked[:7] + sorted([o for o in out if o["commentary"]], key=lambda o: -o["score"])[:1]
    days = {}
    for a in arts:
        if a["i"] is None or a["i"] >= len(bd): continue
        k_ = str(bd[a["i"]]); e_ = days.setdefault(k_, {"n": 0, "pos": 0, "neg": 0})
        e_["n"] += 1; e_["pos"] += a["tone"] == "positive"; e_["neg"] += a["tone"] == "negative"
    series = [{"d": str(d), "close": c, **days.get(str(d), {"n": 0, "pos": 0, "neg": 0})} for d, c in zip(bd, bc) if str(d) >= since]
    group, members, me = await real_peers(pool, tk)
    peer_n = sorted([m_["news_30d"] for m_ in members if m_["news_30d"] is not None])
    kinds = {}
    for o in out: kinds[o["type"]] = kinds.get(o["type"], 0) + 1
    return _clean_json({"ticker": tk, "since": since, "what_happened": sorted(filings_out, key=lambda x: x["filed"], reverse=True),
                        "top_filings": sorted(filings_out, key=lambda x: -x["score"])[:6],
                        "n_articles": len(arts), "n_mentions_only": len(res) - len(arts), "aliases": sorted(a for a in aliases if a), "n_events": len([o for o in out if not o["commentary"]]),
                        "n_commentary": len([o for o in out if o["commentary"]]), "top": top,
                        "timeline": sorted(out, key=lambda o: o["published"], reverse=True)[:80], "series": series, "kinds": kinds,
                        "tones": {t: sum(1 for a in arts if a["tone"] == t) for t in ("positive", "neutral", "negative")},
                        "attention": {"news_30d": me["news_30d"] if me else None, "peer_median_30d": (peer_n[len(peer_n) // 2] if peer_n else None), "peer_group": group},
                        "tone_note": "Tone labels and their one-line reasons are Polygon's automated reading of each article (generated by the data vendor). They describe coverage; they don't predict the price.",
                        "note": "Articles about the same event are merged. Moves are close-to-close on the event day and the day after (news after 4pm ET counts toward the next day)."})


def _clean_json(o):
    import math
    if isinstance(o, float): return o if math.isfinite(o) else None
    if isinstance(o, dict): return {k: _clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)): return [_clean_json(v) for v in o]
    return o



@router.get("/wiki-attention/{ticker}")
async def wiki_attention(ticker: str, request: Request):
    """Public attention: daily English-Wikipedia page views of the company's article (Wikimedia, free).
    The article is found through Wikidata's stock-ticker records (not a name search, so 'Apple' is Apple Inc.)."""
    import httpx, urllib.parse, datetime as _d
    tk = ticker.upper().strip()
    rd = getattr(request.app.state, "redis", None); ck = f"wiki:{tk}"
    try:
        hit = await rd.get(ck) if rd is not None else None
        if hit: return json.loads(hit)
    except Exception: pass
    UA = {"User-Agent": "QuantEdge/1.0 (research site; dileepkreddy5@gmail.com)"}
    q = ('SELECT ?article WHERE { ?item p:P414 ?s . ?s pq:P249 "%s" . ?article schema:about ?item ; '
         'schema:isPartOf <https://en.wikipedia.org/> . } LIMIT 3') % tk.replace('"', '')
    out = {"ticker": tk, "available": False}
    try:
        async with httpx.AsyncClient(timeout=25, headers=UA) as cx:
            b = (await cx.get("https://query.wikidata.org/sparql", params={"query": q, "format": "json"})).json().get("results", {}).get("bindings", [])
            if b:
                url = b[0]["article"]["value"]; title = url.rsplit("/", 1)[-1]
                end = _d.date.today() - _d.timedelta(days=1); start = end - _d.timedelta(days=120)
                pv = (await cx.get(f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/{title}/daily/{start:%Y%m%d}00/{end:%Y%m%d}00")).json().get("items", [])
                series = [{"d": f"{x['timestamp'][:4]}-{x['timestamp'][4:6]}-{x['timestamp'][6:8]}", "views": x["views"]} for x in pv]
                if len(series) >= 30:
                    v = [x["views"] for x in series]; last7 = sum(v[-7:]) / 7; base = sum(v[-97:-7]) / max(1, len(v[-97:-7]))
                    peak = max(series, key=lambda x: x["views"])
                    out = {"ticker": tk, "available": True, "article": urllib.parse.unquote(title).replace("_", " "), "url": url, "series": series,
                           "avg_7d": last7, "avg_prior_90d": base, "ratio": (last7 / base) if base else None, "peak": peak,
                           "note": "Daily page views of the company's English Wikipedia article (human readers only), from Wikimedia. A rise shows public attention, not direction."}
    except Exception as e:
        out = {"ticker": tk, "available": False, "reason": f"Wikipedia data unavailable ({type(e).__name__})"}
    try:
        if rd is not None: await rd.setex(ck, 12 * 3600, json.dumps(out))
    except Exception: pass
    return out
