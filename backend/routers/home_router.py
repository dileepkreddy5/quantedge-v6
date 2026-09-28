"""Homepage data: markets (US index ETFs, world ETFs, six sectors), market mood,
biggest movers, and name/ticker search. Polygon snapshot for today's moves
(timestamped with the quote time it returns), daily_bars for week and history.
Everything cached in Redis so visitor traffic never multiplies API calls."""
from __future__ import annotations
import json, os
from datetime import datetime, timezone
import numpy as np
import httpx
from fastapi import APIRouter, Query, Request

router = APIRouter()
POLY = os.environ.get("POLYGON_API_KEY", "")
SNAP = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"
US = [("SPY", "S&P 500", "via SPY"), ("QQQ", "Nasdaq-100", "via QQQ"), ("DIA", "Dow Jones", "via DIA"),
      ("VTI", "Total US market", "NYSE proxy · via VTI"), ("IWM", "Small caps", "Russell 2000 · via IWM")]
WORLD = [("EWJ", "Japan"), ("EWU", "UK"), ("EWG", "Germany"), ("VGK", "Europe"), ("MCHI", "China"), ("INDA", "India")]
# Largest members of each standard sector; ranked live by our market caps, top 4 shown.
SECTORS = [("Technology", "XLK", ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "CSCO"]),
           ("Healthcare", "XLV", ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "ISRG", "AMGN"]),
           ("Financials", "XLF", ["JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "BRK.B"]),
           ("Energy", "XLE", ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "MPC", "OXY", "WMB"]),
           ("Consumer", "XLY", ["AMZN", "TSLA", "HD", "MCD", "LOW", "BKNG", "TJX", "NKE", "SBUX"]),
           ("Communication", "XLC", ["GOOGL", "META", "NFLX", "TMUS", "DIS", "T", "VZ", "CMCSA"])]


def _nice(name):
    n = (name or "").replace(" Class A Common Stock", "").replace(" Common Stock", "").strip()
    if n.isupper():   # some filers' names arrive in capitals
        n = " ".join(w if w in ("INC.", "INC", "CORP.", "CORP", "LLC", "PLC", "N.V.", "S.A.") else w.capitalize() for w in n.split())
        n = n.replace("INC.", "Inc.").replace("CORP.", "Corp.").replace(" INC", " Inc").replace(" CORP", " Corp")
    return n


async def _cached(request, key, ttl, fn):
    r = getattr(request.app.state, "redis", None)
    if r is not None:
        try:
            hit = await r.get(key)
            if hit: return json.loads(hit)
        except Exception: pass
    val = await fn()
    if r is not None and val is not None:
        try: await r.setex(key, ttl, json.dumps(val, default=str))
        except Exception: pass
    return val


async def _snap(tickers):
    out = {}
    async with httpx.AsyncClient(timeout=25) as cx:
        for i in range(0, len(tickers), 200):
            try:
                r = await cx.get(SNAP, params={"tickers": ",".join(tickers[i:i + 200]), "apiKey": POLY})
                if r.status_code != 200: continue
                for t in r.json().get("tickers", []):
                    day = t.get("day") or {}; mn = t.get("min") or {}
                    price = (t.get("lastTrade") or {}).get("p") or mn.get("c") or day.get("c")
                    out[t["ticker"]] = {"chg_pct": t.get("todaysChangePerc"), "price": price,
                                        "volume": day.get("v") or 0, "updated_ns": t.get("updated")}
            except Exception:
                continue
    return out


async def _hist(pool, tickers, n=22):
    rows = await pool.fetch("""SELECT ticker, d, c FROM (SELECT ticker, d, c, row_number() OVER (PARTITION BY ticker ORDER BY d DESC) rn
                                 FROM daily_bars WHERE ticker = ANY($1)) x WHERE rn <= $2 ORDER BY ticker, d""", tickers, n)
    h = {}
    for r in rows: h.setdefault(r["ticker"], []).append((r["d"], float(r["c"])))
    return h


def _row(tk, snap, hist):
    s = snap.get(tk) or {}; closes = [c for _, c in hist.get(tk, [])]
    if s.get("price") and closes: closes = closes + [float(s["price"])]
    week = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else None
    rets = [closes[i] / closes[i - 1] - 1 for i in range(max(1, len(closes) - 5), len(closes))]
    return {"ticker": tk, "today_pct": round(s["chg_pct"], 2) if s.get("chg_pct") is not None else None,
            "week_pct": round(week, 2) if week is not None else None,
            "avg_day_pct": round(float(np.mean(rets)) * 100, 2) if rets else None,
            "spark": [round(c, 2) for c in closes[-20:]], "updated_ns": s.get("updated_ns")}


@router.get("/home/markets")
async def home_markets(request: Request):
    async def build():
        pool = request.app.state.db
        members = sorted({t for _, _, ts in SECTORS for t in ts})
        tickers = [t for t, *_ in US] + [t for t, _ in WORLD] + [e for _, e, _ in SECTORS] + members
        snap = await _snap(tickers); hist = await _hist(pool, tickers)
        caps = {r["ticker"]: r["market_cap"] for r in await pool.fetch(
            "SELECT ticker, market_cap, name FROM universe WHERE ticker = ANY($1)", members)}
        names = {r["ticker"]: r["name"] for r in await pool.fetch("SELECT ticker, name FROM universe WHERE ticker = ANY($1)", members)}
        upd = max([v.get("updated_ns") or 0 for v in snap.values()] or [0])
        sectors = []
        for name, etf, ts in SECTORS:
            top = sorted([t for t in ts if caps.get(t)], key=lambda t: caps[t] or 0, reverse=True)[:4]
            sectors.append({"name": name, "etf": etf, **{k: v for k, v in _row(etf, snap, hist).items() if k != "ticker"},
                            "companies": [{**_row(t, snap, hist), "name": _nice(names.get(t) or t)} for t in top]})
        return {"as_of": datetime.fromtimestamp(upd / 1e9, tz=timezone.utc).isoformat() if upd else None,
                "us": [{**_row(t, snap, hist), "name": n, "proxy": p} for t, n, p in US],
                "world": [{**_row(t, snap, hist), "name": n} for t, n in WORLD],
                "sectors": sectors,
                "note": ("Indexes and world markets are tracked through the exchange-traded funds that follow them. World funds trade in "
                         "dollars during US hours. Sector members are the largest companies in each standard sector, ranked by market cap.")}
    return await _cached(request, "home:markets", 60, build)


@router.get("/home/mood")
async def home_mood(request: Request):
    """S&P 500 state (trend vs 50-day average, swings vs their 1-year norm) and how
    often SPY was higher 21 sessions later from past days in the same state."""
    async def build():
        rows = await request.app.state.db.fetch("SELECT d, c FROM daily_bars WHERE ticker='SPY' ORDER BY d")
        c = np.array([float(r["c"]) for r in rows]); n = len(c)
        if n < 400: return None
        lr = np.diff(np.log(c)); state = [None] * n
        for i in range(272, n):
            sma = c[i - 49:i + 1].mean(); rv = lr[i - 20:i].std()
            rv_hist = np.array([lr[j - 20:j].std() for j in range(i - 251, i + 1, 5)])
            state[i] = ("rising" if c[i] > sma else "falling", "calm" if rv <= np.median(rv_hist) else "volatile")
        cur = state[-1]; same = [i for i in range(272, n - 21, 5) if state[i] == cur]
        allp = [i for i in range(272, n - 21, 5)]
        hit = float(np.mean([c[i + 21] > c[i] for i in same])) if same else None
        base = float(np.mean([c[i + 21] > c[i] for i in allp]))
        words = {("rising", "calm"): "calm and rising", ("rising", "volatile"): "rising, but with bigger swings",
                 ("falling", "calm"): "drifting lower, calmly", ("falling", "volatile"): "falling, with bigger swings"}
        return {"label": words[cur], "trend": cur[0], "swings": cur[1],
                "higher_month_later_pct": round(hit * 100) if hit is not None else None, "base_pct": round(base * 100),
                "n_similar_days": len(same), "since": str(rows[272]["d"]),
                "method": "S&P 500 (SPY) vs its 50-day average; 20-day swings vs their 1-year median; outcomes every 5th day, 21 sessions ahead."}
    return await _cached(request, "home:mood", 3600, build)


@router.get("/home/movers")
async def home_movers(request: Request):
    async def build():
        pool = request.app.state.db
        big = await pool.fetch("SELECT ticker, name FROM universe WHERE active AND market_cap >= 1e10")
        names = {r["ticker"]: r["name"] for r in big}; snap = await _snap(list(names))
        moved = [(t, s) for t, s in snap.items() if s.get("chg_pct") is not None and s.get("price")]
        moved.sort(key=lambda x: x[1]["chg_pct"])
        pick = moved[:5] + moved[-5:]; tks = [t for t, _ in pick]
        stats = {r["ticker"]: r for r in await pool.fetch("""
            SELECT ticker, max(h) hi, min(l) lo, avg(v) FILTER (WHERE d > CURRENT_DATE - 30) v20
            FROM daily_bars WHERE ticker = ANY($1) AND d > CURRENT_DATE - 365 GROUP BY ticker""", tks)}
        filed = {r["ticker"] for r in await pool.fetch("""SELECT DISTINCT ticker FROM ci_events
            WHERE ticker = ANY($1) AND available_at > NOW() - INTERVAL '24 hours' AND evidence_id IN
            (SELECT id FROM ci_raw_evidence WHERE source_type='SEC')""", tks)}
        def why(t, s):
            st = stats.get(t); out = []
            if st and st["hi"] and s["price"] >= float(st["hi"]): out.append("new 52-week high")
            elif st and st["lo"] and s["price"] <= float(st["lo"]): out.append("new 52-week low")
            if st and st["v20"] and s["volume"] and s["volume"] >= 2 * float(st["v20"]): out.append(f"volume {s['volume'] / float(st['v20']):.1f}× normal so far")
            if t in filed: out.append("SEC filing in last 24h")
            return " · ".join(out)
        fmt = lambda t, s: {"ticker": t, "name": _nice(names.get(t) or t),
                            "today_pct": round(s["chg_pct"], 2), "why": why(t, s)}
        return {"rising": [fmt(t, s) for t, s in reversed(moved[-5:]) if s["chg_pct"] > 0],
                "falling": [fmt(t, s) for t, s in moved[:5] if s["chg_pct"] < 0],
                "universe": len(names), "note": "US companies over $10B market cap."}
    return await _cached(request, "home:movers", 120, build)


@router.get("/search/suggest")
async def search_suggest(request: Request, q: str = Query(..., min_length=1, max_length=40)):
    q = q.strip()
    rows = await request.app.state.db.fetch("""
        SELECT ticker, name FROM universe WHERE active AND (ticker ILIKE $1 || '%' OR name ILIKE '%' || $1 || '%')
        ORDER BY (upper(ticker) = upper($1)) DESC, (ticker ILIKE $1 || '%') DESC, market_cap DESC NULLS LAST LIMIT 8""", q)
    return {"results": [{"ticker": r["ticker"], "name": r["name"]} for r in rows]}
