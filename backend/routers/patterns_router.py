"""Pattern Lab — GET /api/v6/patterns/analogs/{ticker}?window=60

Serves historical-analog distributions from the precomputed window library.
The heavy lifting (508k+ windows) is a nightly artifact; queries are one
vectorized Euclidean pass plus banded DTW on a 300-window shortlist —
sub-second. Distributions ship with library-wide base rates and the
one-macro-era caveat; below 15 episodes a cell is null, not a number.
"""
from __future__ import annotations
import numpy as np
from datetime import date
from fastapi import APIRouter, HTTPException, Query, Request
from quantedge.patterns.engine import PatternLibrary

router = APIRouter()
_LIB = PatternLibrary("/app/models/patterns")


@router.get("/patterns/analogs/{ticker}")
async def pattern_analogs(ticker: str, request: Request,
                          window: int = Query(60, description="20 or 60"),
                          volume: str | None = Query(None), vola: str | None = Query(None),
                          regime: str | None = Query(None), extreme: str | None = Query(None),
                          scope: str | None = Query(None)):
    if window not in (20, 60):
        raise HTTPException(status_code=422, detail="window must be 20 or 60")
    tk = ticker.upper().strip()
    if not tk.replace('-', '').isalpha() or len(tk) > 10:
        raise HTTPException(status_code=422, detail="invalid ticker")

    pool = getattr(request.app.state, "db", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="database not connected")

    rows = await pool.fetch(
        "SELECT d, c, v FROM daily_bars WHERE ticker=$1 ORDER BY d DESC LIMIT $2",
        tk, max(window + 5, 320))
    if len(rows) < window:
        raise HTTPException(status_code=404,
                            detail=f"{tk}: only {len(rows)} bars on record; "
                                   f"{window} needed")
    rows = rows[::-1]
    closes = np.array([r["c"] for r in rows], dtype=np.float64)
    q_end: date = rows[-1]["d"]

    fl = {k: v for k, v in (("volume", volume), ("vola", vola),
                            ("regime", regime), ("extreme", extreme),
                            ("scope", scope)) if v}
    res = _LIB.query(closes, tk, window, q_end, filters=fl or None)
    if res is None:
        raise HTTPException(status_code=503,
                            detail="pattern library not built yet")
    res["ticker"] = tk
    res["as_of"] = q_end.isoformat()

    # ── Current state vector: what the pattern consists of ──
    vols = np.array([float(r["v"] or 0) for r in rows], dtype=np.float64)
    if len(closes) >= 260:
        lr = np.diff(np.log(closes))
        rv21 = float(np.std(lr[-21:]) * np.sqrt(252))
        hist = np.array([np.std(lr[j - 20:j + 1]) for j in range(20, len(lr), 5)])
        sma20, sma50 = closes[-20:].mean(), closes[-50:].mean()
        sma200 = closes[-200:].mean() if len(closes) >= 200 else None
        sl_now = float(np.polyfit(np.arange(20), np.log(closes[-20:]), 1)[0]) * 252
        sl_prev = float(np.polyfit(np.arange(20), np.log(closes[-40:-20]), 1)[0]) * 252
        vroll = np.array([vols[max(0, j - 20):j + 1].mean() for j in range(len(vols))])
        pv_corr = float(np.corrcoef(np.diff(closes[-21:]), vols[-20:])[0, 1]) if vols[-20:].std() > 0 else 0.0
        res["state_vector"] = {
            "price": {
                "trend": "up" if closes[-1] > sma50 else "down",
                "slope_20d_ann_pct": round(sl_now * 100, 1),
                "acceleration": "accelerating" if sl_now > sl_prev else "decelerating",
                "vs_sma20_pct": round((closes[-1] / sma20 - 1) * 100, 2),
                "vs_sma50_pct": round((closes[-1] / sma50 - 1) * 100, 2),
                "vs_sma200_pct": (round((closes[-1] / sma200 - 1) * 100, 2) if sma200 else None),
                "drawdown_pct": round((closes[-1] / closes.max() - 1) * 100, 2),
                "vs_52w_high_pct": round((closes[-1] / closes[-252:].max() - 1) * 100, 2),
                "vs_52w_low_pct": round((closes[-1] / closes[-252:].min() - 1) * 100, 2),
            },
            "momentum": {
                "5d_pct": round((closes[-1] / closes[-6] - 1) * 100, 2),
                "20d_pct": round((closes[-1] / closes[-21] - 1) * 100, 2),
                "60d_pct": round((closes[-1] / closes[-61] - 1) * 100, 2),
            },
            "volatility": {
                "realized_21d_ann_pct": round(rv21 * 100, 1),
                "percentile": round(float((hist < np.std(lr[-21:])).mean()) * 100, 0),
                "direction": ("rising" if np.std(lr[-21:]) > np.std(lr[-42:-21]) else "falling"),
            },
            "multi_scale": (lambda momsigns: {
                "signals": momsigns,
                "alignment_pct": round(100 * max(sum(1 for x in momsigns.values() if x == "bullish"),
                                                 sum(1 for x in momsigns.values() if x == "bearish"))
                                       / len(momsigns), 0),
                "verdict": ("ALIGNED" if len(set(momsigns.values())) == 1 else
                            "MIXED" if max(sum(1 for x in momsigns.values() if x == v)
                                           for v in set(momsigns.values())) >= len(momsigns) - 1
                            else "CONFLICTED"),
            })({
                "5d": "bullish" if closes[-1] > closes[-6] else "bearish",
                "20d": "bullish" if closes[-1] > closes[-21] else "bearish",
                "60d": "bullish" if closes[-1] > closes[-61] else "bearish",
                "252d": ("bullish" if closes[-1] > closes[-253] else "bearish") if len(closes) > 253 else "n/a",
            }),
            "volume": {
                "percentile": round(float((vroll[:-1] < vroll[-1]).mean()) * 100, 0),
                "trend": "rising" if vroll[-1] > vroll[-21] else "falling",
                "price_volume_corr_20d": round(pv_corr, 2),
            },
        }

    # ── Forward-path fan: real forward closes for up to 60 episodes ──
    eps = res.pop("episodes_for_paths", []) or []
    if eps:
        async def _path(e):
            r2 = await pool.fetch(
                "SELECT c FROM daily_bars WHERE ticker=$1 AND d > $2 ORDER BY d LIMIT 61",
                e["ticker"], date.fromisoformat(e["end"]))
            if len(r2) < 20:
                return None
            base = float(r2[0]["c"])
            return [float(x["c"]) / base - 1 for x in r2]
        import asyncio as _aio
        paths = [p for p in await _aio.gather(*[_path(e) for e in eps]) if p]
        if len(paths) >= 10:
            L = min(min(len(p) for p in paths), 61)
            M = np.array([p[:L] for p in paths])
            res["forward_fan"] = {
                "sessions": L, "n_paths": len(paths),
                "median": [round(float(x) * 100, 2) for x in np.median(M, axis=0)],
                "p25": [round(float(x) * 100, 2) for x in np.percentile(M, 25, axis=0)],
                "p75": [round(float(x) * 100, 2) for x in np.percentile(M, 75, axis=0)],
            }
    return res


@router.get("/patterns/formations")
async def formations_library():
    """Classical-formation scan artifact: occurrences, breakout stats and
    forward distributions per formation, measured — never asserted."""
    import json
    from core.artifact_paths import artifact_read_path
    p = artifact_read_path("formations_scan.json")
    if p is None:
        raise HTTPException(status_code=503, detail="formation scan not run yet")
    return json.loads(p.read_text())


@router.get("/patterns/conditions/{ticker}")
async def ticker_conditions(ticker: str, request: Request):
    """Where the ticker sits TODAY on each measured condition (multi-horizon
    momentum, 52w-high distance, volatility percentile), with the historical
    forward distribution of its current quintile vs the unconditional base."""
    import json
    import numpy as np
    from core.artifact_paths import artifact_read_path
    tk = ticker.upper().strip()
    if not tk.replace('-', '').isalpha() or len(tk) > 10:
        raise HTTPException(status_code=422, detail="invalid ticker")
    p = artifact_read_path("conditions_scan.json")
    if p is None:
        raise HTTPException(status_code=503, detail="condition scan not run yet")
    art = json.loads(p.read_text())

    pool = getattr(request.app.state, "db", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="database not connected")
    rows = await pool.fetch(
        "SELECT c FROM daily_bars WHERE ticker=$1 ORDER BY d DESC LIMIT 320", tk)
    if len(rows) < 260:
        raise HTTPException(status_code=404,
                            detail=f"{tk}: {len(rows)} bars on record; 260 needed")
    c = np.array([r["c"] for r in rows], np.float64)[::-1]
    lr = np.diff(np.log(c))
    vol21 = float(np.std(lr[-21:]))
    hist = [float(np.std(lr[j - 20:j + 1])) for j in range(20, len(lr) - 1, 10)]
    vals = {
        "mom_20d": float(c[-1] / c[-21] - 1),
        "mom_60d": float(c[-1] / c[-61] - 1),
        "mom_120d": float(c[-1] / c[-121] - 1),
        "mom_252d": float(c[-1] / c[-253] - 1),
        "dist_52w_high": float(c[-1] / c[-252:].max() - 1),
        "vol_21d_pctile": float((np.array(hist) < vol21).mean()) if hist else 0.5,
    }
    out = {"ticker": tk, "generated": art["generated"], "samples": art["samples"],
           "note": art["note"], "base": art["base"], "conditions": {}}
    for name, v in vals.items():
        spec = art["conditions"].get(name)
        if not spec:
            continue
        q = int(np.digitize([v], spec["quintile_edges"])[0])  # 0..4
        out["conditions"][name] = {
            "value": round(v, 4), "quintile": q + 1,
            "cell": spec["cells"].get(f"Q{q+1}"),
        }
    return out


@router.get("/patterns/evolution/{ticker}")
async def pattern_evolution(ticker: str, request: Request):
    """The ticker's current discrete state (60d trend x vol tercile) and the
    MEASURED historical transition frequencies out of that state, each with
    the +20d return distribution that accompanied it. Counted, not modeled."""
    import json
    import numpy as np
    from core.artifact_paths import artifact_read_path
    tk = ticker.upper().strip()
    p = artifact_read_path("conditions_scan.json")
    if p is None:
        raise HTTPException(status_code=503, detail="condition scan not run yet")
    art = json.loads(p.read_text())
    if "evolution" not in art:
        raise HTTPException(status_code=503, detail="evolution not in current scan artifact")
    pool = getattr(request.app.state, "db", None)
    rows = await pool.fetch(
        "SELECT c FROM daily_bars WHERE ticker=$1 ORDER BY d DESC LIMIT 320", tk)
    if len(rows) < 280:
        raise HTTPException(status_code=404, detail=f"{tk}: insufficient history")
    c = np.array([r["c"] for r in rows], np.float64)[::-1]
    lr = np.diff(np.log(c))
    v21 = np.array([np.std(lr[max(0, j - 20):j + 1]) for j in range(len(lr))])
    vp = float((v21[-251:-1] < v21[-1]).mean())
    volq = 0 if vp <= 0.33 else 2 if vp >= 0.67 else 1
    mom60 = float(c[-1] / c[-61] - 1)
    t = "UP" if mom60 > 0.03 else "DOWN" if mom60 < -0.03 else "FLAT"
    state = f"{t}_{('LOWVOL','MIDVOL','HIGHVOL')[volq]}"
    return {"ticker": tk, "current_state": state,
            "inputs": {"mom_60d_pct": round(mom60 * 100, 2), "vol_pctile": round(vp * 100, 0)},
            "state_definition": art["state_definition"],
            "history": art["evolution"].get(state),
            "all_states": {k: v["n"] for k, v in art["evolution"].items()},
            "note": art["note"], "generated": art["generated"]}


@router.get("/patterns/situation/{ticker}")
async def situation_report(ticker: str, request: Request):
    """Per-ticker 90-day situation report (docs/SITUATION_REPORT.md): the
    current state, what followed that state historically, reported results,
    company intelligence, news tone — each section naming its source — and
    what QuantEdge does NOT have. Assembled from existing engines; predicts
    nothing."""
    import asyncio, httpx
    tk = ticker.upper().strip()
    if not tk.replace('-', '').isalpha() or len(tk) > 10:
        raise HTTPException(status_code=422, detail="invalid ticker")
    B = "http://localhost:8000"
    async with httpx.AsyncClient(timeout=180) as cx:
        async def g(path, **p):
            try:
                r = await cx.get(f"{B}{path}", params=p); return r.json() if r.status_code == 200 else None
            except Exception:
                return None
        async def analyze():
            try:
                r = await cx.post(f"{B}/api/v6/analyze", json={"req": {"ticker": tk, "include_options": False,
                                                                        "include_sentiment": True, "mc_paths": 10000}})
                return (r.json() or {}).get("data") if r.status_code == 200 else None
            except Exception:
                return None
        an, a20, a60, cond, intel, rb, intel_long = await asyncio.gather(
            analyze(), g(f"/api/v6/patterns/analogs/{tk}", window=20), g(f"/api/v6/patterns/analogs/{tk}", window=60),
            g(f"/api/v6/patterns/conditions/{tk}"), g(f"/api/v6/intel/{tk}/timeline", days=90), g("/api/v6/rebound/list"),
            g(f"/api/v6/intel/{tk}/timeline", days=220))
    if an is None:
        raise HTTPException(status_code=404, detail=f"{tk}: no QuantEdge analysis available")

    def dist(a):
        if not a or a.get("insufficient"): return None
        return {"episodes": a.get("episodes"), "outcomes": a.get("distributions"), "base_rates": a.get("base_rates"),
                "excess_vs_spy": a.get("excess_vs_spy"), "episode_date_range": a.get("episode_date_range")}
    # Rebound stage if the ticker is on the board.
    rb_row = None
    for rows in ((rb or {}).get("tiers") or {}).values():
        for r in rows:
            if r.get("ticker") == tk: rb_row = r
    dd = an.get("week_52_high") and an.get("current_price") and (an["current_price"] / an["week_52_high"] - 1)
    if dd is None:   # analysis payload may omit current_price; the state vector has the same figure
        _v = ((a20 or {}).get("state_vector") or {}).get("price", {}).get("vs_52w_high_pct")
        dd = _v / 100 if _v is not None else None
    try:
        from routers.rebound_router import _RECOVERY_BASE_RATE, _dd_bucket
        base_rate = _RECOVERY_BASE_RATE.get(_dd_bucket(abs(dd))) if dd is not None and dd <= -0.35 else None
    except Exception:
        base_rate = None
    cap = [e for e in (intel_long or {}).get("events", []) if e.get("event_type") == "capital_allocation"][:2]
    results = [{"period_end": e["event_date"], "public": e["available_at"][:10],
                "values": {d["field"]: d["value"] for d in e.get("derived", []) if d.get("layer") == "DERIVED" and "trailing" not in d["field"]}}
               for e in cap]
    sent = an.get("sentiment") or {}
    return {
        "ticker": tk, "name": an.get("name"), "as_of": (a20 or {}).get("as_of"),
        "price_state": {"source": "Pattern Lab state vector (daily_bars)", **((a20 or {}).get("state_vector") or {})},
        "what_followed": {"source": "Pattern Lab analog library, non-overlapping episodes; DESCRIPTIVE",
                          "shape_20d": dist(a20), "shape_60d": dist(a60),
                          "conditions": {k: v for k, v in ((cond or {}).get("conditions") or {}).items()
                                         if k in ("mom_60d", "dist_52w_high", "vol_21d_pctile")},
                          "conditions_base": (cond or {}).get("base")},
        "off_highs": ({"source": "rebound scan + measured recovery base rates",
                       "drawdown_from_52w_high_pct": round(dd * 100, 1) if dd is not None else None,
                       "on_rebound_board": bool(rb_row), "rebound_row": rb_row,
                       "recovery_base_rate": base_rate} if dd is not None and dd <= -0.30 else
                      {"source": "rebound scan", "applies": False, "drawdown_from_52w_high_pct": round(dd * 100, 1) if dd is not None else None}),
        "reported_results": {"source": "XBRL filings via Company Intelligence (PRIMARY); ratios from analysis fundamentals",
                             "last_quarters": results,
                             "ratios": {k: an.get(k) for k in ("pe_ratio", "price_to_sales", "gross_margin", "operating_margin",
                                                               "net_margin", "revenue_growth", "earnings_growth", "roic", "debt_to_equity")}},
        "company_intelligence": {"source": "SEC 8-K / Form 4 / 13F / attention (PRIMARY/SECONDARY)",
                                 "material_events": [e for e in (intel or {}).get("events", []) if e.get("significance") == "MATERIAL"][:8],
                                 "insider_open_market": (intel or {}).get("insider_open_market"),
                                 "institutional": next((e["title"] for e in (intel_long or {}).get("events", []) if e.get("event_type") == "institutional_snapshot"), None),
                                 "attention_vs_fundamentals": (intel or {}).get("attention_vs_fundamentals")},
        "news_tone": {"source": f"FinBERT on the analysis's own headline fetch (SECONDARY; scored {sent.get('n_articles_ticker_tagged')} ticker-tagged articles — a much smaller sample than the attention count, which uses the whole feed)",
                      "composite": sent.get("composite"), "label": sent.get("label"), "model": sent.get("model")},
        "no_source": ["analyst estimates / expected results", "management guidance", "earnings-call transcripts",
                      "options positioning (plan returns 403)"],
        "note": "Every distribution is historical frequency with its n and base rate. Nothing here is a prediction.",
    }
