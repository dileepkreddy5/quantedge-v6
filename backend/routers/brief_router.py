"""Daily brief — GET /api/v6/brief/today.

The homepage's answer to "what should I look at today", assembled from what the
system actually recorded: the latest regime read, today's strongest and weakest
ensemble signals, and past calls old enough for their 5-day outcome to have been
realized by the outcome filler. Nothing here is computed at request time beyond
the queries; if a section has no data it returns empty rather than inventing.
"""
from __future__ import annotations
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/brief/indices")
async def brief_indices(request: Request):
    """Index proxies via ETFs on the current Polygon plan: SPY (S&P 500),
    QQQ (Nasdaq-100), XLK (tech sector). Labeled as proxies, not the
    composites — index feeds are a different Polygon product."""
    import httpx, os
    key = os.getenv("POLYGON_API_KEY", "")
    out = []
    async with httpx.AsyncClient(timeout=10) as cx:
        for sym, label in (("SPY", "S&P 500 (SPY)"),
                           ("QQQ", "NASDAQ-100 (QQQ)"),
                           ("XLK", "TECH SECTOR (XLK)")):
            try:
                r = await cx.get(
                    f"https://api.polygon.io/v2/aggs/ticker/{sym}/prev",
                    params={"apiKey": key})
                res = (r.json().get("results") or [None])[0]
                if res and res.get("c") and res.get("o"):
                    chg = (res["c"] - res["o"]) / res["o"]
                    out.append({"symbol": sym, "label": label,
                                "close": res["c"], "change_pct": round(chg * 100, 2)})
            except Exception:
                continue
    return {"indices": out, "note": "previous session, ETF proxies"}


@router.get("/brief/today")
async def brief_today(request: Request):
    pool = getattr(request.app.state, "db", None)
    if pool is None:
        return {"available": False, "reason": "database not connected"}

    async with pool.acquire() as c:
        # Regime: newest SPY read, falling back to the newest row of any ticker.
        regime = await c.fetchrow(
            """SELECT ticker, generated_at, hmm_regime, hmm_confidence,
                      garch_regime, garch_vol_forecast, kalman_trend
               FROM signals WHERE ticker = 'SPY'
               ORDER BY generated_at DESC LIMIT 1""")
        if regime is None:
            regime = await c.fetchrow(
                """SELECT ticker, generated_at, hmm_regime, hmm_confidence,
                          garch_regime, garch_vol_forecast, kalman_trend
                   FROM signals ORDER BY generated_at DESC LIMIT 1""")

        # Today's calls: latest row per ticker in the last 36h, ranked by signal.
        latest = await c.fetch(
            """SELECT DISTINCT ON (ticker)
                      ticker, generated_at, ensemble_signal, ensemble_direction,
                      hmm_regime, xgb_confidence, lgb_confidence, cvar_95,
                      recommended_position
               FROM signals
               WHERE generated_at > now() - interval '36 hours'
               ORDER BY ticker, generated_at DESC""")
        ranked = sorted([dict(r) for r in latest],
                        key=lambda r: r["ensemble_signal"] or 0, reverse=True)

        # Settled calls: old enough that ret_5d exists — the honest scoreboard.
        settled = await c.fetch(
            """SELECT DISTINCT ON (ticker)
                      ticker, generated_at, ensemble_signal, ensemble_direction,
                      ret_5d, barrier_hit
               FROM signals
               WHERE ret_5d IS NOT NULL
               ORDER BY ticker, generated_at DESC LIMIT 8""")

        perf = await c.fetchrow(
            "SELECT * FROM performance_daily ORDER BY date DESC LIMIT 1")

    def _iso(r):
        d = dict(r)
        for k, v in d.items():
            if hasattr(v, "isoformat"):
                d[k] = v.isoformat()
        return d

    settled_rows = []
    for r in settled:
        d = _iso(r)
        sig = d.get("ensemble_signal")
        ret = d.get("ret_5d")
        # A call is 'right' when signal sign matched the realized 5d sign.
        d["call_correct"] = (None if sig is None or ret is None
                             else (sig >= 0) == (ret >= 0))
        settled_rows.append(d)

    return {
        "available": True,
        "regime": _iso(regime) if regime else None,
        "strongest": [_iso(r) for r in ranked[:3]],
        "weakest": [_iso(r) for r in ranked[-3:]][::-1] if len(ranked) > 3 else [],
        "settled": settled_rows,
        "performance": _iso(perf) if perf else None,
        "note": ("Signals cover tickers the system has actually analyzed "
                 "(cache-warmed majors plus user requests), not the full universe. "
                 "Settled calls show the realized 5-day return against the signal's "
                 "direction at the time — including the misses."),
    }
