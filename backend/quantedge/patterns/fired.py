"""Which tickers completed a catalog pattern on the last session — with each
pattern's universe-measured 21-day odds. Runs in the nightly pattern job."""
from __future__ import annotations
import json
import numpy as np
from datetime import date
from loguru import logger
from .formations import _smooth, _extrema, classify_last5
from .candlesticks import detect
from core.artifact_paths import artifact_read_path


async def scan_fired(pool, out_path: str) -> dict:
    fa = artifact_read_path("formations_scan.json"); ca = artifact_read_path("candlestick_scan.json")
    fart = json.loads(fa.read_text()) if fa else {}; cart = json.loads(ca.read_text()) if ca else {}
    tickers = await pool.fetch("""SELECT b.ticker, u.name FROM (SELECT ticker FROM daily_bars GROUP BY ticker HAVING count(*) >= 300) b
                                  LEFT JOIN universe u USING (ticker)""")
    last = await pool.fetchval("SELECT max(d) FROM daily_bars")
    fired = []
    for t in tickers:
        rows = await pool.fetch("SELECT d, o, h, l, c, v FROM daily_bars WHERE ticker=$1 ORDER BY d DESC LIMIT 320", t["ticker"])
        if len(rows) < 260 or rows[0]["d"] != last: continue
        rows = rows[::-1]
        c = np.array([r["c"] for r in rows], np.float64)
        if c[-1] < 3 or np.median(c[-20:] * np.array([float(r["v"] or 0) for r in rows[-20:]])) < 1e6: continue
        o = np.array([r["o"] or r["c"] for r in rows]); h = np.array([r["h"] or r["c"] for r in rows]); l = np.array([r["l"] or r["c"] for r in rows])
        n = len(c)
        for x in detect(o, h, l, c):
            if x["i"] == n - 1 and x["name"] != "doji":
                sc = ((cart.get("patterns") or {}).get(x["name"]) or {}).get("horizons", {}).get("21d", {}).get("all")
                fired.append({"ticker": t["ticker"], "name": t["name"], "family": "candlestick", "pattern": x["name"],
                              "direction": x["direction"], "odds_21d": sc})
        sm = _smooth(c); ext = _extrema(sm)
        if len(ext) >= 5 and ext[-1][0] >= n - 4:
            nm = classify_last5(c, ext, len(ext) - 1)
            if nm:
                f = (fart.get("formations") or {}).get(nm) or {}
                fired.append({"ticker": t["ticker"], "name": t["name"], "family": "formation", "pattern": nm,
                              "direction": "bearish" if nm in ("head_shoulders", "double_top", "triple_top", "rising_wedge", "descending_triangle") else "bullish",
                              "odds_21d": (f.get("distributions") or {}).get("20d"), "follow_through_pct": f.get("follow_through_pct")})
    # rank: measured edge first, unmeasured last
    base = (cart.get("base") or {}).get("21d") or {}
    def edge(r):
        s = r.get("odds_21d"); return abs(s["positive_pct"] - base.get("positive_pct", 50)) if s else -1
    fired.sort(key=edge, reverse=True)
    art = {"generated": date.today().isoformat(), "session": str(last), "n": len(fired), "fired": fired[:400]}
    from pathlib import Path
    Path(out_path).write_text(json.dumps(art))
    logger.info(f"[fired] {len(fired)} pattern completions on {last}")
    return {"n": len(fired)}
