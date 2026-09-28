"""Board cohorts — the track record behind every board (docs/BOARDS_CONTRACT.md).

Nightly: snapshot each board's top-N with the day's close. Then fill forward
returns at 21/63/126/252 sessions as they resolve, alongside the equal-weight
universe and SPY over the SAME dates. Boards display the result — or
'measured over N weeks so far' while it's young. Append-only; outcomes are
filled into their own columns once, never rewritten.
"""
from __future__ import annotations
import httpx
from datetime import date
from loguru import logger

SQL = """
CREATE TABLE IF NOT EXISTS board_cohorts (
    board        TEXT NOT NULL,
    cohort_date  DATE NOT NULL,
    ticker       TEXT NOT NULL,
    rank         INT  NOT NULL,
    tier         TEXT,
    score        DOUBLE PRECISION,
    entry_close  DOUBLE PRECISION,
    ret_21       DOUBLE PRECISION, ret_63 DOUBLE PRECISION, ret_126 DOUBLE PRECISION, ret_252 DOUBLE PRECISION,
    uni_21       DOUBLE PRECISION, uni_63 DOUBLE PRECISION, uni_126 DOUBLE PRECISION, uni_252 DOUBLE PRECISION,
    spy_21       DOUBLE PRECISION, spy_63 DOUBLE PRECISION, spy_126 DOUBLE PRECISION, spy_252 DOUBLE PRECISION,
    PRIMARY KEY (board, cohort_date, ticker)
);
"""
HORIZONS = (21, 63, 126, 252)
TOP_N = 25


async def snapshot(pool) -> dict:
    async with pool.acquire() as c:
        await c.execute(SQL)
        d0 = await c.fetchval("SELECT max(d) FROM daily_bars")
    rows = []
    async with httpx.AsyncClient(timeout=60) as cx:
        try:
            j = (await cx.get("http://localhost:8000/api/v6/scan/tiers")).json()
            for tier, lst in (j.get("tiers") or {}).items():
                for i, r in enumerate(lst[:TOP_N]): rows.append(("multibagger", r["ticker"], i + 1, tier, r.get("score")))
        except Exception as e: logger.warning(f"cohort multibagger: {e}")
        try:
            j = (await cx.get("http://localhost:8000/api/v6/rebound/list")).json()
            for tier, lst in (j.get("tiers") or {}).items():
                for i, r in enumerate(lst[:TOP_N]): rows.append(("rebound", r["ticker"], i + 1, tier, r.get("score")))
        except Exception as e: logger.warning(f"cohort rebound: {e}")
        try:
            j = (await cx.get("http://localhost:8000/api/v6/ascent/top/25")).json()
            for i, r in enumerate((j.get("rows") or [])[:TOP_N]): rows.append(("ascent", r["ticker"], i + 1, r.get("tier"), r.get("ascent_score")))
        except Exception as e: logger.warning(f"cohort ascent: {e}")
    n = 0
    async with pool.acquire() as c:
        for board, tk, rank, tier, score in rows:
            px = await c.fetchval("SELECT c FROM daily_bars WHERE ticker=$1 AND d=$2", tk, d0)
            if px is None: continue
            await c.execute("""INSERT INTO board_cohorts (board, cohort_date, ticker, rank, tier, score, entry_close)
                               VALUES ($1,$2,$3,$4,$5,$6,$7) ON CONFLICT DO NOTHING""",
                            board, d0, tk, rank, tier, float(score) if score is not None else None, float(px))
            n += 1
    logger.info(f"[cohorts] snapshot {d0}: {n} rows across boards")
    return {"date": str(d0), "rows": n}


async def fill_outcomes(pool) -> dict:
    """For each horizon, resolve cohorts whose h-th session after cohort_date exists."""
    filled = 0
    async with pool.acquire() as c:
        await c.execute(SQL)
        spy = await c.fetch("SELECT d, c FROM daily_bars WHERE ticker='SPY' ORDER BY d")
        sd = [r["d"] for r in spy]; sc = {r["d"]: r["c"] for r in spy}; pos = {d: i for i, d in enumerate(sd)}
        for h in HORIZONS:
            pending = await c.fetch(f"SELECT DISTINCT cohort_date FROM board_cohorts WHERE ret_{h} IS NULL")
            for r in pending:
                d0 = r["cohort_date"]
                if d0 not in pos or pos[d0] + h >= len(sd): continue
                d1 = sd[pos[d0] + h]
                uni = await c.fetchval("""SELECT avg(b2.c / b1.c - 1) FROM daily_bars b1 JOIN daily_bars b2 USING (ticker)
                                          WHERE b1.d=$1 AND b2.d=$2 AND b1.c >= 3""", d0, d1)
                spy_r = sc[d1] / sc[d0] - 1
                n = await c.execute(f"""UPDATE board_cohorts bc SET ret_{h} = (SELECT b.c / bc.entry_close - 1 FROM daily_bars b WHERE b.ticker=bc.ticker AND b.d=$2),
                                        uni_{h} = $3, spy_{h} = $4
                                        WHERE bc.cohort_date=$1 AND bc.ret_{h} IS NULL""", d0, d1, uni, spy_r)
                filled += int(n.split()[-1]) if n else 0
    logger.info(f"[cohorts] outcomes filled: {filled}")
    return {"filled": filled}


async def track_record(pool) -> dict:
    async with pool.acquire() as c:
        await c.execute(SQL)
        out = {}
        for board in ("multibagger", "rebound", "ascent"):
            first = await c.fetchval("SELECT min(cohort_date) FROM board_cohorts WHERE board=$1", board)
            ncoh = await c.fetchval("SELECT count(DISTINCT cohort_date) FROM board_cohorts WHERE board=$1", board)
            hz = {}
            for h in HORIZONS:
                r = await c.fetchrow(f"""SELECT count(*) n, avg(ret_{h}) mean_ret, percentile_cont(0.5) WITHIN GROUP (ORDER BY ret_{h}) med_ret,
                                                 avg(CASE WHEN ret_{h} > 0 THEN 1 ELSE 0 END) hit, avg(uni_{h}) uni, avg(spy_{h}) spy,
                                                 avg(CASE WHEN ret_{h} > uni_{h} THEN 1 ELSE 0 END) beat_uni
                                          FROM board_cohorts WHERE board=$1 AND ret_{h} IS NOT NULL""", board)
                hz[f"{h}d"] = ({"n": r["n"], "mean_pct": round(r["mean_ret"] * 100, 2), "median_pct": round(r["med_ret"] * 100, 2),
                               "hit_rate_pct": round(r["hit"] * 100, 1), "universe_pct": round((r["uni"] or 0) * 100, 2),
                               "spy_pct": round((r["spy"] or 0) * 100, 2), "beat_universe_pct": round((r["beat_uni"] or 0) * 100, 1)}
                              if r and r["n"] else None)
            out[board] = {"first_cohort": str(first) if first else None, "cohorts": ncoh,
                          "weeks_measured": (round((date.today() - first).days / 7, 1) if first else 0), "horizons": hz,
                          "note": ("not yet measured — cohorts begin tonight" if not first else
                                   f"measured over {round((date.today() - first).days / 7, 1)} weeks so far; horizons resolve as time passes")}
        return out
