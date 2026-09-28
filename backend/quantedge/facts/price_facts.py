"""Nightly facts sheet, part 1: price facts for every company in the universe.

One row per (ticker, as_of) in company_facts. Everything from daily_bars (5 years,
so "high" = the highest close in 5 years, labelled that way everywhere), plus news
attention from ci_attention_daily. Also writes base rates for the Great-companies-
on-sale tracker: of large companies that fell 20/30/50% from a high, how often they
regained it, and how long it took — measured on the same 5-year history.
"""
from __future__ import annotations
import json
from datetime import date
import numpy as np
from loguru import logger

SQL = """
CREATE TABLE IF NOT EXISTS company_facts (
    ticker TEXT NOT NULL, as_of DATE NOT NULL, name TEXT, tier TEXT, sector TEXT, is_spac BOOLEAN,
    market_cap DOUBLE PRECISION, price DOUBLE PRECISION,
    high_5y DOUBLE PRECISION, high_date DATE, pct_below_high DOUBLE PRECISION, sessions_since_high INT,
    low_since_high DOUBLE PRECISION, low_date DATE, sessions_since_low INT, pct_off_low DOUBLE PRECISION, stage TEXT,
    ret_1d DOUBLE PRECISION, ret_1w DOUBLE PRECISION, ret_1m DOUBLE PRECISION, ret_3m DOUBLE PRECISION,
    ret_6m DOUBLE PRECISION, ret_1y DOUBLE PRECISION,
    weeks_beat_mkt_26 INT, up_weeks_26 INT, vol_ratio_20_60 DOUBLE PRECISION, up_vol_share_20 DOUBLE PRECISION,
    mkt_move_since_high DOUBLE PRECISION, sector_move_since_high DOUBLE PRECISION, drop_cause TEXT, peer_group TEXT,
    news_30d INT, news_180d INT, cik TEXT, primary_listing BOOLEAN, dollar_vol_20 DOUBLE PRECISION,
    fundamentals JSONB,
    PRIMARY KEY (ticker, as_of)
);
CREATE INDEX IF NOT EXISTS idx_cf_asof ON company_facts (as_of);
ALTER TABLE company_facts ADD COLUMN IF NOT EXISTS cik TEXT;
ALTER TABLE company_facts ADD COLUMN IF NOT EXISTS primary_listing BOOLEAN;
ALTER TABLE company_facts ADD COLUMN IF NOT EXISTS dollar_vol_20 DOUBLE PRECISION;
ALTER TABLE company_facts ADD COLUMN IF NOT EXISTS peer_group TEXT;
ALTER TABLE company_facts ADD COLUMN IF NOT EXISTS data_suspect TEXT;
ALTER TABLE company_facts ADD COLUMN IF NOT EXISTS history_note TEXT;
ALTER TABLE company_facts ADD COLUMN IF NOT EXISTS below_200d BOOLEAN;
ALTER TABLE company_facts ADD COLUMN IF NOT EXISTS cross_200d_date DATE;
ALTER TABLE company_facts ADD COLUMN IF NOT EXISTS cross_200d_vol DOUBLE PRECISION;
ALTER TABLE company_facts ADD COLUMN IF NOT EXISTS peer_ret_1m DOUBLE PRECISION;
"""

OVERRIDE = {"GOOGL": "Communication", "GOOG": "Communication", "META": "Communication", "NFLX": "Communication",
            "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "V": "Financials", "MA": "Financials",
            "PYPL": "Financials", "UNH": "Healthcare", "CVS": "Healthcare", "BRK.B": "Financials", "BRK.A": "Financials",
            # investor sector groupings that industry codes get wrong
            "WMT": "Consumer Staples", "COST": "Consumer Staples", "TGT": "Consumer Staples", "DG": "Consumer Staples",
            "DLTR": "Consumer Staples", "BJ": "Consumer Staples", "KR": "Consumer Staples",
            "LRCX": "Technology", "KLAC": "Technology", "TER": "Technology", "ENTG": "Technology", "ONTO": "Technology",
            "TMO": "Healthcare", "DHR": "Healthcare", "A": "Healthcare", "WAT": "Healthcare", "MTD": "Healthcare", "IQV": "Healthcare",
            "GE": "Industrials", "GEV": "Industrials", "HON": "Industrials", "ETN": "Industrials", "EMR": "Industrials",
            "DIS": "Communication", "CMCSA": "Communication", "CHTR": "Communication", "EA": "Communication", "TTWO": "Communication"}


def sector_of(ticker: str, sic) -> str:
    if ticker in OVERRIDE: return OVERRIDE[ticker]
    try: s = int(str(sic)[:4])
    except Exception: return "Other"
    R = [((3020, 3021), "Consumer Discretionary"), ((1300, 1399), "Energy"), ((2900, 2999), "Energy"), ((1000, 1499), "Materials"), ((1500, 1799), "Industrials"),
         ((2000, 2199), "Consumer Staples"), ((2830, 2836), "Healthcare"), ((2840, 2844), "Consumer Staples"),
         ((2200, 2399), "Consumer Discretionary"), ((2500, 2599), "Consumer Discretionary"), ((2700, 2799), "Communication"),
         ((2400, 2899), "Materials"), ((3000, 3099), "Materials"), ((3100, 3199), "Consumer Discretionary"),
         ((3200, 3399), "Materials"), ((3570, 3579), "Technology"), ((3600, 3699), "Technology"),
         ((3711, 3711), "Consumer Discretionary"), ((3714, 3716), "Consumer Discretionary"), ((3840, 3851), "Healthcare"),
         ((3400, 3899), "Industrials"), ((3900, 3999), "Consumer Discretionary"), ((4800, 4899), "Communication"),
         ((4900, 4999), "Utilities"), ((4000, 4799), "Industrials"), ((5122, 5122), "Healthcare"),
         ((5400, 5499), "Consumer Staples"), ((5912, 5912), "Consumer Staples"), ((5200, 5999), "Consumer Discretionary"),
         ((5000, 5199), "Industrials"), ((6324, 6324), "Healthcare"), ((6500, 6599), "Real Estate"), ((6798, 6798), "Real Estate"),
         ((6000, 6799), "Financials"), ((7370, 7379), "Technology"), ((7800, 7899), "Communication"),
         ((8000, 8099), "Healthcare"), ((7000, 7299), "Consumer Discretionary"), ((7900, 7999), "Consumer Discretionary"),
         ((8200, 8299), "Consumer Discretionary"), ((7300, 7369), "Industrials"), ((8700, 8799), "Industrials")]
    for (a, b), name in R:
        if a <= s <= b: return name
    return "Other"


def tier_of(mc) -> str | None:
    if mc is None: return None
    return "large" if mc >= 1e10 else "mid" if mc >= 2e9 else "small" if mc >= 3e8 else "micro"


def _series_break(c):
    """Index AFTER the last impossible one-day move (a spliced series: ticker reused
    or changed, e.g. META was an ETF's symbol before June 2022). None if clean.
    Large but possible moves (a biotech's trial result) are real and kept."""
    if len(c) < 3: return None
    r = c[1:] / c[:-1] - 1
    idx = np.where((r > 3.0) | (r < -0.75))[0]
    return int(idx[-1]) + 1 if len(idx) else None


def _suspect(c, ds, mc):
    """A one-day move no real company of this size makes usually means a broken
    series (ticker change, unadjusted split, reused symbol). Such companies stay off
    every tracker until the history is repaired. Returns a reason or None."""
    if len(c) < 3: return None
    r = c[1:] / c[:-1] - 1; i = int(np.argmax(np.abs(r)))
    limit = 0.6 if (mc or 0) >= 2e9 else 1.5
    if abs(r[i]) > limit: return f"one-day move of {r[i] * 100:+.0f}% on {ds[i + 1]}"
    return None


def _stage(c, lo_i, n) -> str:
    since_low = n - 1 - lo_i
    if since_low <= 10: return "falling"
    sma20 = c[-20:].mean(); sma50 = c[-50:].mean() if n >= 50 else sma20
    slope50 = (c[-50:].mean() - c[-60:-10].mean()) if n >= 60 else 0
    if c[-1] > sma50 and slope50 > 0: return "recovering"
    recent_low = c[-10:].min()
    if recent_low > c[lo_i] * 1.03 and c[-1] > sma20: return "turning"
    return "basing"


async def build_price_facts(pool, as_of: date | None = None) -> dict:
    async with pool.acquire() as con:
        await con.execute(SQL)
        uni = await con.fetch("SELECT ticker, name, sic_code, market_cap, cik FROM universe WHERE active")
        spy = await con.fetch("SELECT d, c FROM daily_bars WHERE ticker='SPY' ORDER BY d")
        dates = [r["d"] for r in await con.fetch("SELECT DISTINCT d FROM daily_bars WHERE ticker='SPY' ORDER BY d")]
        att = {r["ticker"]: (r["a30"], r["a180"]) for r in await con.fetch("""
            SELECT ticker, coalesce(sum(n_articles) FILTER (WHERE d > CURRENT_DATE - 30),0) a30,
                   coalesce(sum(n_articles) FILTER (WHERE d > CURRENT_DATE - 180),0) a180
            FROM ci_attention_daily GROUP BY ticker""")}
    di = {d: i for i, d in enumerate(dates)}; D = len(dates)
    spy_c = np.full(D, np.nan)
    for r in spy: spy_c[di[r["d"]]] = float(r["c"])
    as_of = as_of or dates[-1]
    M = np.full((len(uni), D), np.nan)       # closes aligned to SPY's trading dates, for sector comparisons
    facts, meta = [], []
    for k, u in enumerate(uni):
        tk = u["ticker"]
        async with pool.acquire() as con:
            rows = await con.fetch("SELECT d, h, l, c, v, o FROM daily_bars WHERE ticker=$1 ORDER BY d", tk)
        if len(rows) < 60: continue
        c = np.array([float(r["c"]) for r in rows]); v = np.array([float(r["v"] or 0) for r in rows])
        o = np.array([float(r["o"] or r["c"]) for r in rows]); ds = [r["d"] for r in rows]
        brk = _series_break(c); history_note = None
        if brk is not None:                       # repair: use only the history after the splice
            history_note = f"history from {ds[brk]} (earlier prices belong to a different security under this symbol)"
            c, v, o, ds = c[brk:], v[brk:], o[brk:], ds[brk:]
            if len(c) < 60: continue
        n = len(c)
        for i, d in enumerate(ds):
            j = di.get(d)
            if j is not None: M[k, j] = c[i]
        hi_i = int(np.argmax(c)); lo_i = hi_i + int(np.argmin(c[hi_i:]))
        ret = lambda s: float(c[-1] / c[-1 - s] - 1) if n > s else None
        wk_beat = up_wk = 0; wk_n = 0
        for w in range(26):
            a, b = n - 1 - 5 * w, n - 1 - 5 * (w + 1)
            if b < 0: break
            ja, jb = di.get(ds[a]), di.get(ds[b])
            if ja is None or jb is None or np.isnan(spy_c[ja]) or np.isnan(spy_c[jb]): continue
            sr, mr = c[a] / c[b] - 1, spy_c[ja] / spy_c[jb] - 1
            wk_n += 1; wk_beat += sr > mr; up_wk += sr > 0
        v20 = v[-20:].mean() if n >= 20 else v.mean(); v60 = v[-60:].mean() if n >= 60 else v.mean()
        upv = v[-20:][c[-20:] >= o[-20:]].sum() / max(1.0, v[-20:].sum())
        a30, a180 = att.get(tk, (0, 0))
        # A real trend break, not a stock hovering around its average: above the 200-day
        # average on 80%+ of the 60 sessions before the cross, and still 2%+ below it today.
        below200 = bool(n >= 200 and c[-1] < c[-200:].mean()); cross_i = None
        if below200 and n >= 290:
            cs_ = np.concatenate([[0.0], np.cumsum(c)]); sma = np.full(n, np.nan)
            sma[199:] = (cs_[200:] - cs_[:-200]) / 200.0
            if c[-1] <= 0.98 * sma[-1]:
                for i in range(n - 1, n - 21, -1):
                    if c[i] < sma[i] and c[i - 1] >= sma[i - 1]:
                        if np.mean(c[i - 60:i] > sma[i - 60:i]) >= 0.8: cross_i = i
                        break
        cross_vol = float(v[cross_i] / max(1.0, v[cross_i - 20:cross_i].mean())) if cross_i else None
        mc = u["market_cap"]
        facts.append({"ticker": tk, "name": u["name"], "tier": tier_of(mc), "sector": sector_of(tk, u["sic_code"]),
                      "is_spac": str(u["sic_code"] or "").startswith("6770"), "market_cap": mc, "price": float(c[-1]),
                      "high_5y": float(c[hi_i]), "high_date": ds[hi_i], "pct_below_high": float(c[-1] / c[hi_i] - 1),
                      "sessions_since_high": n - 1 - hi_i, "low_since_high": float(c[lo_i]), "low_date": ds[lo_i],
                      "sessions_since_low": n - 1 - lo_i, "pct_off_low": float(c[-1] / c[lo_i] - 1),
                      "stage": _stage(c, lo_i, n) if c[-1] < c[hi_i] * 0.9 else "near_high",
                      "ret_1d": ret(1), "ret_1w": ret(5), "ret_1m": ret(21), "ret_3m": ret(63), "ret_6m": ret(126), "ret_1y": ret(252),
                      "weeks_beat_mkt_26": wk_beat if wk_n else None, "up_weeks_26": up_wk if wk_n else None,
                      "vol_ratio_20_60": float(v20 / v60) if v60 > 0 else None, "up_vol_share_20": float(upv),
                      "news_30d": int(a30), "news_180d": int(a180), "cik": str(u["cik"]) if u["cik"] else None,
                      "dollar_vol_20": float((c[-20:] * v[-20:]).mean()), "sic": str(u["sic_code"] or ""),
                      "data_suspect": None, "history_note": history_note,
                      "below_200d": below200, "cross_200d_date": ds[cross_i] if cross_i else None, "cross_200d_vol": cross_vol})
        meta.append(k)
        if len(facts) % 1000 == 0: logger.info(f"[facts] {len(facts)} companies…")

    # one listing per company: the most-traded share class (GOOGL over GOOG, BRK.B over BRK.A)
    best = {}
    for f in facts:
        key = f["cik"] or f["ticker"]
        if key not in best or (f["dollar_vol_20"] or 0) > (best[key]["dollar_vol_20"] or 0): best[key] = f
    for f in facts: f["primary_listing"] = best[f["cik"] or f["ticker"]] is f

    # why it fell: market (SPY) and sector-median move over the same dates as the stock's drop
    now_j = D - 1; row_of = {f["ticker"]: meta[i] for i, f in enumerate(facts)}
    # Peers: same 4-digit industry code if it has 5+ established companies, else the
    # 3-digit group, else the sector. "Technology" held Micron (+569%) and Apple (at its
    # high) alongside chip-equipment makers that were all falling together.
    groups: dict[str, list[int]] = {}
    for f in facts:
        if f["tier"] in ("large", "mid") and not f["is_spac"] and f["primary_listing"]:
            r_ = row_of[f["ticker"]]
            if f["sic"][:4]: groups.setdefault("sic4:" + f["sic"][:4], []).append(r_)
            if f["sic"][:3]: groups.setdefault("sic3:" + f["sic"][:3], []).append(r_)
            groups.setdefault("sector:" + f["sector"], []).append(r_)
    def peer_key(f):
        for k in ("sic4:" + f["sic"][:4], "sic3:" + f["sic"][:3]):
            if len(groups.get(k, [])) >= 6: return k
        return "sector:" + f["sector"]
    sect_rows = groups
    cache = {}
    for f in facts:
        j = di.get(f["high_date"])
        if j is None: continue
        f["mkt_move_since_high"] = float(spy_c[now_j] / spy_c[j] - 1) if not np.isnan(spy_c[j]) else None
        pk = peer_key(f); f["peer_group"] = pk
        key = (pk, j)
        if key not in cache:
            rs = [x for x in sect_rows.get(pk, []) if x != row_of[f["ticker"]]]
            mv = M[rs, now_j] / M[rs, j] - 1 if rs else np.array([])
            cache[key] = float(np.nanmedian(mv)) if np.isfinite(mv).sum() >= 5 else None
        f["sector_move_since_high"] = cache[key]
        k1 = (pk, "1m")
        if k1 not in cache:
            rs1 = sect_rows.get(pk, [])
            m1 = M[rs1, now_j] / M[rs1, now_j - 21] - 1 if rs1 and now_j >= 21 else np.array([])
            cache[k1] = float(np.nanmedian(m1)) if np.isfinite(m1).sum() >= 5 else None
        f["peer_ret_1m"] = cache[k1]
        drop = f["pct_below_high"]
        if drop > -0.10: f["drop_cause"] = None
        elif f["mkt_move_since_high"] is not None and f["mkt_move_since_high"] <= 0.6 * drop: f["drop_cause"] = "market"
        elif f["sector_move_since_high"] is not None and f["sector_move_since_high"] <= 0.6 * drop: f["drop_cause"] = "industry"
        else: f["drop_cause"] = "company"

    cols = ["ticker", "name", "tier", "sector", "is_spac", "market_cap", "price", "high_5y", "high_date", "pct_below_high",
            "sessions_since_high", "low_since_high", "low_date", "sessions_since_low", "pct_off_low", "stage",
            "ret_1d", "ret_1w", "ret_1m", "ret_3m", "ret_6m", "ret_1y", "weeks_beat_mkt_26", "up_weeks_26",
            "vol_ratio_20_60", "up_vol_share_20", "mkt_move_since_high", "sector_move_since_high", "drop_cause",
            "news_30d", "news_180d", "cik", "primary_listing", "dollar_vol_20", "peer_group", "data_suspect", "history_note",
            "below_200d", "cross_200d_date", "cross_200d_vol", "peer_ret_1m"]
    ph = ",".join(f"${i + 2}" for i in range(len(cols)))
    upd = ",".join(f"{c}=EXCLUDED.{c}" for c in cols[1:])
    async with pool.acquire() as con:
        await con.executemany(f"""INSERT INTO company_facts (as_of,{",".join(cols)}) VALUES ($1,{ph})
                                  ON CONFLICT (ticker, as_of) DO UPDATE SET {upd}""",
                              [(as_of, *[f.get(c) for c in cols]) for f in facts])

    # history: large companies that fell 20/30/50% from a high — how often, and how fast, they got back
    base = {}
    large_rows = [row_of[f["ticker"]] for f in facts if f["tier"] == "large" and not f["is_spac"] and f["primary_listing"] and not f["data_suspect"]]
    for T in (0.20, 0.30, 0.50):
        eps = []
        for r in large_rows:
            s = M[r]; s = s[np.isfinite(s)]
            if len(s) < 260: continue
            peak, crossed = s[0], False
            for i in range(1, len(s)):
                if s[i] >= peak:
                    peak, crossed = s[i], False; continue
                if not crossed and s[i] <= peak * (1 - T):
                    crossed = True; tgt = peak
                    rec = np.where(s[i:] >= tgt)[0]
                    eps.append({"rec": bool(len(rec)), "days": int(rec[0]) if len(rec) else None, "follow": len(s) - 1 - i})
        full = [e for e in eps if e["follow"] >= 252]
        days = [e["days"] for e in eps if e["rec"]]
        base[f"{int(T * 100)}"] = {"episodes": len(eps), "recovered_pct": round(100 * np.mean([e["rec"] for e in eps]), 1) if eps else None,
                                   "recovered_within_1y_pct": round(100 * np.mean([e["rec"] and e["days"] <= 252 for e in full]), 1) if full else None,
                                   "median_sessions_to_recover": int(np.median(days)) if days else None,
                                   "episodes_with_1y_followup": len(full)}
    from core.artifact_paths import artifact_write_path
    art = {"generated": str(as_of), "window": f"{dates[0]} to {dates[-1]}", "universe": "large companies (over $10B)",
           "note": "An episode starts when a stock first closes the given % below its prior high; recovered = closed back at that high within the data.",
           "buckets": base}
    artifact_write_path("great_on_sale_base.json").write_text(json.dumps(art))
    logger.info(f"[facts] price facts: {len(facts)} companies as of {as_of}; base rates {base}")
    return {"companies": len(facts), "as_of": str(as_of), "base": base}
