"""ML v2, stage 2a — monthly point-in-time price panel.

One row per (month-end, stock): price features computed only from data up to that date, and the
targets — forward 21- and 63-session returns relative to the stock's sector median on that date.
Universe per date: primary US common stocks with price >= $3 and >= $1M average daily dollar volume."""
from __future__ import annotations
import datetime as dt, json, os
import numpy as np, pandas as pd
from loguru import logger

OUT = "/app/models/panel_v2"


async def _matrix(pool, tickers, start):
    parts_c, parts_v = [], []
    for i in range(0, len(tickers), 300):
        rows = await pool.fetch("SELECT ticker, d, c, v FROM daily_bars WHERE ticker = ANY($1) AND d >= $2 AND c > 0", tickers[i:i + 300], start)
        if not rows: continue
        df = pd.DataFrame([(r["ticker"], r["d"], float(r["c"]), float(r["v"] or 0)) for r in rows], columns=["t", "d", "c", "v"])
        parts_c.append(df.pivot_table(index="d", columns="t", values="c")); parts_v.append(df.pivot_table(index="d", columns="t", values="v"))
    C = pd.concat(parts_c, axis=1).sort_index(); V = pd.concat(parts_v, axis=1).reindex(C.index)
    return C, V


async def build_price_panel(pool, start=dt.date(2021, 8, 1)):
    os.makedirs(OUT, exist_ok=True)
    meta = await pool.fetch("""SELECT ticker, sector FROM company_facts WHERE as_of = (SELECT max(as_of) FROM company_facts)
                               AND primary_listing AND NOT is_spac AND data_suspect IS NULL AND tier IN ('large','mid','small')""")
    sector = {r["ticker"]: r["sector"] or "Unknown" for r in meta}
    tickers = sorted(sector)
    C, V = await _matrix(pool, tickers + ["SPY"], start)
    spy = C.pop("SPY") if "SPY" in C else None; V = V.drop(columns=["SPY"], errors="ignore")
    logger.info(f"[ml2] price matrix {C.shape[0]} days x {C.shape[1]} stocks")
    L = np.log(C).diff()
    feats = {
        "mom_12_1": C.shift(21) / C.shift(252) - 1, "mom_6_1": C.shift(21) / C.shift(126) - 1, "rev_1m": C / C.shift(21) - 1,
        "dist_52w_high": C / C.rolling(252, min_periods=200).max() - 1, "trend_200": C / C.rolling(200, min_periods=180).mean() - 1,
        "vol_3m": L.rolling(63, min_periods=50).std() * np.sqrt(252),
        "max_dd_6m": (C / C.rolling(126, min_periods=100).max() - 1).rolling(126, min_periods=100).min(),
        "log_dollar_vol": np.log((C * V).rolling(20, min_periods=15).mean().clip(lower=1)),
        "amihud_1m": np.log1p((L.abs() / (C * V).replace(0, np.nan)).rolling(21, min_periods=15).mean() * 1e9),
    }
    if spy is not None:
        ls = np.log(spy).diff()
        feats["beta_1y"] = L.rolling(252, min_periods=200).cov(ls).div(ls.rolling(252, min_periods=200).var(), axis=0)
    fwd21, fwd63 = C.shift(-21) / C - 1, C.shift(-63) / C - 1
    idx = pd.Series(C.index, index=C.index); month_end = idx.groupby([pd.Index(C.index).map(lambda d: (d.year, d.month))]).max().values
    month_end = [d for d in month_end if d >= start + dt.timedelta(days=380)]
    recs = []
    for d in month_end:
        px = C.loc[d]; dv = np.exp(feats["log_dollar_vol"].loc[d])
        ok = (px >= 3) & (dv >= 1e6) & feats["mom_12_1"].loc[d].notna()
        for t in px.index[ok.values]:
            row = {"date": d, "ticker": t, "sector": sector.get(t, "Unknown")}
            for k, M in feats.items(): row[k] = M.at[d, t]
            row["fwd_21"] = fwd21.at[d, t]; row["fwd_63"] = fwd63.at[d, t]
            recs.append(row)
    P = pd.DataFrame(recs)
    for h in ("21", "63"):   # sector-relative target: return minus the sector median on that date
        P[f"y_{h}"] = P[f"fwd_{h}"] - P.groupby(["date", "sector"])[f"fwd_{h}"].transform("median")
    P.to_parquet(f"{OUT}/prices.parquet", index=False)
    # early look at stage 3: how well each signal alone ranked sector-relative returns (mean monthly rank correlation)
    prev = {}
    for k in feats:
        ics = []
        for d, g in P.dropna(subset=[k, "y_21"]).groupby("date"):
            if len(g) >= 100: ics.append(g[k].rank().corr(g["y_21"].rank()))
        if ics:
            a = np.array(ics); prev[k] = {"ic": float(a.mean()), "t": float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))), "months": len(a)}
    summ = {"rows": len(P), "months": int(P["date"].nunique()), "first": str(min(month_end)), "last": str(max(month_end)),
            "stocks_per_month_median": int(P.groupby("date").size().median()), "preview_ic_21d": prev}
    json.dump(summ, open(f"{OUT}/prices_summary.json", "w"), indent=1, default=str)
    logger.info(f"[ml2] price panel {summ['rows']} rows, {summ['months']} months")
    return summ
