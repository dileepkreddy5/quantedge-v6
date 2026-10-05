"""ML v3, phase B — the factor library: 11 research factor families, monthly, strictly point-in-time.
Prices: 'long' (25y Yahoo, total return + split-adjusted) or 'polygon' (5y, for development).
Fundamentals: SEC quarters (by filing date), balance sheets (first filing), share counts (split-adjusted)."""
from __future__ import annotations
import datetime as dt, json
import numpy as np, pandas as pd
from loguru import logger

OUT = "/app/models/panel_v3"; V2 = "/app/models/panel_v2"
H = [5, 10, 21, 63, 126, 252]
LIT = {   # family: [(feature, sign from the research)]
    "momentum": [("mom_12_1", 1), ("hi52", 1), ("ind_mom", 1)],          # Jegadeesh-Titman 1993; George-Hwang 2004; Moskowitz-Grinblatt 1999
    "reversal": [("rev_1m", -1)],                                         # Jegadeesh 1990
    "earnings": [("sue", 1), ("sales_accel", 1)],                         # Bernard-Thomas 1989
    "profitability": [("gp_assets", 1), ("roe", 1)],                      # Novy-Marx 2013; Asness-Frazzini-Pedersen 2019
    "investment": [("asset_growth", -1)],                                 # Cooper-Gulen-Schill 2008
    "issuance": [("net_issuance", -1)],                                   # Pontiff-Woodgate 2008
    "accruals": [("accruals", -1)],                                       # Sloan 1996
    "value": [("ey", 1), ("bm", 1), ("sy", 1), ("cfy", 1)],               # Fama-French 1992, 2015
    "size": [("log_mcap", -1)],                                           # Banz 1981
    "low_risk": [("ivol", -1), ("beta", -1)],                             # Ang et al. 2006; Frazzini-Pedersen 2014
    "liquidity": [("amihud", 1)],                                         # Amihud 2002
}


async def _prices(pool, source):
    if source == "long":
        import pyarrow.parquet as pq
        T = pq.read_table(f"{OUT}/prices_long.parquet", read_dictionary=["ticker"]).to_pandas(date_as_object=False).drop_duplicates(["ticker", "d"])
        di = pd.DatetimeIndex(sorted(T["d"].unique())); tk = list(T["ticker"].cat.categories); r = di.get_indexer(T["d"]); c = T["ticker"].cat.codes.values
        def _mat(col):
            a = np.full((len(di), len(tk)), np.nan, dtype="float32"); a[r, c] = T[col].values.astype("float32"); return pd.DataFrame(a, index=di, columns=tk)
        C, CS, V = _mat("c"), _mat("cs"), _mat("v"); del T
    else:
        from quantedge.ml2.panel_prices import _matrix
        tk = [r["ticker"] for r in await pool.fetch("""SELECT ticker FROM company_facts WHERE as_of=(SELECT max(as_of) FROM company_facts)
                                                        AND primary_listing AND NOT is_spac AND tier IN ('large','mid','small')""")]
        C, V = await _matrix(pool, tk + ["SPY"], dt.date(2021, 8, 1)); C.index = pd.to_datetime(C.index); V.index = C.index; CS = C
    return C.astype("float32"), CS.reindex_like(C).astype("float32"), V.reindex_like(C).astype("float32")


async def build_factors(pool, source="polygon"):
    C, CS, V = await _prices(pool, source)
    sector = {r["ticker"]: (r["sector"] or "Unknown") for r in await pool.fetch("SELECT ticker, sector FROM company_facts WHERE as_of=(SELECT max(as_of) FROM company_facts)")}
    M = C.pop("SPY").values; CS = CS.drop(columns="SPY", errors="ignore"); V = V.drop(columns="SPY", errors="ignore")
    tick = np.array(C.columns); A, AS, VV = C.values, CS.values, V.values; dates = C.index
    with np.errstate(all="ignore"):
        L = np.vstack([np.full((1, A.shape[1]), np.nan, "float32"), np.diff(np.log(A), axis=0)]); LM = np.r_[np.nan, np.diff(np.log(M))]
    ym = dates.year * 12 + dates.month; me = np.where(np.r_[ym[1:] != ym[:-1], True])[0]; me = me[me >= 252]
    recs = []
    with np.errstate(all="ignore"):
        for i in me:
            c = A[i]; dv = np.nanmean(AS[i - 20:i + 1] * VV[i - 20:i + 1], axis=0)
            ok = np.isfinite(c) & np.isfinite(A[i - 252]) & (AS[i] >= 3) & (dv >= 1e6)
            if ok.sum() < 50: continue
            w, wm = L[i - 62:i + 1], LM[i - 62:i + 1]; wy, wmy = L[i - 251:i + 1], LM[i - 251:i + 1]
            beta = np.nanmean((wy - np.nanmean(wy, 0)) * (wmy - np.nanmean(wmy))[:, None], 0) / np.nanvar(wmy)
            covq = np.nanmean((w - np.nanmean(w, 0)) * (wm - np.nanmean(wm))[:, None], 0)
            f = {"mom_12_1": A[i - 21] / A[i - 252] - 1, "mom_6_1": A[i - 21] / A[i - 126] - 1, "rev_1m": c / A[i - 21] - 1,
                 "hi52": c / np.nanmax(A[i - 251:i + 1], 0), "vol_3m": np.nanstd(w, 0) * np.sqrt(252), "beta": beta,
                 "ivol": np.sqrt(np.clip(np.nanvar(w, 0) - covq ** 2 / np.nanvar(wm), 0, None)) * np.sqrt(252),
                 "amihud": np.log1p(np.nanmean(np.abs(L[i - 20:i + 1]) / (AS[i - 20:i + 1] * VV[i - 20:i + 1] + 1), 0) * 1e9),
                 "log_dv": np.log(dv), "price_s": AS[i]}
            for h in H: f[f"fwd_{h}"] = (A[i + h] / c - 1) if i + h < len(A) else np.full(len(c), np.nan)
            df = pd.DataFrame(f); df["ticker"] = tick; df["date"] = dates[i]; recs.append(df[ok])
    P = pd.concat(recs, ignore_index=True); P["sector"] = P["ticker"].map(sector).fillna("Unknown")
    P["ind_mom"] = P.groupby(["date", "sector"])["mom_6_1"].transform("median")
    for h in H: P[f"y_{h}"] = P[f"fwd_{h}"] - P.groupby(["date", "sector"])[f"fwd_{h}"].transform("median")
    logger.info(f"[ml3] price factors: {len(P):,} rows, {P['date'].nunique()} months")
    # --- fundamentals, merged by the date they became public ---
    Q = pd.read_parquet(f"{V2}/fund_quarters.parquet"); Q["end"] = pd.to_datetime(Q["end"]); Q["filed"] = pd.to_datetime(Q["filed"])
    Q = Q.sort_values(["ticker", "end"]).drop_duplicates(["ticker", "end"]).reset_index(drop=True); g = Q.groupby("ticker")
    ok4 = (Q["end"] - g["end"].shift(3)).dt.days.between(250, 300); gap4 = (Q["end"] - g["end"].shift(4)).dt.days.between(350, 380)
    for k in ("sales", "gross_profit", "net_income", "op_cash_flow"):
        Q[k + "_ttm"] = g[k].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True).where(ok4)
    prev_s = g["sales"].shift(4)
    Q["sales_yoy"] = (Q["sales"] / prev_s - 1).where(gap4 & (prev_s > 0)); Q["sales_accel"] = Q["sales_yoy"] - Q.groupby("ticker")["sales_yoy"].shift(1)
    Q["_d4"] = (Q["net_income"] - g["net_income"].shift(4)).where(gap4)
    Q["sue"] = (Q["_d4"] / Q.groupby("ticker")["_d4"].rolling(8, min_periods=4).std().reset_index(level=0, drop=True)).clip(-10, 10)
    Q["avail"] = Q.groupby("ticker")["filed"].cummax()
    keep = ["ticker", "avail", "sales_ttm", "gross_profit_ttm", "net_income_ttm", "op_cash_flow_ttm", "sales_yoy", "sales_accel", "sue"]
    P = pd.merge_asof(P.sort_values("date"), Q[keep].dropna(subset=["avail"]).sort_values("avail"), left_on="date", right_on="avail",
                      by="ticker", direction="backward", tolerance=pd.Timedelta(days=200))
    B = pd.read_parquet(f"{V2}/fund_balance.parquet").dropna(subset=["filed"]); B["end"] = pd.to_datetime(B["end"]); B["filed"] = pd.to_datetime(B["filed"])
    B = B.sort_values(["ticker", "end"]).reset_index(drop=True); B["bavail"] = B.groupby("ticker")["filed"].cummax()
    prev = B[["ticker", "end", "assets"]].rename(columns={"end": "end_prev", "assets": "assets_prev"}).sort_values("end_prev")
    B["_k"] = B["end"] - pd.Timedelta(days=365)
    B = pd.merge_asof(B.sort_values("_k"), prev, left_on="_k", right_on="end_prev", by="ticker", direction="nearest", tolerance=pd.Timedelta(days=45))
    B["asset_growth"] = (B["assets"] / B["assets_prev"] - 1).where(B["assets_prev"] > 0)
    P = pd.merge_asof(P.sort_values("date"), B[["ticker", "bavail", "assets", "equity", "asset_growth"]].sort_values("bavail"),
                      left_on="date", right_on="bavail", by="ticker", direction="backward", tolerance=pd.Timedelta(days=200))
    from quantedge.ml2.panel_fund import _split_adjust
    S = pd.read_parquet(f"{V2}/fund_shares.parquet"); rows = []
    for t, gq in S.groupby("ticker"):
        rows += [(t, pd.Timestamp(fd), v) for fd, v in _split_adjust(sorted(zip(gq["filed"], gq["shares"])))]
    SH = pd.DataFrame(rows, columns=["ticker", "sh_filed", "sh"]).sort_values("sh_filed")
    P = pd.merge_asof(P.sort_values("date"), SH, left_on="date", right_on="sh_filed", by="ticker", direction="backward", tolerance=pd.Timedelta(days=200))
    P["_d1"] = P["date"] - pd.Timedelta(days=365)
    P = pd.merge_asof(P.sort_values("_d1"), SH.rename(columns={"sh_filed": "sh1_filed", "sh": "sh1"}), left_on="_d1", right_on="sh1_filed",
                      by="ticker", direction="backward", tolerance=pd.Timedelta(days=200))
    with np.errstate(all="ignore"):
        mc = P["price_s"] * P["sh"]
        P["net_issuance"] = np.log(P["sh"] / P["sh1"]); P["log_mcap"] = np.log(mc)
        P["ey"] = P["net_income_ttm"] / mc; P["sy"] = P["sales_ttm"] / mc; P["cfy"] = P["op_cash_flow_ttm"] / mc; P["bm"] = P["equity"] / mc
        P["gp_assets"] = P["gross_profit_ttm"] / P["assets"]; P["roe"] = P["net_income_ttm"] / P["equity"].where(P["equity"] > 0)
        P["accruals"] = (P["net_income_ttm"] - P["op_cash_flow_ttm"]) / P["assets"]
    P = P.replace([np.inf, -np.inf], np.nan).drop(columns=["avail", "bavail", "sh_filed", "sh1_filed", "_d1"], errors="ignore").sort_values(["date", "ticker"]).reset_index(drop=True)
    P.to_parquet(f"{OUT}/factors_{source}.parquet", index=False)
    cov = {f: round(float(P[f].notna().mean()), 2) for fam in LIT.values() for f, _ in fam}
    logger.info(f"[ml3] factor panel ({source}): {len(P):,} rows · coverage {cov}")
    return {"rows": len(P), "months": int(P["date"].nunique()), "coverage": cov}
