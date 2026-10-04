"""ML v2, stage 2c — merge point-in-time fundamentals into the monthly price panel.

For each (month-end, stock) only quarters FILED by that date are used. Share counts are split-adjusted
(Nvidia's 10-for-1 in 2024 made raw counts jump 10x) so market value and yields are consistent with
split-adjusted prices. Companies without cover-page share counts (e.g. Rivian, which reports by class)
fall back to each quarter's diluted weighted-average share count."""
from __future__ import annotations
import json, os
import numpy as np, pandas as pd
from loguru import logger

OUT = "/app/models/panel_v2"


def _split_adjust(sh):
    """sh: list of (filed, shares) sorted by filed. Adjust earlier counts for split-sized jumps."""
    sh = [list(x) for x in sh]
    for i in range(1, len(sh)):
        a, b = sh[i - 1][1], sh[i][1]
        if not a or not b: continue
        r = b / a
        for k in (2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50):
            for kk in (k, 1 / k):
                if abs(r / kk - 1) < 0.08:
                    for j in range(i): sh[j][1] *= kk
                    break
            else: continue
            break
    return sh


def _diluted_shares(cik):
    from quantedge.fundamentals.edgar_bulk import company_facts_from_bulk
    import datetime as dt
    f = company_facts_from_bulk(str(cik)) or {}
    out = []
    for tag in ("WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic"):
        for u in (((f.get("facts") or {}).get("us-gaap") or {}).get(tag) or {}).get("units", {}).get("shares", []):
            try:
                span = (dt.date.fromisoformat(u["end"]) - dt.date.fromisoformat(u["start"])).days
                if 80 <= span <= 100 and u.get("val") and u.get("filed"): out.append((u["filed"], float(u["val"])))
            except Exception: pass
        if out: break
    return sorted(set(out))


async def build_fund_panel(pool):
    P = pd.read_parquet(f"{OUT}/prices.parquet"); Q = pd.read_parquet(f"{OUT}/fund_quarters.parquet"); S = pd.read_parquet(f"{OUT}/fund_shares.parquet")
    P["date"] = pd.to_datetime(P["date"]); Q["filed_d"] = pd.to_datetime(Q["filed"]); Q["end_d"] = pd.to_datetime(Q["end"])
    shares = {t: _split_adjust(sorted(zip(g["filed"], g["shares"]))) for t, g in S.groupby("ticker")}
    missing = sorted(set(P["ticker"]) - set(shares))
    if missing:
        ciks = {r["ticker"]: r["cik"] for r in await pool.fetch("SELECT ticker, cik FROM universe WHERE ticker = ANY($1) AND cik IS NOT NULL", missing)}
        for t in missing:
            try:
                d = _diluted_shares(ciks[t]) if t in ciks else []
                if d: shares[t] = _split_adjust(d)
            except Exception: pass
    logger.info(f"[ml2] share history for {len(shares)} companies ({len(missing)} needed the diluted-count fallback)")
    closes = {}
    for i in range(0, len(P["ticker"].unique()), 300):
        chunk = list(P["ticker"].unique()[i:i + 300])
        for r in await pool.fetch("SELECT ticker, d, c FROM daily_bars WHERE ticker = ANY($1) AND d >= '2021-06-01' AND c > 0 ORDER BY ticker, d", chunk):
            closes.setdefault(r["ticker"], []).append((pd.Timestamp(r["d"]), float(r["c"])))
    feats = []
    Qg = {t: g.sort_values("end_d") for t, g in Q.groupby("ticker")}
    for t, rows in P.groupby("ticker"):
        q = Qg.get(t); sh = shares.get(t, []); cl = closes.get(t, [])
        cd = np.array([c[0] for c in cl], dtype="datetime64[ns]") if cl else None; cv = np.array([c[1] for c in cl]) if cl else None
        for idx, r in rows.iterrows():
            d = r["date"]; f = {"_i": idx}
            if q is not None:
                k = q[q["filed_d"] <= d]
                if len(k) >= 5:
                    l4, p4 = k.tail(4), k.iloc[-8:-4] if len(k) >= 8 else None
                    s4 = l4["sales"].sum()
                    def yoy(row):
                        prev = k[(k["end_d"] - row["end_d"]).abs().sub(pd.Timedelta(days=365)).abs() <= pd.Timedelta(days=20)]
                        prev = k[((row["end_d"] - k["end_d"]).dt.days - 365).abs() <= 20]
                        return (row["sales"] / prev["sales"].iloc[0] - 1) if len(prev) and prev["sales"].iloc[0] > 0 else np.nan
                    ys = [yoy(k.iloc[-j]) for j in (1, 2, 3, 4) if len(k) >= j]
                    f["sales_yoy"] = ys[0]; f["sales_accel"] = ys[0] - ys[1] if len(ys) > 1 else np.nan
                    st = 0
                    for j in range(len(ys) - 1):
                        if not np.isnan(ys[j]) and not np.isnan(ys[j + 1]) and ys[j] > ys[j + 1]: st += 1
                        else: break
                    f["accel_streak"] = st
                    if s4 > 0:
                        f["gross_margin_ttm"] = l4["gross_profit"].sum() / s4 if l4["gross_profit"].notna().all() else np.nan
                        f["op_margin_ttm"] = l4["op_income"].sum() / s4 if l4["op_income"].notna().all() else np.nan
                        f["net_margin_ttm"] = l4["net_income"].sum() / s4 if l4["net_income"].notna().all() else np.nan
                        if p4 is not None and p4["sales"].sum() > 0 and p4["op_income"].notna().all() and l4["op_income"].notna().all():
                            f["op_margin_chg_1y"] = f["op_margin_ttm"] - p4["op_income"].sum() / p4["sales"].sum()
                        ni, oc = l4["net_income"].sum(), l4["op_cash_flow"].sum()
                        if l4["net_income"].notna().all() and l4["op_cash_flow"].notna().all():
                            f["accruals"] = (ni - oc) / s4
                            f["cash_backing"] = float(np.clip(oc / abs(ni), -3, 3)) if ni else np.nan
                        sh_now = [v for fd, v in sh if pd.Timestamp(fd) <= d]
                        if sh_now and sh_now[-1] > 0:
                            mcap = r_price = None
                            if cl:
                                j = np.searchsorted(cd, np.datetime64(d), side="right") - 1
                                if j >= 0: mcap = cv[j] * sh_now[-1]
                            if mcap:
                                f["log_mcap"] = np.log(mcap)
                                if l4["net_income"].notna().all(): f["earnings_yield"] = ni / mcap
                                f["sales_yield"] = s4 / mcap
                    fd = k["filed_d"].iloc[-1]; age = (d - fd).days; f["days_since_filing"] = age
                    if cl and age <= 100:
                        j0 = np.searchsorted(cd, np.datetime64(fd), side="left") - 1; j2 = j0 + 3; jd = np.searchsorted(cd, np.datetime64(d), side="right") - 1
                        if j0 >= 0 and j2 < len(cv) and jd >= j2:
                            f["filing_reaction"] = cv[j2] / cv[j0] - 1; f["post_filing_drift"] = cv[jd] / cv[j2] - 1
            feats.append(f)
    F = pd.DataFrame(feats).set_index("_i")
    P = P.join(F)
    P = P.sort_values(["ticker", "date"])
    P["ey_vs_own_3y"] = P.groupby("ticker")["earnings_yield"].transform(lambda s: s.rolling(36, min_periods=12).apply(lambda w: (w[:-1] < w[-1]).mean() if len(w) > 1 else np.nan, raw=True))
    P.to_parquet(f"{OUT}/panel.parquet", index=False)
    feat_cols = [c for c in P.columns if c not in ("date", "ticker", "sector", "fwd_21", "fwd_63", "y_21", "y_63")]
    prev = {}
    for h in ("21", "63"):
        for c in feat_cols:
            ics = []
            for dte, g in P.dropna(subset=[c, f"y_{h}"]).groupby("date"):
                if len(g) >= 100: ics.append(g[c].rank().corr(g[f"y_{h}"].rank()))
            if len(ics) >= 12:
                a = np.array(ics); prev.setdefault(c, {})[h] = {"ic": float(a.mean()), "t": float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))), "months": len(a),
                                                                 "coverage": float(P[c].notna().mean())}
    summ = {"rows": len(P), "features": feat_cols, "preview": prev}
    json.dump(summ, open(f"{OUT}/panel_summary.json", "w"), indent=1, default=str)
    logger.info(f"[ml2] panel with fundamentals: {len(P)} rows, {len(feat_cols)} features")
    return {"rows": len(P), "features": len(feat_cols)}
