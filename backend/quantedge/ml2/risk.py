"""ML v2 — risk model. Predicts next-month and next-quarter volatility and the chance of a 15%+
peak-to-trough fall within 3 months. Walk-forward (train on the past, gap, test the next 6 months),
judged against strong simple baselines: past volatility, and historical drop rates by volatility decile."""
from __future__ import annotations
import datetime as dt, json
import numpy as np, pandas as pd
from loguru import logger

OUT = "/app/models/panel_v2"
DROP = {"date", "ticker", "sector", "fwd_21", "fwd_63", "y_21", "y_63"}


async def add_risk_targets(pool):
    from quantedge.ml2.panel_prices import _matrix
    P = pd.read_parquet(f"{OUT}/panel.parquet"); P["date"] = pd.to_datetime(P["date"])
    C, _ = await _matrix(pool, sorted(P["ticker"].unique()), dt.date(2021, 8, 1)); C.index = pd.to_datetime(C.index)
    L = np.log(C).diff()
    fv21 = L.rolling(21, min_periods=18).std().shift(-21) * np.sqrt(252)
    fv63 = L.rolling(63, min_periods=55).std().shift(-63) * np.sqrt(252)
    pos = {d: i for i, d in enumerate(C.index)}; A = C.values; cols = {t: j for j, t in enumerate(C.columns)}
    dd = {}
    for d in P["date"].unique():
        i = pos.get(pd.Timestamp(d))
        if i is None or i + 63 >= len(A): continue
        W = A[i:i + 64]; runmax = np.fmax.accumulate(W, axis=0); dd[pd.Timestamp(d)] = np.nanmin(W / runmax - 1, axis=0)
    P["fut_vol_21"] = [fv21.at[d, t] if t in fv21.columns else np.nan for d, t in zip(P["date"], P["ticker"])]
    P["fut_vol_63"] = [fv63.at[d, t] if t in fv63.columns else np.nan for d, t in zip(P["date"], P["ticker"])]
    P["fut_dd_63"] = [dd[d][cols[t]] if (d in dd and t in cols) else np.nan for d, t in zip(P["date"], P["ticker"])]
    P["drop15_63"] = np.where(P["fut_dd_63"].notna(), (P["fut_dd_63"] <= -0.15).astype(float), np.nan)
    P.to_parquet(f"{OUT}/panel_risk.parquet", index=False)
    return {"rows": len(P), "with_vol21": int(P["fut_vol_21"].notna().sum()), "with_dd": int(P["fut_dd_63"].notna().sum()),
            "drop15_rate": float(P["drop15_63"].mean())}


def _auc(y, p):
    y = np.asarray(y); p = np.asarray(p); pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0: return np.nan
    r = pd.Series(p).rank().values
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def evaluate_risk():
    import lightgbm as lgb
    from quantedge.ml2.evaluate import nw_t
    P = pd.read_parquet(f"{OUT}/panel_risk.parquet"); P["date"] = pd.to_datetime(P["date"])
    feats = [c for c in P.columns if c not in DROP and not c.startswith(("fut_", "drop15"))]
    starts = [pd.Timestamp(s) for s in ("2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01")]
    out = {"features": feats, "vol": {}, "drop": {}}
    for tgt, gapm in (("fut_vol_21", 1), ("fut_vol_63", 3)):
        D = P.dropna(subset=[tgt, "vol_3m"]); ics_m, ics_b, err_m, err_b = [], [], [], []
        for s in starts:
            tr = D["date"] < s - pd.DateOffset(months=gapm); te = (D["date"] >= s) & (D["date"] < s + pd.DateOffset(months=6))
            if not te.any(): continue
            m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
            m.fit(D.loc[tr, feats], np.log(D.loc[tr, tgt].clip(lower=0.02)))
            pm = np.exp(m.predict(D.loc[te, feats])); pb = D.loc[te, "vol_3m"].values; y = D.loc[te, tgt].values
            T = pd.DataFrame({"d": D.loc[te, "date"].values, "m": pm, "b": pb, "y": y})
            for _, g in T.groupby("d"):
                if len(g) >= 100: ics_m.append(g["m"].rank().corr(g["y"].rank())); ics_b.append(g["b"].rank().corr(g["y"].rank()))
            err_m += list(np.abs(pm / y - 1)); err_b += list(np.abs(pb / y - 1))
        lag = 2 if gapm == 3 else 0
        out["vol"][tgt] = {"model_ic": float(np.mean(ics_m)), "model_t": float(nw_t(ics_m, lag)), "baseline_ic": float(np.mean(ics_b)),
                           "model_median_error": float(np.nanmedian(err_m)), "baseline_median_error": float(np.nanmedian(err_b)), "months": len(ics_m)}
        logger.info(f"[ml2 risk] {tgt}: {out['vol'][tgt]}")
    D = P.dropna(subset=["drop15_63", "vol_3m"]); ys, pm_all, pb_all = [], [], []
    for s in starts:
        tr = D["date"] < s - pd.DateOffset(months=3); te = (D["date"] >= s) & (D["date"] < s + pd.DateOffset(months=6))
        if not te.any(): continue
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        m.fit(D.loc[tr, feats], D.loc[tr, "drop15_63"].astype(int)); pm = m.predict_proba(D.loc[te, feats])[:, 1]
        bins = pd.qcut(D.loc[tr, "vol_3m"], 10, labels=False, duplicates="drop"); rate = D.loc[tr, "drop15_63"].groupby(bins).mean()
        edges = D.loc[tr, "vol_3m"].groupby(bins).max().values
        pb = rate.reindex(np.searchsorted(edges, D.loc[te, "vol_3m"].values).clip(0, len(rate) - 1)).values
        ys += list(D.loc[te, "drop15_63"].values); pm_all += list(pm); pb_all += list(pb)
    y, pm, pb = np.array(ys), np.array(pm_all), np.array(pb_all); base = float(y.mean())
    cal = pd.DataFrame({"p": pm, "y": y}); cal["bin"] = pd.qcut(cal["p"], 10, labels=False, duplicates="drop")
    out["drop"] = {"base_rate": base, "auc_model": _auc(y, pm), "auc_baseline": _auc(y, pb),
                   "brier_model": float(np.mean((pm - y) ** 2)), "brier_baseline": float(np.mean((pb - y) ** 2)), "brier_base_rate": float(np.mean((base - y) ** 2)),
                   "calibration": [{"predicted": float(g["p"].mean()), "actual": float(g["y"].mean()), "n": int(len(g))} for _, g in cal.groupby("bin")],
                   "n_tested": int(len(y))}
    json.dump(out, open(f"{OUT}/eval_risk.json", "w"), indent=1, default=str)
    return out
