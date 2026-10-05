"""ML v2 — risk at every horizon (1w, 2w, 1m, 3m, 6m, 1y): expected volatility (model vs past-volatility
baseline), the likely price range (80%, calibrated on how real returns spread around the volatility forecast),
and the chance of a 15% / 25% fall within the horizon (volatility-decile tables). Walk-forward validated."""
from __future__ import annotations
import datetime as dt, json, math
import numpy as np, pandas as pd
from loguru import logger
from quantedge.ml2.risk import _auc
from quantedge.ml2.evaluate import nw_t

OUT = "/app/models/panel_v2"
H = {5: "1 week", 10: "2 weeks", 21: "1 month", 63: "3 months", 126: "6 months", 252: "1 year"}
DROP = {"date", "ticker", "sector", "fwd_21", "fwd_63", "y_21", "y_63"}
LGB = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
STARTS = ["2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01"]


async def add_targets(pool):
    from quantedge.ml2.panel_prices import _matrix
    P = pd.read_parquet(f"{OUT}/panel.parquet"); P["date"] = pd.to_datetime(P["date"])
    C, _ = await _matrix(pool, sorted(P["ticker"].unique()), dt.date(2021, 8, 1)); C.index = pd.to_datetime(C.index)
    L = np.log(C).diff(); A = C.values; pos = {d: i for i, d in enumerate(C.index)}; col = {t: j for j, t in enumerate(C.columns)}
    di = np.array([pos.get(d, -1) for d in P["date"]]); ci = np.array([col.get(t, -1) for t in P["ticker"]]); ok = (di >= 0) & (ci >= 0)
    for h in H:
        fv = (L.rolling(h, min_periods=max(4, int(h * 0.85))).std().shift(-h) * np.sqrt(252)).values
        fr = (C.shift(-h) / C - 1).values
        P[f"fv_{h}"] = np.where(ok, fv[di.clip(0), ci.clip(0)], np.nan); P[f"fr_{h}"] = np.where(ok, fr[di.clip(0), ci.clip(0)], np.nan)
        dd = np.full(len(P), np.nan)
        for d in P["date"].unique():
            i = pos.get(pd.Timestamp(d))
            if i is None or i + h >= len(A): continue
            W = A[i:i + h + 1]; m = np.nanmin(W / np.fmax.accumulate(W, axis=0) - 1, axis=0)
            rows = np.where((P["date"].values == np.datetime64(d)) & ok)[0]; dd[rows] = m[ci[rows]]
        P[f"fd_{h}"] = dd
        logger.info(f"[ml2 horizons] targets {h}d: {np.isfinite(P[f'fv_{h}']).sum():,} rows")
    P.to_parquet(f"{OUT}/panel_h.parquet", index=False)
    return {"rows": len(P)}


def _feats(P): return [c for c in P.columns if c not in DROP and not c.startswith(("fut_", "drop15", "fv_", "fr_", "fd_"))]


def _drop_eval(tr, te, h, thr):
    d = tr.dropna(subset=[f"fd_{h}", "vol_3m"]).copy(); d["b"] = pd.qcut(d["vol_3m"], 10, labels=False, duplicates="drop")
    edges = d.groupby("b")["vol_3m"].max().values; rate = d.groupby("b")[f"fd_{h}"].apply(lambda s: float((s <= -thr).mean())).values
    t = te.dropna(subset=[f"fd_{h}", "vol_3m"])
    return rate[np.searchsorted(edges, t["vol_3m"].values).clip(0, len(rate) - 1)], (t[f"fd_{h}"] <= -thr).astype(float).values


def evaluate_horizons():
    import lightgbm as lgb
    P = pd.read_parquet(f"{OUT}/panel_h.parquet"); P["date"] = pd.to_datetime(P["date"]); feats = _feats(P); out = {}
    for h, label in H.items():
        D = P.dropna(subset=[f"fv_{h}", f"fr_{h}", "vol_3m"]); gap = pd.DateOffset(months=max(1, math.ceil(h / 21)))
        icm, icb, em, eb, cover, n_test, folds = [], [], [], [], [], 0, 0; dr = {0.15: ([], []), 0.25: ([], [])}
        for s in map(pd.Timestamp, STARTS):
            tr = D[D["date"] < s - gap]; te = D[(D["date"] >= s) & (D["date"] < s + pd.DateOffset(months=6))]
            if len(te) < 500 or len(tr) < 5000: continue
            folds += 1
            m = lgb.LGBMRegressor(**LGB).fit(tr[feats], np.log(tr[f"fv_{h}"].clip(lower=0.02)))
            ptr, pte = np.exp(m.predict(tr[feats])), np.exp(m.predict(te[feats]))
            z = np.log1p(tr[f"fr_{h}"].clip(lower=-0.99)) / (ptr * math.sqrt(h / 252)); q10, q90 = np.nanpercentile(z, [10, 90])
            lr = np.log1p(te[f"fr_{h}"].clip(lower=-0.99)).values; s_ = pte * math.sqrt(h / 252)
            cover.append(float(np.mean((lr >= q10 * s_) & (lr <= q90 * s_)))); n_test += len(te)
            T = pd.DataFrame({"d": te["date"].values, "m": pte, "b": te["vol_3m"].values, "y": te[f"fv_{h}"].values})
            for _, g in T.groupby("d"):
                if len(g) >= 100: icm.append(g["m"].rank().corr(g["y"].rank())); icb.append(g["b"].rank().corr(g["y"].rank()))
            em += list(np.abs(pte / te[f"fv_{h}"].values - 1)); eb += list(np.abs(te["vol_3m"].values / te[f"fv_{h}"].values - 1))
            for thr in dr:
                p, y = _drop_eval(tr, te, h, thr); dr[thr][0].extend(p); dr[thr][1].extend(y)
        if not folds: out[h] = {"label": label, "available": False}; continue
        res = {"label": label, "available": True, "folds": folds, "n_test": n_test,
               "vol": {"model_ic": float(np.mean(icm)), "baseline_ic": float(np.mean(icb)), "model_error": float(np.nanmedian(em)), "baseline_error": float(np.nanmedian(eb))},
               "range80_coverage": float(np.mean(cover)), "drop": {}}
        for thr, (p, y) in dr.items():
            p, y = np.array(p), np.array(y)
            cal = pd.DataFrame({"p": p, "y": y}); cal["b"] = pd.qcut(cal["p"].rank(method="first"), 5, labels=False)
            res["drop"][str(int(thr * 100))] = {"base_rate": float(y.mean()), "auc": _auc(y, p), "brier": float(np.mean((p - y) ** 2)),
                                                "brier_base_rate": float(np.mean((y.mean() - y) ** 2)),
                                                "calibration": [{"predicted": float(g["p"].mean()), "actual": float(g["y"].mean())} for _, g in cal.groupby("b")]}
        res["passes"] = bool(res["vol"]["model_ic"] > res["vol"]["baseline_ic"] and res["vol"]["model_error"] < res["vol"]["baseline_error"] and 0.74 <= res["range80_coverage"] <= 0.86)
        out[h] = res; logger.info(f"[ml2 horizons] {label}: {res['vol']} coverage {res['range80_coverage']:.2f} passes={res['passes']}")
    json.dump(out, open(f"{OUT}/eval_horizons.json", "w"), indent=1, default=str)
    return {h: (r.get("passes"), r.get("range80_coverage")) for h, r in out.items()}


async def serve_horizons(pool):
    """Final models on all labelled history; forecasts for every stock at the latest panel date. 1 year uses the
    past-volatility baseline (the model didn't beat it). Ranges are stored as multipliers on the live price."""
    import lightgbm as lgb
    P = pd.read_parquet(f"{OUT}/panel_h.parquet"); P["date"] = pd.to_datetime(P["date"]); feats = _feats(P)
    latest = P["date"].max(); now = P[P["date"] == latest].copy(); ev = json.load(open(f"{OUT}/eval_horizons.json"))
    recs = {t: {} for t in now["ticker"]}; base_cov = None
    for h in H:
        D = P.dropna(subset=[f"fv_{h}", f"fr_{h}", "vol_3m"]); use_model = bool(ev.get(str(h), {}).get("passes"))
        if use_model:
            m = lgb.LGBMRegressor(**LGB).fit(D[feats], np.log(D[f"fv_{h}"].clip(lower=0.02)))
            ptr, pnow = np.exp(m.predict(D[feats])), np.exp(m.predict(now[feats]))
        else:
            ptr, pnow = D["vol_3m"].values, now["vol_3m"].fillna(now["vol_3m"].median()).values
            if h == 252:   # walk-forward coverage of the baseline range, so the page can show its own record
                cov = []
                for s in map(pd.Timestamp, STARTS):
                    tr = D[D["date"] < s - pd.DateOffset(months=12)]; te = D[(D["date"] >= s) & (D["date"] < s + pd.DateOffset(months=6))]
                    if len(te) < 500 or len(tr) < 5000: continue
                    z = np.log1p(tr[f"fr_{h}"].clip(lower=-0.99)) / (tr["vol_3m"] * math.sqrt(h / 252)); q10, q90 = np.nanpercentile(z, [10, 90])
                    lr = np.log1p(te[f"fr_{h}"].clip(lower=-0.99)).values; s_ = te["vol_3m"].values * math.sqrt(h / 252)
                    cov.append(float(np.mean((lr >= q10 * s_) & (lr <= q90 * s_))))
                base_cov = float(np.mean(cov)) if cov else None
        z = np.log1p(D[f"fr_{h}"].clip(lower=-0.99)).values / (ptr * math.sqrt(h / 252)); q10, q50, q90 = np.nanpercentile(z, [10, 50, 90])
        sc = pnow * math.sqrt(h / 252)
        tabs = {}
        for thr in (0.15, 0.25):
            d = D.dropna(subset=[f"fd_{h}"]).copy(); d["b"] = pd.qcut(d["vol_3m"], 10, labels=False, duplicates="drop")
            edges = d.groupby("b")["vol_3m"].max().values; rate = d.groupby("b")[f"fd_{h}"].apply(lambda s: float((s <= -thr).mean())).values
            tabs[thr] = rate[np.searchsorted(edges, now["vol_3m"].fillna(now["vol_3m"].median()).values).clip(0, len(rate) - 1)]
        for i, t in enumerate(now["ticker"]):
            recs[t][h] = (float(pnow[i]), float(np.exp(q10 * sc[i])), float(np.exp(q50 * sc[i])), float(np.exp(q90 * sc[i])),
                          float(tabs[0.15][i]), float(tabs[0.25][i]), "model" if use_model else "baseline")
    await pool.execute("""CREATE TABLE IF NOT EXISTS ml_risk_h (as_of DATE, ticker TEXT, horizon INT, vol DOUBLE PRECISION, lo_mult DOUBLE PRECISION,
        mid_mult DOUBLE PRECISION, hi_mult DOUBLE PRECISION, drop15 DOUBLE PRECISION, drop25 DOUBLE PRECISION, method TEXT, PRIMARY KEY (as_of, ticker, horizon))""")
    rows = [(latest.date(), t, h, *v) for t, hv in recs.items() for h, v in hv.items()]
    await pool.executemany("""INSERT INTO ml_risk_h VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) ON CONFLICT (as_of, ticker, horizon) DO UPDATE SET
        vol=EXCLUDED.vol, lo_mult=EXCLUDED.lo_mult, mid_mult=EXCLUDED.mid_mult, hi_mult=EXCLUDED.hi_mult, drop15=EXCLUDED.drop15, drop25=EXCLUDED.drop25, method=EXCLUDED.method""", rows)
    if base_cov is not None and "252" in ev: ev["252"]["baseline_range80_coverage"] = base_cov
    json.dump(ev, open(f"{OUT}/serving_h.json", "w"), indent=1, default=str)
    logger.info(f"[ml2 horizons] served {len(recs)} stocks x {len(H)} horizons as of {latest.date()}")
    return {"as_of": str(latest.date()), "stocks": len(recs), "baseline_1y_coverage": base_cov}
