"""ML v2 — serving. Trains the final risk models on all labelled history, forecasts every stock at the
latest panel date, and stores the evidence beside the forecasts. Volatility: LightGBM (beat the past-volatility
baseline walk-forward). Drop risk: the validated volatility-decile table (the model only matched it)."""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from loguru import logger
from quantedge.ml2.risk import _auc

OUT = "/app/models/panel_v2"
DROP = {"date", "ticker", "sector", "fwd_21", "fwd_63", "y_21", "y_63"}
LGB = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def _drop_table(D, thr):
    d = D.dropna(subset=["fut_dd_63", "vol_3m"]).copy(); d["dec"] = pd.qcut(d["vol_3m"], 10, labels=False, duplicates="drop")
    return d.groupby("dec")["vol_3m"].max().values, d.groupby("dec")["fut_dd_63"].apply(lambda s: float((s <= -thr).mean())).values


def _drop_eval(P, thr):
    """Walk-forward check of the volatility-decile drop table at a threshold (same folds as the models)."""
    ys, ps = [], []
    for s in [pd.Timestamp(x) for x in ("2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01")]:
        tr = P[P["date"] < s - pd.DateOffset(months=3)]; te = P[(P["date"] >= s) & (P["date"] < s + pd.DateOffset(months=6))].dropna(subset=["fut_dd_63", "vol_3m"])
        if te.empty: continue
        edges, rate = _drop_table(tr, thr)
        ps += list(rate[np.searchsorted(edges, te["vol_3m"].values).clip(0, len(rate) - 1)]); ys += list((te["fut_dd_63"] <= -thr).astype(float).values)
    y, p = np.array(ys), np.array(ps)
    cal = pd.DataFrame({"p": p, "y": y}); cal["b"] = pd.qcut(cal["p"].rank(method="first"), 5, labels=False)
    return {"threshold": thr, "base_rate": float(y.mean()), "auc": _auc(y, p), "brier": float(np.mean((p - y) ** 2)),
            "brier_base_rate": float(np.mean((y.mean() - y) ** 2)), "n_tested": int(len(y)),
            "calibration": [{"predicted": float(g["p"].mean()), "actual": float(g["y"].mean())} for _, g in cal.groupby("b")]}


async def build_serving(pool):
    import lightgbm as lgb
    P = pd.read_parquet(f"{OUT}/panel_risk.parquet"); P["date"] = pd.to_datetime(P["date"])
    feats = [c for c in P.columns if c not in DROP and not c.startswith(("fut_", "drop15"))]
    latest = P["date"].max(); now = P[P["date"] == latest].copy()
    for tgt in ("fut_vol_21", "fut_vol_63"):
        D = P.dropna(subset=[tgt]); m = lgb.LGBMRegressor(**LGB).fit(D[feats], np.log(D[tgt].clip(lower=0.02)))
        now[tgt.replace("fut_", "pred_")] = np.exp(m.predict(now[feats]))
    for thr in (0.15, 0.25):
        edges, rate = _drop_table(P, thr)
        now[f"drop{int(thr * 100)}"] = rate[np.searchsorted(edges, now["vol_3m"].fillna(now["vol_3m"].median()).values).clip(0, len(rate) - 1)]
    now["vol21_pct"] = now["pred_vol_21"].rank(pct=True)
    await pool.execute("""CREATE TABLE IF NOT EXISTS ml_risk_forecast (as_of DATE, ticker TEXT, vol_21 DOUBLE PRECISION, vol_63 DOUBLE PRECISION,
        vol_21_pct DOUBLE PRECISION, drop15 DOUBLE PRECISION, drop25 DOUBLE PRECISION, PRIMARY KEY (as_of, ticker))""")
    rows = [(latest.date(), r.ticker, float(r.pred_vol_21), float(r.pred_vol_63), float(r.vol21_pct), float(r.drop15), float(r.drop25)) for r in now.itertuples()]
    await pool.executemany("""INSERT INTO ml_risk_forecast VALUES ($1,$2,$3,$4,$5,$6,$7) ON CONFLICT (as_of, ticker) DO UPDATE SET
        vol_21=EXCLUDED.vol_21, vol_63=EXCLUDED.vol_63, vol_21_pct=EXCLUDED.vol_21_pct, drop15=EXCLUDED.drop15, drop25=EXCLUDED.drop25""", rows)
    ev = {"as_of": str(latest.date()), "stocks": len(rows), "risk": json.load(open(f"{OUT}/eval_risk.json")),
          "drop15_table": _drop_eval(P, 0.15), "drop25_table": _drop_eval(P, 0.25),
          "returns": {h: json.load(open(f"{OUT}/eval_{h}d.json")) for h in (21, 63)}}
    json.dump(ev, open(f"{OUT}/serving.json", "w"), indent=1, default=str)
    logger.info(f"[ml2] served {len(rows)} risk forecasts as of {latest.date()}")
    return {"as_of": str(latest.date()), "stocks": len(rows), "drop25_auc": ev["drop25_table"]["auc"]}
