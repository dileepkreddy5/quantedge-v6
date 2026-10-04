"""ML v2, stage 3 — walk-forward evaluation. Train strictly on the past, test on the next 6 months,
with a gap equal to the horizon (no overlap leakage). Contenders: best single signal, a simple sign-learned
composite, and LightGBM. Newey-West t-stats correct for overlapping multi-month returns."""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from loguru import logger

OUT = "/app/models/panel_v2"
DROP = {"date", "ticker", "sector", "fwd_21", "fwd_63", "y_21", "y_63"}


def nw_t(x, lag):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    if n < 6: return np.nan
    m = x.mean(); e = x - m; s = (e @ e) / n
    for L in range(1, lag + 1): s += 2 * (1 - L / (lag + 1)) * (e[L:] @ e[:-L]) / n
    return m / np.sqrt(s / n) if s > 0 else np.nan


def ranks(df, cols):
    R = df[cols].groupby(df["date"]).rank(pct=True)
    return R.fillna(0.5)


def date_ics(dates, pred, y):
    d = pd.DataFrame({"d": dates, "p": pred, "y": y}).dropna()
    return d.groupby("d").apply(lambda g: g["p"].rank().corr(g["y"].rank()) if len(g) >= 100 else np.nan).dropna()


def spread(dates, pred, y):
    d = pd.DataFrame({"d": dates, "p": pred, "y": y}).dropna(); out = []
    for _, g in d.groupby("d"):
        if len(g) < 100: continue
        q = g["p"].rank(pct=True); out.append(g.loc[q >= 0.9, "y"].mean() - g.loc[q <= 0.1, "y"].mean())
    return float(np.mean(out)) if out else np.nan


def evaluate(h=63):
    import lightgbm as lgb
    P = pd.read_parquet(f"{OUT}/panel.parquet"); P["date"] = pd.to_datetime(P["date"])
    feats = [c for c in P.columns if c not in DROP]
    ycol = f"y_{h}"; P = P.dropna(subset=[ycol])
    P["yr"] = P.groupby("date")[ycol].rank(pct=True)
    X = ranks(P, feats)
    gap = pd.DateOffset(months=3 if h == 63 else 1); lag = 2 if h == 63 else 0
    starts = [pd.Timestamp(s) for s in ("2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01")]
    res = {"single": [], "composite": [], "lightgbm": []}; per_fold = []
    for s in starts:
        e = s + pd.DateOffset(months=6)
        tr = P["date"] < (s - gap); te = (P["date"] >= s) & (P["date"] < e)
        if te.sum() == 0 or tr.sum() == 0: continue
        # direction and strength of each signal, learned on training data only
        tr_ic = {c: date_ics(P.loc[tr, "date"], X.loc[tr, c], P.loc[tr, ycol]) for c in feats}
        tr_t = {c: nw_t(v.values, lag) for c, v in tr_ic.items()}
        best = max(feats, key=lambda c: abs(tr_t[c]) if not np.isnan(tr_t[c]) else -1)
        top = [c for c in sorted(feats, key=lambda c: -abs(tr_t[c]) if not np.isnan(tr_t[c]) else 0)[:8]]
        sgn = {c: np.sign(tr_ic[c].mean()) for c in feats}
        p_single = X.loc[te, best] * sgn[best]
        p_comp = sum(X.loc[te, c] * sgn[c] for c in top) / len(top)
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        m.fit(X.loc[tr, feats], P.loc[tr, "yr"]); p_lgb = m.predict(X.loc[te, feats])
        for name, p in (("single", p_single), ("composite", p_comp), ("lightgbm", p_lgb)):
            ic = date_ics(P.loc[te, "date"], p, P.loc[te, ycol]); res[name].append((ic, P.loc[te, "date"], p, P.loc[te, ycol]))
        per_fold.append({"test": f"{s:%Y-%m}", "train_rows": int(tr.sum()), "best_single": best, "composite": top,
                         "ic": {n: float(res[n][-1][0].mean()) for n in res}})
        logger.info(f"[ml2] fold {s:%Y-%m}: " + ", ".join(f"{n} {res[n][-1][0].mean():+.3f}" for n in res))
    summ = {"horizon_days": h, "folds": per_fold, "results": {}}
    for n, parts in res.items():
        ics = pd.concat([p[0] for p in parts]); d = pd.concat([p[1] for p in parts]); pr = np.concatenate([np.asarray(p[2]) for p in parts]); y = pd.concat([p[3] for p in parts])
        summ["results"][n] = {"ic": float(ics.mean()), "t_nw": float(nw_t(ics.values, lag)), "hit": float((ics > 0).mean()), "months": int(len(ics)),
                              "top_minus_bottom_decile": spread(d.values, pr, y.values)}
    json.dump(summ, open(f"{OUT}/eval_{h}d.json", "w"), indent=1, default=str)
    return summ
