"""ML v3, phase C/D — the QuantEdge Factor Model and the ML models, walk-forward at six horizons.
Composite: each month, every family's weight is a Bayesian blend of the literature prior and that family's
own past rank-IC (exponentially weighted, 12-month half-life = factor momentum), using only results that were
already known. ML (Gu-Kelly-Xiu set): gradient-boosted trees, elastic net, neural net — retrained yearly on
the past only — plus their ensemble. Judged against an equal-weight literature composite."""
from __future__ import annotations
import json, math
import numpy as np, pandas as pd
from loguru import logger
from scipy.stats import norm
from quantedge.ml2.evaluate import nw_t
from quantedge.ml3.factors import LIT, H, OUT

EXTRA = ["vol_3m", "log_dv", "mom_6_1", "sales_yoy"]
PRIOR_MU = 0.015
VARIANTS = {"composite": dict(tau=0.0075, half_life=36, clip=True),        # v2: trust the research prior more; shrink, never flip
            "composite_fm": dict(tau=0.015, half_life=12, clip=False)}     # v1: follows recent factor performance freely


def _weights(IC, lag, tau, half_life, clip):
    V = IC.values; W = np.zeros_like(V, dtype=float)
    for k in range(len(V)):
        hist = V[:max(0, k - lag)]
        for j in range(V.shape[1]):
            x = hist[:, j]; x = x[~np.isnan(x)]
            if len(x) < 6: mu = PRIOR_MU
            else:
                w = 0.5 ** ((len(x) - 1 - np.arange(len(x))) / half_life); m = (w * x).sum() / w.sum()
                v = (w * (x - m) ** 2).sum() / w.sum(); ne = w.sum() ** 2 / (w ** 2).sum(); se2 = max(v / ne, 1e-8)
                mu = (PRIOR_MU / tau ** 2 + m / se2) / (1 / tau ** 2 + 1 / se2)
            W[k, j] = max(mu, 0.0) if clip else mu
    return pd.DataFrame(W, index=IC.index, columns=IC.columns)


def _calib(P, score, h, test):
    """Walk-forward calibration of 'chance to beat the sector': each year's table is learned from earlier test years only."""
    yc = f"y_{h}"; s = pd.Series(score); rp = s.groupby(P["date"]).rank(pct=True); dec = np.minimum((rp * 10).fillna(-1).astype(int), 9)
    beat = (P[yc] > 0).astype(float).where(P[yc].notna()); yrs = P["date"].dt.year; lagd = pd.DateOffset(months=math.ceil(h / 21))
    pp = pd.Series(np.nan, index=P.index)
    for yr in sorted(set(yrs[test])):
        past = test & (P["date"] < pd.Timestamp(f"{yr}-01-01") - lagd) & s.notna()
        if past.sum() < 5000: continue
        rate = beat[past].groupby(dec[past]).mean(); cur = test & (yrs == yr) & s.notna(); pp[cur] = dec[cur].map(rate)
    m = pp.notna() & beat.notna()
    if m.sum() < 1000: return None
    return {"brier": float(((pp[m] - beat[m]) ** 2).mean()), "brier_coinflip": float(((0.5 - beat[m]) ** 2).mean()), "n": int(m.sum()),
            "by_decile": [{"decile": d + 1, "predicted": float(pp[m & (dec == d)].mean()), "actual": float(beat[m & (dec == d)].mean())} for d in range(10)]}


def _z(P, cols):
    g = P.groupby(["date", "sector"]); n = g["ticker"].transform("size"); gd = P.groupby("date"); out = {}
    for c in cols:
        r = g[c].rank(pct=True).where(n >= 8, gd[c].rank(pct=True))
        out[c] = pd.Series(norm.ppf(r.clip(0.005, 0.995)), index=P.index).astype("float32")
    return pd.DataFrame(out).fillna(0.0)


def _date_ic(dates, x, y):
    d = pd.DataFrame({"d": dates, "x": x, "y": y}).dropna()
    return d.groupby("d").apply(lambda g: g["x"].rank().corr(g["y"].rank()) if len(g) >= 50 else np.nan)


def _metrics(P, pred, h, mask):
    k = max(1, math.ceil(h / 21))
    T = pd.DataFrame({"d": P.loc[mask, "date"].values, "p": np.asarray(pred)[mask.values], "y": P.loc[mask, f"y_{h}"].values,
                      "sz": P.loc[mask, "log_mcap"].values}).dropna(subset=["p", "y"])
    ics, ls, bysz = [], [], {"large": [], "mid": [], "small": []}
    for _, g in T.groupby("d"):
        if len(g) < 50: continue
        ics.append(g["p"].rank().corr(g["y"].rank())); q = g["p"].rank(pct=True)
        ls.append(g.loc[q >= 0.9, "y"].mean() - g.loc[q <= 0.1, "y"].mean())
        gz = g.dropna(subset=["sz"])
        if len(gz) >= 90:
            t = pd.qcut(gz["sz"].rank(method="first"), 3, labels=["small", "mid", "large"])
            for lab in bysz:
                s = gz[t == lab]
                if len(s) >= 30: bysz[lab].append(s["p"].rank().corr(s["y"].rank()))
    if len(ics) < 6: return None
    ics, ls = np.array(ics), np.array(ls); nov = ls[::k]; n_ind = len(nov)
    return {"ic": float(ics.mean()), "t": float(nw_t(ics, k - 1)) if n_ind >= 8 else None, "independent_windows": n_ind,
            "hit": float((ics > 0).mean()), "months": len(ics),
            "ls_per_period": float(ls.mean()), "ls_annual": float(ls.mean() * 252 / h),
            "ls_sharpe": float(nov.mean() / nov.std(ddof=1) * math.sqrt(12 / k)) if n_ind >= 8 and nov.std(ddof=1) > 0 else None,
            "ic_by_size": {s: (float(np.mean(v)) if v else None) for s, v in bysz.items()}}


def evaluate(source="polygon", min_train_years=2, retrain_every=1, horizons=None):
    import lightgbm as lgb, torch
    from sklearn.linear_model import ElasticNet
    from sklearn.neural_network import MLPRegressor
    from quantedge.ml3.ipca import fit_ipca
    from quantedge.ml3.cae import fit_cae, predict_cae
    torch.set_num_threads(2)
    P = pd.read_parquet(f"{OUT}/factors_{source}.parquet"); P["date"] = pd.to_datetime(P["date"]); P = P.sort_values(["date", "ticker"]).reset_index(drop=True)
    feats = sorted({f for fam in LIT.values() for f, _ in fam}); Z = _z(P, feats + [e for e in EXTRA if e in P])
    fam = pd.DataFrame({k: sum(s * Z[f] for f, s in v) / len(v) for k, v in LIT.items()})
    dates = np.array(sorted(P["date"].unique())); test_start = pd.Timestamp(dates[0]) + pd.DateOffset(years=min_train_years)
    out = {"source": source, "rows": len(P), "months": len(dates), "first": str(dates[0])[:10], "last": str(dates[-1])[:10],
           "test_from": str(test_start.date()), "retrain_every_years": retrain_every, "horizons": {}}
    names = ("lgbm", "enet", "nn3", "ranker", "ipca", "cae"); rng = np.random.default_rng(0)
    for h in (horizons or H):
        yc = f"y_{h}"; lag = math.ceil(h / 21); test = (P["date"] >= test_start) & P[yc].notna()
        if test.sum() < 1000: logger.info(f"[ml3] h={h}: not enough labelled test data"); continue
        IC = pd.DataFrame({f: _date_ic(P["date"], fam[f], P[yc]) for f in fam}).reindex(dates)
        res, scores = {}, {}
        for name, cfg in VARIANTS.items():
            W = _weights(IC, lag, **cfg); Wr = W.loc[P["date"]].values; sw = np.abs(Wr).sum(1); sw[sw == 0] = 1
            scores[name] = (fam.values * Wr).sum(1) / sw; res[name + "_weights_latest"] = {f: float(W.iloc[-1][f]) for f in fam.columns}
        scores["equal_weight"] = fam.values.mean(1)
        X = pd.concat([Z, fam], axis=1); X["composite"] = scores["composite"]; Xv = X.values.astype("float32"); Zv = Z.values.astype("float32")
        Zc = np.c_[Zv, np.ones(len(Zv), dtype="float32")]
        yz = pd.Series(norm.ppf(P.groupby("date")[yc].rank(pct=True).clip(0.005, 0.995)), index=P.index).values
        preds = {n: np.full(len(P), np.nan) for n in names}
        tyears = sorted(set(P.loc[test, "date"].dt.year))
        for yr in tyears[::retrain_every]:
            tr = ((P["date"] < pd.Timestamp(f"{yr}-01-01") - pd.DateOffset(months=lag)) & P[yc].notna()).values
            te = (test & P["date"].dt.year.between(yr, yr + retrain_every - 1)).values
            if tr.sum() < 20000 or te.sum() == 0: continue
            ytr = yz[tr]; sub = np.sort(rng.choice(np.where(tr)[0], min(int(tr.sum()), 250000), replace=False))
            preds["lgbm"][te] = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=31, min_child_samples=500, subsample=0.8,
                                                  subsample_freq=1, colsample_bytree=0.8, verbose=-1).fit(Xv[tr], ytr).predict(Xv[te])
            preds["enet"][te] = ElasticNet(alpha=0.0005, l1_ratio=0.5, max_iter=2000).fit(Xv[tr], ytr).predict(Xv[te])
            preds["nn3"][te] = np.mean([MLPRegressor(hidden_layer_sizes=(32, 16, 8), alpha=1e-3, early_stopping=True, max_iter=60, random_state=sd)
                                        .fit(Xv[sub], yz[sub]).predict(Xv[te]) for sd in range(3)], axis=0)
            lab = (P.loc[tr, yc].groupby(P.loc[tr, "date"]).rank(pct=True) * 5).clip(upper=4.999).astype(int).values
            grp = P.loc[tr].groupby("date", sort=True).size().values
            preds["ranker"][te] = lgb.LGBMRanker(objective="rank_xendcg", n_estimators=250, learning_rate=0.05, num_leaves=31, min_child_samples=500,
                                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1).fit(Xv[tr], lab, group=grp).predict(Xv[te])
            G, lam = fit_ipca(Zc[tr].astype("float64"), ytr, P["date"].values[tr], K=3); preds["ipca"][te] = Zc[te] @ G @ lam
            m, lamc = fit_cae(Zv[sub], yz[sub], P["date"].values[sub], K=3); preds["cae"][te] = predict_cae(m, lamc, Zv[te])
            logger.info(f"[ml3] h={h} retrain {yr}: {int(tr.sum()):,} training rows → tested {int(te.sum()):,}")
        for n in names: scores[n] = preds[n]
        rk = pd.DataFrame({n: pd.Series(preds[n]).groupby(P["date"]).rank(pct=True) for n in names}).mean(axis=1).values
        scores["ensemble"] = np.where(test, rk, np.nan)
        for n, sc in scores.items(): res[n] = _metrics(P, sc, h, test)
        res["calibration"] = {n: _calib(P, scores[n], h, test) for n in ("composite", "lgbm", "ensemble")}
        res["family_ic"] = {f: float(IC[f].mean()) for f in fam.columns}
        out["horizons"][str(h)] = res; json.dump(out, open(f"{OUT}/eval_ml3_{source}.json", "w"), indent=1, default=str)
        logger.info(f"[ml3] h={h}: " + ", ".join(f"{n} {r['ic']:+.3f}" + (f" (t {r['t']:+.1f})" if r.get("t") is not None else " (t n/a)")
                                                 for n, r in res.items() if isinstance(r, dict) and r and "ic" in r))
    return {h: {n: round(r["ic"], 3) for n, r in v.items() if isinstance(r, dict) and r and "ic" in r} for h, v in out["horizons"].items()}
