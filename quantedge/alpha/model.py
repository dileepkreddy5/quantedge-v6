"""Cross-sectional model + walk-forward validation (Stage F).

Gu-Kelly-Xiu (2020) found gradient-boosted trees among the top performers on
US equities. We use them here. The ONLY honest test is walk-forward:
  train on months [0..k], predict month k+1, measure rank-IC on REAL forward
  returns, roll forward. Never train on data at/after the prediction month.

Rank-IC = Spearman correlation between predicted score and realized forward
return, per prediction-month, averaged over months. Positive, stable IC =
real cross-sectional signal. The frozen gate decides ship / no-ship.

If xgboost is unavailable, falls back to ridge regression so the pipeline is
always runnable; the driver reports which backend was used.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

from quantedge.alpha.features import FEATURE_NAMES


def _spearman(pred: List[float], real: List[float]) -> Optional[float]:
    pairs = [(p, r) for p, r in zip(pred, real) if p is not None and r is not None]
    if len(pairs) < 5:
        return None
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        rk = [0.0] * len(xs)
        for r, i in enumerate(order):
            rk[i] = r
        return rk
    pr = ranks([p for p, _ in pairs])
    rr = ranks([r for _, r in pairs])
    n = len(pairs)
    mp, mr = sum(pr)/n, sum(rr)/n
    cov = sum((a-mp)*(b-mr) for a, b in zip(pr, rr))
    vp = sum((a-mp)**2 for a in pr); vr = sum((b-mr)**2 for b in rr)
    if vp == 0 or vr == 0:
        return None
    return cov / math.sqrt(vp*vr)


def _impute(rows: List[Dict], feat: List[str]) -> Dict[str, float]:
    med = {}
    for f in feat:
        vals = sorted(r[f] for r in rows if r.get(f) is not None)
        med[f] = vals[len(vals)//2] if vals else 0.0
    return med


def _matrix(rows: List[Dict], feat: List[str], med: Dict[str, float]):
    return [[(r[f] if r.get(f) is not None else med[f]) for f in feat] for r in rows]


class CrossSectionalModel:
    def __init__(self, feat: List[str] = None):
        self.feat = feat or FEATURE_NAMES
        self.backend = None
        self.med = {}

    def fit(self, rows: List[Dict], label_key: str):
        train = [r for r in rows if r.get(label_key) is not None]
        if len(train) < 100:
            raise ValueError(f"too few labeled rows: {len(train)}")
        self.med = _impute(train, self.feat)
        X = _matrix(train, self.feat, self.med)
        y = [r[label_key] for r in train]
        try:
            import xgboost as xgb
            self.backend = "xgboost"
            dtr = xgb.DMatrix(X, label=y, feature_names=self.feat)
            self.model = xgb.train(
                {"max_depth": 4, "eta": 0.05, "subsample": 0.8,
                 "colsample_bytree": 0.8, "objective": "reg:squarederror",
                 "min_child_weight": 20},
                dtr, num_boost_round=200)
        except Exception:
            self.backend = "ridge"
            self.model = _ridge_fit(X, y, l2=10.0)
        return self

    def predict(self, rows: List[Dict]) -> List[float]:
        X = _matrix(rows, self.feat, self.med)
        if self.backend == "xgboost":
            import xgboost as xgb
            return list(self.model.predict(xgb.DMatrix(X, feature_names=self.feat)))
        return _ridge_predict(self.model, X)

    def importances(self) -> Dict[str, float]:
        if self.backend == "xgboost":
            sc = self.model.get_score(importance_type="gain")
            tot = sum(sc.values()) or 1.0
            return {f: round(sc.get(f, 0.0)/tot, 4) for f in self.feat}
        w = self.model["w"]
        tot = sum(abs(x) for x in w) or 1.0
        return {f: round(abs(w[i])/tot, 4) for i, f in enumerate(self.feat)}


def _ridge_fit(X, y, l2=10.0):
    n, p = len(X), len(X[0])
    means = [sum(row[j] for row in X)/n for j in range(p)]
    stds = [(sum((row[j]-means[j])**2 for row in X)/n) ** 0.5 or 1.0 for j in range(p)]
    Xs = [[(row[j]-means[j])/stds[j] for j in range(p)] for row in X]
    ymean = sum(y)/n
    yc = [v - ymean for v in y]
    XtX = [[sum(Xs[k][i]*Xs[k][j] for k in range(n)) + (l2 if i==j else 0)
            for j in range(p)] for i in range(p)]
    Xty = [sum(Xs[k][i]*yc[k] for k in range(n)) for i in range(p)]
    w = _solve(XtX, Xty)
    return {"w": w, "means": means, "stds": stds, "ymean": ymean, "p": p}


def _ridge_predict(m, X):
    out = []
    for row in X:
        z = sum(m["w"][j]*((row[j]-m["means"][j])/m["stds"][j]) for j in range(m["p"]))
        out.append(m["ymean"] + z)
    return out


def _solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-12:
            M[col][col] = 1e-12
        for r in range(n):
            if r != col:
                f = M[r][col]/M[col][col]
                for c in range(col, n+1):
                    M[r][c] -= f*M[col][c]
    return [M[i][n]/M[i][i] for i in range(n)]


def walk_forward(panel_by_date: Dict[str, List[Dict]], label_key: str,
                 min_train_months: int = 12) -> Dict:
    raw_key = "_" + label_key.replace("_rank", "_raw")
    dates = sorted(panel_by_date)
    ics: List[Tuple[str, float]] = []
    agg_importance: Dict[str, float] = {f: 0.0 for f in FEATURE_NAMES}
    n_fits = 0
    backend = None
    for i in range(min_train_months, len(dates)):
        test_d = dates[i]
        train_rows = [r for d in dates[:i] for r in panel_by_date[d]]
        test_rows = panel_by_date[test_d]
        try:
            m = CrossSectionalModel().fit(train_rows, label_key)
        except ValueError:
            continue
        preds = m.predict(test_rows)
        real = [r.get(raw_key) for r in test_rows]
        ic = _spearman(preds, real)
        if ic is not None:
            ics.append((test_d, round(ic, 4)))
        for f, v in m.importances().items():
            agg_importance[f] = agg_importance.get(f, 0.0) + v
        n_fits += 1
        backend = m.backend
    ic_vals = [v for _, v in ics]
    mean_ic = sum(ic_vals)/len(ic_vals) if ic_vals else None
    t = None
    if len(ic_vals) >= 3:
        mu = mean_ic
        sd = (sum((x-mu)**2 for x in ic_vals)/(len(ic_vals)-1)) ** 0.5
        t = mu/(sd/len(ic_vals)**0.5) if sd > 0 else None
    imp = {f: round(v/n_fits, 4) for f, v in agg_importance.items()} if n_fits else {}
    return {
        "backend": backend,
        "n_test_months": len(ic_vals),
        "mean_rank_ic": round(mean_ic, 4) if mean_ic is not None else None,
        "ic_t_stat": round(t, 2) if t is not None else None,
        "positive_month_frac": round(sum(1 for x in ic_vals if x > 0)/len(ic_vals), 3) if ic_vals else None,
        "monthly_ic": ics,
        "feature_importance": dict(sorted(imp.items(), key=lambda kv: -kv[1])),
    }
