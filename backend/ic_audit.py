"""Read-only IC audit: recompute OLD vs NEW figures from the saved panel + models.
No training, no writes. Proves the sample-mismatch diagnosis before patching."""
import json, glob
import numpy as np, pandas as pd, joblib
from pathlib import Path
from ml.training.train_panel_models import (
    cross_sectional_rank_ic, per_date_ics, hac_t_stat)

MD = Path("/app/models/panel")
rep = json.load(open(MD / "training_report.json"))
feat = json.load(open(MD / "feature_names.json"))
cols = feat["features"] if isinstance(feat, dict) else feat
split_date = np.datetime64(pd.Timestamp(rep["split_date"]))

p = sorted(glob.glob("/app/models/panels/panel_*.parquet"))[-1]
df = pd.read_parquet(p)
print(f"panel={p}  rows={len(df)}  split={rep['split_date']}\n")

def thin(dates_used, min_gap):
    """Indices of a non-overlapping subsample, taken FROM the scored series."""
    keep, last = [], None
    for i, d in enumerate(dates_used):
        if last is None or (pd.Timestamp(d) - pd.Timestamp(last)).days >= min_gap:
            keep.append(i); last = d
    return keep

for h in (5, 10, 21, 63, 126, 252):
    lab = f"label_{h}d"
    if lab not in df.columns: continue
    sub = df.dropna(subset=[lab]).copy()
    X = sub[cols].fillna(0.5).values.astype(np.float64)
    y = sub[lab].values.astype(np.float64)
    dts = sub["date"].values
    vm = dts >= split_date
    Xv, yv, dv = X[vm], y[vm], dts[vm]

    xgb = joblib.load(MD / f"xgb_{h}d.joblib"); lgb = joblib.load(MD / f"lgb_{h}d.joblib")
    xp, lp = xgb.predict(Xv), lgb.predict(Xv)

    vd = np.sort(np.unique(dv)); mid = vd[len(vd)//2] if len(vd) >= 4 else None
    if mid is not None:
        fitm, scm = dv < mid, dv >= mid
        wx, _, _ = cross_sectional_rank_ic(dv[fitm], xp[fitm], yv[fitm], horizon_days=h)
        wl, _, _ = cross_sectional_rank_ic(dv[fitm], lp[fitm], yv[fitm], horizon_days=h)
    else:
        fitm = np.zeros(len(dv), bool); scm = np.ones(len(dv), bool); wx = wl = 0.0
    wx, wl = max(wx, 0.0), max(wl, 0.0)
    ens = 0.5*xp + 0.5*lp if wx+wl <= 0 else (wx*xp + wl*lp)/(wx+wl)

    # ---- OLD (as shipped) ----
    o_x, _, _ = cross_sectional_rank_ic(dv, xp, yv, horizon_days=h)      # FULL val
    o_l, _, _ = cross_sectional_rank_ic(dv, lp, yv, horizon_days=h)      # FULL val
    o_e, _, o_nd = cross_sectional_rank_ic(dv[scm], ens[scm], yv[scm], horizon_days=h)  # HALF
    o_ics, _ = per_date_ics(dv[scm], ens[scm], yv[scm])                  # UNTHINNED
    sd = np.sort(np.unique(dv[scm]))
    step = max(1, int(np.median([(pd.Timestamp(sd[i+1])-pd.Timestamp(sd[i])).days
                                 for i in range(len(sd)-1)]))) if len(sd) > 1 else 5
    o_t, _, o_ok = hac_t_stat(o_ics, int(np.ceil(h/step)))

    # ---- NEW (one series, scoring half, thinned by index) ----
    n_x, _ = per_date_ics(dv[scm], xp[scm], yv[scm])
    n_l, _ = per_date_ics(dv[scm], lp[scm], yv[scm])
    n_e, used = per_date_ics(dv[scm], ens[scm], yv[scm])
    idx = thin(used, max(h, step))
    n_t, _, n_ok = hac_t_stat(n_e, int(np.ceil(h/step)))
    ind = float(np.mean([n_e[i] for i in idx])) if idx else 0.0

    print(f"--- {h}d ({rep['horizons'][str(h)]['horizon_label']})  step={step}d ---")
    print(f"  OLD  xgb {o_x:+.4f}(full val)  lgb {o_l:+.4f}(full val)  ens {o_e:+.4f}(half)"
          f"  t {o_t:+.2f}  n_indep {o_nd}  n_all {len(o_ics)}  stable={o_ok}")
    print(f"  NEW  xgb {np.mean(n_x):+.4f}  lgb {np.mean(n_l):+.4f}  ens_all {np.mean(n_e):+.4f}"
          f"  t {n_t:+.2f}  ens_indep {ind:+.4f}  n_indep {len(idx)}  n_all {len(n_e)}  stable={n_ok}")
    print(f"  sign agree (ens_all vs t): {np.sign(np.mean(n_e))==np.sign(n_t)}"
          f" | OLD agreed: {np.sign(o_e)==np.sign(o_t)}\n")
