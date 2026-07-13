# Research Journal — Cross-sectional alpha (alpha_xs): PASS

**Date:** 2026-07-13
**Model:** alpha_xs (Stage F, Gu-Kelly-Xiu style)
**Verdict:** PASS (frozen gate, params_alpha.yaml alpha_xs.kill_threshold)

## Real numbers — full universe walk-forward
Store: 13.8M bars, 2021-07-13 → 2026-07-10 (Polygon 5y plan floor).
43 monthly as_of dates; 31 walk-forward test months (12-month min train).
Ridge backend (xgboost not in container; ridge is a valid linear baseline).
Labels = cross-sectional rank of REAL forward 3-month returns. No lookahead
(walk-forward proven in _test_model). Liquidity floor $2M ADV.

- mean rank-IC:        0.0867
- IC t-stat:           3.47   (>= 3.0 required)
- positive months:     0.71   (>= 0.55 required)
- test months:         31     (>= 6 required)
- ALL gate checks TRUE -> PASS

Top features (avg gain importance): downside_dev 0.16, r_3m 0.12,
log_adv 0.10, vol_21d 0.08, vol_63d 0.07, dd_3y 0.06, up_day_share 0.06,
r_1m 0.05. The model independently recovered the low-volatility and
momentum anomalies — consistent with Gu-Kelly-Xiu (2020). Sensible loadings,
not noise.

## Honest scope
- IC 0.087 is a REAL but MODEST edge. Rank correlation with fwd returns ~9%.
  Valuable in aggregate across many names; not a per-stock predictor.
- An IC is NOT yet money. Net-of-cost portfolio performance is untested. The
  next unit (portfolio construction) measures whether this IC becomes return
  after costs — separate frozen gate, separate verdict.
- Ridge linear baseline. xgboost may raise IC further; to be tested in-container.
- Price + liquidity features only so far; fundamentals not yet wired (should
  add, not subtract, signal).

## Decision
alpha_xs is a VALIDATED cross-sectional signal on available data. It advances
to portfolio construction + net-of-cost backtest before any live/shipped use.
First unit in the project to pass an honest pre-committed gate.
