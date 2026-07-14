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

## Portfolio backtest (Stage H): FAIL — long-only vs SPY

Top-decile equal-weight long book, monthly rebalance, net of committed costs,
vs SPY over identical 3m windows. Real walk-forward predictions.

- gross period excess: -0.0156  (book LOST to SPY before costs)
- net period excess:   -0.0161
- ann excess, Sharpe, t-stat, drawdown: all FAIL
- VERDICT: FAIL

Reading (honest): the IC (0.087) proves cross-sectional ranking information,
but an equal-weight top-decile book vs a CAP-WEIGHTED mega-cap index (SPY) is
a benchmark mismatch on a 2022-2025 window dominated by mega-cap tech. The
book lost pre-cost, so costs are not the cause. This is a result about THIS
construction + benchmark, not proof the signal is worthless.

## Pre-registered next test (decided before running): long-short spread
The textbook isolation of a cross-sectional signal: top-decile MINUS
bottom-decile return, which removes the benchmark entirely and measures
whether liked stocks beat disliked stocks. Frozen gate below. ONE run.
If flat -> signal is real but not monetizable long-only; stop.

## Long-short spread (pre-registered): FAIL — signal not tradeable

Top-decile minus bottom-decile, 31 walk-forward periods, net of 2x costs.

- ann spread:      0.0002   (dead flat)
- t-stat:          0.0
- positive frac:   0.516    (coin flip)
- max drawdown:    -0.78    (wildly unstable)
- gross period spread 0.0014 -> net 0.00005  (costs not the cause; gross ~0)
- VERDICT: FAIL

## Final honest verdict on alpha_xs (all three tests done)
- IC (whole cross-section):     PASS (0.087, t 3.47)
- Long-only vs SPY:             FAIL (benchmark mismatch + thin signal)
- Long-short decile spread:     FAIL (no tail separation, flat, unstable)

Reconciliation: the IC is real but lives in the MIDDLE of the ranking, not
the tradable tails. The extreme deciles do not separate. A signal can carry
average rank information yet be dominated by idiosyncratic noise at the tails
where a book is actually formed. -78% spread drawdown confirms instability.

CONCLUSION: alpha_xs is a statistically real but NOT TRADEABLE signal on
available data (2021-2026, Polygon 5y floor). It does NOT ship as a live
strategy. Three pre-committed tests, three honest verdicts, no re-slicing.
The feature-importance finding (low-vol + momentum dominate) stands as a real,
honest observation about this market and is the durable output.

STOP condition reached: no further constructions tested on this window
(that would be fishing). Real paths forward, both requiring inputs I do not
have from here: (1) deeper price history to test across real bear regimes;
(2) richer features (fundamentals wired into the panel) tested under the SAME
frozen discipline. Neither is a tweak; both are new, separately-gated work.

## Fundamentals-enhanced (final on this model): no improvement
IC 0.0851 (t 3.5) vs 0.0867 price-only — statistically identical. Fundamental
features rank LOW (growth_streak 0.046, bottom of top-8; others below). Top
drivers unchanged: downside_dev, r_3m, vol. Long-short still flat (0.0023, t
0.02). Long-only worse (-0.072, t -2.65).

CONCLUSION: value/quality/growth add no cross-sectional signal on 2021-2026.
The cross-sectional model is fully explored: real IC, untradeable tails,
fundamentals redundant. CLOSED on available data. Durable finding stands:
low-vol + momentum dominate this market's cross-section.
