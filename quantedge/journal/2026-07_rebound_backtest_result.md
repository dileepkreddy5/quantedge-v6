# Research Journal — REBOUND backtest #1: FAIL

**Date:** 2026-07-12
**Unit:** 004_rebound
**Verdict:** FAIL (frozen gate, params.yaml rebound.kill_threshold)

## Real numbers (no adjustment)
Window: 6 quarterly as_of dates, 2024-06-30 → 2025-12-31.
Regimes (SPY trailing-6m): BULL, BULL, FLAT, FLAT, BULL, BULL.
Control = prefilter-passers that FAILED the gates (beaten-down base rate).
Net of committed round-trip cost (16 bps).

- 6m:  dates=6  passers=391  spread=-0.0266  t=-0.72  lift=0.88  win 0.322 vs 0.366
- 12m: dates=4  passers=233  spread=-0.1303  t=-1.78  lift=0.92  win 0.335 vs 0.366

Gate checks: lift>=2.0 FALSE; t>=3.0 FALSE; regimes>=2 TRUE; dates>=3 TRUE.

Per-stage 6m spread (net):
- FALLING     -0.0233  (n=185)
- BASING      -0.0778  (n=79)
- TURNING     +0.0119  (n=13)
- RECOVERING  +0.0218  (n=114)

## What this means (straight)
On 2024–2025 data the full composite score did NOT beat the beaten-down
control; passing the gate slightly HURT returns (lift < 1). The strategy as
specified is not validated and does NOT ship.

The signal inside the failure: buying while a stock is still FALLING/BASING
lost money; only stocks that had STOPPED falling (TURNING/RECOVERING) were
positive. Consistent with the falling-knife literature.

## Two honest limitations (not excuses)
1. NO BEAR REGIME in the window. The core thesis (De Bondt-Thaler overreaction
   → mean reversion) is strongest coming out of drawdowns. This window had no
   crash to rebound from. The plan's 5-year data floor prevents testing 2018/
   2020/2022. A fair test of the CORE thesis requires deeper history.
2. Insider component scored None historically (conservative); survivorship via
   current CIK map shared by both groups (spread is the protected stat).

## Actions
- Full composite REBOUND: NOT validated, does not ship. Recorded as FAIL.
- Pre-registered second test (below): stage-restricted, separate frozen bar,
  ONE shot, declared before seeing the result.
- Deeper-history test of the core thesis pending real data (Polygon depth).
