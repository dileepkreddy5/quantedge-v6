# Research Journal — Cohen-Frazzini directional check

**Date:** 2026-06
**Unit:** network.customer_momentum (research/002)
**Hypothesis:** A customer shock in month M predicts the supplier's return in
month M+1 (suppliers' prices lag customer news). Mechanism: investor
inattention to economic links (Cohen-Frazzini 2008).

## Setup
- 11 hand-verified supplier->customer edges (confirmed, >=10% revenue weight).
- 2 deliberately-bad seed edges (1 unconfirmed, 1 immaterial) — both correctly
  excluded by the score guards before fetch. Guards work on real data.
- Real Polygon adjusted daily closes; 3 shock/next-month window pairs.

## Result
- Supplier next-month return after + customer shock: +8.01% (n=13)
- Supplier next-month return after - customer shock: -0.88% (n=20)
- Directional spread: +8.88% — sign CONSISTENT with CF.

## What this shows / does NOT show
- SHOWS: the data pipeline + harness guards are wired correctly; a documented
  anomaly appears with the predicted SIGN on live prices.
- DOES NOT SHOW: a tradable edge or a reproduced t-stat. Caveats:
  1. Windows not independent — most edges share AAPL; ~3 real observations.
  2. Effect confounded with semiconductor sector beta; link-specific alpha
     not isolated.
  3. Windows chosen post hoc — a degree of freedom.

## Decision
- PROCEED to build on the pipeline (milestone intent satisfied: pipeline
  trustworthy on a known result).
- Do NOT promote anything. The unit stays status: research. Real validation
  needs the full universe + survivorship-free history + a pre-committed
  window schedule + sector-neutralization (Phase 5).
- Next: automate EDGAR customer-edge extraction; add sector-neutral returns.
