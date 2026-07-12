# Research Journal — REBOUND thresholds committed

**Date:** 2026-07-12
**Unit:** rebound.discounted_quality (Stage C, step 11)

All REBOUND parameters and kill-thresholds are hereby frozen in params.yaml
BEFORE any backtest has been designed or run. Mechanism: markets overreact to
multi-year bad stretches (De Bondt-Thaler 1985); within the beaten-down set,
improving fundamentals separate rebounds from value traps (Piotroski 2000);
expensed R&D hides value (Chan-Lakonishok-Sougiannis 2001); opportunistic
insider clusters and post-decline buybacks confirm (Cohen-Malloy-Pomorski
2012; Ikenberry-Lakonishok-Vermaelen 1995).

Any future change to these numbers requires a new entry here stating the
reason. The kill-threshold is never edited after seeing a result it judges.
The backtest (step 21) will be built AGAINST these bars, per-stage and
per-tier, net of the committed cost model.

## Amendment — 2026-07-12: step-18 (13F holder-count) deferred

13F cannot be fetched per-issuer: it requires ingesting every fund's
quarterly filing and inverting the dataset — a separate bulk-data project
for a signal that is 45-days lagged by construction. Deferred to the
post-launch roadmap. REBOUND's confirmation layer ships on volume,
buybacks, and insider clusters (the stronger three). No threshold changed.
