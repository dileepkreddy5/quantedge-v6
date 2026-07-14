# Research Journal — Recovery-to-high test (unit 005): FAIL

**Date:** 2026-07-13
**Thesis:** buy beaten-down stocks with good financials, they recover to prior
high more often/faster than beaten-down stocks with weak financials.
**Verdict:** FAIL (frozen gate, params_recovery.yaml)

## Real numbers — 4 quarterly cohorts 2024-09..2025-06, 1y tracking
Survivorship-aware (delisted = failure). Healthy = growth streak>=4 & F>=5.
Control = beaten-down but fails health gate.

- HEALTHY: n=123, recovered to prior high 9.8%, median 243d
- CONTROL: n=1477, recovered 11.7%, median 194.5d
- edge -1.9%, z -0.63 -> FAIL

By drawdown bucket (healthy vs control recovery rate):
- 35-50%: 13.6% vs 23.9%  (edge -10.2%, z -1.84)
- 50-70%:  7.1% vs 12.2%  (edge  -5.0%, z -0.97)
- 70+%:    0.0% vs  4.8%  (edge  -4.8%, z -0.86)

## Honest reading
Good financials did NOT predict better recovery — healthy names recovered
LESS, in every bucket. Not significant, but zero signal in the predicted
direction. Likely cause: 2024-2025 is a bull window with no bear bottom. The
names that fell and bounced were beaten-down speculative (weak-fundamental)
names re-rating on risk appetite; the healthy-but-down names fell for real
company-specific reasons and stayed down. The thesis needs a bear-market
bottom (indiscriminate quality selloff -> quality recovers first), absent here.

## Decision
Three tests now (REBOUND, cross-sectional, recovery-event) all FAIL to find a
tradeable/predictive edge on the 5-year Polygon window — all blocked by the
same missing bear regime. The recovery thesis does NOT ship as a validated
signal. What DOES ship: the scan as an honest SCREENER — "down 35%+ with
strong fundamentals" — labeled research, not a buy signal, with recovery data
shown but no recovery claim. No re-slicing past this.
