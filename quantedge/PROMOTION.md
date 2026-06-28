# PROMOTION.md — The Gate

> The promotion gate is the ONLY bridge from research to production.
> (Build Manual v1.0, §9, §15, Table 20.)

A signal may contribute to a production rating ONLY after it has, in order:

1. A stated economic mechanism — why should it work? (§3). A 10-K footnote
   and a peer-reviewed paper are judged the same way; fame is not a reason.
2. Survived out-of-sample on the point-in-time harness — scored only on data
   after its training boundary, with zero look-ahead (§13).
3. Cleared the pre-committed kill-threshold — every bar in
   params.yaml: kill_threshold, fixed before results were seen (§15, §22.2).
4. Survived realistic costs — net-of-cost lift, never gross alone (§13).

A unit that misses any bar CANNOT be promoted, regardless of how compelling
its story is. Math COMBINES gate-passed signals; it never invents a signal
from raw financials (Table 21).

## The mechanical gate

status: research -> validated -> production. The transition to production is
allowed ONLY when harness/promotion_gate.py returns PASS against the unit's
validation block and params.yaml. The live app (backend/, frontend/) may
import ONLY status: production units.

## What touching the live site requires

Per the Locked Build Plan (Table 32), the first user-visible tab is Phase 6,
and only after a unit has passed this gate. Until then:
- Research/harness code is NEVER imported by backend/ or frontend/.
- main (which the VPS deploys via git pull origin main) is not changed by
  research work. Promotion is a deliberate, logged merge — not a default.

## The reproduction milestone (Table 29)

Before trusting the harness on novel ideas, reproduce a known published
result — the Cohen-Frazzini customer-momentum effect. If a prize-winning
anomaly cannot be even partially reproduced with our data and harness, the
pipeline is broken.
