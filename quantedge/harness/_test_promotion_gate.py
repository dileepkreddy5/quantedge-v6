"""Proof the gate refuses to promote anything that misses the frozen bar.
Run: PYTHONPATH=. python this_file."""
import yaml
from quantedge.harness.promotion_gate import check

params = yaml.safe_load(open("quantedge/params.yaml"))

# A unit that clears EVERY bar (lift>=2, t>=3, calib<=0.20, regimes>=2, mechanism, costs).
clean = dict(mechanism="Gross profitability predicts long-run returns (Novy-Marx).",
             lift=2.6, t_stat=3.4, calibration_max_dev=0.12,
             n_regimes_passed=3, cost_survived=True)
d = check(clean, params)
print("clean unit ->", "PASS" if d else "FAIL", "| reasons:", d.reasons)
assert d.passed, "a fully-qualified unit must pass"

# Strong numbers but NO mechanism — the 'AI stock score' failure mode. Must reject.
no_mech = dict(clean, mechanism="")
d = check(no_mech, params)
print("no mechanism ->", "PASS" if d else "FAIL", "|", d.reasons[0])
assert not d.passed and "mechanism" in d.reasons[0].lower()

# Beautiful in-sample numbers but never costed. Must reject (§13).
no_costs = dict(clean, cost_survived=False)
assert not check(no_costs, params).passed
print("gross-only (no cost survival) -> FAIL                     OK")

# Lift 1.9 — just under the 2.0 bar. Must reject, NOT round up.
low_lift = dict(clean, lift=1.9)
assert not check(low_lift, params).passed
print("lift 1.9 (< 2.0) -> FAIL (not rounded up)                 OK")

# t-stat 2.8 — would pass a naive t>2 test, but our bar is 3.0. Must reject.
low_t = dict(clean, t_stat=2.8)
assert not check(low_t, params).passed
print("t-stat 2.8 (< 3.0) -> FAIL                                OK")

# Held in only 1 regime — rests on one decade. Must reject.
one_regime = dict(clean, n_regimes_passed=1)
assert not check(one_regime, params).passed
print("1 regime (< 2) -> FAIL                                    OK")

# A unit failing MANY bars names ALL of them, not just the first.
junk = dict(mechanism="", lift=1.1, t_stat=0.5, calibration_max_dev=0.5,
            n_regimes_passed=0, cost_survived=False)
d = check(junk, params)
print(f"junk unit names {len(d.reasons)} distinct failures (expect 6)")
assert len(d.reasons) == 6

print("\nPASS — the gate cannot be talked into promoting an unqualified unit.")
