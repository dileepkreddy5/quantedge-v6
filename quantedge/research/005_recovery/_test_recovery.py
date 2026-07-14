from datetime import date, timedelta
import importlib.util, os
spec = importlib.util.spec_from_file_location("rec", os.path.join(os.path.dirname(__file__), "recovery.py"))
rec = importlib.util.module_from_spec(spec); spec.loader.exec_module(rec)

PARAMS = {"recovery": {"min_hit_rate_edge": 0.10, "min_z_stat": 2.0, "min_events": 20}}


def bars_from(vals, start=date(2022,1,3)):
    out, d = [], start
    for v in vals:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        out.append((d, float(v))); d += timedelta(days=7)
    return out


def test_prior_high_and_drawdown():
    b = bars_from([100, 120, 80, 60]); as_of = b[-1][0]
    assert rec.prior_high(b, as_of, 3*365) == 120
    assert abs(rec.drawdown_at(b, as_of, 3*365) - 0.5) < 1e-9
    print("  prior high 120, drawdown 50%")


def test_recovery_outcome_hit_and_miss():
    b = bars_from([60, 70, 90, 110, 125])
    r = rec.recovery_outcome(b, b[0][0], 120, 400)
    assert r["matured"] and r["recovered"] and r["days"] > 0
    b2 = bars_from([60, 62, 65, 63, 64])
    r2 = rec.recovery_outcome(b2, b2[0][0], 120, 400)
    assert r2["matured"] and not r2["recovered"]
    print(f"  recovery in {r['days']}d; non-recovery flagged")


def test_no_lookahead():
    b = bars_from([130, 60, 70, 125])
    r = rec.recovery_outcome(b, b[1][0], 120, 400)
    assert r["recovered"] and r["days"] > 0
    print("  past bars excluded")


def test_evaluate_edge_and_gate():
    events = []
    for i in range(60):
        events.append({"healthy": True, "outcome": {"matured": True, "recovered": i < 42, "days": 90 if i < 42 else None}})
    for i in range(60):
        events.append({"healthy": False, "outcome": {"matured": True, "recovered": i < 18, "days": 120 if i < 18 else None}})
    r = rec.evaluate(events, PARAMS)
    assert r["verdict"] == "PASS" and abs(r["hit_rate_edge"] - 0.40) < 1e-9 and r["z_stat"] > 2.0
    print(f"  strong edge: {r['healthy']['hit_rate']} vs {r['control']['hit_rate']}, z {r['z_stat']} -> PASS")


def test_evaluate_no_edge_fails():
    events = []
    for i in range(60):
        events.append({"healthy": True, "outcome": {"matured": True, "recovered": i < 30, "days": 90 if i < 30 else None}})
    for i in range(60):
        events.append({"healthy": False, "outcome": {"matured": True, "recovered": i < 29, "days": 100 if i < 29 else None}})
    r = rec.evaluate(events, PARAMS)
    assert r["verdict"] == "FAIL"
    print(f"  no edge: {r['healthy']['hit_rate']} vs {r['control']['hit_rate']}, z {r['z_stat']} -> FAIL")


if __name__ == "__main__":
    for t in (test_prior_high_and_drawdown, test_recovery_outcome_hit_and_miss,
              test_no_lookahead, test_evaluate_edge_and_gate, test_evaluate_no_edge_fails):
        print(f"── {t.__name__}"); t()
    print("ALL RECOVERY TESTS PASSED")
