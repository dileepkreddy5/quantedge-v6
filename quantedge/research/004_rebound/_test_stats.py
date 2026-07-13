"""Tests for backtest statistics. PYTHONPATH=. python this_file."""
import random
# package name starts with a digit -> load stats by path
import importlib.util, os
spec = importlib.util.spec_from_file_location(
    "rb_stats", os.path.join(os.path.dirname(__file__), "stats.py"))
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)


def test_excess_net():
    assert abs(st.excess_net(0.20, 0.10, 0.01) - 0.09) < 1e-9
    assert st.excess_net(None, 0.1, 0.01) is None
    print("  net excess math + None propagation")


def test_t_stat_honest_about_small_n():
    assert st.t_stat_over_dates([0.05, 0.04]) is None       # <3 dates: no claim
    t = st.t_stat_over_dates([0.05, 0.06, 0.04, 0.05, 0.06])
    assert t is not None and t > 10                          # tight, consistent
    t2 = st.t_stat_over_dates([0.05, -0.04, 0.06, -0.05, 0.02])
    assert t2 is not None and abs(t2) < 2                    # noisy -> weak t
    print("  <3 dates refuses a t-stat; consistency drives t, not magnitude")


def test_summarize_detects_real_edge():
    random.seed(1)
    events = []
    for d in ("2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31", "2025-06-30"):
        for _ in range(40):   # passers: +8% mean excess pre-cost
            events.append({"as_of": d, "passed": True,
                           "ret_6m": 0.08 + random.gauss(0, 0.10) + 0.03,
                           "spy_6m": 0.03})
        for _ in range(120):  # control: 0% mean excess
            events.append({"as_of": d, "passed": False,
                           "ret_6m": random.gauss(0, 0.10) + 0.03,
                           "spy_6m": 0.03})
    s = st.summarize(events, "6m", roundtrip_cost=0.01)
    assert s["n_dates"] == 5 and s["n_pass_events"] == 200
    assert s["mean_spread_per_date"] > 0.05, s
    assert s["t_stat"] and s["t_stat"] > 3, s
    assert s["lift"] and s["lift"] > 1.2, s
    print(f"  edge detected: spread {s['mean_spread_per_date']}, "
          f"t {s['t_stat']}, lift {s['lift']}")


def test_summarize_finds_nothing_in_noise():
    random.seed(2)
    events = []
    for d in ("2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31", "2025-06-30"):
        for p in (True,) * 40 + (False,) * 120:
            events.append({"as_of": d, "passed": p,
                           "ret_6m": random.gauss(0.03, 0.10), "spy_6m": 0.03})
    s = st.summarize(events, "6m", roundtrip_cost=0.01)
    assert s["t_stat"] is None or abs(s["t_stat"]) < 3, s
    print(f"  pure noise: t {s['t_stat']} (no false edge)")


if __name__ == "__main__":
    for t in (test_excess_net, test_t_stat_honest_about_small_n,
              test_summarize_detects_real_edge, test_summarize_finds_nothing_in_noise):
        print(f"── {t.__name__}")
        t()
    print("ALL BACKTEST-STATS TESTS PASSED")
