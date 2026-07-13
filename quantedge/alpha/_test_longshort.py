import random
from quantedge.alpha.longshort import backtest_long_short
random.seed(5)

def make_month(as_of, signal, n=200):
    preds, realized = [], {}
    for i in range(n):
        s = random.gauss(0, 1)
        preds.append((f"S{i}", s)); realized[f"S{i}"] = signal * s * 0.02 + random.gauss(0, 0.08)
    return {"as_of": as_of, "preds": preds, "realized": realized}

def test_working_signal_positive_spread():
    months = [make_month(f"20{y}-{m:02d}", 1.5) for y in (23,24) for m in range(1,13)]
    r = backtest_long_short(months, 0.1, 0.1, 0.0016)
    assert r["ok"] and r["ann_spread"] > 0 and r["t_stat"] > 2.5, r
    assert r["mean_period_spread"] < r["gross_mean_period_spread"]
    print(f"  working: ann spread {r['ann_spread']}, Sharpe {r['sharpe']}, t {r['t_stat']}")

def test_no_signal_flat():
    months = [make_month(f"20{y}-{m:02d}", 0.0) for y in (23,24) for m in range(1,13)]
    r = backtest_long_short(months, 0.1, 0.1, 0.0016)
    assert r["ok"] and abs(r["t_stat"]) < 2.5, r["t_stat"]
    print(f"  no signal: ann spread {r['ann_spread']}, t {r['t_stat']}")

if __name__ == "__main__":
    for t in (test_working_signal_positive_spread, test_no_signal_flat):
        print(f"── {t.__name__}"); t()
    print("ALL LONGSHORT TESTS PASSED")
