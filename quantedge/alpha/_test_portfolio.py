"""Tests for portfolio construction + net-of-cost backtest. PYTHONPATH=. python."""
import random
from quantedge.alpha.portfolio import _decile_longs, _book_return, backtest_long_book
random.seed(3)


def test_decile_selects_top():
    preds = [(f"T{i}", i/100) for i in range(100)]
    longs = _decile_longs(preds, 0.1)
    assert len(longs) == 10 and "T99" in longs and "T0" not in longs
    print("  top decile: 10 names, highest scores selected")


def test_book_return_skips_missing():
    r = _book_return(["A","B","C"], {"A":0.10,"B":0.20,"C":None})
    assert abs(r - 0.15) < 1e-9
    print("  book return skips missing, no fabrication:", r)


def make_month(as_of, signal, n=200):
    preds, realized = [], {}
    for i in range(n):
        score = random.gauss(0, 1)
        ret = signal * score * 0.02 + random.gauss(0, 0.08)
        preds.append((f"S{i}", score)); realized[f"S{i}"] = ret
    spy = random.gauss(0.02, 0.04)
    for t in realized: realized[t] += spy
    return {"as_of": as_of, "preds": preds, "realized": realized, "spy": spy}


def test_working_signal_beats_spy_net():
    months = [make_month(f"2023-{m:02d}", 1.5) for m in range(1,13)]
    months += [make_month(f"2024-{m:02d}", 1.5) for m in range(1,13)]
    r = backtest_long_book(months, 0.1, 0.0016)
    assert r["ok"] and r["ann_excess_return"] > 0 and r["sharpe_excess"] > 0.5, r
    assert r["mean_period_excess"] < r["gross_mean_period_excess"], r
    print(f"  working: ann excess {r['ann_excess_return']}, Sharpe {r['sharpe_excess']}, "
          f"turnover {r['avg_turnover']}, hit {r['hit_rate']}")


def test_no_signal_no_excess():
    months = [make_month(f"2023-{m:02d}", 0.0) for m in range(1,13)]
    months += [make_month(f"2024-{m:02d}", 0.0) for m in range(1,13)]
    r = backtest_long_book(months, 0.1, 0.0016)
    assert r["ok"] and abs(r["t_stat"]) < 2.5, r["t_stat"]
    print(f"  no signal: ann excess {r['ann_excess_return']}, t {r['t_stat']}")


def test_costs_reduce_return():
    months = [make_month(f"2023-{m:02d}", 1.5) for m in range(1,13)]
    free = backtest_long_book(months, 0.1, 0.0)
    costly = backtest_long_book(months, 0.1, 0.01)
    assert costly["ann_excess_return"] < free["ann_excess_return"]
    print(f"  costs bite: free {free['ann_excess_return']} -> costly {costly['ann_excess_return']}")


if __name__ == "__main__":
    for t in (test_decile_selects_top, test_book_return_skips_missing,
              test_working_signal_beats_spy_net, test_no_signal_no_excess,
              test_costs_reduce_return):
        print(f"── {t.__name__}"); t()
    print("ALL PORTFOLIO TESTS PASSED")
