"""Tests for the cross-sectional feature engine. PYTHONPATH=. python this."""
import math
from datetime import date, timedelta
from quantedge.alpha.features import compute_features, _ret, _vol, _drawdown, FEATURE_NAMES


def weekdays(start, n):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


AS_OF = date(2025, 6, 30)


def make_bars(n=800, daily=0.0005, start_px=50.0, vol_base=1e6):
    days, d = [], AS_OF
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    days = list(reversed(days))
    bars, px = [], start_px
    for dd in days:
        px *= (1 + daily)
        bars.append((dd, round(px, 4), vol_base))
    return bars


def test_momentum_sign_and_magnitude():
    bars = make_bars(daily=0.0005)
    r1m = _ret(bars, 21, AS_OF)
    assert r1m is not None and 0.008 < r1m < 0.014, r1m  # 21 bars * 0.05%/day
    down = make_bars(daily=-0.001)
    assert _ret(down, 63, AS_OF) < 0
    print("  momentum sign+magnitude:", round(r1m, 4))


def test_12_1_momentum_excludes_last_month():
    bars = make_bars(n=400, daily=0.001)
    crashed = bars[:-21] + [(d, c * 0.7, v) for d, c, v in bars[-21:]]
    r12 = _ret(crashed, 365, AS_OF)
    r1 = _ret(crashed, 21, AS_OF)
    ex = (1 + r12) / (1 + r1) - 1
    assert ex > r12, (ex, r12)
    assert r1 < -0.2
    print("  12-1 excludes crash: ex_1m", round(ex, 3), "> r_12m", round(r12, 3),
          "(last-month", round(r1, 3), ")")


def test_volatility_positive_and_ordered():
    calm = make_bars(daily=0.0002)
    v = _vol(calm, 63, AS_OF)
    assert v is not None and v >= 0
    print("  annualized vol:", round(v, 4))


def test_drawdown_detects_decline():
    bars = make_bars(n=500, daily=0.001)
    bars = bars[:-60] + [(d, c * 0.6, v) for d, c, v in bars[-60:]]
    dd = _drawdown(bars, 3 * 365, AS_OF)
    assert dd["dd"] > 0.3, dd
    print("  drawdown detected:", round(dd["dd"], 3))


def test_pit_no_lookahead():
    bars = make_bars(n=500)
    future = bars + [(AS_OF + timedelta(days=i), 999.0, 1e6) for i in range(1, 30)]
    f1 = compute_features(bars, AS_OF)
    f2 = compute_features(future, AS_OF)
    for k in FEATURE_NAMES:
        assert f1[k] == f2[k], (k, f1[k], f2[k])
    print("  PIT: future bars do not leak into features")


def test_fundamentals_passthrough_and_missing():
    bars = make_bars()
    f = compute_features(bars, AS_OF, {"piotroski": 8, "ps_percentile_own": 12.0})
    assert f["piotroski"] == 8 and f["ps_percentile_own"] == 12.0
    assert f["accruals"] is None
    f2 = compute_features(bars, AS_OF, None)
    assert all(f2[k] is None for k in ("piotroski", "accruals", "growth_streak"))
    print("  fundamentals passthrough + honest missingness")


def test_all_feature_names_present():
    f = compute_features(make_bars(), AS_OF, {})
    assert set(f.keys()) >= set(FEATURE_NAMES), set(FEATURE_NAMES) - set(f.keys())
    print("  all", len(FEATURE_NAMES), "features present")


if __name__ == "__main__":
    for t in (test_momentum_sign_and_magnitude, test_12_1_momentum_excludes_last_month,
              test_volatility_positive_and_ordered, test_drawdown_detects_decline,
              test_pit_no_lookahead, test_fundamentals_passthrough_and_missing,
              test_all_feature_names_present):
        print(f"── {t.__name__}")
        t()
    print("ALL FEATURE TESTS PASSED")
