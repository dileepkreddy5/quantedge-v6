"""Tests for the REBOUND confirmation layer. Run: PYTHONPATH=. python this_file."""
from datetime import date, timedelta
import yaml, os, random

from quantedge.fundamentals.rebound.confirm import (
    volume_signals, buyback_confirm, compute_confirm,
)

PARAMS = yaml.safe_load(open(os.path.join(
    os.path.dirname(__file__), "..", "..", "params.yaml")))

AS_OF = date(2026, 7, 10)
random.seed(7)


def _weekdays_back(n):
    out, d = [], AS_OF
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return list(reversed(out))


def build_bars(accumulating: bool):
    """120 bars. Baseline: 1M vol, flat drift. Last month: if accumulating,
    2x volume concentrated on up days; else quiet and directionless."""
    days = _weekdays_back(120)
    bars, px = [], 50.0
    for i, d in enumerate(days):
        last_month = i >= len(days) - 21
        if accumulating and last_month:
            up = random.random() < 0.68           # mostly up days
            px *= 1.006 if up else 0.997
            vol = (2_400_000 if up else 900_000)  # heavy vol ON the up days
        else:
            up = random.random() < 0.5
            px *= 1.002 if up else 0.998
            vol = random.uniform(0.9e6, 1.1e6)
        bars.append((d, round(px, 4), vol))
    return bars


def test_volume_accumulation_detected():
    v = volume_signals(build_bars(accumulating=True), AS_OF, PARAMS)
    assert v["ok"], v
    assert v["vol_1m_ratio"] > 1.4, v
    assert v["up_day_share_1m"] > 0.60, v
    assert v["accum_streak_weeks"] >= 3, v
    print("  1m vol ratio:", v["vol_1m_ratio"], "up-day share:", v["up_day_share_1m"],
          "streak:", v["accum_streak_weeks"], "wk")


def test_volume_quiet_not_flagged():
    v = volume_signals(build_bars(accumulating=False), AS_OF, PARAMS)
    assert v["ok"], v
    assert 0.7 < v["vol_1m_ratio"] < 1.3, v
    assert v["accum_streak_weeks"] <= 2, v
    print("  quiet stock: ratio", v["vol_1m_ratio"], "streak", v["accum_streak_weeks"])


def test_baseline_not_self_inflated():
    """A month-long volume spike must not raise its own baseline: the ratio
    must reflect spike/quiet, not spike/(mix)."""
    bars = build_bars(accumulating=True)
    v = volume_signals(bars, AS_OF, PARAMS)
    # recent-month avg ~1.9M, TRUE baseline ~1.0M => ratio must be ~1.9, not ~1.4
    assert v["vol_1m_ratio"] > 1.5, v
    print("  baseline excludes recent window — ratio", v["vol_1m_ratio"])


def q_series(n, start_val, start=date(2023, 6, 30)):
    out, e = [], start
    for i in range(n):
        out.append((e, start_val, e + timedelta(days=40)))
        e = e + timedelta(days=91)
    return [(e, v, f) for e, v, f in out if e <= AS_OF]


def test_buyback():
    bb = buyback_confirm(q_series(10, 30e6), market_cap=2_000e6,
                         as_of=AS_OF, params=PARAMS)
    assert bb["ok"] and bb["active_through_decline"], bb
    assert abs(bb["buyback_ttm"] - 120e6) < 1e6, bb        # 4 x 30M
    assert abs(bb["buyback_to_mktcap"] - 0.06) < 0.005, bb
    print("  TTM buyback:", bb["buyback_ttm"], "=", bb["buyback_to_mktcap"], "of mktcap")

    # stale buybacks (stopped 2y ago) must NOT count as active
    old = [(e - timedelta(days=730), v, f - timedelta(days=730))
           for e, v, f in q_series(6, 30e6)]
    bb2 = buyback_confirm(old, market_cap=2_000e6, as_of=AS_OF, params=PARAMS)
    assert bb2["ok"] and bb2["active_through_decline"] is False, bb2
    print("  stale buyback correctly inactive")


def test_full_summary():
    r = compute_confirm(build_bars(accumulating=True), q_series(10, 30e6),
                        market_cap=2_000e6, as_of=AS_OF, params=PARAMS)
    assert r["n_confirmations"] >= 3, r
    print("  summary:", r["reason"])


if __name__ == "__main__":
    for t in (test_volume_accumulation_detected, test_volume_quiet_not_flagged,
              test_baseline_not_self_inflated, test_buyback, test_full_summary):
        print(f"── {t.__name__}")
        t()
    print("ALL CONFIRM TESTS PASSED")
