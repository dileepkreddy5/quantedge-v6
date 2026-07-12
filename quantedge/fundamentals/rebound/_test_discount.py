"""Tests for the REBOUND discount layer. Run: PYTHONPATH=. python this_file.

Synthetic fixtures with hand-computable answers — the point is to prove the
MATH and the PIT discipline, not to hit the network.
"""
from datetime import date, timedelta
import yaml, os

from quantedge.fundamentals.rebound.bulk_extra import ttm_points, knowable
from quantedge.fundamentals.rebound.discount import (
    drawdown_structure, valuation_vs_own_history, compute_discount,
)

PARAMS = yaml.safe_load(open(os.path.join(
    os.path.dirname(__file__), "..", "..", "params.yaml")))

AS_OF = date(2026, 7, 10)


def _weekdays(start: date, n: int):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def build_closes():
    """4y series: climbs 50->100 over 2y, crashes to 45 over 1y, bases ~1y.
    Ends near 49.5 => drawdown from high ~= 50.5%."""
    days = _weekdays(AS_OF - timedelta(days=4 * 365), 1050)  # cover through AS_OF
    closes = []
    for i, d in enumerate(days):
        if i < 500:
            px = 50 + 50 * i / 499                    # climb to 100
        elif i < 750:
            px = 100 - 55 * (i - 500) / 249           # crash to 45
        else:
            px = 45 + 4.5 * (i - 750) / 299           # slow base to ~49.5
        closes.append((d, round(px, 4)))
    return [c for c in closes if c[0] <= AS_OF]


def q_rev_series():
    """17 quarters of revenue growing 4%/q, filed ~40d after period end.
    History must span the expensive pre-crash era."""
    out = []
    e = date(2022, 6, 30)   # history must span the expensive era pre-crash
    v = 100e6
    for _ in range(17):
        out.append((e, v, e + timedelta(days=40)))
        e = e + timedelta(days=91)
        v *= 1.04
    return [(e, v, f) for e, v, f in out if e <= AS_OF]


def test_drawdown():
    dd = drawdown_structure(build_closes(), AS_OF)
    assert dd["ok"]
    assert abs(dd["drawdown_from_3y_high"] - 0.505) < 0.02, dd["drawdown_from_3y_high"]
    assert dd["days_since_1y_low"] > 200          # low was set ~1y ago, basing since
    assert dd["pct_off_low"] > 0.05               # ~10% off the low
    assert dd["dd_percentile_own_history"] > 55   # deeper than most of its own history
    print("  drawdown:", dd["drawdown_from_3y_high"], "underwater days:", dd["days_underwater"])


def test_ttm_pit():
    q = q_rev_series()
    ttm = ttm_points(q)
    assert len(ttm) >= 6
    # PIT: last TTM window's knowable date = the LATEST of its 4 filings
    last = ttm[-1]
    assert last[2] == max(f for _, _, f in q[-4:])
    # A window whose final quarter files AFTER as_of must be invisible
    cutoff = last[2] - timedelta(days=1)
    visible = knowable(ttm, cutoff)
    assert all(f <= cutoff for _, _, f in visible)
    assert len(visible) == len(ttm) - 1
    print("  ttm points:", len(ttm), "PIT exclusion verified")


def test_valuation_vs_own_history():
    closes = build_closes()
    q = q_rev_series()
    shares = [(date(2022, 12, 31), 10_000_000, date(2023, 1, 30))]  # constant 10M shares
    # Price fell ~50% while TTM revenue GREW => P/S now must be near the
    # bottom of its own history.
    val = valuation_vs_own_history(closes, q, shares, [], [], AS_OF)
    assert val["ok"], val
    assert val["ps_vs_5y_median"] < 0.75, val
    assert val["ps_percentile_own"] <= 25.0, val
    print("  P/S now:", val["ps_ttm_now"], "vs median:", val["ps_vs_5y_median"],
          "percentile:", val["ps_percentile_own"])


def test_layer_verdict():
    closes = build_closes()
    q = q_rev_series()
    shares = [(date(2022, 12, 31), 10_000_000, date(2023, 1, 30))]
    r = compute_discount(closes, q, shares, [], [], AS_OF, PARAMS)
    assert r["qualifies"] is True, r
    assert "from 3y high" in r["reason"]
    print("  verdict:", r["reason"])

    # An 80%-retracement stock (barely down) must NOT qualify
    shallow = [(d, c * 1.9) if i > 700 else (d, c)
               for i, (d, c) in enumerate(closes)]
    r2 = compute_discount(shallow, q, shares, [], [], AS_OF, PARAMS)
    assert r2["qualifies"] is False
    print("  shallow-drawdown correctly rejected")


if __name__ == "__main__":
    for t in (test_drawdown, test_ttm_pit, test_valuation_vs_own_history, test_layer_verdict):
        print(f"── {t.__name__}")
        t()
    print("ALL DISCOUNT TESTS PASSED")
