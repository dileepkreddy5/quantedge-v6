"""Tests for the REBOUND disqualifier layer. Run: PYTHONPATH=. python this_file."""
from datetime import date, timedelta
import yaml, os

from quantedge.fundamentals.rebound.disqualify import (
    check_revenue_shrinking, check_leverage, check_dilution,
    check_cash_runway, compute_disqualifiers,
)

PARAMS = yaml.safe_load(open(os.path.join(
    os.path.dirname(__file__), "..", "..", "params.yaml")))

AS_OF = date(2026, 7, 10)


def q_series(n, start_val, yoy, start=date(2022, 6, 30)):
    out, e, v = [], start, start_val
    for i in range(n):
        out.append((e, v, e + timedelta(days=40)))
        v *= (1 + yoy) ** 0.25
        e = e + timedelta(days=91)
    return [(e, v, f) for e, v, f in out if e <= AS_OF]


def _annual(**kw):
    yrs = [2023, 2024, 2025]
    return {k: dict(zip(yrs, v)) for k, v in kw.items()}


def test_revenue_shrinking():
    grow = check_revenue_shrinking(q_series(16, 200e6, 0.10), AS_OF)
    assert grow["ok"] and grow["flag"] is False, grow
    shrink = check_revenue_shrinking(q_series(16, 200e6, -0.08), AS_OF)
    assert shrink["ok"] and shrink["flag"] is True, shrink
    print("  growing passes, shrinking flagged (ttm chg", shrink["ttm_change"], ")")


def test_leverage():
    ok = check_leverage(_annual(liabilities=[400e6, 410e6, 420e6],
                                assets=[1000e6, 1050e6, 1100e6]), 0.10)
    assert ok["flag"] is False
    spike = check_leverage(_annual(liabilities=[400e6, 420e6, 650e6],
                                   assets=[1000e6, 1020e6, 1040e6]), 0.10)
    assert spike["flag"] is True, spike
    print("  stable passes, +", spike["leverage_delta_yoy"], "leverage flagged")


def test_dilution_and_split_guard():
    base = date(2024, 6, 30)
    sh = [(base, 10e6, base + timedelta(days=10)),
          (base + timedelta(days=365), 10.3e6, base + timedelta(days=375))]
    ok = check_dilution(sh, AS_OF, 0.10)
    assert ok["flag"] is False, ok
    sh_dil = [(base, 10e6, base + timedelta(days=10)),
              (base + timedelta(days=365), 11.8e6, base + timedelta(days=375))]
    dil = check_dilution(sh_dil, AS_OF, 0.10)
    assert dil["flag"] is True, dil
    sh_split = [(base, 10e6, base + timedelta(days=10)),
                (base + timedelta(days=365), 40e6, base + timedelta(days=375))]
    split = check_dilution(sh_split, AS_OF, 0.10)
    assert split["flag"] is False and "split" in split.get("note", ""), split
    print("  +3% passes, +18% flagged, 4x split correctly ignored")


def test_cash_runway():
    cash = [(date(2026, 3, 31), 120e6, date(2026, 5, 10))]
    healthy = check_cash_runway(cash, _annual(op_cash_flow=[50e6, 60e6, 75e6]),
                                AS_OF, 18)
    assert healthy["flag"] is False and healthy["burning"] is False
    # burning $120M/yr with $120M cash = 12 months < 18 => flagged
    burner = check_cash_runway(cash, _annual(op_cash_flow=[-90e6, -100e6, -120e6]),
                               AS_OF, 18)
    assert burner["flag"] is True and abs(burner["runway_months"] - 12.0) < 0.1, burner
    # burning with NO visible cash balance => flagged (unverifiable survival)
    blind = check_cash_runway([], _annual(op_cash_flow=[-90e6, -100e6, -120e6]),
                              AS_OF, 18)
    assert blind["flag"] is True
    print("  runway", burner["runway_months"], "mo flagged; invisible-cash burner flagged")


def test_full_verdict():
    good = compute_disqualifiers(
        q_series(16, 200e6, 0.10),
        [(date(2025, 6, 30), 10e6, date(2025, 7, 10)),
         (date(2026, 6, 30), 10.2e6, date(2026, 7, 8))],
        [(date(2026, 3, 31), 200e6, date(2026, 5, 10))],
        _annual(liabilities=[400e6, 405e6, 410e6], assets=[1000e6, 1050e6, 1100e6],
                op_cash_flow=[50e6, 60e6, 75e6]),
        AS_OF, PARAMS)
    assert good["disqualified"] is False, good
    print("  clean company:", good["reason"])

    knife = compute_disqualifiers(
        q_series(16, 200e6, -0.10),
        [(date(2025, 6, 30), 10e6, date(2025, 7, 10)),
         (date(2026, 6, 30), 11.9e6, date(2026, 7, 8))],
        [(date(2026, 3, 31), 40e6, date(2026, 5, 10))],
        _annual(liabilities=[400e6, 430e6, 700e6], assets=[1000e6, 1010e6, 1020e6],
                op_cash_flow=[-30e6, -60e6, -90e6]),
        AS_OF, PARAMS)
    assert knife["disqualified"] is True
    assert set(knife["flags"]) == {"revenue_shrinking", "leverage_spiking",
                                   "dilution", "cash_runway"}, knife["flags"]
    print("  knife:", knife["reason"])


if __name__ == "__main__":
    for t in (test_revenue_shrinking, test_leverage, test_dilution_and_split_guard,
              test_cash_runway, test_full_verdict):
        print(f"── {t.__name__}")
        t()
    print("ALL DISQUALIFIER TESTS PASSED")
