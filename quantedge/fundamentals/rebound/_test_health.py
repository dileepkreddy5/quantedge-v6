"""Tests for the REBOUND health layer. Run: PYTHONPATH=. python this_file."""
from datetime import date, timedelta
import yaml, os

from quantedge.fundamentals.rebound.health import (
    growth_streak, margin_trajectory, rd_factor, roic_direction, compute_health,
)

PARAMS = yaml.safe_load(open(os.path.join(
    os.path.dirname(__file__), "..", "..", "params.yaml")))

AS_OF = date(2026, 7, 10)


def q_series(n, start_val, growths, start=date(2022, 6, 30), filed_lag=40):
    """n quarters; growths[i] = YoY growth applied per year boundary via
    quarterly compounding of (1+g)^(1/4)."""
    out, e, v = [], start, start_val
    for i in range(n):
        out.append((e, v, e + timedelta(days=filed_lag)))
        g = growths[min(i // 4, len(growths) - 1)]
        v *= (1 + g) ** 0.25
        e = e + timedelta(days=91)
    return [(e, v, f) for e, v, f in out if e <= AS_OF]


def test_growth_streak_basic():
    # 16 quarters, always growing ~12%/yr => streak spans every matchable quarter
    q = q_series(16, 200e6, [0.12])
    gs = growth_streak(q, AS_OF, 25e6)
    assert gs["ok"] and gs["streak"] >= 8, gs
    assert gs["latest_yoy"] > 0.10
    print("  streak:", gs["streak"], "latest yoy:", gs["latest_yoy"])


def test_growth_streak_breaks_on_decline():
    # grew for 2y, then SHRANK for the most recent year => streak == 0
    q = q_series(16, 200e6, [0.15, 0.15, -0.10, -0.10])
    gs = growth_streak(q, AS_OF, 25e6)
    assert gs["ok"] and gs["streak"] == 0, gs
    print("  declining company correctly shows streak 0")


def test_growth_streak_drops_artifact():
    # inject a bogus $5M quarter between ~$230M quarters — clean_quarters must
    # drop it so the streak SURVIVES the artifact
    q = q_series(16, 200e6, [0.12])
    bogus_end = q[10][0] + timedelta(days=1)
    q_bad = sorted(q + [(bogus_end, 5e6, bogus_end + timedelta(days=40))])
    gs_clean = growth_streak(q, AS_OF, 25e6)
    gs_bad = growth_streak(q_bad, AS_OF, 25e6)
    assert gs_bad["streak"] >= gs_clean["streak"] - 1, (gs_bad, gs_clean)
    print("  artifact quarter dropped; streak preserved:", gs_bad["streak"])


def test_growth_streak_pit():
    # the newest quarter files AFTER as_of => must not count toward the streak
    q = q_series(16, 200e6, [0.12])
    late = list(q)
    e_new = late[-1][0] + timedelta(days=91)
    late.append((e_new, late[-1][1] * 1.03, AS_OF + timedelta(days=30)))  # files later
    gs = growth_streak(late, AS_OF, 25e6)
    gs_base = growth_streak(q, AS_OF, 25e6)
    assert gs["streak"] == gs_base["streak"], (gs, gs_base)
    print("  unfiled quarter invisible (PIT)")


def test_margin_trajectory():
    q_rev = q_series(12, 300e6, [0.10])
    # gross profit margin climbing 40% -> 48%
    q_gp = [(e, v * (0.40 + 0.01 * i), f) for i, (e, v, f) in enumerate(q_rev)]
    mt = margin_trajectory(q_gp, q_rev, AS_OF)
    assert mt["ok"] and mt["expanding"], mt
    assert mt["gross_margin_slope_per_q"] > 0.005
    print("  margin now:", mt["gross_margin_now"], "slope/q:", mt["gross_margin_slope_per_q"])


def test_rd_factor():
    rd_q = q_series(10, 30e6, [0.30])       # R&D growing 30%/yr
    r = rd_factor(rd_q, market_cap=1_000e6, as_of=AS_OF)
    assert r["ok"] and r["rd_to_mktcap"] > 0.10, r
    assert r.get("rd_yoy_growth", 0) > 0.20
    print("  R&D/mktcap:", r["rd_to_mktcap"], "R&D yoy:", r["rd_yoy_growth"])


def _annual(ni, ocf, a, li, rev, gp, ca, cl, sh):
    yrs = list(range(2022, 2022 + len(ni)))
    z = lambda xs: dict(zip(yrs, xs))
    return {"net_income": z(ni), "op_cash_flow": z(ocf), "assets": z(a),
            "liabilities": z(li), "revenue": z(rev), "gross_profit": z(gp),
            "cur_assets": z(ca), "cur_liab": z(cl), "shares": z(sh)}


def test_roic_and_full_verdict():
    known = _annual(
        ni=[50e6, 70e6, 95e6], ocf=[80e6, 100e6, 130e6],
        a=[900e6, 950e6, 1000e6], li=[400e6, 390e6, 380e6],
        rev=[800e6, 900e6, 1020e6], gp=[320e6, 380e6, 460e6],
        ca=[300e6, 330e6, 380e6], cl=[150e6, 150e6, 150e6],
        sh=[10e6, 10e6, 10e6])
    ro = roic_direction(known)
    assert ro["ok"] and ro["improving"], ro

    q_rev = q_series(16, 220e6, [0.13])
    q_gp = [(e, v * (0.42 + 0.005 * i), f) for i, (e, v, f) in enumerate(q_rev)]
    rd_q = q_series(12, 25e6, [0.25])
    r = compute_health(q_rev, q_gp, rd_q, known, market_cap=2_000e6,
                       ticker="TEST", as_of=AS_OF, params=PARAMS)
    assert r["qualifies"] is True, (r["growth"], r["piotroski"])
    assert r["piotroski"] >= 7
    print("  verdict:", r["reason"])

    # a shrinking, low-quality company must FAIL the gate
    known_bad = _annual(
        ni=[50e6, 20e6, -10e6], ocf=[40e6, 10e6, -20e6],
        a=[900e6, 950e6, 1000e6], li=[400e6, 500e6, 620e6],
        rev=[800e6, 700e6, 600e6], gp=[300e6, 240e6, 180e6],
        ca=[300e6, 260e6, 210e6], cl=[150e6, 180e6, 220e6],
        sh=[10e6, 11e6, 12.5e6])
    q_bad = q_series(16, 220e6, [-0.12])
    r2 = compute_health(q_bad, q_gp, rd_q, known_bad, market_cap=500e6,
                        ticker="BAD", as_of=AS_OF, params=PARAMS)
    assert r2["qualifies"] is False
    print("  deteriorating company correctly rejected")


if __name__ == "__main__":
    for t in (test_growth_streak_basic, test_growth_streak_breaks_on_decline,
              test_growth_streak_drops_artifact, test_growth_streak_pit,
              test_margin_trajectory, test_rd_factor, test_roic_and_full_verdict):
        print(f"── {t.__name__}")
        t()
    print("ALL HEALTH TESTS PASSED")
