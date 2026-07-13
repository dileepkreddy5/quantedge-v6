"""Tests for the score assembly. Run: PYTHONPATH=. python this_file."""
import yaml, os
from quantedge.fundamentals.rebound.rebound_score import score_candidate

PARAMS = yaml.safe_load(open(os.path.join(
    os.path.dirname(__file__), "..", "..", "params.yaml")))


def strong_layers():
    discount = {"qualifies": True, "valuation_history": True,
                "reason": "-61% from 3y high; P/S 0.7 = 45% of own 5y median",
                "drawdown": {"ok": True, "drawdown_from_3y_high": 0.75,
                             "days_since_1y_low": 120, "pct_off_low": 0.08},
                "valuation": {"ok": True, "ps_percentile_own": 0.0}}
    health = {"qualifies": True, "piotroski": 9, "accruals": -0.03,
              "reason": "8 straight growth quarters; F-score 9/9",
              "growth": {"ok": True, "streak": 8},
              "margin": {"ok": True, "expanding": True},
              "rd": {"ok": True, "rd_to_mktcap": 0.08, "rd_yoy_growth": 0.2},
              "roic": {"ok": True, "improving": True}}
    disq = {"disqualified": False, "unverified": []}
    confirm = {"n_confirmations": 3, "reason": "vol 2x; 65% up-days; 5-wk streak",
               "volume": {"ok": True, "accum_streak_weeks": 5,
                          "up_day_share_1m": 0.70, "vol_1m_ratio": 2.0},
               "buyback": {"ok": True, "active_through_decline": True,
                           "buyback_to_mktcap": 0.03}}
    insider = {"ok": True, "cluster": True, "n_buyers": 3,
               "reason": "3 insiders bought $2,050,000 open-market"}
    stage = {"stage": "TURNING", "reason": "based 120d; 5-wk streak"}
    return discount, health, disq, confirm, insider, stage


def test_perfect_candidate_scores_100():
    r = score_candidate(*strong_layers(), PARAMS)
    assert r["passes"] is True
    assert r["score"] == 100.0, (r["score"], r["components"])
    assert abs(sum(r["components"].values()) - 100.0) < 0.01
    print("  perfect candidate = 100.0; components sum verified")
    print("  thesis:", r["thesis"][:100])


def test_gates_are_hard():
    d, h, dq, c, i, s = strong_layers()
    for broken in ("discount", "health", "knife_filter"):
        dd, hh, dqq = d, h, dq
        if broken == "discount":
            dd = {**d, "qualifies": False}
        elif broken == "health":
            hh = {**h, "qualifies": False}
        else:
            dqq = {**dq, "disqualified": True}
        r = score_candidate(dd, hh, dqq, c, i, s, PARAMS)
        assert r["passes"] is False and broken in r["failed"], (broken, r)
    print("  each gate independently blocks (no partial credit)")


def test_opacity_penalized():
    d, h, dq, c, i, s = strong_layers()
    dq2 = {**dq, "unverified": ["cash_runway", "dilution"]}
    r_full = score_candidate(d, h, dq, c, i, s, PARAMS)
    r_op = score_candidate(d, h, dq2, c, i, s, PARAMS)
    assert r_op["score"] < r_full["score"], (r_op["score"], r_full["score"])
    assert r_op["components"]["verifiable"] == 2.0   # 4 * (1 - 2/4)
    print("  2 unverifiable checks cost", r_full["score"] - r_op["score"], "pts")


def test_no_insider_no_points_no_crash():
    d, h, dq, c, _, s = strong_layers()
    r = score_candidate(d, h, dq, c, None, s, PARAMS)
    assert r["passes"] and r["components"]["insider"] == 0.0
    assert r["score"] == 96.0, r["score"]
    print("  insider=None handled; score", r["score"])


if __name__ == "__main__":
    for t in (test_perfect_candidate_scores_100, test_gates_are_hard,
              test_opacity_penalized, test_no_insider_no_points_no_crash):
        print(f"── {t.__name__}")
        t()
    print("ALL SCORE TESTS PASSED")
