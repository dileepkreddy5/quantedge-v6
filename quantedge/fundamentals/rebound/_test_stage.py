"""Tests for the stage detector. Run: PYTHONPATH=. python this_file."""
import yaml, os
from quantedge.fundamentals.rebound.stage import classify_stage

PARAMS = yaml.safe_load(open(os.path.join(
    os.path.dirname(__file__), "..", "..", "params.yaml")))

DD = lambda days, off: {"ok": True, "days_since_1y_low": days, "pct_off_low": off}
VOL = lambda streak, up: {"ok": True, "accum_streak_weeks": streak, "up_day_share_1m": up}


def test_all_stages():
    cases = [
        (DD(15, 0.03),  VOL(0, 0.48), "FALLING"),
        (DD(90, 0.06),  VOL(1, 0.50), "BASING"),
        (DD(90, 0.08),  VOL(4, 0.52), "TURNING"),    # via streak
        (DD(120, 0.10), VOL(0, 0.61), "TURNING"),    # via up-day share
        (DD(200, 0.31), VOL(0, 0.50), "RECOVERING"),
        (DD(15, 0.30),  VOL(0, 0.50), "RECOVERING"), # off-low dominates
    ]
    for dd, vol, expect in cases:
        r = classify_stage(dd, vol, PARAMS)
        assert r["stage"] == expect, (r, expect)
        print(f"  {expect:<10} <- {r['reason']}")


def test_missing_volume_degrades_gracefully():
    r = classify_stage(DD(90, 0.06), {"ok": False}, PARAMS)
    assert r["stage"] == "BASING"          # base holds; no accum evidence claimed
    r2 = classify_stage({"ok": False}, VOL(4, 0.6), PARAMS)
    assert r2["stage"] == "UNKNOWN"
    print("  missing volume -> BASING; missing prices -> UNKNOWN")


if __name__ == "__main__":
    for t in (test_all_stages, test_missing_volume_degrades_gracefully):
        print(f"── {t.__name__}")
        t()
    print("ALL STAGE TESTS PASSED")
