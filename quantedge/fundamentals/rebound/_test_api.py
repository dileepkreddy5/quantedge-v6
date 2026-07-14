"""Tests for the rebound live API logic. PYTHONPATH=. python this."""
from quantedge.fundamentals.rebound.api import (
    recovery_progress, build_list_payload, DISCLAIMER)


def test_recovery_progress_math():
    r = recovery_progress(60, 90, 120)
    assert r["progress_pct"] == 50.0
    assert not r["reached_high"]
    assert r["upside_to_high_pct"] == round((120/90-1)*100, 1)
    r2 = recovery_progress(60, 125, 120)
    assert r2["reached_high"]
    r3 = recovery_progress(60, 50, 120)
    assert r3["progress_pct"] == 0.0
    print("  recovery math: 50% progress, reached-high flag, clamps at 0")


def test_build_list_payload_shape():
    art = {"as_of": "2026-07-10", "generated": "2026-07-10T02:00:00Z",
           "total_passed": 127,
           "stage_counts": {"FALLING": 60, "RECOVERING": 40},
           "tiers": {"large": [{
               "ticker": "ADBE", "name": "Adobe Inc", "score": 72.2,
               "stage": "FALLING", "tier": "large", "drawdown": 0.65,
               "thesis": "-65% from 3y high; 49 growth quarters",
               "price": 350.0, "prior_high": 700.0}]}}
    def price_lookup(t):
        return 420.0 if t == "ADBE" else None
    p = build_list_payload(art, price_lookup)
    row = p["tiers"]["large"][0]
    assert row["ticker"] == "ADBE"
    assert row["drawdown_from_high_pct"] == 65.0
    assert row["current_price"] == 420.0
    assert row["recovery"]["progress_pct"] == 20.0
    assert p["disclaimer"] == DISCLAIMER
    assert p["total_passed"] == 127
    print("  payload: ADBE 65% off high, recovery 20%, disclaimer present")


def test_no_price_lookup_graceful():
    art = {"tiers": {"mid": [{"ticker": "ELF", "name": "e.l.f.", "score": 83.5,
            "stage": "RECOVERING", "tier": "mid", "drawdown": 0.65,
            "thesis": "x", "price": 100.0, "prior_high": 200.0}]}}
    p = build_list_payload(art, None)
    row = p["tiers"]["mid"][0]
    assert "recovery" not in row
    assert row["score"] == 83.5
    print("  no price lookup -> recovery omitted, never faked")


if __name__ == "__main__":
    for t in (test_recovery_progress_math, test_build_list_payload_shape,
              test_no_price_lookup_graceful):
        print(f"── {t.__name__}"); t()
    print("ALL API TESTS PASSED")
