"""Tests for missing-Q4 synthesis (the ADBE-2021 bug). PYTHONPATH=. python this."""
from datetime import date, timedelta
from quantedge.fundamentals.rebound.bulk_extra import (
    synthesize_missing_quarters, ttm_points, knowable,
)


def fy(y):  # fiscal year ends Nov-30, quarters end Feb/May/Aug
    return [date(y, 2, 28), date(y, 5, 31), date(y, 8, 31), date(y, 11, 30)]


def build(missing_q4_from=2023):
    """Quarterly rows Q1-Q3 always; Q4 as a real duration only BEFORE the
    cutover year (mimics Adobe's tagging change). Annuals every year."""
    quarterly, annual = [], []
    for y in range(2020, 2027):
        ends = fy(y)
        vals = [100 + 10 * (y - 2020) + i for i in range(4)]   # gently growing
        for i, e in enumerate(ends[:3]):
            quarterly.append((e, float(vals[i]), e + timedelta(days=40)))
        if y < missing_q4_from:
            quarterly.append((ends[3], float(vals[3]), ends[3] + timedelta(days=60)))
        annual.append((ends[3], float(sum(vals)), ends[3] + timedelta(days=60)))
    return sorted(quarterly), sorted(annual)


def test_q4_reconstructed_exactly():
    q, a = build()
    full = synthesize_missing_quarters(q, a)
    missing_years = [y for y in range(2023, 2027)]
    for y in missing_years:
        e = fy(y)[3]
        row = [r for r in full if r[0] == e]
        assert row, f"Q4 {y} not synthesized"
        expected = 100 + 10 * (y - 2020) + 3
        assert abs(row[0][1] - expected) < 1e-9, (row, expected)
    print(f"  {len(missing_years)} missing Q4s reconstructed exactly")


def test_ttm_resumes_through_present():
    q, a = build()
    broken = ttm_points(q)                      # without synthesis
    fixed = ttm_points(synthesize_missing_quarters(q, a))
    assert fixed[-1][0] > broken[-1][0], (fixed[-1], broken[-1])
    assert fixed[-1][0] == date(2026, 11, 30)   # newest complete window (synth Q4)
    print(f"  TTM windows: {len(broken)} broken -> {len(fixed)} fixed; "
          f"latest {fixed[-1][0]}")


def test_synthesized_q4_pit():
    """A synthesized Q4 must not be knowable before the ANNUAL filed."""
    q, a = build()
    full = synthesize_missing_quarters(q, a)
    q4_2024 = [r for r in full if r[0] == fy(2024)[3]][0]
    annual_filed = [f for e, _, f in a if e == fy(2024)[3]][0]
    assert q4_2024[2] == annual_filed
    day_before = annual_filed - timedelta(days=1)
    assert all(r[0] != fy(2024)[3] for r in knowable(full, day_before))
    print("  synthesized Q4 invisible until its 10-K files (PIT)")


def test_real_q4_untouched():
    q, a = build(missing_q4_from=2023)
    full = synthesize_missing_quarters(q, a)
    e = fy(2021)[3]                              # real Q4 exists for 2021
    rows = [r for r in full if r[0] == e]
    assert len(rows) == 1 and rows[0] in q       # original row, not replaced
    print("  real Q4 durations left untouched")


if __name__ == "__main__":
    for t in (test_q4_reconstructed_exactly, test_ttm_resumes_through_present,
              test_synthesized_q4_pit, test_real_q4_untouched):
        print(f"── {t.__name__}")
        t()
    print("ALL Q4-SYNTHESIS TESTS PASSED")
