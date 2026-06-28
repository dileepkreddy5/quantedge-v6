"""Proof the PIT loader cannot see the future. Run directly: python this_file."""
from datetime import date
from quantedge.data.sources.base import FundamentalFact
from quantedge.harness.pit_loader import PITLoader

# Three NVDA gross-profit facts, filed at different times:
facts = [
    FundamentalFact("NVDA", "gross_profit", date(2020, 9, 30), 900.0,  date(2020, 11, 15), "edgar"),  # past
    FundamentalFact("NVDA", "gross_profit", date(2020, 12, 31), 1000.0, date(2021, 2, 15),  "edgar"),  # FUTURE on our cutoff
    FundamentalFact("NVDA", "gross_profit", date(2020, 6, 30), 800.0,  date(2020, 8, 14),  "edgar"),  # older past
]
loader = PITLoader(facts)

CUTOFF = date(2021, 1, 10)  # between the Q3 filing (visible) and Q4 filing (NOT yet)

visible = loader.as_of(CUTOFF)
print(f"As of {CUTOFF}, visible facts: {len(visible)} (expect 2)")
for f in visible:
    print(f"   period {f.fiscal_period} filed {f.available_date} value {f.value}")

assert len(visible) == 2, "FAIL: wrong number of visible facts"
assert all(f.available_date <= CUTOFF for f in visible), "FAIL: a future fact leaked!"
assert not any(f.fiscal_period == date(2020, 12, 31) for f in visible), \
    "FAIL: the Q4 number (filed 2021-02-15) must be invisible on 2021-01-10"

latest = loader.latest("NVDA", "gross_profit", CUTOFF)
print(f"Latest knowable value on {CUTOFF}: {latest.value} (expect 900.0, the Q3 filing)")
assert latest.value == 900.0, "FAIL: latest() returned the wrong vintage"

# And confirm the future becomes visible once we move the cutoff past the filing:
later = loader.latest("NVDA", "gross_profit", date(2021, 3, 1))
print(f"Latest knowable value on 2021-03-01: {later.value} (expect 1000.0, Q4 now filed)")
assert later.value == 1000.0, "FAIL: Q4 should be visible after its filing date"

print("\nPASS — the loader shows the past, hides the future, and respects filing vintage.")
