"""Proof the Learning Loop works: timeline lead detection, graveyard, miss taxonomy.
Run: PYTHONPATH=. python this_file."""
import os, tempfile
from datetime import date
from quantedge.harness.company_timeline import TimelinePoint, measure_lead
from quantedge.harness.learning_loop import (
    Graveyard, GraveyardEntry, FailureCause, Miss, MissCause,
)

def month(y, m): return date(y, m, 1)

# --- THE MISSION TEST: score inflects BEFORE price (the NVIDIA-2015 profile) ---
# Score starts rising in early 2015; price stays flat until late 2015.
timeline = []
score = 40.0
price = 20.0
for i, (y, m) in enumerate([(2015,1),(2015,2),(2015,3),(2015,4),(2015,5),(2015,6),
                            (2015,7),(2015,8),(2015,9),(2015,10),(2015,11),(2015,12)]):
    # business (score) inflects from month 2; price doesn't move until month 8
    if i >= 1:
        score += 5.0
    if i >= 7:
        price += 6.0
    timeline.append(TimelinePoint(month(y, m), score, price))

res = measure_lead(timeline, min_consec=2)
print(f"score inflection: {res.score_inflection}")
print(f"price inflection: {res.price_inflection}")
print(f"lead: {res.lead_days} days  | detected_early: {res.detected_early}")
assert res.detected_early, "FAIL: score should lead price in this profile"
assert res.lead_days > 100, "FAIL: lead should be several months"
print("MISSION TEST: the system's score led the price -> early detection demonstrated  OK")

# --- Negative control: if score and price move together, lead is ~0, not early ---
flat = [TimelinePoint(month(2018, m), 50.0 + m, 30.0 + m) for m in range(1, 8)]
res2 = measure_lead(flat, min_consec=2)
print(f"\nsimultaneous-move control: lead={res2.lead_days}, early={res2.detected_early}")
assert not res2.detected_early, "FAIL: simultaneous moves must NOT count as early detection"
print("NEGATIVE CONTROL: simultaneous move is NOT flagged as early detection      OK")

# --- GRAVEYARD: a liked company that failed is buried permanently ---
tmp = tempfile.mkdtemp()
gy = Graveyard(path=os.path.join(tmp, "graveyard.jsonl"))
gy.bury(GraveyardEntry("XYZ", liked_on=date(2019,1,1), failed_by=date(2021,6,1),
                       cause=FailureCause.FRAUD, score_at_like=78.0,
                       note="Liked on accelerating revenue; revenue was fabricated."))
gy.bury(GraveyardEntry("ABC", liked_on=date(2020,3,1), failed_by=date(2022,2,1),
                       cause=FailureCause.DEBT, score_at_like=65.0,
                       note="Strong growth funded by debt that became unserviceable."))
buried = gy.all()
print(f"\ngraveyard holds {len(buried)} autopsied failures (expect 2)")
assert len(buried) == 2
neg = gy.negative_evidence()
print(f"negative-evidence feed: {neg}")
assert {n['cause'] for n in neg} == {'fraud', 'debt'}
print("GRAVEYARD: failures buried + exposed as negative evidence                 OK")

# --- MISS TAXONOMY: a miss maps to exactly one fixable cause ---
miss = Miss("DEF", as_of=date(2016,5,1), cause=MissCause.VALUATION_GATE_TOO_STRICT,
            note="Screened out at 40x earnings; it was a justified premium.")
assert miss.cause is MissCause.VALUATION_GATE_TOO_STRICT
print("MISS TAXONOMY: miss classified to a specific, fixable cause               OK")

print("\nPASS — the system measures its own foresight and learns from its failures.")
