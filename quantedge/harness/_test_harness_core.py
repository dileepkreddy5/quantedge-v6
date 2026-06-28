"""Proof for labeler, costs, splitter. Run: PYTHONPATH=. python this_file."""
import yaml
from datetime import date
from quantedge.harness.labeler import label, Outcome, Exit, Label
from quantedge.harness.costs import apply_costs, round_trip_cost_bps
from quantedge.harness.splitter import Split

params = yaml.safe_load(open("quantedge/params.yaml"))

# --- LABELER: the survivorship-critical cases (§14, Table 30) ---
# A bankrupt company is a LOSS, not a dropped row — the core survivorship fix.
bankrupt = Outcome(Exit.BANKRUPT_DELISTED, horizon_reached=True, bankrupt=True)
assert label(bankrupt, params) is Label.NON_WINNER
print("bankrupt -> NON_WINNER (counted as a loss, not dropped)  OK")

# A premium buyout whose total return cleared 2x is a WINNER.
buyout = Outcome(Exit.ACQUIRED_PREMIUM, horizon_reached=True, total_excess_return_vs_sector=2.4)
assert label(buyout, params) is Label.WINNER
print("premium buyout @2.4x -> WINNER                            OK")

# A premium buyout that fell short is NOT a winner.
weak_buyout = Outcome(Exit.ACQUIRED_PREMIUM, horizon_reached=True, total_excess_return_vs_sector=1.3)
assert label(weak_buyout, params) is Label.NON_WINNER
print("premium buyout @1.3x -> NON_WINNER                        OK")

# Still trading, passed all three tests -> WINNER.
strong = Outcome(Exit.STILL_TRADING, horizon_reached=True,
                 total_excess_return_vs_sector=2.5, revenue_multiple=2.2,
                 roic_improving=True, terminal_dilution_pct=10)
assert label(strong, params) is Label.WINNER
print("still trading, all tests pass -> WINNER                   OK")

# Strong returns but financed by dilution > 40% -> survival test fails.
diluted = Outcome(Exit.STILL_TRADING, horizon_reached=True,
                  total_excess_return_vs_sector=3.0, revenue_multiple=2.5,
                  roic_improving=True, terminal_dilution_pct=55)
assert label(diluted, params) is Label.NON_WINNER
print("3x return but 55% dilution -> NON_WINNER (survival test)  OK")

# Horizon not reached -> CENSORED (using it would be look-ahead).
unfinished = Outcome(Exit.STILL_TRADING, horizon_reached=False)
assert label(unfinished, params) is Label.CENSORED
print("horizon not reached -> CENSORED                           OK")

# --- COSTS: gross is never shown alone (§13) ---
rt = round_trip_cost_bps(params)  # 2 * (1+5+2) = 16 bps
assert rt == 16.0
net = apply_costs(0.10, turnover=1.0, params=params)
assert abs(net - (0.10 - 0.0016)) < 1e-12
print(f"round-trip cost {rt} bps; 10% gross -> {net:.4f} net      OK")

# --- SPLITTER: leakage is structurally impossible ---
ok = Split(date(2010,1,1), date(2015,12,31), date(2016,1,1), date(2020,12,31))
assert ok.is_train(date(2012,6,1)) and ok.is_test(date(2018,6,1))
print("valid time-split accepted                                 OK")

try:
    Split(date(2010,1,1), date(2016,12,31), date(2016,6,1), date(2020,1,1))  # overlap
    raise SystemExit("FAIL: overlapping split was allowed")
except ValueError:
    print("overlapping split rejected (no train/test leakage)       OK")

print("\nPASS — labeler, costs, and splitter all enforce their discipline.")
