"""First real Cohen-Frazzini directional check on live prices.
Run: export POLYGON=...; PYTHONPATH=. python this_file

CF logic: a CUSTOMER shock in month M predicts the SUPPLIER's return in
month M+1 (the supplier's price lags the customer news). We measure, across
seed edges, whether customer-shock and next-month supplier-return line up.
This is a directional sanity check, NOT a full reproduction (§ caveat).
"""
import os, sys, time
from datetime import date
sys.path.insert(0, os.path.dirname(__file__))
from seed_graph import SEED_EDGES
from quantedge.data.sources.polygon_prices import daily_closes

# Two non-overlapping windows: shock month, then the following month.
WINDOWS = [
    ("2023-05", date(2023,5,1), date(2023,5,31), date(2023,6,1), date(2023,6,30)),
    ("2023-09", date(2023,9,1), date(2023,9,30), date(2023,10,1),date(2023,10,31)),
    ("2024-01", date(2024,1,1), date(2024,1,31), date(2024,2,1), date(2024,2,29)),
]

def ret(ticker, a, b):
    c = daily_closes(ticker, a, b)
    if len(c) < 2: return None
    return c[-1][1] / c[0][1] - 1.0

edges = [e for e in SEED_EDGES if e.confirmed and e.weight >= 0.10]
price_cache = {}

def cached_ret(t, a, b):
    k = (t, a, b)
    if k not in price_cache:
        price_cache[k] = ret(t, a, b)
    return price_cache[k]

rows = []
for label, sa, sb, na, nb in WINDOWS:
    for e in edges:
        cust_shock = cached_ret(e.to_ticker, sa, sb)        # customer, shock month
        supp_next  = cached_ret(e.from_ticker, na, nb)      # supplier, NEXT month
        if cust_shock is None or supp_next is None:
            continue
        rows.append((label, e.from_ticker, e.to_ticker, e.weight, cust_shock, supp_next))
        print(f"{label} {e.from_ticker:5s}<-{e.to_ticker:5s} "
              f"cust_shock={cust_shock:+.2%}  supp_next={supp_next:+.2%}")

# Directional test: does a positive customer shock associate with a positive
# next-month supplier return? Compare supplier returns after +shocks vs -shocks.
pos = [r[5] for r in rows if r[4] > 0]
neg = [r[5] for r in rows if r[4] <= 0]
print(f"\nobservations: {len(rows)}")
if pos: print(f"avg supplier next-month return AFTER + customer shock: {sum(pos)/len(pos):+.2%}  (n={len(pos)})")
if neg: print(f"avg supplier next-month return AFTER - customer shock: {sum(neg)/len(neg):+.2%}  (n={len(neg)})")
if pos and neg:
    spread = sum(pos)/len(pos) - sum(neg)/len(neg)
    print(f"CF directional spread (should be POSITIVE if effect present): {spread:+.2%}")
    print("\nDirectional sign is", "CONSISTENT with CF." if spread > 0 else "NOT consistent (noise or backwards).")
print("\nNote: 11 edges over 3 windows is a sanity check, not a t-stat. Full")
print("reproduction needs the universe + survivorship-free history (Phase 5).")
