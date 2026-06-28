"""Generate the research artifact the /research page serves.
Runs the CF directional check and writes a JSON the backend reads.
Run: export POLYGON=...; PYTHONPATH=. python this_file"""
import os, sys, json
from datetime import date, datetime
sys.path.insert(0, os.path.dirname(__file__))
from seed_graph import SEED_EDGES
from quantedge.data.sources.polygon_prices import daily_closes

WINDOWS = [
    ("2023-05", date(2023,5,1), date(2023,5,31), date(2023,6,1), date(2023,6,30)),
    ("2023-09", date(2023,9,1), date(2023,9,30), date(2023,10,1),date(2023,10,31)),
    ("2024-01", date(2024,1,1), date(2024,1,31), date(2024,2,1), date(2024,2,29)),
]
cache = {}
def ret(t,a,b):
    k=(t,a,b)
    if k not in cache:
        c=daily_closes(t,a,b); cache[k]=(c[-1][1]/c[0][1]-1.0) if len(c)>=2 else None
    return cache[k]

edges=[e for e in SEED_EDGES if e.confirmed and e.weight>=0.10]
rows=[]
for label,sa,sb,na,nb in WINDOWS:
    for e in edges:
        cs=ret(e.to_ticker,sa,sb); sn=ret(e.from_ticker,na,nb)
        if cs is None or sn is None: continue
        rows.append({"window":label,"supplier":e.from_ticker,"customer":e.to_ticker,
                     "weight":e.weight,"customer_shock":round(cs,4),"supplier_next":round(sn,4)})

pos=[r["supplier_next"] for r in rows if r["customer_shock"]>0]
neg=[r["supplier_next"] for r in rows if r["customer_shock"]<=0]
avg=lambda x: round(sum(x)/len(x),4) if x else None
artifact={
  "unit":"network.customer_momentum","status":"research","generated":datetime.utcnow().isoformat()+"Z",
  "title":"Cohen-Frazzini customer-momentum — directional check",
  "mechanism":"Investors are inattentive to economic links; a customer shock predicts the supplier's lagged return.",
  "n_edges":len(edges),"n_observations":len(rows),
  "avg_supplier_after_pos_shock":avg(pos),"avg_supplier_after_neg_shock":avg(neg),
  "directional_spread":round((avg(pos)-avg(neg)),4) if pos and neg else None,
  "rows":rows,
  "honest_caveats":[
    "NOT a tradable edge and NOT a reproduced t-stat.",
    "Windows are not independent — most edges share AAPL as customer (~3 real observations).",
    "Effect is confounded with semiconductor sector beta; link-specific alpha not isolated.",
    "Windows were chosen post hoc — a researcher degree of freedom.",
    "Full validation needs the universe + survivorship-free history + pre-committed windows (Phase 5)."
  ],
  "promotion":"Unit is status: research. It has NOT passed the gate. Not investment advice."
}
out=os.path.join(os.path.dirname(__file__),"cf_artifact.json")
json.dump(artifact,open(out,"w"),indent=2)
print("wrote",out)
print("spread:",artifact["directional_spread"],"| obs:",artifact["n_observations"])
