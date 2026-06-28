"""Generate the cap-tier scan artifact the live tab will serve.
Run: export POLYGON=...; PYTHONPATH=. python this_file"""
import os, sys, json
from datetime import datetime
sys.path.insert(0, '.')
from quantedge.fundamentals.universe_tiers import sec_ticker_cik_map
from quantedge.fundamentals.scanner import scan_batch

cikmap = sec_ticker_cik_map()

# A real curated starter set per tier (scaling to full universe is the scan job).
# Tiers reflect current rough cap; the scanner stores the live read.
TIERS = {
  "small": ["RKLB","IONQ","HIMS","CAVA","APP","RDDT","ASTS","OKLO","SMR","TMDX"],
  "mid":   ["CELH","SOFI","DKNG","ROKU","U","AFRM","TOST","RIVN","CHWY","ELF"],
  "large": ["NVDA","SMCI","PLTR","AMD","AAPL","MSFT","AVGO","CRWD","NOW","PANW"],
}

out = {"generated": datetime.utcnow().isoformat()+"Z", "tiers": {},
       "disclaimer": "Shortlist filter ranked by quarterly growth + quiet price. "
                     "NOT a predictor; tail inflections are unpredictable from "
                     "prior fundamentals (proven by backtest). Not investment advice."}

for tier, tickers in TIERS.items():
    items = [(t, cikmap[t]) for t in tickers if t in cikmap]
    ranked = scan_batch(items)
    out["tiers"][tier] = ranked
    print(f"{tier}: scored {len(ranked)} names")

path = "backend/research_data/scan_artifact.json"
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(out, open(path, "w"), indent=2)
print("wrote", path)
