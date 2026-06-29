"""Enrich the top-100 names with price ladder + volume (only ~300 names, cheap).

The bulk scan skips per-ticker price/volume for speed (it scores thousands).
But the SURVIVING top 100/tier = 300 names — cheap to enrich with Polygon
per-ticker calls, gently rate-limited. Adds the ladder and volume the UI shows.
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from quantedge.fundamentals.price_ladder import price_ladder
from quantedge.fundamentals.volume_signal import volume_signal

PATH = os.path.join(os.path.dirname(__file__), "..", "..", "backend", "research_data", "scan_artifact.json")

def enrich():
    d = json.load(open(PATH))
    total = sum(len(d["tiers"][t]) for t in d["tiers"])
    done = 0
    for tier in d["tiers"]:
        for r in d["tiers"][tier]:
            tk = r["ticker"]
            try:
                r["price_ladder"] = price_ladder(tk)
                r.update(volume_signal(tk))
                r.setdefault("quiet_price", None)
                pl = r["price_ladder"]
                # quiet if 6-mo-ish (3m) move is modest
                m3 = pl.get("3m")
                r["quiet_price"] = (m3 is not None and m3 < 0.20)
                r["price_move_6mo"] = m3
            except Exception:
                r["price_ladder"] = {}
            done += 1
            if done % 50 == 0:
                print(f"enriched {done}/{total}", flush=True)
            time.sleep(0.08)  # gentle on Polygon
    json.dump(d, open(PATH, "w"), indent=2)
    print(f"DONE — enriched {done} names with ladder + volume")

if __name__ == "__main__":
    enrich()
