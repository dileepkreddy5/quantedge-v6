"""Proof of the network-momentum score. Run from repo root:
PYTHONPATH=. python quantedge/research/002_network_momentum/_test_score.py"""
import os, sys, yaml
sys.path.insert(0, os.path.dirname(__file__))   # so 'score' resolves locally
from datetime import date
from score import Neighbor, network_momentum, explain
from quantedge.data.sources.base import Edge

params = yaml.safe_load(open("quantedge/params.yaml"))

# A confirmed, material customer link (28% of supplier revenue) with a +shock.
big = Neighbor(Edge("SUPPLIER","BIGCUST", weight=0.28, tier=1, source="10-K",
                    confirmed=True, as_of=date(2020,1,1)), shock=0.10)
# An UNCONFIRMED text-inferred edge — must be ignored (§8.1 hallucination guard).
ghost = Neighbor(Edge("SUPPLIER","RUMOR", weight=0.50, tier=1, source="text",
                      confirmed=False, as_of=date(2020,1,1)), shock=0.20)
# A confirmed but immaterial edge (3% < 10% floor) — must be ignored (§22.3).
tiny = Neighbor(Edge("SUPPLIER","SMALL", weight=0.03, tier=1, source="10-K",
                     confirmed=True, as_of=date(2020,1,1)), shock=0.30)

score, drivers = network_momentum("SUPPLIER", [big, ghost, tiny],
                                  centrality=1.0, attention_gate=1.5, params=params)
print(f"score = {score:.4f}")
print(f"drivers counted: {len(drivers)} (expect 1 — only the confirmed material edge)")
assert len(drivers) == 1, "FAIL: unconfirmed or immaterial edge leaked into the score"
assert drivers[0]["neighbor"] == "BIGCUST"
# 0.10 * 0.28 * 1.0(tier1) * 1.0(centrality) * 1.5(attention) = 0.042
assert abs(score - 0.042) < 1e-9, f"FAIL: score math wrong ({score})"
print("only the confirmed, material edge scored                  OK")
print("explanation:", explain("SUPPLIER", drivers))
assert "BIGCUST" in explain("SUPPLIER", drivers) and "28%" in explain("SUPPLIER", drivers)
print("score names its own driver (explain contract)            OK")

# Attention gate does its job: an obscure name (higher gate) scores higher.
s_obscure, _ = network_momentum("S",[big],centrality=1.0,attention_gate=2.0,params=params)
s_covered, _ = network_momentum("S",[big],centrality=1.0,attention_gate=1.0,params=params)
assert s_obscure > s_covered
print("attention gate up-weights the under-covered name         OK")

print("\nPASS — CF score: confirmed+material only, explainable, attention-gated.")
