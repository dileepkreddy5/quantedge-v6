"""Network-momentum score — the Cohen-Frazzini reproduction (Manual §5.2, Table 11).

network_momentum(C) = sum over neighbors N of:
    shock(N)                  # recent unpriced move in the neighbor
  * edge_materiality(C, N)    # revenue-weighted, NOT equal-weighted
  * distance_decay(tier)      # 1st > 2nd > 3rd order
  * centrality(C)             # position in the whole graph
  * attention_gate(C)         # UP-weight small/obscure names

Every term is from the literature (§5.2). The score is explainable by
construction: it can name WHICH neighbor, at WHAT revenue weight, drove it
(the explain() contract, Table 12).
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from quantedge.data.sources.base import Edge


@dataclass
class Neighbor:
    edge: Edge
    shock: float          # neighbor's recent unpriced return move


def _decay(tier: int, params: dict) -> float:
    d = params["network_graph"]["distance_decay"]
    return {1: d["tier_1"], 2: d["tier_2"], 3: d["tier_3"]}.get(tier, 0.0)


def network_momentum(company: str, neighbors: list[Neighbor], centrality: float,
                     attention_gate: float, params: dict):
    """Return (score, drivers) where drivers explains the score (Table 12)."""
    floor = params["network_graph"]["edge_materiality_floor_pct"] / 100.0
    score = 0.0
    drivers = []
    for nb in neighbors:
        # §8.1 + §22.3: only confirmed edges above the materiality floor score.
        if not nb.edge.confirmed:
            continue
        if nb.edge.weight < floor:
            continue
        contrib = (nb.shock
                   * nb.edge.weight
                   * _decay(nb.edge.tier, params)
                   * centrality
                   * attention_gate)
        score += contrib
        drivers.append({
            "neighbor": nb.edge.to_ticker,
            "revenue_weight": nb.edge.weight,
            "tier": nb.edge.tier,
            "shock": nb.shock,
            "contribution": contrib,
        })
    drivers.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return score, drivers


def explain(company: str, drivers: list[dict]) -> str:
    """Name the dominant driver in plain language (Manual Table 12)."""
    if not drivers:
        return f"{company}: no confirmed, material network catalyst."
    top = drivers[0]
    direction = "positive" if top["shock"] > 0 else "negative"
    return (f"{company} ranks on a {direction} shock from {top['neighbor']} "
            f"({top['revenue_weight']*100:.0f}% of revenue, tier {top['tier']}).")
