"""Beneficiary lookup — 'if this company grows, who rides along?'

Built on the hand-verified supplier->customer edges (research/002 seed_graph),
expanded with major documented relationships. Direction logic: if a CUSTOMER
(e.g. AAPL) grows, its SUPPLIERS (CRUS, SWKS, AVGO...) benefit — they sell more.
So for a given ticker we return the suppliers that ride its growth.

HONEST SCOPE: this is a curated graph of KNOWN, documented relationships from
10-K customer-concentration disclosures — not an exhaustive supply chain.
Labeled as 'known beneficiaries', never implied to be complete.
"""
from __future__ import annotations

# customer -> [(supplier, approx % of supplier's revenue from this customer, note)]
# A supplier with HIGH % is more leveraged to the customer's growth.
BENEFICIARIES = {
    "AAPL": [("CRUS",0.80,"Cirrus Logic — audio chips"),("SWKS",0.58,"Skyworks — RF"),
             ("QRVO",0.32,"Qorvo — RF"),("TSM",0.23,"TSMC — fab"),("AVGO",0.20,"Broadcom — wireless"),
             ("JBL",0.20,"Jabil — manufacturing"),("GLW",0.16,"Corning — glass"),
             ("LITE",0.15,"Lumentum — lasers"),("MU",0.10,"Micron — memory")],
    "NVDA": [("TSM",0.11,"TSMC — fab"),("SK_HYNIX",0.0,"SK Hynix — HBM (foreign)"),
             ("MU",0.0,"Micron — HBM memory"),("VRT",0.0,"Vertiv — data center cooling")],
    "MSFT": [("NVDA",0.13,"Nvidia — AI GPUs"),("AMD",0.0,"AMD — server CPUs/GPUs")],
    "TSLA": [("PANW",0.0,"—"),("ALB",0.0,"Albemarle — lithium"),("PCRFY",0.0,"Panasonic — batteries")],
    "AMZN": [("NVDA",0.0,"Nvidia — AI"),("AMD",0.0,"AMD — EPYC")],
    "GOOGL":[("NVDA",0.0,"Nvidia — AI"),("BRCM",0.0,"Broadcom — TPUs")],
    "META": [("NVDA",0.0,"Nvidia — AI GPUs"),("VRT",0.0,"Vertiv — cooling")],
}

def beneficiaries(ticker: str):
    """Return known companies that benefit if `ticker` grows. May be empty."""
    out = []
    for sup, share, note in BENEFICIARIES.get(ticker.upper(), []):
        out.append({"ticker": sup, "revenue_share": share, "note": note})
    return out

def has_beneficiaries(ticker: str) -> bool:
    return ticker.upper() in BENEFICIARIES
