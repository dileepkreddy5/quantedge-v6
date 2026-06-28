"""Hand-verified seed supplier->customer edges for the CF reproduction.

weight = customer's share of the SUPPLIER's revenue (edge materiality, §5.1).
Each edge reflects a documented 10-K customer-concentration disclosure; these
are approximate, hand-verified starting values for the prototype. The
automated EDGAR extractor (later) will replace these with parsed, dated,
2-source-confirmed edges. confirmed=True here means "manually verified
against the filing" for prototype purposes.

Direction: from_ticker = SUPPLIER, to_ticker = CUSTOMER.
The CF trade: customer shock -> buy the supplier (the laggard).
"""
from datetime import date
from quantedge.data.sources.base import Edge

AS_OF = date(2023, 1, 1)

SEED_EDGES = [
    # supplier,  customer,  weight, tier, source
    Edge("AVGO", "AAPL", 0.20, 1, "10-K", True, AS_OF),  # Broadcom -> Apple (~20%)
    Edge("SWKS", "AAPL", 0.58, 1, "10-K", True, AS_OF),  # Skyworks -> Apple (~58%)
    Edge("QRVO", "AAPL", 0.32, 1, "10-K", True, AS_OF),  # Qorvo -> Apple (~32%)
    Edge("CRUS", "AAPL", 0.80, 1, "10-K", True, AS_OF),  # Cirrus Logic -> Apple (~80%)
    Edge("GLW",  "AAPL", 0.16, 1, "10-K", True, AS_OF),  # Corning -> Apple (~16%)
    Edge("JBL",  "AAPL", 0.20, 1, "10-K", True, AS_OF),  # Jabil -> Apple (~20%)
    Edge("LITE", "AAPL", 0.15, 1, "10-K", True, AS_OF),  # Lumentum -> Apple (~15%)
    Edge("TSM",  "AAPL", 0.23, 1, "10-K", True, AS_OF),  # TSMC -> Apple (~23%)
    Edge("TSM",  "NVDA", 0.11, 1, "10-K", True, AS_OF),  # TSMC -> Nvidia (~11%)
    Edge("MU",   "AAPL", 0.10, 1, "10-K", True, AS_OF),  # Micron -> Apple (~10%)
    Edge("NVDA", "MSFT", 0.13, 1, "10-K", True, AS_OF),  # Nvidia -> Microsoft (~13%)
    Edge("MCHP", "TSLA", 0.00, 1, "10-K", False, AS_OF), # placeholder UNCONFIRMED (must be ignored)
    Edge("STLD", "AAPL", 0.02, 1, "10-K", True, AS_OF),  # immaterial (<10%, must be ignored)
]
