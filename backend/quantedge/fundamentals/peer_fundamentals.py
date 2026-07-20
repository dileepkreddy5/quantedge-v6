"""Compute peer fundamental factors from a bulk companyfacts blob.
Reads pit_from_bulk output {metric:[(fy,val,filed)]} and market_cap, returns fundamental
factors (margins, returns, growth, valuation, quality) for peer-relative percentile scoring.
Every factor is None if its inputs are missing — never fabricated."""
from .bulk_adapter import pit_from_bulk

def _latest(series):
    """latest (fy,val,filed) value from a metric series."""
    return series[-1][1] if series else None

def _prior(series):
    """prior-year value (second-to-last)."""
    return series[-2][1] if series and len(series)>=2 else None

def _safe_div(a,b):
    if a is None or b is None or b==0: return None
    return a/b

def compute_peer_fundamentals(facts: dict, market_cap=None) -> dict:
    """Returns a dict of fundamental factors. Missing inputs -> factor omitted (not faked)."""
    if not facts: return {}
    pit = pit_from_bulk(facts)
    f = {}

    rev = _latest(pit.get("revenue",[])); rev_prior = _prior(pit.get("revenue",[]))
    ni = _latest(pit.get("net_income",[])); ni_prior = _prior(pit.get("net_income",[]))
    gp = _latest(pit.get("gross_profit",[]))
    assets = _latest(pit.get("assets",[])); liab = _latest(pit.get("liabilities",[]))
    ca = _latest(pit.get("cur_assets",[])); cl = _latest(pit.get("cur_liab",[]))
    ocf = _latest(pit.get("op_cash_flow",[]))
    equity = (assets - liab) if (assets is not None and liab is not None) else None

    # ---- Margins ----
    f["fund_gross_margin"] = _safe_div(gp, rev)
    f["fund_net_margin"] = _safe_div(ni, rev)
    f["fund_ocf_margin"] = _safe_div(ocf, rev)

    # ---- Returns ----
    f["fund_roa"] = _safe_div(ni, assets)
    f["fund_roe"] = _safe_div(ni, equity)
    # ROIC approx: NI / (equity + total liabilities-as-invested-capital proxy). Use NI/(equity+liab) as rough ROIC
    f["fund_roic_approx"] = _safe_div(ni, (equity + liab) if (equity is not None and liab is not None) else None)

    # ---- Growth (YoY) ----
    if rev is not None and rev_prior not in (None,0):
        f["fund_revenue_growth"] = rev/rev_prior - 1
    if ni is not None and ni_prior not in (None,0) and ni_prior>0:
        f["fund_earnings_growth"] = ni/ni_prior - 1

    # ---- Valuation (needs market_cap) ----
    if market_cap:
        f["fund_pe"] = _safe_div(market_cap, ni) if (ni and ni>0) else None
        f["fund_ps"] = _safe_div(market_cap, rev)
        f["fund_pb"] = _safe_div(market_cap, equity) if (equity and equity>0) else None
        f["fund_ocf_yield"] = _safe_div(ocf, market_cap)
        f["fund_earnings_yield"] = _safe_div(ni, market_cap) if (ni and ni>0) else None

    # ---- Quality ----
    f["fund_current_ratio"] = _safe_div(ca, cl)
    f["fund_asset_turnover"] = _safe_div(rev, assets)

    return {k:v for k,v in f.items() if v is not None}

if __name__=="__main__":
    # test on real MSFT bulk facts (CIK 0000789019)
    from .edgar_bulk import company_facts_from_bulk, download_bulk
    import sys
    download_bulk()  # ensure bulk file present
    facts = company_facts_from_bulk("0000789019")  # MSFT
    if not facts:
        print("MSFT facts not found in bulk - is companyfacts.zip present?"); sys.exit(1)
    ff = compute_peer_fundamentals(facts, market_cap=3.5e12)
    print(f"MSFT fundamental factors: {len(ff)} computed")
    for k,v in sorted(ff.items()):
        print(f"  {k:24s} {round(v,4)}")
