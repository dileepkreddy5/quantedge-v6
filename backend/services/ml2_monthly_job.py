"""Monthly ML v2 refresh: rebuild the point-in-time panel, risk targets, retrain and re-forecast."""
async def run_ml2_monthly(pool):
    from quantedge.ml2.panel_prices import build_price_panel
    from quantedge.ml2.fund_history import build_fund_history
    from quantedge.ml2.panel_fund import build_fund_panel
    from quantedge.ml2.risk import add_risk_targets
    from quantedge.ml2.serve import build_serving
    a = await build_price_panel(pool); b = await build_fund_history(pool); c = await build_fund_panel(pool)
    d = await add_risk_targets(pool); e = await build_serving(pool)
    return {"panel_rows": a["rows"], "fund_companies": b["with_quarters"], "served": e}
