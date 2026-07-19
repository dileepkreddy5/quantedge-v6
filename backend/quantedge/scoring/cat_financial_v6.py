"""Financial Intelligence catalog v6 — finalized 12-category spec, weights sum to 18.00.
Each signal maps to a REAL feature from financial_features.py. Scoring: hybrid
percentile-vs-peers + absolute bands + floors/caps.
"""
def _s(id,label,field,weight,good,great,hib=True,status="live",
       floor=None,floor_score=30.0,cap=None,cap_score=85.0,peer_key=None,evidence=""):
    d={"id":id,"label":label,"field":field,"weight":weight,"higher_is_better":hib,
       "good":good,"great":great,"status":status,"evidence":evidence}
    if floor is not None: d["floor"]=floor; d["floor_score"]=floor_score
    if cap is not None: d["cap"]=cap; d["cap_score"]=cap_score
    if peer_key: d["peer_key"]=peer_key
    return d

CATEGORIES = {
 "income_statement": ("Income Statement Intelligence", 3.00, [
   _s("gross_margin","Gross margin","gross_margin",0.40,0.30,0.60,peer_key="gross_margin",evidence="gross profit / revenue TTM"),
   _s("operating_margin","Operating margin","operating_margin",0.40,0.10,0.30,peer_key="operating_margin",evidence="operating income / revenue TTM"),
   _s("net_margin","Net margin","net_margin",0.35,0.05,0.25,peer_key="net_margin",evidence="net income / revenue TTM"),
   _s("ebitda_margin","EBITDA margin","ebitda_margin",0.30,0.15,0.40,evidence="(op income + D&A) / revenue"),
   _s("gross_margin_stability","Gross margin stability","gross_margin_stability",0.30,0.6,0.95,evidence="1 - CoV gross margin (pricing power)"),
   _s("operating_margin_stability","Operating margin stability","operating_margin_stability",0.30,0.6,0.95,evidence="1 - CoV operating margin"),
   _s("gross_margin_trend","Gross margin trend","gross_margin_trend",0.25,0.0,0.01,evidence="slope of gross margin over time"),
   _s("operating_margin_trend","Operating margin trend","operating_margin_trend",0.25,0.0,0.01,evidence="slope of operating margin"),
   _s("cogs_ratio","COGS efficiency","cogs_ratio",0.25,0.70,0.35,hib=False,evidence="cost of revenue / revenue"),
   _s("rd_intensity","R&D intensity","rd_intensity",0.20,0.03,0.15,evidence="R&D / revenue (innovation investment)"),
 ]),
 "cash_flow": ("Cash Flow Intelligence", 2.50, [
   _s("fcf_margin","Free cash flow margin","fcf_margin",0.45,0.05,0.25,evidence="(OCF - capex) / revenue"),
   _s("fcf_conversion","FCF conversion","fcf_conversion",0.40,0.7,1.0,evidence="FCF / net income"),
   _s("ocf_to_net_income","OCF / net income","ocf_to_net_income",0.35,0.9,1.2,evidence="operating cash flow / net income"),
   _s("owner_earnings_yield","Owner earnings yield","owner_earnings_yield",0.35,0.03,0.07,evidence="Buffett owner earnings / market cap"),
   _s("fcf_stability","FCF stability","fcf_stability",0.30,0.6,0.9,evidence="1 - CoV of FCF"),
   _s("capex_intensity","Capex intensity","capex_intensity",0.30,0.10,0.03,hib=False,evidence="capex / revenue (lower = asset-light)"),
   _s("fcf_growth","FCF growth","fcf_growth",0.35,0.05,0.20,evidence="YoY free cash flow growth"),
 ]),
 "balance_sheet": ("Balance Sheet Intelligence", 2.20, [
   _s("equity_ratio","Equity ratio","equity_ratio",0.40,0.3,0.6,evidence="equity / total assets"),
   _s("goodwill_ratio","Goodwill ratio","goodwill_ratio",0.35,0.40,0.10,hib=False,evidence="goodwill / assets (acquisition reliance)"),
   _s("asset_turnover","Asset turnover","asset_turnover",0.40,0.5,1.2,evidence="revenue / assets"),
   _s("current_ratio","Current ratio","current_ratio",0.35,1.2,2.0,cap=3.0,cap_score=80,evidence="current assets / current liabilities"),
   _s("cash_ratio","Cash ratio","cash_ratio",0.35,0.3,0.8,evidence="cash / current liabilities"),
   _s("dso","Days sales outstanding","dso",0.35,60,30,hib=False,evidence="receivables collection days"),
 ]),
 "profitability": ("Profitability Intelligence", 2.00, [
   _s("roic","Return on invested capital","roic",0.45,0.08,0.20,peer_key="roic",floor=0.0,floor_score=25,evidence="NOPAT / invested capital"),
   _s("roic_ex_goodwill","ROIC ex-goodwill","roic_ex_goodwill",0.35,0.10,0.25,evidence="true return on tangible capital"),
   _s("roic_wacc_spread","ROIC - WACC spread","roic_wacc_spread",0.45,0.0,0.15,floor=0.0,floor_score=30,evidence="value creation test (ROIC above cost of capital)"),
   _s("roe","Return on equity","roe",0.30,0.10,0.25,peer_key="roe",evidence="net income / equity (DuPont)"),
   _s("roa","Return on assets","roa",0.25,0.03,0.12,evidence="net income / assets"),
   _s("roic_stability","ROIC stability","roic_stability",0.20,0.6,0.9,evidence="1 - CoV of ROIC (moat durability)"),
 ]),
 "capital_allocation": ("Capital Allocation Intelligence", 1.80, [
   _s("reinvestment_rate","Reinvestment rate","reinvestment_rate",0.35,0.2,0.5,evidence="capex / operating cash flow"),
   _s("roic_trend","ROIC trend","roic_trend",0.40,0.0,0.005,evidence="direction of ROIC (allocation skill)"),
   _s("shareholder_yield","Shareholder yield","shareholder_yield",0.40,0.02,0.06,evidence="(dividends + buybacks) / market cap"),
   _s("share_count_trend","Share count trend","share_count_trend",0.35,0.0,-0.02,hib=False,evidence="diluted share change (buybacks reduce)"),
   _s("sbc_dilution_ratio","SBC dilution","sbc_dilution_ratio",0.30,0.15,0.03,hib=False,evidence="stock comp / net income"),
 ]),
 "financial_health": ("Financial Health Intelligence", 1.50, [
   _s("altman_z","Altman Z-score","altman_z",0.45,1.8,3.5,cap=6.0,cap_score=95,evidence="bankruptcy risk composite"),
   _s("piotroski_f","Piotroski F-score","piotroski_f",0.45,4,8,evidence="9-point fundamental quality"),
   _s("net_debt_to_ebitda","Net debt / EBITDA","net_debt_to_ebitda",0.35,2.5,0.5,hib=False,evidence="(debt - cash) / EBITDA"),
   _s("debt_to_ebitda","Debt / EBITDA","debt_to_ebitda",0.25,3.0,1.0,hib=False,evidence="leverage vs earnings"),
 ]),
 "liquidity_solvency": ("Liquidity & Solvency Intelligence", 1.20, [
   _s("quick_ratio","Quick ratio","quick_ratio",0.35,1.0,1.5,evidence="(current assets - inventory) / current liab"),
   _s("debt_to_equity","Debt to equity","debt_to_equity",0.35,1.0,0.3,hib=False,evidence="total debt / equity"),
   _s("cash_conversion_cycle","Cash conversion cycle","cash_conversion_cycle",0.30,60,0,hib=False,evidence="DSO + DIO - DPO (working capital efficiency)"),
   _s("dpo","Days payable","dpo",0.20,30,60,evidence="supplier financing days"),
 ]),
 "accounting_quality": ("Accounting Quality Intelligence", 1.00, [
   _s("beneish_m","Beneish M-score","beneish_m",0.40,-1.78,-2.5,hib=False,evidence="earnings manipulation probability"),
   _s("accruals_ratio","Accruals ratio","accruals_ratio",0.35,0.05,-0.05,hib=False,evidence="(NI - OCF) / assets (Sloan)"),
   _s("cash_earnings_gap","Cash-earnings gap","cash_earnings_gap",0.25,0.2,0.05,hib=False,evidence="gap between earnings and cash"),
 ]),
 "earnings_quality": ("Earnings Quality Intelligence", 0.90, [
   _s("earnings_quality","Earnings quality (OCF/NI)","earnings_quality",0.40,0.8,1.1,evidence="operating cash flow backs earnings"),
   _s("net_margin_stability","Net margin stability","net_margin_stability",0.30,0.6,0.95,evidence="1 - CoV of net margin"),
   _s("effective_tax_rate","Effective tax rate","effective_tax_rate",0.20,0.30,0.15,hib=False,evidence="normalized tax rate"),
 ]),
 "growth": ("Growth Intelligence", 0.80, [
   _s("revenue_growth","Revenue growth YoY","revenue_growth",0.30,0.05,0.20,peer_key="rev_growth",floor=-0.05,floor_score=25,evidence="TTM revenue growth"),
   _s("revenue_cagr_3y","Revenue 3y CAGR","revenue_cagr_3y",0.25,0.05,0.15,evidence="3-year revenue CAGR"),
   _s("earnings_cagr_3y","Earnings 3y CAGR","earnings_cagr_3y",0.25,0.05,0.15,evidence="3-year net income CAGR"),
   _s("revenue_stability","Revenue stability","revenue_stability",0.20,0.6,0.9,evidence="consistency of revenue growth"),
 ]),
 "efficiency": ("Efficiency Intelligence", 0.60, [
   _s("dio","Days inventory","dio",0.30,90,30,hib=False,evidence="inventory days"),
   _s("asset_turnover_eff","Asset turnover","asset_turnover",0.35,0.5,1.2,evidence="revenue / assets"),
   _s("equity_multiplier","Equity multiplier","equity_multiplier",0.20,3.0,1.5,hib=False,evidence="assets / equity (leverage)"),
 ]),
 "shareholder_returns": ("Shareholder Returns Intelligence", 0.50, [
   _s("buyback_yield","Buyback yield","buyback_yield",0.30,0.0,0.04,evidence="net buybacks / market cap"),
   _s("dividend_yield","Dividend yield","dividend_yield",0.20,0.0,0.03,evidence="dividends / market cap"),
   _s("dividend_coverage","Dividend coverage","dividend_coverage",0.20,1.5,3.0,evidence="FCF / dividends"),
 ]),
}
FINANCIAL_INTELLIGENCE = {"label":"Financial Intelligence","weight":18.0,"categories":CATEGORIES}
