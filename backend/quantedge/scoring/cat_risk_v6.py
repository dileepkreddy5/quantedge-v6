"""Risk Intelligence catalog v6 — 10 categories, weight 6.00, 43 signals.
Credit models (Altman/Ohlson), leverage, liquidity, earnings-quality (Sloan/Beneish),
tail risk, volatility structure, business-model, valuation, governance, macro.
Risk direction encoded per signal: high Altman-Z = safe (hib=True); high leverage = risky (hib=False)."""
def _s(id,label,field,weight,good,great,hib=True,status="live",evidence=""):
    return {"id":id,"label":label,"field":field,"weight":weight,"higher_is_better":hib,
            "good":good,"great":great,"status":status,"evidence":evidence}

CATEGORIES = {
 "bankruptcy": ("Bankruptcy & Solvency Risk", 1.10, [
   _s("altman_z","Altman Z-score","altman_z",0.30,3.0,6.0,evidence=">3 safe, <1.8 distress zone"),
   _s("bankruptcy_prob","Bankruptcy probability","bankruptcy_prob",0.25,0.10,0.01,hib=False,evidence="Ohlson-derived default probability"),
   _s("ohlson_o","Ohlson O-score","ohlson_o",0.15,0.0,-4.0,hib=False,evidence="higher = more distress risk"),
   _s("ebit_to_debt","EBIT-to-debt coverage","ebit_to_debt",0.15,0.3,2.0,evidence="operating income vs total debt"),
   _s("piotroski","Piotroski health (partial)","piotroski_partial",0.15,1,3,evidence="fundamental health checks passed"),
 ]),
 "leverage": ("Financial Leverage Risk", 0.80, [
   _s("debt_to_equity","Debt-to-equity","debt_to_equity",0.22,1.0,0.2,hib=False,evidence="leverage vs equity"),
   _s("net_debt_ebitda","Net debt / EBITDA","net_debt_to_ebitda",0.22,3.5,0.5,hib=False,evidence="years to repay debt"),
   _s("debt_ebitda","Debt / EBITDA","debt_to_ebitda",0.18,4.0,1.0,hib=False,evidence="gross leverage"),
   _s("equity_ratio","Equity ratio","equity_ratio",0.20,0.3,0.6,evidence="equity cushion vs assets"),
   _s("leverage_trend","Leverage trend","leverage_trend",0.18,0.05,-0.02,hib=False,evidence="debt/assets change YoY"),
 ]),
 "liquidity": ("Liquidity Risk", 0.60, [
   _s("current_ratio","Current ratio","current_ratio",0.25,1.0,2.5,evidence="current assets vs liabilities"),
   _s("quick_ratio","Quick ratio","quick_ratio",0.22,0.7,1.8,evidence="liquid assets vs current liabilities"),
   _s("cash_ratio","Cash ratio","cash_ratio",0.18,0.2,1.0,evidence="cash vs current liabilities"),
   _s("cash_to_debt","Cash-to-debt","cash_to_debt",0.20,0.2,1.5,evidence="cash coverage of total debt"),
   _s("wc_ratio","Working-capital ratio","working_capital_ratio",0.15,0.0,0.2,evidence="net working capital vs assets"),
 ]),
 "earnings_quality": ("Earnings Quality Risk", 0.70, [
   _s("sloan","Sloan accruals","sloan_accruals",0.25,0.1,-0.05,hib=False,evidence="(NI-OCF)/assets; high=low quality"),
   _s("cash_conversion","Cash conversion","cash_conversion",0.22,0.7,1.2,evidence="OCF/NI; low=earnings not cash-backed"),
   _s("earn_vol","Earnings volatility","earnings_volatility",0.20,0.8,0.2,hib=False,evidence="quarterly NI variability"),
   _s("beneish_dsri","Beneish DSRI","beneish_dsri",0.16,1.5,1.0,hib=False,evidence="receivables index; >1.4 manipulation flag"),
   # beneish_gmi was computed and then discarded — no catalog entry, so the
   # margin-deterioration half of the Beneish screen never reached a score.
   _s("beneish_gmi","Beneish GMI","beneish_gmi",0.12,1.2,1.0,hib=False,evidence="gross margin index; >1.2 means margins deteriorating year on year"),
   _s("rev_ocf_div","Revenue-OCF divergence","revenue_ocf_divergence",0.17,0.2,-0.05,hib=False,evidence="revenue up but cash flow down"),
 ]),
 "tail_risk": ("Market & Tail Risk", 0.70, [
   _s("max_dd","Max drawdown","max_drawdown",0.25,-0.5,-0.15,evidence="worst peak-to-trough (closer to 0=safer)"),
   _s("cvar","5% CVaR","cvar_5pct",0.22,-0.06,-0.02,evidence="expected loss in worst 5% of days"),
   _s("worst_month","Worst month","worst_month",0.20,-0.35,-0.1,evidence="worst 21-day return"),
   _s("beta","Beta (magnitude)","abs_beta",0.18,1.5,0.8,hib=False,evidence="|beta| — distance from market-independent, either direction"),
   _s("worst_day","Worst day","worst_day",0.15,-0.1,-0.03,evidence="single worst daily return"),
 ]),
 "volatility": ("Volatility Structure", 0.55, [
   _s("ann_vol","Annualized volatility","annualized_vol",0.30,0.5,0.2,hib=False,evidence="return standard deviation"),
   _s("downside_dev","Downside deviation","downside_deviation",0.28,0.4,0.15,hib=False,evidence="volatility of losses only"),
   _s("semivar","Semi-variance","semivariance",0.22,0.08,0.02,hib=False,evidence="below-mean return dispersion"),
   _s("vol_cluster","Volatility clustering","vol_clustering",0.20,0.3,0.05,hib=False,evidence="persistence of volatility shocks"),
 ]),
 "business_model": ("Business Model Risk", 0.50, [
   _s("op_margin","Operating margin","operating_margin",0.25,0.05,0.3,evidence="profitability level"),
   _s("cap_intensity","Capital intensity","capital_intensity",0.22,0.2,0.05,hib=False,evidence="capex/revenue; high=capital-hungry"),
   _s("asset_turnover","Asset turnover","asset_turnover",0.23,0.3,1.0,evidence="revenue efficiency of assets"),
 ]),
 "valuation_risk": ("Valuation Risk", 0.75, [
   _s("val_downside","Valuation downside risk","valuation_downside_risk",0.35,0.5,0.15,hib=False,evidence="multiple-compression exposure"),
   _s("pe","P/E ratio","pe_ratio",0.30,30,12,hib=False,evidence="high P/E = more downside on miss"),
   _s("pb","Price-to-book","price_to_book",0.20,8,2,hib=False,evidence="premium to book value"),
   _s("earn_yield","Earnings yield","earnings_yield",0.15,0.02,0.06,evidence="inverse P/E; higher=cheaper"),
 ]),
 "governance": ("Governance & Dilution Risk", 0.15, [
   _s("dilution","Share dilution","share_dilution",0.35,0.03,-0.01,hib=False,evidence="share count growth YoY"),
   _s("sbc_intensity","SBC intensity","sbc_intensity",0.30,0.08,0.01,hib=False,evidence="stock comp / revenue"),
   _s("debt_buyback","Debt-funded buyback","debt_funded_buyback_flag",0.35,1,0,hib=False,evidence="buybacks while leverage rising"),
 ]),
 "macro": ("Macro Sensitivity", 0.15, [
   _s("rate_sens","Rate sensitivity","rate_sensitivity",0.35,0.4,0.1,hib=False,evidence="leverage-driven rate exposure"),
 ]),
}
RISK_INTELLIGENCE = {"label":"Risk Intelligence","weight":6.0,"categories":CATEGORIES}

def risk_rating(score):
    if score is None: return "Unrated"
    if score>=78: return "Low Risk"
    if score>=63: return "Moderate Risk"
    if score>=47: return "Elevated Risk"
    if score>=32: return "High Risk"
    return "Severe Risk"
