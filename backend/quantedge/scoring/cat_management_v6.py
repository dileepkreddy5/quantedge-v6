"""Management Intelligence catalog v6 — 9 categories, weight 6.00."""
def _s(id,label,field,weight,good,great,hib=True,status="live",evidence=""):
    return {"id":id,"label":label,"field":field,"weight":weight,"higher_is_better":hib,
            "good":good,"great":great,"status":status,"evidence":evidence}

CATEGORIES = {
 "capital_allocation":("Capital Allocation", 1.05,[
   _s("fcf_gen","FCF generation","fcf_generation",0.20,0.08,0.25,evidence="free cash flow / revenue"),
   _s("total_payout","Total payout yield","total_payout_yield",0.18,0.02,0.06,evidence="buybacks + dividends / mcap"),
   _s("buyback_yield","Buyback yield","buyback_yield",0.15,0.01,0.05,evidence="buybacks / market cap"),
   _s("roic_level","ROIC level","roic_level",0.22,0.08,0.20,evidence="return on invested capital"),
   _s("div_consistency","Dividend consistency","dividend_consistency",0.12,0.5,1.0,evidence="quarters paying dividend"),
   _s("div_growth","Dividend growth","dividend_growth",0.13,0.0,0.1,evidence="dividend growth YoY")]),
 "insider_activity":("Insider Activity", 0.90,[
   _s("buy_sell_ratio","Buy/sell txn ratio","insider_buy_sell_ratio",0.20,0.15,0.5,evidence="share of insider txns that are buys"),
   _s("buy_value_ratio","Buy value ratio","insider_buy_value_ratio",0.20,0.15,0.5,evidence="buy $ vs total insider $"),
   _s("net_value","Net insider value","insider_net_value_norm",0.15,-0.005,0.002,evidence="net buy value / mcap"),
   _s("cluster","Cluster buying","insider_cluster_buying",0.15,0,1,evidence="3+ insiders buying"),
   _s("officer_net","Officer net activity","insider_officer_net",0.12,-2,2,evidence="officer buys minus sells"),
   _s("any_buying","Any insider buying","insider_any_buying",0.10,0,1,evidence="at least one open-market buy"),
   _s("sell_pressure","Insider sell pressure","insider_sell_pressure",0.08,0.02,0.002,hib=False,evidence="sell value / mcap; lower better")]),
 "effectiveness":("Management Effectiveness", 0.95,[
   _s("margin_trend","Margin trend","margin_trend",0.28,-0.01,0.03,evidence="operating margin expansion"),
   _s("at_trend","Asset turnover trend","asset_turnover_trend",0.18,-0.02,0.05,evidence="asset efficiency improving"),
   _s("rev_consistency","Revenue consistency","revenue_consistency",0.20,0.5,0.85,evidence="steady revenue growth"),
   _s("roe_level","ROE level","roe_level",0.20,0.10,0.25,evidence="return on equity"),
   _s("margin_current","Current margin","margin_current",0.10,0.10,0.30,evidence="operating margin level"),
   _s("roic_trend","ROIC trend","roic_trend",0.16,-0.01,0.03,evidence="capital returns improving"),
   _s("incremental_roic","Incremental ROIC","incremental_roic",0.14,0.05,0.25,evidence="return on newly deployed capital")]),
 "alignment":("Shareholder Alignment", 0.75,[
   _s("share_change","Share count change","share_count_change",0.30,0.01,-0.02,hib=False,evidence="buyback (shrink) vs dilution"),
   _s("sbc_intensity","SBC intensity","sbc_intensity",0.25,0.08,0.01,hib=False,evidence="stock comp / revenue; lower better"),
   _s("net_buyback","Net buyback vs SBC","net_buyback_vs_sbc",0.25,0.0,0.03,evidence="real buybacks net of dilution"),
   ]),
 "execution":("Execution Quality", 0.65,[
   _s("cash_conv","Cash conversion","cash_conversion",0.30,0.7,1.2,evidence="OCF / net income"),
   _s("wc_ratio","Working capital ratio","working_capital_ratio",0.20,1.0,2.0,evidence="current ratio"),
   _s("fcf_stability","FCF margin stability","fcf_margin_stability",0.25,0.5,0.85,evidence="consistent cash generation"),
   _s("rev_growth","Revenue growth","revenue_growth_ttm",0.25,0.05,0.20,evidence="TTM revenue growth")]),
 "governance":("Governance & Discipline", 0.55,[
   _s("debt_disc","Debt discipline","debt_discipline",0.35,-0.1,0.05,evidence="controlled debt growth"),
   _s("reinvest","Reinvestment rate","reinvestment_rate",0.30,0.05,0.25,evidence="capex / OCF"),
   _s("payout_ratio","Payout ratio","payout_ratio",0.35,0.9,0.5,hib=False,evidence="payout / FCF; sustainable if <1")]),
 "profitability_quality":("Profitability Quality", 0.45,[
   _s("fcf_gen2","FCF margin","fcf_generation",0.40,0.08,0.25,evidence="free cash flow margin"),
   _s("roic2","ROIC","roic_level",0.35,0.08,0.20,evidence="capital returns"),
   _s("roe2","ROE","roe_level",0.25,0.10,0.25,evidence="equity returns")]),
}
CATEGORIES["earnings_quality"]=("Earnings Quality", 0.70,[
   _s("accrual_q","Accrual quality","accrual_quality",0.35,-0.05,0.05,evidence="low accruals = clean earnings"),
   _s("eq_consistency","Earnings-quality consistency","earnings_quality_consistency",0.35,0.5,0.85,evidence="OCF consistently >= NI"),
   _s("rd_eff","R&D efficiency","rd_efficiency",0.30,0.0,3.0,evidence="revenue growth per R&D dollar")])
MANAGEMENT_INTELLIGENCE={"label":"Management Intelligence","weight":6.0,"categories":CATEGORIES}

def management_rating(score):
    if score is None: return "Unrated"
    if score>=72: return "Excellent"
    if score>=58: return "Strong"
    if score>=44: return "Adequate"
    if score>=30: return "Weak"
    return "Poor"
