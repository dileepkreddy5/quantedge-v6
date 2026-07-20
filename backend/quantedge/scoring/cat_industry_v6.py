"""Industry Intelligence catalog v6 — 10 categories, weight 6.00, 39 signals.
Sector classification/scale, sector-relative performance, sector momentum/cycle, industry
position (peer-bucket percentiles), RS leadership, sector risk, maturity. 2 needs_source
(fundamental peer valuation) until peer-fundamental enrichment."""
def _s(id,label,field,weight,good,great,hib=True,status="live",evidence=""):
    return {"id":id,"label":label,"field":field,"weight":weight,"higher_is_better":hib,
            "good":good,"great":great,"status":status,"evidence":evidence}

CATEGORIES = {
 "sector_class": ("Sector Classification & Scale", 0.50, [
   _s("size_tier","Size tier","size_tier",0.30,1,3,evidence="large(3)/mid(2)/small(1) cap"),
   _s("mcap_b","Market cap ($B)","market_cap_b",0.25,5,100,evidence="absolute scale"),
   _s("peer_count","Sector peer count","sector_peer_count",0.20,20,100,evidence="sector breadth in coverage"),
   _s("employees","Employee scale","employee_scale",0.25,5000,100000,evidence="workforce size"),
 ]),
 "sector_rel_perf": ("Sector-Relative Performance", 0.90, [
   _s("rs_3m","RS vs sector (3m)","rs_sector_3m",0.25,-0.05,0.15,evidence="outperformance vs sector ETF"),
   _s("rs_6m","RS vs sector (6m)","rs_sector_6m",0.20,-0.05,0.2,evidence="6-month relative strength"),
   _s("rs_1y","RS vs sector (1y)","rs_sector_1y",0.15,-0.1,0.3,evidence="1-year relative strength"),
   _s("rs_1m","RS vs sector (1m)","rs_sector_1m",0.12,-0.03,0.08,evidence="1-month relative strength"),
   _s("up_capture","Up-capture vs sector","up_capture",0.15,0.9,1.3,evidence="gains vs sector in up moves"),
   _s("down_capture","Down-capture vs sector","down_capture",0.13,1.1,0.7,hib=False,evidence="losses vs sector in down moves"),
 ]),
 "sector_momentum": ("Sector Momentum & Cycle", 0.80, [
   _s("sec_trend_3m","Sector trend (3m)","sector_trend_3m",0.25,-0.05,0.1,evidence="sector ETF 3-month trend"),
   _s("sec_trend_6m","Sector trend (6m)","sector_trend_6m",0.20,-0.05,0.15,evidence="sector ETF 6-month trend"),
   _s("sec_vs_spy","Sector vs market (3m)","sector_vs_spy_3m",0.25,-0.05,0.05,evidence="sector rotation vs SPY"),
   _s("sec_52w","Sector 52w position","sector_52w_position",0.15,0.3,0.8,evidence="sector within its yearly range"),
   _s("in_favor","Sector in favor","sector_in_favor",0.15,0,1,evidence="positive trend + outperforming market"),
 ]),
 "industry_position": ("Industry Position", 0.90, [
   _s("mom_pct","Momentum percentile","momentum_pctile_sector",0.20,0.4,0.8,evidence="3m momentum rank vs sector peers"),
   _s("sharpe_pct","Sharpe percentile","sharpe_pctile_sector",0.18,0.4,0.8,evidence="risk-adjusted return rank in sector"),
   _s("quality_pct","Quality percentile","quality_pctile_sector",0.15,0.4,0.8,evidence="Piotroski rank vs sector"),
   _s("mom12_pct","12m momentum percentile","mom12_pctile_sector",0.12,0.4,0.8,evidence="long momentum rank in sector"),
   _s("voladj_pct","Vol-adj return percentile","vol_adj_pctile_sector",0.08,0.4,0.8,evidence="vol-adjusted return rank"),
   _s("mom1m_pct","1m momentum percentile","mom1m_pctile_sector",0.07,0.4,0.8,evidence="short momentum rank in sector"),
   _s("mom6m_pct","6m momentum percentile","mom6m_pctile_sector",0.08,0.4,0.8,evidence="6-month momentum rank"),
   _s("altman_pct","Altman-Z percentile","altman_pctile_sector",0.07,0.4,0.8,evidence="solvency rank vs sector"),
 ]),
 "sector_structure": ("Sector Structure & Rank", 0.30, [
   _s("composite_rank","Composite sector rank","composite_sector_rank",0.30,0.4,0.8,evidence="average percentile across all factors"),
   _s("top_quartile","Top-quartile factor share","top_quartile_flags",0.20,0.2,0.6,evidence="share of factors in sector top quartile"),
   _s("bottom_quartile","Bottom-quartile share","bottom_quartile_flags",0.15,0.4,0.1,hib=False,evidence="share of factors in sector bottom quartile"),
   _s("mom_vs_avg","Momentum vs sector avg","momentum_vs_sector_avg",0.20,-2,8,evidence="momentum above sector average"),
   _s("obv_pct","OBV accumulation percentile","obv_pctile_sector",0.08,0.4,0.8,evidence="accumulation rank vs sector"),
   _s("ma200_pct","Trend (MA200) percentile","ma200_pctile_sector",0.07,0.4,0.8,evidence="long-trend rank vs sector"),
 ]),
 "rs_leadership": ("RS Leadership", 0.70, [
   _s("is_leader","Sector leader","is_sector_leader",0.30,0,1,evidence="outperforming sector"),
   _s("rs_mag","RS magnitude","rs_magnitude",0.25,-0.05,0.15,evidence="size of outperformance"),
   _s("lead_rank","Leadership rank","leadership_rank",0.25,0.4,0.85,evidence="momentum percentile in sector"),
   _s("beta_sector","Beta to sector","beta_to_sector",0.20,1.5,0.9,hib=False,evidence="sensitivity to sector moves"),
 ]),
 "sector_risk": ("Sector Risk & Cyclicality", 0.60, [
   _s("sec_vol","Sector volatility","sector_volatility",0.30,0.35,0.15,hib=False,evidence="sector ETF volatility"),
   _s("sec_dd","Sector drawdown","sector_drawdown",0.25,-0.3,-0.1,evidence="sector recent max drawdown"),
   _s("sec_beta_mkt","Sector beta to market","sector_beta_to_market",0.20,1.4,0.8,hib=False,evidence="sector systematic risk"),
   _s("defensive","Defensive sector","is_defensive",0.15,0,1,evidence="staples/utilities/healthcare"),
   _s("corr_sector","Correlation to sector","correlation_to_sector",0.10,0.9,0.5,hib=False,evidence="independence from sector"),
 ]),
 "maturity": ("Maturity & Lifecycle", 0.30, [
   _s("years_public","Years public","years_public",0.40,2,25,evidence="time since IPO"),
   _s("cap_eff","Capital efficiency","capital_efficiency",0.35,2,20,evidence="market cap per employee ($M)"),
   _s("mcap_emp","Mcap per employee","mcap_per_employee_m",0.25,2,20,evidence="scale efficiency"),
 ]),
 "sector_valuation": ("Sector Valuation Context", 0.40, [
   _s("rel_pe","Relative P/E vs sector","rel_pe_vs_sector",0.50,1.2,0.8,hib=False,status="needs_source",evidence="P/E vs sector median (needs peer fundamentals)"),
   _s("rel_growth","Relative growth vs sector","rel_growth_vs_sector",0.50,-0.05,0.1,status="needs_source",evidence="growth vs sector median (needs peer fundamentals)"),
 ]),
 "sector_flows": ("Sector Flows", 0.40, [
   _s("sec_mom_1m","Sector momentum (1m)","sector_momentum_1m",0.40,-0.03,0.05,evidence="recent sector direction"),
   _s("sec_accel","Sector acceleration","sector_acceleration",0.35,-0.03,0.03,evidence="sector momentum accelerating"),
   _s("sec_rotation","Sector rotation","sector_vs_spy_3m",0.25,-0.05,0.05,evidence="flows into/out of sector"),
 ]),
 "geo_structural": ("Geographic & Structural", 0.20, [
   _s("cap_eff2","Capital efficiency","capital_efficiency",0.50,2,20,evidence="mcap per employee"),
   _s("size2","Size tier","size_tier",0.50,1,3,evidence="market-cap tier"),
 ]),
}
INDUSTRY_INTELLIGENCE = {"label":"Industry Intelligence","weight":6.0,"categories":CATEGORIES}

def industry_rating(score):
    if score is None: return "Unrated"
    if score>=72: return "Sector Leader"
    if score>=58: return "Above Sector"
    if score>=42: return "In-Line"
    if score>=28: return "Below Sector"
    return "Sector Laggard"
