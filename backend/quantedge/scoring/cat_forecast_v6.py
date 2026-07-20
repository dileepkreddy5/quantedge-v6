"""Forecast Intelligence catalog v6 — 11 categories, weight 4.00, 53 signals."""
def _s(id,label,field,weight,good,great,hib=True,status="live",evidence=""):
    return {"id":id,"label":label,"field":field,"weight":weight,"higher_is_better":hib,
            "good":good,"great":great,"status":status,"evidence":evidence}

CATEGORIES = {
 "earnings_trajectory":("Earnings Trajectory",0.50,[
   _s("earn_accel","Earnings acceleration","earnings_accel",0.16,0.0,0.08,evidence="earnings growth 2nd derivative"),
   _s("earn_recent","Recent earnings growth","earnings_growth_recent",0.16,0.0,0.12,evidence="latest QoQ earnings growth"),
   _s("earn_stab","Earnings growth stability","earnings_growth_stability",0.14,0.5,0.85,evidence="consistent earnings"),
   _s("earn_slope","Earnings trend slope","earnings_trend_slope",0.14,0.0,0.08,evidence="earnings trajectory"),
   _s("earn_pos","Earnings positivity","earnings_positivity",0.10,0.6,1.0,evidence="profitable quarters"),
   _s("eps_recent","EPS growth recent","eps_growth_recent",0.16,0.0,0.1,evidence="per-share earnings growth"),
   _s("eps_accel","EPS acceleration","eps_accel",0.14,0.0,0.06,evidence="EPS growth accelerating")]),
 "revenue_trajectory":("Revenue Trajectory",0.45,[
   _s("rev_persist","Revenue persistence","revenue_growth_persistence",0.18,0.5,0.85,evidence="steady revenue growth"),
   _s("rev_accel","Revenue acceleration","revenue_accel",0.16,0.0,0.04,evidence="growth speeding up"),
   _s("rev_recent","Recent revenue growth","revenue_growth_recent",0.16,0.0,0.08,evidence="latest QoQ growth"),
   _s("rev_slope","Revenue trend slope","revenue_trend_slope",0.14,0.0,0.06,evidence="revenue trajectory"),
   _s("yoy_recent","YoY growth","yoy_growth_recent",0.12,0.0,0.15,evidence="year-over-year growth"),
   _s("yoy_accel","YoY acceleration","yoy_accel",0.12,0.0,0.03,evidence="YoY growth accelerating"),
   _s("stacked","2-year stacked growth","two_year_stacked_growth",0.12,0.1,0.4,evidence="base-effect normalized growth")]),
 "growth_quality":("Growth Quality",0.40,[
   _s("growth_consist","Growth consistency","growth_consistency",0.25,0.6,0.9,evidence="positive-growth quarters"),
   _s("yoy_stab","YoY stability","yoy_growth_stability",0.25,0.5,0.85,evidence="predictable seasonal pattern"),
   _s("cash_conv_trend","Cash conversion trend","cash_conversion_trend",0.25,0.0,0.05,evidence="earnings increasingly cash-backed"),
   _s("cash_conv_lvl","Cash conversion level","cash_conversion_level",0.25,0.8,1.2,evidence="OCF / net income")]),
 "operating_leverage":("Operating Leverage",0.45,[
   _s("inc_margin","Incremental op margin","incremental_op_margin",0.35,0.15,0.4,evidence="operating income per new $ revenue"),
   _s("op_leverage","Operating leverage","operating_leverage",0.25,1.0,2.0,evidence="%ΔOI / %Δrevenue"),
   _s("margin_traj","Op margin trajectory","op_margin_trajectory",0.20,0.0,0.005,evidence="margins expanding"),
   _s("gm_traj","Gross margin trajectory","gross_margin_trajectory",0.20,0.0,0.004,evidence="gross margin trend")]),
 "compounding":("Intrinsic Compounding",0.50,[
   _s("roiic","ROIIC","roiic",0.35,0.1,0.3,evidence="return on incremental invested capital"),
   _s("intrinsic","Intrinsic growth rate","intrinsic_growth_proxy",0.30,0.05,0.2,evidence="ROIC × reinvestment"),
   _s("rule40","Rule of 40","rule_of_40",0.20,0.4,0.6,evidence="growth + margin"),
   _s("rule40_pass","Rule of 40 pass","rule_of_40_pass",0.15,0,1,evidence="clears 40% threshold")]),
 "cashflow_forecast":("Cash Flow Forecast",0.40,[
   _s("ocf_recent","OCF growth recent","ocf_growth_recent",0.25,0.0,0.1,evidence="operating cash flow growth"),
   _s("ocf_persist","OCF persistence","ocf_growth_persistence",0.20,0.5,0.85,evidence="steady cash generation"),
   _s("ocf_slope","OCF trend slope","ocf_trend_slope",0.20,0.0,0.08,evidence="cash flow trajectory"),
   _s("fcf_margin","FCF margin proxy","fcf_margin_proxy",0.20,0.08,0.25,evidence="OCF minus capex / revenue"),
   _s("cash_traj","Cash trajectory","cash_trajectory",0.15,0.0,0.05,evidence="balance-sheet cash trend")]),
 "momentum_quality":("Momentum Quality",0.40,[
   _s("mom6","6-month momentum","momentum_6m",0.20,0.0,0.2,evidence="trailing 6mo return"),
   _s("mom12","12-month momentum","momentum_12m",0.18,0.0,0.3,evidence="trailing 12mo return"),
   _s("mom_qual","Momentum quality","momentum_quality",0.22,0.0,0.5,evidence="return / volatility (durable momentum)"),
   _s("mom_align","Momentum alignment","momentum_alignment",0.15,0,1,evidence="3/6mo same direction"),
   _s("trend_accel","Trend acceleration","trend_acceleration",0.13,0.0,0.005,evidence="trend speeding up"),
   _s("up_days","Up-day ratio","up_day_ratio",0.12,0.5,0.6,evidence="share of up days")]),
 "trend_forecast":("Trend Forecast",0.25,[
   _s("trend_slope","Price trend slope","price_trend_slope",0.35,0.0,0.1,evidence="projected 60-day trend"),
   _s("price_ma50","Price vs MA50","price_vs_ma50",0.30,0.0,0.1,evidence="above 50-day average"),
   _s("golden","Golden cross","ma_golden_cross",0.35,0.0,0.05,evidence="50 vs 200-day MA")]),
 "mean_reversion":("Mean Reversion",0.25,[
   _s("mr_pull","Mean-reversion pull","mean_reversion_pull",0.35,-0.1,0.1,evidence="pull toward 200-day MA"),
   _s("dist_high","Distance from 52w high","dist_from_52w_high",0.35,-0.2,-0.02,evidence="proximity to highs"),
   _s("rsi","RSI signal","rsi_signal",0.30,0.5,0.55,hib=False,evidence="overbought if >0.7")]),
 "leading_indicators":("Leading Indicators",0.25,[
   _s("wc_trend","Working capital trend","working_capital_trend",0.35,-0.05,0.05,evidence="working-capital trajectory"),
   _s("tax_stab","Tax-rate stability","tax_rate_stability",0.30,0.5,0.9,evidence="no one-time tax distortions"),
   _s("ext_mean","Extension from mean","extension_from_mean",0.35,0.25,0.05,hib=False,evidence="over-extension risk")]),
 "forecast_confidence":("Forecast Confidence",0.15,[
   _s("fc_conf","Forecast confidence","forecast_confidence",0.30,0.5,0.85,evidence="low volatility = predictable"),
   _s("ft_agree","Fundamental-technical agreement","fundamental_technical_agree",0.25,0,1,evidence="growth & momentum aligned"),
   _s("fwd_comp","Forward composite","forward_composite",0.25,0.5,1.0,evidence="growth+momentum+margin aligned"),
   _s("quality_anchor","Quality anchor","quality_anchor",0.20,0.08,0.2,evidence="ROIC supports reliability")]),
}
FORECAST_INTELLIGENCE={"label":"Forecast Intelligence","weight":4.0,"categories":CATEGORIES}

def forecast_rating(score):
    if score is None: return "Unrated"
    if score>=70: return "Bullish Outlook"
    if score>=56: return "Constructive"
    if score>=44: return "Neutral"
    if score>=30: return "Cautious"
    return "Bearish Outlook"
