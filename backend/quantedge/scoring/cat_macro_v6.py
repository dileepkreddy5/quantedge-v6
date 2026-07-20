"""Macro Sensitivity catalog v6 — 8 categories, weight 3.00.
Scores macro RESILIENCE — lower absolute sensitivity to macro shocks = more insulated = better.
Some signals are neutral-informational (betas can be good either direction depending on regime)."""
def _s(id,label,field,weight,good,great,hib=True,status="live",evidence=""):
    return {"id":id,"label":label,"field":field,"weight":weight,"higher_is_better":hib,
            "good":good,"great":great,"status":status,"evidence":evidence}

CATEGORIES = {
 "rate_sensitivity":("Rate Sensitivity",0.55,[
   _s("rate_sens_abs","Rate sensitivity","rate_sensitivity_abs",0.40,0.5,0.1,hib=False,evidence="exposure to rate moves (lower=insulated)"),
   _s("rate_beta","Rate beta (TLT)","rate_beta",0.30,-0.3,0.3,evidence="bond-like behavior; +=benefits from falling rates"),
   _s("rate_corr","Rate correlation","rate_correlation",0.30,-0.5,0.3,evidence="correlation to long bonds")]),
 "inflation":("Inflation Exposure",0.45,[
   _s("inflation_hedge","Inflation hedge (gold)","inflation_hedge",0.40,-0.2,0.3,evidence="gold correlation; +=inflation hedge"),
   _s("oil_corr","Oil correlation","oil_correlation",0.30,0.4,-0.1,hib=False,evidence="energy-cost exposure"),
   _s("gold_beta","Gold beta","gold_beta",0.30,-0.2,0.2,evidence="commodity-inflation sensitivity")]),
 "dollar":("Dollar Sensitivity",0.40,[
   _s("dollar_sens","Dollar sensitivity","dollar_sensitivity_abs",0.45,0.4,0.1,hib=False,evidence="FX exposure (lower=insulated)"),
   _s("dollar_beta","Dollar beta (UUP)","dollar_beta",0.30,-0.4,0.1,evidence="strong-dollar impact; -=multinational hurt"),
   _s("dollar_corr","Dollar correlation","dollar_correlation",0.25,-0.4,0.1,evidence="USD correlation")]),
 "growth_cycle":("Growth Cycle",0.45,[
   _s("market_beta","Market beta","market_beta",0.35,1.5,0.9,hib=False,evidence="systematic risk"),
   _s("smallcap_beta","Small-cap beta (IWM)","smallcap_beta",0.30,1.3,0.7,hib=False,evidence="cyclical/risk-appetite exposure"),
   _s("cyclical_corr","Cyclical correlation","cyclical_correlation",0.35,0.8,0.4,hib=False,evidence="economic-cycle sensitivity")]),
 "risk_regime":("Risk Regime",0.40,[
   _s("credit_beta","Credit beta (HYG)","credit_beta",0.40,1.0,0.4,hib=False,evidence="risk-on/off sensitivity"),
   _s("risk_on_corr","Risk-on correlation","risk_on_correlation",0.30,0.7,0.3,hib=False,evidence="credit-spread sensitivity"),
   _s("beta_stability","Beta stability","beta_stability",0.30,0.5,0.85,evidence="consistent market exposure")]),
 "commodity":("Commodity Exposure",0.30,[
   _s("oil_beta","Oil beta","oil_beta",0.50,0.5,-0.1,hib=False,evidence="oil-price sensitivity"),
   _s("gold_beta2","Gold beta","gold_beta",0.50,-0.2,0.2,evidence="commodity correlation")]),
 "factor_exposure":("Factor Exposure",0.30,[
   _s("value_tilt","Value tilt","value_tilt",0.35,-0.2,0.5,evidence="value-factor loading"),
   _s("momentum_tilt","Momentum tilt","momentum_tilt",0.35,0.0,0.5,evidence="momentum-factor loading"),
   _s("market_corr","Market correlation","market_correlation",0.30,0.9,0.5,hib=False,evidence="independence from market")]),
 "defensiveness":("Defensiveness & Resilience",0.15,[
   _s("defensiveness","Defensiveness","defensiveness",0.40,0.3,0.7,evidence="low-beta defensive character"),
   _s("macro_resilience","Macro resilience","macro_resilience",0.60,0.5,0.85,evidence="insulation from macro shocks")]),
}
MACRO_INTELLIGENCE={"label":"Macro Sensitivity","weight":3.0,"categories":CATEGORIES}

def macro_rating(score):
    if score is None: return "Unrated"
    if score>=68: return "Insulated"
    if score>=54: return "Resilient"
    if score>=42: return "Balanced"
    if score>=30: return "Exposed"
    return "Highly Exposed"
