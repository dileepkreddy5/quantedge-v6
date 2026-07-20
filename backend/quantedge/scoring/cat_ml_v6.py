"""ML Models Intelligence catalog v6 — 6 categories, weight 1.00.
Transparent factor-ensemble (value/momentum/quality/growth/low-vol sub-models + meta)."""
def _s(id,label,field,weight,good,great,hib=True,status="live",evidence=""):
    return {"id":id,"label":label,"field":field,"weight":weight,"higher_is_better":hib,
            "good":good,"great":great,"status":status,"evidence":evidence}

CATEGORIES = {
 "quality_model":("Quality Factor Model",0.20,[
   _s("q_score","Quality model score","quality_model_score",0.70,0.5,0.75,evidence="ROIC/ROE/margin/cash-conversion factor"),
   _s("q_signal","Quality signal","quality_signal",0.30,0,1,evidence="bullish/bearish/neutral")]),
 "momentum_model":("Momentum Factor Model",0.18,[
   _s("m_score","Momentum model score","momentum_model_score",0.70,0.5,0.75,evidence="6m/3m/quality/trend factor"),
   _s("m_signal","Momentum signal","momentum_signal",0.30,0,1,evidence="bullish/bearish/neutral")]),
 "growth_model":("Growth Factor Model",0.18,[
   _s("g_score","Growth model score","growth_model_score",0.70,0.5,0.75,evidence="revenue/earnings growth + Rule-of-40 factor"),
   _s("g_signal","Growth signal","growth_signal",0.30,0,1,evidence="bullish/bearish/neutral")]),
 "value_model":("Value Factor Model",0.16,[
   _s("v_score","Value model score","value_model_score",0.70,0.5,0.75,evidence="earnings-yield/cash-yield/PE factor"),
   _s("v_signal","Value signal","value_signal",0.30,0,1,evidence="bullish/bearish/neutral")]),
 "lowvol_model":("Low-Volatility Model",0.10,[
   _s("lv_score","Low-vol model score","lowvol_model_score",1.0,0.5,0.75,evidence="forecast-confidence/beta/extension factor")]),
 "ensemble_meta":("Ensemble Meta-Model",0.18,[
   _s("ensemble","Ensemble score","ensemble_score",0.30,0.5,0.7,evidence="weighted multi-factor blend"),
   _s("up_prob","Up-probability","up_probability",0.25,0.5,0.7,evidence="logistic-scaled directional probability"),
   _s("agreement","Model agreement","model_agreement",0.20,0.6,0.9,evidence="sub-model consensus"),
   _s("conviction","Ensemble conviction","ensemble_conviction",0.15,0.2,0.6,evidence="distance from neutral"),
   _s("n_bullish","Models bullish","n_models_bullish",0.10,2,4,evidence="count of bullish sub-models")]),
}
ML_INTELLIGENCE={"label":"ML Models Intelligence","weight":1.0,"categories":CATEGORIES}

def ml_rating(score):
    if score is None: return "Unrated"
    if score>=68: return "Strong Buy Signal"
    if score>=56: return "Buy Signal"
    if score>=44: return "Hold Signal"
    if score>=32: return "Sell Signal"
    return "Strong Sell Signal"
