"""Market Intelligence — peer-relative scoring of price/momentum/liquidity signals.
Reuses peer_stats bucketed factors, scores each as PERCENTILE vs sector peers.
Single ownership: price-based only (no fundamentals/valuation double-count).
"""
from __future__ import annotations
import math
from typing import Optional, Dict, List, Any

def percentile_vs_peers(value, peer_values, higher_is_better=True):
    vals=[v for v in peer_values if v is not None and isinstance(v,(int,float)) and math.isfinite(v)]
    if value is None or not isinstance(value,(int,float)) or not math.isfinite(value) or len(vals)<5: return None
    below=sum(1 for x in vals if x<value); equal=sum(1 for x in vals if x==value)
    pct=100.0*(below+0.5*equal)/len(vals)
    return round(pct if higher_is_better else 100.0-pct, 1)

MARKET_SIGNALS = {
 "trend_momentum": ("Trend & Momentum", 0.85, [
   ("mom_3m","3-month momentum","mom_3m",0.35,True,"price momentum vs sector"),
   ("mom_6m","6-month momentum","mom_6m",0.30,True,"medium-term momentum vs sector"),
   ("mom_12_1","12-1 momentum","mom_12_1",0.35,True,"classic 12-1 momentum factor vs sector"),
   ("ma_alignment","MA alignment","ma_alignment",0.30,True,"moving-average trend structure"),
 ]),
 "trend_persistence": ("Trend Persistence", 0.50, [
   ("hurst","Hurst exponent","hurst",0.40,True,"trend persistence (>0.5 trending)"),
   ("pct_above_ma50","% above 50-day MA","pct_above_ma50",0.30,True,"price vs 50-day average"),
   ("pct_above_ma200","% above 200-day MA","pct_above_ma200",0.30,True,"price vs 200-day average"),
 ]),
 "risk_adjusted": ("Risk-Adjusted Performance", 0.60, [
   ("sharpe_3m","3-month Sharpe","sharpe_3m",0.55,True,"risk-adjusted return vs sector"),
   ("vol_adj_return","Vol-adjusted return","vol_adj_return",0.45,True,"return per unit volatility vs sector"),
 ]),
 "relative_strength": ("Relative Strength", 0.70, [
   ("rs_6m","6-month relative strength","mom_6m",0.50,True,"performance vs sector median"),
   ("rs_12m","12-month relative strength","mom_12_1",0.50,True,"long-term relative strength vs sector"),
 ]),
 "liquidity_flow": ("Liquidity & Flow", 0.40, [
   ("amihud","Amihud liquidity","amihud",0.40,False,"price impact (lower = more liquid)"),
   ("volume_surge","Volume surge","volume_surge",0.30,True,"recent volume vs baseline"),
   ("obv_slope","OBV slope","obv_slope_norm",0.30,True,"on-balance-volume accumulation"),
 ]),
 "short_term": ("Short-Term Signal", 0.25, [
   ("mom_1m","1-month momentum","mom_1m",1.0,True,"near-term price action vs sector"),
 ]),
}

def _band(value, good, great, hib=True, floor=None, floor_score=15, cap=None, cap_score=90):
    """Absolute-band score 0-100 for deep signals not in peer factors."""
    v=value
    if v is None or not isinstance(v,(int,float)): return None
    import math
    if not math.isfinite(v): return None
    if not hib: v,good,great=-v,-good,-great
    if v<=good: s=max(0.0,40.0+(v-(good-(great-good)))/((great-good) or 1)*25.0)
    elif v<=great: s=65.0+(v-good)/((great-good) or 1)*25.0
    else: s=min(100.0,90.0+(v-great)/((great-good) or 1)*10.0)
    if floor is not None and ((hib and value<floor) or (not hib and value>floor)): s=min(s,floor_score)
    if cap is not None and ((hib and value>cap) or (not hib and value<cap)): s=max(s,cap_score)
    return round(max(0.0,min(100.0,s)),1)

# Deep category specs: (id, label, weight, [(sig_id, label, source_key, weight, good, great, hib, evidence)])
DEEP_CATEGORIES = {
 "volatility": ("Volatility Intelligence", 0.60, [
   ("vol_percentile","Volatility percentile","vol_percentile",0.28,80,40,False,"current vol vs own history (lower=calmer)"),
   ("vol_trend","Volatility trend","vol_trend",0.20,0.2,-0.1,False,"rising vol is a risk"),
   ("beta","Beta","beta",0.18,1.4,0.9,False,"market sensitivity (lower=defensive)"),
   ("up_down_vol_ratio","Up/down vol ratio","up_down_vol_ratio",0.16,0.9,1.3,True,"upside vs downside volatility (higher=favorable)"),
   ("downside_vol","Downside volatility","downside_vol",0.10,0.35,0.15,False,"annualized downside deviation"),
   ("vol_of_vol","Vol of vol","vol_of_vol",0.08,0.4,0.15,False,"volatility stability (lower=steadier)"),
 ]),
 "trading_risk": ("Trading Risk", 0.55, [
   ("max_drawdown","Max drawdown","max_drawdown",0.20,-0.35,-0.15,True,"worst peak-to-trough (closer to 0 better)"),
   ("var_5pct","5% Value-at-Risk","var_5pct",0.18,-0.04,-0.02,True,"daily downside tail"),
   ("gap_risk","Gap risk frequency","gap_risk_freq",0.12,0.20,0.08,False,"frequency of >2% daily moves"),
   ("downside_dev","Downside deviation","downside_dev",0.12,0.30,0.15,False,"annualized downside volatility"),
   ("ulcer_index","Ulcer index","ulcer_index",0.14,0.20,0.05,False,"depth+duration of drawdowns (lower=better)"),
   ("pain_index","Pain index","pain_index",0.10,0.15,0.04,False,"average drawdown"),
   ("recovery_factor","Recovery factor","recovery_factor",0.08,0.5,3.0,True,"return per unit drawdown"),
   ("tail_ratio","Tail ratio","tail_ratio",0.06,0.9,1.3,True,"upside tail vs downside tail"),
 ]),
 "volume_accum": ("Volume & Accumulation", 0.30, [
   ("obv_slope","OBV accumulation","obv_slope",0.22,-0.5,1.0,True,"on-balance-volume trend"),
   ("vwap_distance","VWAP distance","vwap_distance",0.16,-0.05,0.03,True,"price vs volume-weighted avg"),
   ("rvol","Relative volume","rvol",0.14,0.5,1.5,True,"volume vs 21-day average"),
   ("up_volume_ratio","Up-volume ratio","up_volume_ratio",0.18,0.4,0.6,True,"volume on up days (accumulation)"),
   ("volume_trend","Volume trend","volume_trend",0.12,-0.1,0.2,True,"recent vs baseline volume"),
   ("accumulation_days","Accumulation days","accumulation_days",0.10,5,12,True,"up-days on higher volume (of 20)"),
   ("mfi_proxy","Money flow index","mfi_proxy",0.08,40,65,True,"money flow (accumulation vs distribution)"),
 ]),
 "short_interest": ("Short Interest", 0.25, [
   ("days_to_cover","Days to cover","days_to_cover",0.50,5,1.5,False,"short interest / avg volume (lower=less bearish bet)"),
   ("si_trend","Short interest trend","short_interest_trend",0.50,0.15,-0.1,False,"rising short interest is bearish"),
 ]),
}

def score_market_deep(deep_data):
    """Score the 4 deep categories from real market_deep outputs."""
    cats=[]
    src_map={"volatility":deep_data.get("volatility") or {},
             "trading_risk":deep_data.get("trading_risk") or {},
             "volume_accum":deep_data.get("volume") or {},
             "short_interest":deep_data.get("short_interest") or {}}
    for cid,(label,wt,sigs) in DEEP_CATEGORIES.items():
        source=src_map[cid]
        scored=[]
        for sid,slabel,skey,sw,good,great,hib,ev in sigs:
            val=source.get(skey)
            score=_band(val,good,great,hib)
            scored.append({"id":sid,"label":slabel,"weight":sw,"raw_value":val,"score":score,
                           "status":"live","evidence":ev,"method":"absolute" if score is not None else "missing"})
        sc=[s for s in scored if s["score"] is not None]
        tw=sum(s["weight"] for s in scored); aw=sum(s["weight"] for s in sc)
        cscore=round(sum(s["score"]*s["weight"] for s in sc)/aw,1) if aw>0 else None
        cats.append({"id":cid,"label":label,"weight":wt,"score":cscore,
                     "confidence":round(aw/tw,3) if tw else 0,
                     "n_signals":len(sigs),"n_scored":len(sc),"signals":scored})
    return cats

def score_market(me_factors, peer_factors_list, deep_data=None):
    cats=[]
    for cid,(label,wt,sigs) in MARKET_SIGNALS.items():
        scored=[]
        for sid,slabel,fkey,sw,hib,ev in sigs:
            sw=float(sw); val=me_factors.get(fkey)
            peer_vals=[pf.get(fkey) for pf in peer_factors_list]
            score=percentile_vs_peers(val, peer_vals, hib) if val is not None else None
            scored.append({"id":sid,"label":slabel,"weight":sw,"raw_value":val,
                           "score":score,"status":"live","evidence":ev,
                           "method":"peer_percentile" if score is not None else "missing"})
        sc=[s for s in scored if s["score"] is not None]
        tw=sum(s["weight"] for s in scored); aw=sum(s["weight"] for s in sc)
        cat_score=round(sum(s["score"]*s["weight"] for s in sc)/aw,1) if aw>0 else None
        cats.append({"id":cid,"label":label,"weight":wt,"score":cat_score,
                     "confidence":round(aw/tw,3) if tw else 0,
                     "n_signals":len(sigs),"n_scored":len(sc),"signals":scored})
    if deep_data:
        cats.extend(score_market_deep(deep_data))
    sc=[c for c in cats if c["score"] is not None]
    tw=sum(c["weight"] for c in cats); aw=sum(c["weight"] for c in sc)
    market_score=round(sum(c["score"]*c["weight"] for c in sc)/aw,1) if aw>0 else None
    return {"label":"Market Intelligence","weight":5.0,"score":market_score,
            "confidence":round(aw/tw,3) if tw else 0,"categories":cats}

def market_rating(score):
    if score is None: return "Unrated"
    if score>=75: return "Strong Momentum"
    if score>=58: return "Positive Momentum"
    if score>=42: return "Neutral"
    if score>=25: return "Weak Momentum"
    return "Downtrend"
