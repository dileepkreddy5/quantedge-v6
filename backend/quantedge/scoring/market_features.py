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
 "trend_momentum": ("Trend & Momentum", 1.30, [
   ("mom_3m","3-month momentum","mom_3m",0.35,True,"price momentum vs sector"),
   ("mom_6m","6-month momentum","mom_6m",0.30,True,"medium-term momentum vs sector"),
   ("mom_12_1","12-1 momentum","mom_12_1",0.35,True,"classic 12-1 momentum factor vs sector"),
   ("ma_alignment","MA alignment","ma_alignment",0.30,True,"moving-average trend structure"),
 ]),
 "trend_persistence": ("Trend Persistence", 0.80, [
   ("hurst","Hurst exponent","hurst",0.40,True,"trend persistence (>0.5 trending)"),
   ("pct_above_ma50","% above 50-day MA","pct_above_ma50",0.30,True,"price vs 50-day average"),
   ("pct_above_ma200","% above 200-day MA","pct_above_ma200",0.30,True,"price vs 200-day average"),
 ]),
 "risk_adjusted": ("Risk-Adjusted Performance", 1.00, [
   ("sharpe_3m","3-month Sharpe","sharpe_3m",0.55,True,"risk-adjusted return vs sector"),
   ("vol_adj_return","Vol-adjusted return","vol_adj_return",0.45,True,"return per unit volatility vs sector"),
 ]),
 "relative_strength": ("Relative Strength", 0.90, [
   ("rs_6m","6-month relative strength","mom_6m",0.50,True,"performance vs sector median"),
   ("rs_12m","12-month relative strength","mom_12_1",0.50,True,"long-term relative strength vs sector"),
 ]),
 "liquidity_flow": ("Liquidity & Flow", 0.60, [
   ("amihud","Amihud liquidity","amihud",0.40,False,"price impact (lower = more liquid)"),
   ("volume_surge","Volume surge","volume_surge",0.30,True,"recent volume vs baseline"),
   ("obv_slope","OBV slope","obv_slope_norm",0.30,True,"on-balance-volume accumulation"),
 ]),
 "short_term": ("Short-Term Signal", 0.40, [
   ("mom_1m","1-month momentum","mom_1m",1.0,True,"near-term price action vs sector"),
 ]),
}

def score_market(me_factors, peer_factors_list):
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
