"""Valuation Intelligence endpoint — GET /api/v6/valuation/{ticker}.
Reuses the Financial data pipeline (merged quarters + features) as single source,
adds real CAPM beta from Polygon prices, runs all valuation models, scores the
10-category catalog. Analyst-estimate signals honest needs_source.
"""
from __future__ import annotations
import math
from typing import Optional, Dict, List, Any
import httpx
from fastapi import APIRouter, HTTPException, Request, Depends
from loguru import logger

from ml.fundamentals.quality_engine import fetch_quarterly_financials, estimate_wacc
from quantedge.scoring.edgar_fetch import fetch_edgar_supplement
from quantedge.scoring.hybrid_merge import merge_quarters
from quantedge.scoring.financial_features import compute_financial_features
from quantedge.scoring.valuation_features import compute_valuation_features, compute_capm_beta
from quantedge.scoring.compute import score_signal
from quantedge.scoring.cat_valuation_v6 import CATEGORIES, valuation_rating
from auth.cognito_auth import get_optional_user, CognitoUser
from core.config import settings

router = APIRouter()
_POLY = "https://api.polygon.io"

def _san(o):
    if isinstance(o, float): return o if math.isfinite(o) else None
    if isinstance(o, dict): return {k: _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_san(v) for v in o]
    return o

def _roll(children):
    scored=[c for c in children if c.get("score") is not None]
    tw=sum(c["weight"] for c in children); aw=sum(c["weight"] for c in scored)
    if not scored or tw==0: return None,0.0
    return round(sum(c["score"]*c["weight"] for c in scored)/aw,1), round(aw/tw,3)

async def _get_market_cap_and_price(ticker: str, api_key: str):
    """Returns (market_cap, price). Price from prev-close; market cap from field or shares x price."""
    try:
        async with httpx.AsyncClient(timeout=12) as c:
            r = await c.get(f"{_POLY}/v3/reference/tickers/{ticker}?apiKey={api_key}")
            res = (r.json() or {}).get("results", {}) if r.status_code==200 else {}
            mc = res.get("market_cap")
            shares = res.get("weighted_shares_outstanding") or res.get("share_class_shares_outstanding")
            pr = await c.get(f"{_POLY}/v2/aggs/ticker/{ticker}/prev?apiKey={api_key}")
            price = None
            if pr.status_code==200:
                results=(pr.json() or {}).get("results",[])
                if results: price=results[0].get("c")
            if not mc and shares and price: mc = float(shares)*float(price)
            return (float(mc) if mc else None, float(price) if price else None)
    except Exception:
        return (None, None)

async def _get_beta_and_history(ticker: str, api_key: str):
    """Real CAPM beta from ~1.5yr daily returns vs SPY, plus the stock price history."""
    try:
        import datetime as dt
        end = dt.date.today(); start = end - dt.timedelta(days=550)
        async with httpx.AsyncClient(timeout=15) as c:
            async def closes(sym):
                u=f"{_POLY}/v2/aggs/ticker/{sym}/range/1/day/{start.isoformat()}/{end.isoformat()}?adjusted=true&sort=asc&limit=800&apiKey={api_key}"
                r=await c.get(u)
                if r.status_code!=200: return []
                return [b["c"] for b in (r.json() or {}).get("results",[])]
            s_px = await closes(ticker); m_px = await closes("SPY")
        def rets(px): return [(px[i]/px[i-1]-1) for i in range(1,len(px))] if len(px)>1 else []
        beta = compute_capm_beta(rets(s_px), rets(m_px))
        return beta, s_px
    except Exception:
        return None, []

def score_valuation(features):
    cats=[]
    for cid,(label,wt,sigs) in CATEGORIES.items():
        scored=[]
        for spec in sigs:
            val=features.get(spec["field"])
            res=score_signal(val,spec,None)
            scored.append({"id":spec["id"],"label":spec["label"],"weight":spec["weight"],
                           "status":spec["status"],"evidence":spec["evidence"],
                           "raw_value":val,**res})
        cs,cc=_roll(scored)
        cats.append({"id":cid,"label":label,"weight":wt,"score":cs,"confidence":cc,
                     "n_signals":len(sigs),"n_scored":sum(1 for s in scored if s["score"] is not None),
                     "signals":scored})
    vs,vc=_roll(cats)
    return {"label":"Valuation Intelligence","weight":10.0,"score":vs,"confidence":vc,"categories":cats}

async def compute_valuation_intelligence(ticker: str, api_key: str) -> Dict[str, Any]:
    """Reusable: fetch data, compute, score. Called by endpoint AND aggregator."""
    ticker=ticker.upper().strip()
    try:
        pq=await fetch_quarterly_financials(ticker,api_key,limit=24)
    except Exception:
        pq=[]
    if not pq:
        return {"ticker":ticker,"available":False,"reason":"no financial statements"}
    try:
        ed=await fetch_edgar_supplement(ticker,years_back=6)
    except Exception:
        ed={}
    merged=merge_quarters(pq,ed)
    market_cap, price = await _get_market_cap_and_price(ticker, api_key)
    beta, price_history = await _get_beta_and_history(ticker, api_key)
    wacc_dict = estimate_wacc(beta=beta)
    fin_features = compute_financial_features(merged, market_cap=market_cap, wacc=wacc_dict.get("mid"))
    val_features = compute_valuation_features(merged, fin_features, price, market_cap, wacc_dict, beta)
    if not val_features:
        return {"ticker":ticker,"available":False,"reason":"insufficient data for valuation"}
    from quantedge.scoring.valuation_deep import compute_valuation_deep
    deep = compute_valuation_deep(merged, fin_features, val_features, price, market_cap,
                                  wacc_dict.get("mid"), price_history=price_history)
    val_features.update(deep)
    tree=score_valuation(val_features)
    n_scored=sum(c["n_scored"] for c in tree["categories"])
    n_total=sum(c["n_signals"] for c in tree["categories"])
    return {"ticker":ticker,"available":True,"intelligence":"valuation",
            "score":tree["score"],"confidence":tree["confidence"],
            "valuation_rating":valuation_rating(tree["score"]),
            "weight_in_conviction":10.0,
            "coverage":{"scored":n_scored,"total":n_total},
            "beta_used":beta,"wacc_used":wacc_dict.get("mid"),"current_price":price,
            "market_cap":market_cap,"tree":tree,
            "key_metrics":{k:val_features.get(k) for k in
                ["fair_value","margin_of_safety","upside_to_fair","intrinsic_consensus",
                 "dcf_weighted","dcf_bull","dcf_base","dcf_bear","reverse_dcf_implied_growth",
                 "mult_pe","mult_ev_ebitda","buy_zone","sell_zone","epv_per_share","graham_number",
                 "residual_income_value","ddm_value","nav_per_share","dcf_scenarios_above_price",
                 "pe_vs_history","pe_historical_avg"]},
            "sensitivity":val_features.get("dcf_sensitivity"),
            "assumptions":{k:val_features.get(k) for k in
                ["assumption_revenue_cagr","assumption_wacc","assumption_terminal_growth",
                 "assumption_operating_margin","assumption_tax_rate","assumption_fcf_margin",
                 "assumption_forecast_years","assumption_beta"]},
            "cases":{"bear":val_features.get("dcf_bear"),"base":val_features.get("dcf_base"),
                "bull":val_features.get("dcf_bull"),"probabilities":val_features.get("case_probabilities"),
                "confidence_low":val_features.get("confidence_range_low"),
                "confidence_high":val_features.get("confidence_range_high")},
            "expected_return":{k:val_features.get(k) for k in
                ["exp_return_growth","exp_return_dividend","exp_return_buyback",
                 "exp_return_margin","exp_return_multiple","expected_total_return"]},
            "model_agreement":{"dispersion_cv":val_features.get("model_dispersion_cv"),
                "consensus_overvalued":val_features.get("model_consensus_overvalued"),
                "agreement_score":val_features.get("model_agreement_score")},
            "driver_waterfall":val_features.get("driver_waterfall"),
            "value_range":{"current_price":price,
                "methods":{"dcf_bear":val_features.get("dcf_bear"),"dcf_base":val_features.get("dcf_base"),
                    "dcf_bull":val_features.get("dcf_bull"),"epv":val_features.get("epv_per_share"),
                    "graham":val_features.get("graham_number"),"residual_income":val_features.get("residual_income_value"),
                    "ddm":val_features.get("ddm_value"),"nav":val_features.get("nav_per_share")}}}

@router.get("/valuation/{ticker}")
async def get_valuation(ticker: str, http_request: Request,
                        current_user: Optional[CognitoUser]=Depends(get_optional_user)):
    api_key=getattr(settings,"POLYGON_API_KEY","") or ""
    if not api_key:
        raise HTTPException(503,"data source unavailable")
    result=await compute_valuation_intelligence(ticker, api_key)
    return {"data":_san(result)}
