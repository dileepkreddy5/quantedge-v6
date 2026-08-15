"""ML Models Intelligence — GET /api/v6/mlmodels/{ticker}. Transparent factor-ensemble."""
from __future__ import annotations
import os, math, datetime as dt
from typing import Optional, Dict, Any
import httpx
from fastapi import APIRouter, HTTPException, Request, Depends
from ml.fundamentals.quality_engine import fetch_quarterly_financials, estimate_wacc
from quantedge.scoring.edgar_fetch import fetch_edgar_supplement
from quantedge.scoring.hybrid_merge import merge_quarters
from quantedge.scoring.financial_features import compute_financial_features
from quantedge.scoring.forecast_features import compute_forecast_features
from quantedge.scoring.ml_features import compute_ml_features
from quantedge.scoring.compute import score_signal
from quantedge.scoring.cat_ml_v6 import CATEGORIES, ml_rating
from auth.cognito_auth import get_optional_user, CognitoUser
from core.config import settings

router = APIRouter()

def _san(o):
    if isinstance(o,float): return o if math.isfinite(o) else None
    if isinstance(o,dict): return {k:_san(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [_san(v) for v in o]
    return o

def _roll(children):
    sc=[c for c in children if c.get("score") is not None]
    tw=sum(c["weight"] for c in children); aw=sum(c["weight"] for c in sc)
    if not sc or tw==0: return None,0.0
    return round(sum(c["score"]*c["weight"] for c in sc)/aw,1), round(aw/tw,3)

def score_ml(features):
    cats=[]
    for cid,(label,wt,sigs) in CATEGORIES.items():
        scored=[]
        for spec in sigs:
            val=features.get(spec["field"])
            res=score_signal(val,spec,None)
            scored.append({"id":spec["id"],"label":spec["label"],"weight":spec["weight"],
                           "status":spec["status"],"evidence":spec["evidence"],"raw_value":val,**res})
        cs,cc=_roll(scored)
        cats.append({"id":cid,"label":label,"weight":wt,"score":cs,"confidence":cc,
                     "n_signals":len(sigs),"n_scored":sum(1 for s in scored if s["score"] is not None),"signals":scored})
    s,c=_roll(cats)
    return {"label":"ML Models Intelligence","weight":1.0,"score":s,"confidence":c,"categories":cats}

async def _closes(ticker, api_key, days=400):
    try:
        end=dt.date.today(); start=end-dt.timedelta(days=days)
        async with httpx.AsyncClient(timeout=12) as c:
            u=f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}?adjusted=true&sort=asc&limit=500&apiKey={api_key}"
            r=await c.get(u)
            if r.status_code==200: return [b["c"] for b in (r.json() or {}).get("results",[])]
    except Exception: pass
    return []

async def compute_ml_intelligence(ticker: str, api_key: str) -> Dict[str,Any]:
    ticker=ticker.upper().strip()
    pq=await fetch_quarterly_financials(ticker, api_key, limit=14)
    if not pq: return {"ticker":ticker,"available":False,"reason":"no financial data"}
    ed=await fetch_edgar_supplement(ticker, years_back=5)
    merged=merge_quarters(pq, ed)
    if not merged or len(merged)<6: return {"ticker":ticker,"available":False,"reason":"insufficient history"}
    closes=await _closes(ticker, api_key)
    mcap=None
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r=await c.get(f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={api_key}")
            if r.status_code==200: mcap=((r.json() or {}).get("results",{}) or {}).get("market_cap")
    except Exception: pass
    fin=compute_financial_features(merged, market_cap=mcap, wacc=estimate_wacc(beta=None)["mid"])
    fc=compute_forecast_features(merged, closes, fin)
    # merge all feature sources for the ensemble
    combined={}
    combined.update(fin if isinstance(fin,dict) else {})
    combined.update(fc if isinstance(fc,dict) else {})
    if mcap: combined["market_cap"]=mcap
    feats=compute_ml_features(combined)
    if not feats:
        return {"ticker":ticker,"available":False,"reason":"insufficient features for ensemble"}
    tree=score_ml(feats)
    n_scored=sum(c["n_scored"] for c in tree["categories"]); n_total=sum(c["n_signals"] for c in tree["categories"])
    return {"ticker":ticker,"available":True,"intelligence":"mlmodels",
            "score":tree["score"],"confidence":tree["confidence"],
            "ml_rating":ml_rating(tree["score"]),"weight_in_conviction":1.0,
            "methodology":"Transparent factor-ensemble: interpretable value/momentum/quality/growth/low-vol sub-models blended into a directional signal. Not a black-box neural network.",
            "coverage":{"scored":n_scored,"total":n_total},"tree":tree,
            "key_metrics":{k:feats.get(k) for k in
                ["ensemble_score","up_probability","model_agreement","ensemble_conviction","n_models_bullish",
                 "quality_model_score","momentum_model_score","growth_model_score","value_model_score","lowvol_model_score"]}}

@router.get("/ml/panel/skill")
async def panel_skill():
    """Published out-of-sample skill of the cross-sectional panel ensemble.

    Reads the training report written by ml.training.train_panel_models. No
    computation, no external calls — the artifact is regenerated by the nightly
    retrain. Every figure is reported with the sample it was measured on:
    ic_all_dates is the mean per-date rank-IC across all scoring dates and is
    paired with the Newey-West t computed on that same series; ic_independent is
    the subsample of non-overlapping windows, null below 5 windows because a mean
    of fewer than five Spearman coefficients is not an information coefficient.
    """
    import json
    from pathlib import Path
    rp = Path(os.getenv("MODEL_DIR", "/app/models")) / "panel" / "training_report.json"
    if not rp.exists():
        raise HTTPException(status_code=503, detail="panel has not been trained yet")
    try:
        rep = json.loads(rp.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"unreadable training report: {e}")

    horizons = []
    for hkey, v in sorted((rep.get("horizons") or {}).items(), key=lambda kv: int(kv[0])):
        ic = v.get("ic_all_dates") or v.get("oos_rank_ic") or {}
        horizons.append({
            "horizon_days": int(hkey),
            "label": v.get("horizon_label"),
            "ic_all_dates": ic.get("ensemble"),
            "ic_xgboost": ic.get("xgboost"),
            "ic_lightgbm": ic.get("lightgbm"),
            "ic_t_stat": v.get("ic_t_stat"),
            "ic_hit_rate": v.get("ic_hit_rate"),
            "n_scoring_dates": v.get("n_scoring_dates"),
            "n_independent_windows": v.get("n_independent_val_dates"),
            "ic_independent": v.get("ic_independent"),
            "independent_measurable": v.get("independent_measurable"),
            "hac_stable": v.get("hac_stable"),
            "hac_lag_obs": v.get("hac_lag_obs"),
            "blend_weights": v.get("blend_weights"),
            "reliable": v.get("reliable"),
            "confidence_note": v.get("confidence_note"),
        })
    return _san({
        "trained_at": rep.get("trained_at"),
        "panel": rep.get("panel"),
        "n_tickers": rep.get("n_tickers"),
        "n_features": rep.get("n_features"),
        "split_date": rep.get("split_date"),
        "environment": rep.get("environment"),
        "n_train": (horizons[0] or {}).get("n_train") if horizons else None,
        "horizons": horizons,
        "any_reliable": any(h.get("reliable") for h in horizons),
        "method": ("Cross-sectional rank-IC on held-out dates the models never saw. "
                   "Significance is a Newey-West t on the per-date IC series, widened "
                   "for the overlap induced by forward-return windows. A horizon is "
                   "marked reliable only with |t| >= 2, a stable HAC correction, and a "
                   "positive IC."),
        "disclaimer": ("Research output, not investment advice. Forecast horizons that "
                       "report reliable=false have not demonstrated statistically "
                       "significant out-of-sample skill."),
    })


@router.get("/mlmodels/{ticker}")
async def get_mlmodels(ticker: str, http_request: Request,
                       current_user: Optional[CognitoUser]=Depends(get_optional_user)):
    api_key=getattr(settings,"POLYGON_API_KEY","") or ""
    if not api_key: raise HTTPException(503,"data source unavailable")
    return {"data":_san(await compute_ml_intelligence(ticker, api_key))}
