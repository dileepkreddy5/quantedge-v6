"""
QuantEdge Price Oracle — FastAPI Router
═══════════════════════════════════════
Plugs into your existing FastAPI backend at /api/v1/oracle/predict

Flow:
  1. Receive ticker from frontend
  2. Run PriceOracleEngine.compute() → real ML numbers
  3. Send computed data to Claude → narrative synthesis
  4. Return combined response to frontend

Endpoint: POST /api/v1/oracle/predict
"""

import os
import json
import asyncio
import httpx
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import logging

from .engine import PriceOracleEngine

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Price Oracle"])

# ── Request / Response models ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    ticker:       str         = Field(..., description="US equity ticker symbol", example="AAPL")
    account_size: float       = Field(25000, description="Portfolio size in USD for Kelly sizing")
    horizon:      str         = Field("1m",  description="Primary horizon: 1w|1m|3m|6m")

class HorizonPrediction(BaseModel):
    bear_price:      float
    base_price:      float
    bull_price:      float
    expected_price:  float
    expected_return: float
    prob_positive:   float
    vol_forecast:    float
    var_95_1d:       float

class PredictResponse(BaseModel):
    ticker:        str
    company_name:  str
    sector:        str
    current_price: float
    signal:        str
    conviction:    int
    kelly_pct:     float
    horizons:      dict
    levels:        dict
    features_summary: dict
    garch_summary: dict
    options_summary: dict
    narrative:     dict       # Claude-generated text
    models_used:   list
    disclaimer:    str

# ── Claude synthesis prompt ───────────────────────────────────────────────────

CLAUDE_SYSTEM = """You are the Chief Quantitative Analyst at a top-tier hedge fund.
You have just received a complete computational package from our quant engine containing:
- Real GJR-GARCH(1,1) volatility forecasts
- 10,000-path Monte Carlo price simulations
- 40+ technical and statistical features
- IC-weighted directional signals
- Options market data

Your job is to SYNTHESIZE this data into actionable narrative.
You do NOT generate price numbers — those come from the models.
You DO provide:
1. Sharp interpretation of what the data means
2. Primary thesis (why this setup exists)
3. Key risks (what could break the thesis)
4. Regime assessment (does current market environment help or hurt)
5. Entry guidance (specific, actionable)
6. Catalyst identification (what events could drive price)

Return ONLY valid JSON with this exact structure:
{
  "company_name": "Full company name",
  "sector": "Sector",
  "primary_thesis": "2-3 sentences. What the data says about this stock RIGHT NOW.",
  "bull_thesis": "1-2 sentences. Why the bull case plays out.",
  "bear_thesis": "1-2 sentences. Why the bear case plays out.",
  "regime_assessment": "1-2 sentences. How current market regime affects this trade.",
  "entry_guidance": "Specific guidance: wait for X / buy on open / scale in at these levels.",
  "catalysts": ["catalyst 1", "catalyst 2", "catalyst 3"],
  "tail_risks": ["risk 1", "risk 2", "risk 3"],
  "technical_narrative": "1 sentence on key technical level or pattern.",
  "options_narrative": "1 sentence on what options market implies (or 'Options data unavailable').",
  "hurst_interpretation": "1 sentence: trending vs mean-reverting and what it means.",
  "recommended_horizon": "1w" or "1m" or "3m" or "6m",
  "setup_grade": "A+" or "A" or "B+" or "B" or "C" or "D",
  "urgency": "ACT NOW" or "WATCH" or "PASS",
  "one_line_summary": "One crisp sentence a trader can remember."
}"""

def build_claude_prompt(ticker: str, computed: dict, account_size: float) -> str:
    f  = computed["features"]
    g  = computed["garch"]
    s  = computed["signal"]
    h  = computed["horizons"]
    op = computed["options"]
    co = computed["company"]
    er = computed["expected_returns"]
    
    return f"""Synthesize this quantitative analysis for {ticker}:

COMPANY: {co.get('name','?')} | Sector: {co.get('sector','?')} | Market Cap: ${co.get('market_cap_b','?')}B
Beta: {co.get('beta','?')} | P/E: {co.get('pe_ratio','?')} | Short Float: {co.get('short_pct','?')}
Analyst Target: ${co.get('analyst_target','?')}

CURRENT PRICE: ${computed['current_price']:.2f}

GARCH VOLATILITY MODEL ({g['model']}):
- Current realized vol: {g['current_vol']:.1f}%
- Long-run equilibrium vol: {g['long_run_vol']:.1f}%
- Persistence: {g['persistence']:.4f} (>{0.95:.2f} = highly persistent clustering)
- Asymmetric (bad news amplifies vol more): {g['asymmetric']}

EXPECTED RETURN ESTIMATES (annualized):
- Historical: {er.get('mu_historical','?')}%
- CAPM: {er.get('mu_capm','?')}%
- Momentum-adjusted: {er.get('mu_momentum_adj','?')}%
- Ensemble: {er.get('mu_ensemble','?')}%

MONTE CARLO PRICE TARGETS (10,000 paths, Student-t innovations):
1 Week:  Bear ${h['1w']['bear_price']} | Base ${h['1w']['base_price']} | Bull ${h['1w']['bull_price']} | P(up)={h['1w']['prob_positive']:.1%}
1 Month: Bear ${h['1m']['bear_price']} | Base ${h['1m']['base_price']} | Bull ${h['1m']['bull_price']} | P(up)={h['1m']['prob_positive']:.1%}
3 Month: Bear ${h['3m']['bear_price']} | Base ${h['3m']['base_price']} | Bull ${h['3m']['bull_price']} | P(up)={h['3m']['prob_positive']:.1%}
6 Month: Bear ${h['6m']['bear_price']} | Base ${h['6m']['base_price']} | Bull ${h['6m']['bull_price']} | P(up)={h['6m']['prob_positive']:.1%}

DIRECTIONAL SIGNALS (IC-weighted):
Composite score: {s['composite_score']:.3f} (-1=strong sell, +1=strong buy)
Signal: {s['signal']} | Conviction: {s['conviction']}/100
Components: {json.dumps({k: round(v,3) for k,v in s['components'].items()})}

TECHNICAL FEATURES:
RSI(14): {f.get('rsi_14','?'):.1f} | RSI(5): {f.get('rsi_5','?'):.1f}
Momentum 5d: {f.get('mom_5d','?'):.2f}% | 21d: {f.get('mom_21d','?'):.2f}% | 63d: {f.get('mom_63d','?'):.2f}% | 252d: {f.get('mom_252d','?'):.2f}%
Distance from MA50: {f.get('dist_ma50','?'):.2f}% | MA200: {f.get('dist_ma200','?'):.2f}%
Golden cross: {'YES' if f.get('golden_cross',0)>0.5 else 'NO'}
Bollinger %B: {f.get('bb_pct','?'):.2f} | BB Width: {f.get('bb_width','?'):.2f}%
Relative Volume: {f.get('rel_vol','?'):.2f}x

STATISTICAL / RISK:
Hurst exponent: {f.get('hurst_exponent','?'):.3f} (>0.5=trending, <0.5=mean-reverting)
Fracdiff d*: {f.get('fracdiff_d','?')} (memory preservation)
Skewness: {f.get('skewness','?'):.3f} | Excess Kurtosis: {f.get('excess_kurtosis','?'):.3f}
1d VaR(95%): {f.get('var_95_1d','?'):.2f}% | CVaR(95%): {f.get('cvar_95_1d','?'):.2f}%
Drawdown (current): {f.get('drawdown_pct','?'):.1f}%
Regime class: {f.get('regime_class','?')}/4 (0=bear, 4=bull)

OPTIONS MARKET:
{'Available: Put/Call ratio: ' + str(round(op.get('put_call_vol','?'),2)) + 
 ' | IV Skew (put-call): ' + str(round(op.get('iv_skew','?'),3)) +
 ' | Smart money bias: ' + str(op.get('smart_money_bias','?'))
 if op.get('iv_available') else 'Options data not available'}

KELLY POSITION SIZING (c=0.25 fraction):
Recommended position: {computed['kelly']['kelly_fraction_pct']:.2f}% of ${account_size:,.0f} portfolio
= ${account_size * computed['kelly']['kelly_fraction_pct'] / 100:,.0f}

Now synthesize this into actionable narrative. Be specific. Reference the actual numbers."""


# ── FastAPI endpoints ─────────────────────────────────────────────────────────

engine = PriceOracleEngine()  # singleton

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Full price prediction pipeline:
    1. Fetch real market data (yFinance)
    2. Run ML computations (GARCH, Monte Carlo, features)
    3. Claude synthesizes narrative
    4. Return combined response
    """
    ticker = request.ticker.upper().strip()
    
    if len(ticker) > 6 or not ticker.isalpha():
        raise HTTPException(400, f"Invalid ticker: {ticker}")
    
    # ── Step 1: Run computational engine ──────────────────────────────────
    try:
        logger.info(f"Running price oracle for {ticker}")
        loop = asyncio.get_running_loop()
        computed = await loop.run_in_executor(
            None, engine.compute, ticker
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Engine error for {ticker}: {e}", exc_info=True)
        raise HTTPException(500, f"Computation failed: {str(e)}")
    
    # ── Step 2: Claude synthesis ───────────────────────────────────────────
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    narrative = {}
    
    if anthropic_key:
        try:
            prompt = build_claude_prompt(ticker, computed, request.account_size)
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": anthropic_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 1500,
                        "system": CLAUDE_SYSTEM,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                )
            raw  = resp.json().get("content", [{}])[0].get("text", "{}")
            s, e = raw.find("{"), raw.rfind("}")
            if s != -1:
                narrative = json.loads(raw[s:e+1])
        except Exception as ex:
            logger.warning(f"Claude synthesis failed: {ex}")
            narrative = {
                "primary_thesis": "Narrative synthesis unavailable.",
                "one_line_summary": computed["signal"]["signal"],
            }
    else:
        narrative = {"primary_thesis": "Set ANTHROPIC_API_KEY for narrative synthesis."}
    
    # ── Step 3: Assemble response ──────────────────────────────────────────
    f  = computed["features"]
    s  = computed["signal"]
    k  = computed["kelly"]
    co = computed["company"]
    g  = computed["garch"]
    op = computed["options"]
    
    return PredictResponse(
        ticker        = ticker,
        company_name  = narrative.get("company_name", co.get("name", ticker)),
        sector        = narrative.get("sector", co.get("sector", "Unknown")),
        current_price = computed["current_price"],
        signal        = s["signal"],
        conviction    = s["conviction"],
        kelly_pct     = round(k["kelly_fraction_pct"], 2),
        
        horizons = {
            label: {
                "bear_price":     h["bear_price"],
                "base_price":     h["base_price"],
                "bull_price":     h["bull_price"],
                "expected_price": h["expected_price"],
                "expected_return":h["expected_return"],
                "prob_positive":  h["prob_positive"],
                "vol_forecast":   h["vol_forecast_ann"],
                "var_95_1d":      h["var_95_1d"],
                "bear_return":    h["bear_return_pct"],
                "base_return":    h["base_return_pct"],
                "bull_return":    h["bull_return_pct"],
            }
            for label, h in computed["horizons"].items()
        },
        
        levels = computed["levels"],
        
        features_summary = {
            "rsi_14":        round(f.get("rsi_14", 50), 1),
            "rsi_5":         round(f.get("rsi_5", 50), 1),
            "mom_5d":        round(f.get("mom_5d", 0), 2),
            "mom_21d":       round(f.get("mom_21d", 0), 2),
            "mom_63d":       round(f.get("mom_63d", 0), 2),
            "mom_252d":      round(f.get("mom_252d", 0), 2),
            "rel_vol":       round(f.get("rel_vol", 1), 2),
            "hurst":         round(f.get("hurst_exponent", 0.5), 3),
            "dist_ma50":     round(f.get("dist_ma50", 0), 2),
            "dist_ma200":    round(f.get("dist_ma200", 0), 2),
            "drawdown":      round(f.get("drawdown_pct", 0), 2),
            "rv21":          round(f.get("rv21", 20), 2),
            "skewness":      round(f.get("skewness", 0), 3),
            "excess_kurtosis": round(f.get("excess_kurtosis", 0), 3),
            "var_95_1d":     round(f.get("var_95_1d", -2), 3),
            "bb_pct":        round(f.get("bb_pct", 0.5), 3),
            "golden_cross":  bool(f.get("golden_cross", 0) > 0.5),
            "regime_class":  int(f.get("regime_class", 2)),
            "fracdiff_d":    f.get("fracdiff_d", 0.5),
            "acf_lag1":      round(f.get("acf_lag1", 0), 3),
        },
        
        garch_summary = {
            "model":       g["model"],
            "current_vol": g["current_vol"],
            "long_run_vol":g["long_run_vol"],
            "persistence": g["persistence"],
            "asymmetric":  g["asymmetric"],
        },
        
        options_summary = {
            "available":      op.get("iv_available", False),
            "put_call_vol":   op.get("put_call_vol", 1.0),
            "iv_skew":        op.get("iv_skew", 0.0),
            "smart_money":    op.get("smart_money_bias", "FLAT"),
            "options_signal": op.get("options_signal", "NEUTRAL"),
        },
        
        narrative = narrative,
        
        models_used = computed["models_used"],
        
        disclaimer = (
            "All price targets are probabilistic outputs from statistical models, "
            "not guaranteed predictions. Past statistical patterns do not guarantee "
            "future results. This is not financial advice. "
            "Models: GJR-GARCH(1,1), Monte Carlo (10K paths), "
            "IC-weighted signals, Fractional Kelly."
        )
    )


@router.get("/health")
async def health():
    return {"status": "ok", "engine": "QuantEdge Price Oracle v1.0"}


@router.get("/methodology")
async def methodology():
    """Returns complete methodology documentation."""
    return {
        "models": {
            "volatility": {
                "name": "GJR-GARCH(1,1) with Student-t errors",
                "paper": "Glosten, Jagannathan & Runkle (1993)",
                "what_it_does": "Forecasts conditional volatility. Captures vol clustering and asymmetry (bad news spikes vol more than good news). Student-t handles fat tails.",
                "parameters": "omega (long-run variance), alpha (ARCH), gamma (asymmetry), beta (GARCH), nu (tail thickness)",
            },
            "price_distribution": {
                "name": "Monte Carlo GBM + Student-t innovations",
                "paths": 10000,
                "what_it_does": "Simulates 10,000 price paths. P10=bear case, P50=base case, P90=bull case. Not a single point prediction.",
                "key_insight": "Price prediction is inherently probabilistic. Distributional output is more honest than a single number.",
            },
            "signals": {
                "name": "IC-weighted signal aggregation",
                "paper": "Grinold & Kahn (1999) — Active Portfolio Management",
                "signals": ["63d momentum (IC=0.04)", "RSI mean reversion (IC=0.03)", "MACD (IC=0.025)", "Volume confirmation (IC=0.02)", "Hurst persistence (IC=0.03)", "Volatility regime (IC=0.02)", "Golden cross (IC=0.025)"],
            },
            "position_sizing": {
                "name": "Fractional Kelly criterion",
                "paper": "Thorp (1962). Kelly (1956).",
                "fraction": 0.25,
                "what_it_does": "Quarter-Kelly is conservative. Maximizes long-run geometric growth while limiting drawdown.",
            },
            "memory": {
                "name": "Hurst exponent (R/S analysis)",
                "paper": "Hurst (1951)",
                "interpretation": "H>0.55: trending (momentum works). H<0.45: mean-reverting (RSI/fade works). H≈0.5: random walk.",
            },
            "stationarity": {
                "name": "Fractional differentiation",
                "paper": "Lopez de Prado (2018) Ch.5",
                "what_it_does": "Finds minimum d* to make price series stationary while preserving maximum memory. Avoids throwing away information by overdifferencing.",
            },
        },
        "data_sources": {
            "price_data": "yFinance (2-year history, daily OHLCV)",
            "fundamentals": "yFinance (P/E, beta, market cap, short interest)",
            "options": "yFinance (nearest expiry chain)",
            "limitations": [
                "yFinance has survivorship bias (delisted stocks missing)",
                "No point-in-time fundamentals (look-ahead bias in financials)",
                "Options data may have stale quotes outside market hours",
                "For $1M+ trading: use Polygon.io or Bloomberg instead",
            ],
        },
        "academic_references": [
            "Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.",
            "Glosten, L., Jagannathan, R., Runkle, D. (1993). GJR-GARCH. Journal of Finance.",
            "Grinold, R., Kahn, R. (1999). Active Portfolio Management. McGraw-Hill.",
            "Jegadeesh, N., Titman, S. (1993). Returns to Buying Winners. Journal of Finance.",
            "Moskowitz, T., Ooi, Y., Pedersen, L. (2012). Time Series Momentum. Journal of Financial Economics.",
            "Fama, E., French, K. (2018). Choosing Factors. Journal of Financial Economics.",
            "Thorp, E. (1962). Beat the Dealer. Kelly Criterion.",
            "Hurst, H. (1951). Long-term Storage Capacity of Reservoirs. ASCE Transactions.",
            "Almgren, R., Chriss, N. (2001). Optimal Execution. Journal of Risk.",
        ],
    }
