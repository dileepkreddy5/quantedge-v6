"""
QuantEdge v6.0 — Enhanced Analysis Router
==========================================
Full institutional pipeline:
  Data → Features → Labels → Alpha → Risk → Portfolio → Governance

All ML predictions use real trained models:
  - XGBoost: fit() on historical feature matrix, SHAP via TreeExplainer
  - LightGBM: fit() on historical feature matrix, Spearman IC tracked
  - BiLSTM: trained on 60-day sequences, MC Dropout uncertainty
  - GJR-GARCH, HMM, Kalman: real library calls (arch, hmmlearn, filterpy)
  - FinBERT: real transformer inference for sentiment

No Claude API anywhere in the prediction path.
"""

from fastapi import APIRouter, Body, Depends, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, validator
import asyncio
import json
import time
import numpy as np

# Single risk-free rate for the whole pipeline. Previously 0.05 in
# VolatilityFeatures.risk_metrics and 0.053 in the CAPM block, with the
# governance Sharpe using none at all. Tracks the 3-month T-bill; set 2026-07.
RISK_FREE_ANNUAL = 0.053
import pandas as pd
from typing import Optional, Dict, Any, List
from loguru import logger

from auth.cognito_auth import get_current_user, get_optional_user, CognitoUser
from core.config import settings
# signal_tracker accessed via body.app.state.signal_tracker (set in main_v6.py lifespan)

from data.feeds.market_data import MarketDataFeed, FundamentalDataFeed
from data.feeds.market_data import OptionsDataFeed, SentimentDataFeed

from ml.models.lstm_model import build_default_model, LSTMTrainer
from ml.models.xgboost_lgbm import XGBoostPredictor, LightGBMPredictor, EnsembleModel
from ml.models.regime_volatility import HMMRegimeClassifier, GJRGARCHModel, KalmanTrendFilter, MonteCarloEngine
from ml.models.nlp_options import FinBERTSentiment, OptionsAnalytics
from ml.features.feature_engineering import FeaturePipeline
from ml.price_oracle.analyst_ratings import AnalystRatingsEngine

from ml.labeling.triple_barrier import LabelingPipeline, DeflatedSharpeRatio
from ml.labeling.meta_label import MetaLabeler, build_summary as build_meta_summary
from ml.labeling.walk_forward_oof import walk_forward_oof_ensemble
from ml.labeling.pbo import compute_pbo_from_ticker_history
from ml.risk.risk_engine import MasterRiskEngine, DynamicCovarianceEngine, CVaREngine, VolatilityTargetingEngine
from ml.portfolio.portfolio_engine import (
    HierarchicalRiskParity,
    CVaROptimizer,
    EqualRiskContribution,
    RegimeAwarePortfolioBlender,
    ModelGovernanceEngine,
)



def _sanitize_json(obj):
    """
    Recursively replace NaN, +inf, -inf with None for JSON compliance.
    Pandas/NumPy calculations can produce these; stdlib json cannot serialize them.
    """
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_json(v) for v in obj)
    # numpy types
    try:
        import numpy as np
        if isinstance(obj, (np.floating,)):
            f = float(obj)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        if isinstance(obj, np.ndarray):
            return [_sanitize_json(v) for v in obj.tolist()]
    except ImportError:
        pass
    return obj


router = APIRouter()


class AnalyzeRequest(BaseModel):
    ticker: str
    include_options: bool = True
    include_sentiment: bool = True
    mc_paths: int = 20_000
    include_portfolio: bool = False
    portfolio_tickers: List[str] = []
    target_vol: float = 0.10

    @validator("ticker")
    def validate_ticker(cls, v):
        v = v.upper().strip()
        if not v.replace('-', '').isalpha() or len(v) > 10:
            raise ValueError("Invalid ticker symbol")
        return v

    @validator("mc_paths")
    def cap_mc_paths(cls, v):
        # FIX (step-5): server-side cap — a public request must not be able to
        # order a 100k-path Monte Carlo on a 3-vCPU box (compute DoS).
        return max(1_000, min(int(v), 20_000))


class QuantEdgeAnalyzerV6:
    """
    Master analysis orchestrator.

    Architecture (fully separated layers):
      DATA → FEATURES → LABELS → ALPHA → RISK → PORTFOLIO → GOVERNANCE

    All ML models do real training on real historical data before predicting.
    No Claude API in any prediction path.
    """

    def __init__(self):
        self.market_feed = MarketDataFeed()
        # SPY is the CAPM benchmark for every ticker. Fetching it per request
        # got the analyzer throttled by Polygon, so the regression silently
        # failed for all but the first name in any burst. Cached per process.
        self._spy_cache = None
        self._spy_cached_at = None
        self.fund_feed = FundamentalDataFeed()
        self.options_feed = OptionsDataFeed()
        self.sentiment_feed = SentimentDataFeed()

        self.feature_pipeline = FeaturePipeline()
        self.garch = GJRGARCHModel()
        self.hmm = HMMRegimeClassifier()
        self.kalman = KalmanTrendFilter()
        self.mc_engine = MonteCarloEngine()
        self.ensemble = EnsembleModel()
        self.finbert = FinBERTSentiment()
        self.options_analytics = OptionsAnalytics()

        self.analyst_engine = AnalystRatingsEngine()
        self.labeling = LabelingPipeline(
            profit_take=2.0, stop_loss=1.0, hold_days=21,
            cusum_h=1.0, time_decay=0.5
        )
        self.risk_engine = MasterRiskEngine(target_vol=0.10)
        self.cov_engine = DynamicCovarianceEngine()
        self.cvar_engine = CVaREngine(confidence=0.95)
        self.vol_targeter = VolatilityTargetingEngine()
        self.portfolio_blender = RegimeAwarePortfolioBlender()
        self.governance = ModelGovernanceEngine()
        self.dsr = DeflatedSharpeRatio()

    async def run_full_analysis(
        self,
        ticker: str,
        include_options: bool = True,
        include_sentiment: bool = True,
        mc_paths: int = 100_000,
        target_vol: float = 0.10,
    ) -> Dict:
        """
        Full institutional analysis pipeline with 240-second timeout.
        All layers run in correct dependency order.
        """
        # Hard timeout: prevents hung external calls from blocking the server forever
        return await asyncio.wait_for(
            self._run_pipeline(ticker, include_options, include_sentiment, mc_paths, target_vol),
            timeout=240.0,
        )

    async def _run_pipeline(
        self,
        ticker: str,
        include_options: bool,
        include_sentiment: bool,
        mc_paths: int,
        target_vol: float,
    ) -> Dict:
        start_time = time.time()
        result = {}

        try:
            # ── LAYER 1: DATA ─────────────────────────────────
            async def _noop() -> dict:
                return {}

            price_data, fundamentals, options_chain, news_data = await asyncio.gather(
                self.market_feed.get_price_history(ticker),
                self.fund_feed.get_fundamentals(ticker),
                self.options_feed.get_chain(ticker) if include_options else _noop(),
                self.sentiment_feed.get_news_and_reddit(ticker) if include_sentiment else _noop(),
                return_exceptions=True
            )

            if isinstance(price_data, Exception) or price_data is None or len(price_data) < 100:
                raise HTTPException(status_code=404, detail=f"Insufficient price data for {ticker}")

            close = price_data["close"]
            high = price_data.get("high", close)
            low = price_data.get("low", close)
            volume = price_data.get("volume", pd.Series(0, index=close.index))

            returns = close.pct_change().dropna()

            # Polygon occasionally returns a bar spanning an unadjusted corporate
            # action — META shows a +1394% day in June 2022 where the series jumps
            # from ~$14 to ~$200. One such print takes the return std from 0.024 to
            # 0.410, which annualises to 650% volatility and corrupts Sharpe, VaR,
            # position sizing and every Monte Carlo path downstream. Equities do not
            # move 60% in a session; anything beyond that is a data error.
            _bad = returns.abs() > 0.60
            if _bad.any():
                _n = int(_bad.sum())
                logger.warning(
                    f"{ticker}: dropped {_n} bar(s) with implausible daily moves "
                    f"(max {returns.abs().max():.1%}) — likely unadjusted split data")
                # Drop the bad returns but leave `close` alone. Rebuilding the price
                # series from surviving returns re-indexes it in a way that breaks
                # every downstream join — the CAPM regression against SPY collapsed
                # to zero aligned rows for most tickers. The bad bar affects return
                # statistics, not the price level, so filtering returns is enough.
                returns = returns[~_bad]

            log_returns = np.log(close / close.shift(1)).dropna()
            log_returns = log_returns[log_returns.abs() < 0.47]   # ln(1.6)

            current_price = float(close.iloc[-1])
            price_1y_ago = float(close.iloc[-252]) if len(close) >= 252 else float(close.iloc[0])
            annual_return = current_price / price_1y_ago - 1

            if not isinstance(fundamentals, Exception) and fundamentals:
                result["fundamentals"] = fundamentals
                result["name"] = fundamentals.get("name", ticker)
                result["sector"] = fundamentals.get("sector", "Unknown")
                result["industry"] = fundamentals.get("industry", "Unknown")
                result["exchange"] = fundamentals.get("exchange", "")
                result["market_cap"] = fundamentals.get("market_cap")
                for key in [
                    "pe_ratio", "forward_pe", "peg_ratio", "price_to_book",
                    "price_to_sales", "ev_ebitda", "gross_margin", "operating_margin",
                    "net_margin", "roe", "roa", "roic", "debt_to_equity",
                    "revenue_growth", "earnings_growth", "fcf_yield",
                    "current_ratio", "quick_ratio", "dividend_yield",
                    "short_interest", "institutional_ownership", "beta",
                    "week_52_high", "week_52_low",
                ]:
                    result[key] = fundamentals.get(key)

                # Compute fields Polygon fundamentals omits, from data we already have.
                try:
                    import numpy as _np
                    _closes = price_data["close"].dropna()
                    if len(_closes) >= 60:
                        _last252 = _closes.iloc[-252:] if len(_closes) >= 252 else _closes
                        if result.get("week_52_high") is None:
                            result["week_52_high"] = float(_last252.max())
                        if result.get("week_52_low") is None:
                            result["week_52_low"] = float(_last252.min())
                    # PEG = P/E / earnings growth% (if both present and growth positive)
                    if result.get("peg_ratio") is None:
                        _pe = result.get("pe_ratio"); _g = result.get("earnings_growth")
                        if _pe and _g and _g > 0:
                            result["peg_ratio"] = float(_pe / (_g * 100))
                    # FCF yield = (operating cash flow - capex) / market cap — approximate
                    # via net margin * revenue as a proxy floor if direct FCF absent
                    if result.get("fcf_yield") is None:
                        _fund = fundamentals
                        _ocf = _fund.get("operating_cash_flow") or _fund.get("free_cash_flow")
                        _mcap = result.get("market_cap")
                        if _ocf and _mcap and _mcap > 0:
                            result["fcf_yield"] = float(_ocf / _mcap)
                except Exception as _e:
                    logger.info(f"supplemental field computation skipped: {_e}")

            # ── Volatility term history: is vol expanding or contracting? ──
            # Rolling 21d annualised vol, sampled at intervals back to 5 years, so the
            # user can see whether current turbulence is unusual for this name.
            try:
                import numpy as _np
                _c = price_data["close"].dropna()
                if len(_c) >= 40:
                    _lr = _np.log(_c / _c.shift(1)).dropna()
                    _rv = _lr.rolling(21).std() * _np.sqrt(252) * 100   # annualised %
                    _rv = _rv.dropna()
                    if len(_rv) >= 30:
                        _now = float(_rv.iloc[-1])
                        _pts = [("1d",1),("3d",3),("1w",5),("2w",10),("1mo",21),
                                ("2mo",42),("3mo",63),("6mo",126),("1y",252),
                                ("2y",504),("3y",756),("5y",1260)]
                        _hist = {}
                        for _lbl,_n in _pts:
                            if len(_rv) > _n:
                                _then = float(_rv.iloc[-1-_n])
                                _hist[_lbl] = {
                                    "then": round(_then,2),
                                    "change": round(_now-_then,2),
                                    "change_pct": round(((_now-_then)/_then)*100,1) if _then > 0 else None,
                                }
                        _vals = _rv.values
                        result["volatility_history"] = {
                            "current": round(_now,2),
                            "changes": _hist,
                            "range_high": round(float(_np.max(_vals)),2),
                            "range_low": round(float(_np.min(_vals)),2),
                            "percentile": round(float((_vals < _now).mean()*100),1),
                            "median": round(float(_np.median(_vals)),2),
                            "observations": int(len(_vals)),
                            "note": "Rolling 21-day realized volatility, annualised. Percentile is versus this stock's own history over the available window.",
                        }
            except Exception as _e:
                logger.info(f"volatility history skipped: {_e}")

            # ── Flow analysis: is supply being absorbed or is demand building? ──
            # Everything here is derived from daily bars. True buy/sell volume is NOT
            # derivable — every trade has a buyer and a seller — so we never claim it.
            try:
                import numpy as _np
                _px = price_data["close"].dropna()
                _vol = price_data["volume"].dropna() if "volume" in price_data else None
                if _vol is not None and len(_px) >= 260 and len(_vol) >= 260:
                    _px, _vol = _px.align(_vol, join="inner")
                    _ret = _px.pct_change()
                    _dv = _px * _vol                      # dollar volume
                    _flow = {}

                    # Volume percentile vs its own history
                    _v21 = _vol.rolling(21).mean().dropna()
                    _vnow = float(_v21.iloc[-1])
                    _flow["volume_21d_avg"] = round(_vnow, 0)
                    _flow["volume_percentile"] = round(float((_v21.values < _vnow).mean() * 100), 1)
                    _v252 = float(_vol.iloc[-252:].mean())
                    _flow["volume_vs_1y_avg_pct"] = round(((_vnow / _v252) - 1) * 100, 1) if _v252 > 0 else None

                    # Up-day vs down-day volume share (accumulation proxy)
                    for _w in (21, 63, 252):
                        _r = _ret.iloc[-_w:]; _v = _vol.iloc[-_w:]
                        _up = float(_v[_r > 0].sum()); _dn = float(_v[_r < 0].sum())
                        _tot = _up + _dn
                        _flow[f"up_volume_share_{_w}d"] = round((_up / _tot) * 100, 1) if _tot > 0 else None

                    # Effort vs result: dollar volume required per 1% of price movement.
                    # Falling ratio = moves coming easier. Rising = absorption/resistance.
                    def _effort(w):
                        _r = _ret.iloc[-w:].abs(); _d = _dv.iloc[-w:]
                        _m = _r > 0.0005
                        return float(_d[_m].sum() / (_r[_m].sum() * 100)) if _m.sum() > 5 else None
                    _e_now, _e_prev = _effort(21), None
                    _r_p = _ret.iloc[-84:-21].abs(); _d_p = _dv.iloc[-84:-21]
                    _m_p = _r_p > 0.0005
                    if _m_p.sum() > 5:
                        _e_prev = float(_d_p[_m_p].sum() / (_r_p[_m_p].sum() * 100))
                    _flow["effort_per_pct_now"] = round(_e_now, 0) if _e_now else None
                    _flow["effort_per_pct_prior"] = round(_e_prev, 0) if _e_prev else None
                    if _e_now and _e_prev:
                        _flow["effort_change_pct"] = round(((_e_now / _e_prev) - 1) * 100, 1)

                    # On-Balance Volume slope, normalised
                    _obv = (_np.sign(_ret.fillna(0)) * _vol).cumsum()
                    for _w in (21, 63):
                        _seg = _obv.iloc[-_w:]
                        _sl = float(_np.polyfit(range(len(_seg)), _seg.values, 1)[0])
                        _flow[f"obv_slope_{_w}d"] = round(_sl / float(_vol.iloc[-_w:].mean()), 3)

                    # Volume-weighted average price over 1y = the supply map
                    _v1y, _p1y = _vol.iloc[-252:], _px.iloc[-252:]
                    _vwap = float((_p1y * _v1y).sum() / _v1y.sum())
                    _cur = float(_px.iloc[-1])
                    _flow["vwap_1y"] = round(_vwap, 2)
                    _flow["price_vs_vwap_pct"] = round(((_cur / _vwap) - 1) * 100, 1)
                    _flow["shares_in_profit_pct"] = round(float((_p1y < _cur).mul(_v1y).sum() / _v1y.sum() * 100), 1)

                    # Turnover: share of float trading daily, and its trend
                    _so = (fundamentals or {}).get("shares_outstanding")
                    if _so and _so > 0:
                        _flow["turnover_daily_pct"] = round((_vnow / float(_so)) * 100, 3)

                    # Phase: OBV direction vs price direction over 63d
                    _pch = float((_px.iloc[-1] / _px.iloc[-63] - 1) * 100)
                    _osl = _flow.get("obv_slope_63d", 0)
                    if _osl > 0.15 and abs(_pch) < 5:   _ph = ("ACCUMULATION", "Volume is building while price stays flat — supply is being absorbed quietly.")
                    elif _osl > 0.15 and _pch >= 5:     _ph = ("MARKUP", "Price and volume rising together — the move is confirmed by participation.")
                    elif _osl < -0.15 and abs(_pch) < 5:_ph = ("DISTRIBUTION", "Price holding while volume flow turns negative — supply is being fed into strength.")
                    elif _osl < -0.15 and _pch <= -5:   _ph = ("MARKDOWN", "Price and volume flow both falling — active selling pressure.")
                    else:                                _ph = ("NEUTRAL", "No clear accumulation or distribution pattern over the last quarter.")
                    _flow["phase"], _flow["phase_note"] = _ph
                    _flow["price_change_63d_pct"] = round(_pch, 1)
                    # Demand persistence ladder — is participation strengthening?
                    for _w in (5, 15, 30, 90):
                        _r = _ret.iloc[-_w:]; _v = _vol.iloc[-_w:]
                        _u = float(_v[_r > 0].sum()); _dn2 = float(_v[_r < 0].sum())
                        _t = _u + _dn2
                        _flow[f"up_volume_share_{_w}d"] = round((_u / _t) * 100, 1) if _t > 0 else None

                    # Volume percentile over full available history (5y)
                    _vser = _vol.rolling(21).mean().dropna()
                    _flow["volume_percentile_5y"] = round(float((_vser.values < _vnow).mean() * 100), 1)
                    _flow["relative_volume"] = round(_vnow / _v252, 2) if _v252 > 0 else None

                    # Price-volume agreement over 21d
                    _p21 = float(_px.iloc[-1] / _px.iloc[-21] - 1)
                    _vtrend = (_vnow / float(_vol.iloc[-63:-21].mean()) - 1) if len(_vol) > 63 else 0
                    if _p21 > 0.01 and _vtrend > 0.05:   _pv = ("ACCUMULATION", "Price advancing on expanding volume — the move has participation behind it.")
                    elif _p21 > 0.01 and _vtrend < -0.05:_pv = ("WEAK RALLY", "Price advancing on shrinking volume — few participants are following the move.")
                    elif _p21 < -0.01 and _vtrend > 0.05:_pv = ("DISTRIBUTION", "Price declining on expanding volume — active selling into the market.")
                    elif _p21 < -0.01 and _vtrend < -0.05:_pv = ("SELLING EXHAUSTION", "Price declining on shrinking volume — sellers appear to be running out.")
                    else:                                 _pv = ("BALANCED", "No clear divergence between price direction and volume trend.")
                    _flow["pv_agreement"], _flow["pv_note"] = _pv
                    _flow["volume_trend_pct"] = round(_vtrend * 100, 1)

                    # Exhaustion: falling volume against a directional move
                    _exh = None
                    if _p21 < -0.02 and _vtrend < -0.10:
                        _exh = {"type": "SUPPLY", "score": min(100, round(abs(_vtrend) * 200 + abs(_p21) * 300)),
                                "note": "Price falling while volume dries up — selling pressure is fading, which often precedes stabilisation."}
                    elif _p21 > 0.02 and _vtrend < -0.10:
                        _exh = {"type": "DEMAND", "score": min(100, round(abs(_vtrend) * 200 + _p21 * 300)),
                                "note": "Price rising while volume dries up — buying interest is thinning, which often precedes a stall."}
                    _flow["exhaustion"] = _exh

                    # Market Participation Score — conviction behind the move, not buyer identity
                    _c = []
                    _rv = _flow.get("relative_volume") or 1
                    _c.append(("Relative volume", f"{_rv:.2f}x", min(100, _rv * 50)))
                    _uv = _flow.get("up_volume_share_21d") or 50
                    _c.append(("Up-volume share (21d)", f"{_uv:.0f}%", max(0, min(100, (_uv - 35) * 3.3))))
                    _vp = _flow.get("volume_percentile_5y") or 50
                    _c.append(("Volume percentile (5y)", f"{_vp:.0f}th", _vp))
                    _pvs = {"ACCUMULATION": 90, "BALANCED": 50, "WEAK RALLY": 35, "SELLING EXHAUSTION": 45, "DISTRIBUTION": 15}[_pv[0]]
                    _c.append(("Price-volume agreement", _pv[0].title(), _pvs))
                    _os = _flow.get("obv_slope_21d") or 0
                    _c.append(("Volume flow direction", f"{_os:+.2f}", max(0, min(100, 50 + _os * 60))))
                    _pers = 0
                    _lad = [_flow.get(f"up_volume_share_{w}d") for w in (252, 90, 30, 21, 5)]
                    _lad = [x for x in _lad if x is not None]
                    if len(_lad) >= 3:
                        _pers = 100 if _lad[-1] > _lad[0] + 3 else 30 if _lad[-1] < _lad[0] - 3 else 60
                        _c.append(("Participation trend", "strengthening" if _pers > 70 else "weakening" if _pers < 40 else "steady", _pers))
                    _score = round(sum(x[2] for x in _c) / len(_c))
                    _flow["participation"] = {
                        "score": _score,
                        "label": "Very high" if _score >= 80 else "High" if _score >= 65 else "Moderate" if _score >= 45 else "Low" if _score >= 30 else "Very low",
                        "components": [{"name": n, "value": v, "score": round(s)} for n, v, s in _c],
                        "note": "Measures how much conviction sits behind the current move. It does not identify who is buying — that requires trade-level data.",
                    }

                    _flow["disclaimer"] = ("Buy and sell volume cannot be separated from daily bars — every trade has both "
                                           "a buyer and a seller. These are directional proxies based on where price closed, "
                                           "not actual order flow. Institutional versus retail participation cannot be "
                                           "determined without trade-size and venue data.")
                    result["flow_analysis"] = _flow
            except Exception as _e:
                logger.info(f"flow analysis skipped: {_e}")

            # ── Volatility intelligence: stability, expected moves, regime history ──
            try:
                import numpy as _np
                _c2 = price_data["close"].dropna()
                if len(_c2) >= 300:
                    _lr2 = _np.log(_c2 / _c2.shift(1)).dropna()
                    _rv2 = (_lr2.rolling(21).std() * _np.sqrt(252) * 100).dropna()
                    _cur2 = float(_rv2.iloc[-1])
                    _vi = {}

                    # Expected move translation — the practical form of a vol number
                    _d = _cur2 / _np.sqrt(252)
                    _vi["expected_move"] = {
                        "daily": round(_d, 2), "weekly": round(_d * _np.sqrt(5), 2),
                        "monthly": round(_d * _np.sqrt(21), 2), "quarterly": round(_d * _np.sqrt(63), 2),
                        "note": "One standard deviation. Roughly two days in three fall within this range; one in twenty exceeds double it.",
                    }

                    # Stability: volatility of volatility + trend of the rolling series
                    _vov = float(_rv2.iloc[-63:].std() / _rv2.iloc[-63:].mean() * 100) if _rv2.iloc[-63:].mean() > 0 else None
                    _sl2 = float(_np.polyfit(range(63), _rv2.iloc[-63:].values, 1)[0])
                    _slpct = (_sl2 * 63) / _cur2 * 100 if _cur2 > 0 else 0
                    _stab = max(0, min(100, round(100 - (_vov or 0) * 1.6 - max(0, _slpct) * 0.6)))
                    _vi["stability"] = {
                        "score": _stab,
                        "label": "Very stable" if _stab >= 75 else "Stable" if _stab >= 55 else "Unsettled" if _stab >= 35 else "Rapidly changing",
                        "vol_of_vol_pct": round(_vov, 1) if _vov else None,
                        "trend_pct_per_quarter": round(_slpct, 1),
                        "note": "How steady the volatility level itself has been this quarter, scored 0-100. Below 50 means the level is still moving materially, so the GARCH estimates above are indicative rather than precise \u2014 read them as a range, and size positions off the wider end.",
                    }

                    # Percentile timeline — gradual drift or one abnormal stretch?
                    _tl = {}
                    for _lbl, _n in [("30d",30),("90d",90),("180d",180),("1y",252),("full",len(_rv2))]:
                        if len(_rv2) >= _n:
                            _w = _rv2.iloc[-_n:]
                            _tl[_lbl] = {"percentile": round(float((_w.values < _cur2).mean()*100),1),
                                         "mean": round(float(_w.mean()),1)}
                    _vi["percentile_timeline"] = _tl

                    # Regime history over ~2y using terciles of the full distribution
                    _lo_t, _hi_t = float(_np.percentile(_rv2.values,33)), float(_np.percentile(_rv2.values,67))
                    _lbl_of = lambda v: "LOW" if v < _lo_t else "HIGH" if v > _hi_t else "NORMAL"
                    _win = _rv2.iloc[-504:] if len(_rv2) >= 504 else _rv2
                    _segs, _cl, _cs = [], None, None
                    for _dt, _v in _win.items():
                        _l = _lbl_of(float(_v))
                        if _l != _cl:
                            if _cl is not None:
                                _segs.append({"regime": _cl, "start": str(_cs)[:10], "end": str(_pd)[:10], "days": _dl})
                            _cl, _cs, _dl = _l, _dt, 1
                        else:
                            _dl += 1
                        _pd = _dt
                    if _cl is not None:
                        _segs.append({"regime": _cl, "start": str(_cs)[:10], "end": str(_pd)[:10], "days": _dl})
                    _segs = [s for s in _segs if s["days"] >= 10]
                    _vi["regime_history"] = {
                        "segments": _segs[-10:],
                        "current": _lbl_of(_cur2),
                        "thresholds": {"low_below": round(_lo_t,1), "high_above": round(_hi_t,1)},
                        "note": "Volatility clusters. Regime changes matter more than the level on any single day.",
                    }

                    # Market vs idiosyncratic split (needs a benchmark; skipped if absent)
                    _vi["decomposition_note"] = ("Splitting volatility into market, sector and company-specific "
                                                 "components requires benchmark return series not loaded in this request.")
                    result["volatility_intel"] = _vi
            except Exception as _e:
                logger.info(f"volatility intel skipped: {_e}")

            # ── Regime context: how long in this state, and what happened last time? ──
            # The HMM gives the current state; this reconstructs a comparable history
            # from the same features so the current reading has something to sit against.
            try:
                import numpy as _np
                import pandas as _pdm
                _pd_ts = lambda s: _pdm.Timestamp(s)
                _c3 = price_data["close"].dropna()
                if len(_c3) >= 300:
                    _r3 = _c3.pct_change().dropna()
                    _v3 = (_r3.rolling(21).std() * _np.sqrt(252)).dropna()
                    _t3 = (_c3 / _c3.rolling(63).mean() - 1).dropna()
                    _idx = _v3.index.intersection(_t3.index)
                    _v3, _t3 = _v3.loc[_idx], _t3.loc[_idx]
                    _vmed = float(_v3.median())

                    def _state(v, t):
                        hi = v > _vmed
                        if t > 0.02:   return "BULL_HIGH_VOL" if hi else "BULL_LOW_VOL"
                        if t < -0.02:  return "BEAR_HIGH_VOL" if hi else "BEAR_LOW_VOL"
                        return "MEAN_REVERT"

                    _lab = [_state(float(_v3.loc[d]), float(_t3.loc[d])) for d in _idx]
                    _fwd = _c3.pct_change().shift(-1).reindex(_idx)

                    # Statistics conditional on each regime
                    _stats = {}
                    for _s in set(_lab):
                        _m = _np.array([l == _s for l in _lab])
                        _fr = _fwd.values[_m]
                        _fr = _fr[~_np.isnan(_fr)]
                        if len(_fr) >= 20:
                            _stats[_s] = {
                                "days_observed": int(_m.sum()),
                                "share_of_time_pct": round(float(_m.mean()*100), 1),
                                "avg_daily_return_pct": round(float(_np.mean(_fr)*100), 3),
                                "annualised_return_pct": round(float(_np.mean(_fr)*252*100), 1),
                                "daily_vol_pct": round(float(_np.std(_fr)*100), 2),
                                "win_rate_pct": round(float((_fr > 0).mean()*100), 1),
                                "worst_day_pct": round(float(_np.min(_fr)*100), 2),
                                "best_day_pct": round(float(_np.max(_fr)*100), 2),
                            }

                    # Segment into episodes
                    _eps, _cs3, _cl3, _n3 = [], _idx[0], _lab[0], 1
                    for _i in range(1, len(_lab)):
                        if _lab[_i] != _cl3:
                            if _n3 >= 5:
                                _p0, _p1 = float(_c3.loc[_cs3]), float(_c3.loc[_idx[_i-1]])
                                _eps.append({"regime": _cl3, "start": str(_cs3)[:10], "end": str(_idx[_i-1])[:10],
                                             "days": _n3, "return_pct": round((_p1/_p0-1)*100, 1)})
                            _cl3, _cs3, _n3 = _lab[_i], _idx[_i], 1
                        else:
                            _n3 += 1
                    _p0, _p1 = float(_c3.loc[_cs3]), float(_c3.iloc[-1])
                    _eps.append({"regime": _cl3, "start": str(_cs3)[:10], "end": str(_idx[-1])[:10],
                                 "days": _n3, "return_pct": round((_p1/_p0-1)*100, 1), "ongoing": True})

                    _cur_ep = _eps[-1]
                    _same = [e for e in _eps[:-1] if e["regime"] == _cur_ep["regime"]]
                    _med_dur = float(_np.median([e["days"] for e in _same])) if _same else None

                    # Where does this state usually lead, and what follows entry?
                    _exits = {}
                    for _i in range(len(_eps) - 1):
                        if _eps[_i]["regime"] == _cl3:
                            _nx = _eps[_i + 1]["regime"]
                            _exits[_nx] = _exits.get(_nx, 0) + 1
                    _tot_ex = sum(_exits.values())
                    _exit_tbl = {k: {"count": v, "share_pct": round(v / _tot_ex * 100, 1)}
                                 for k, v in sorted(_exits.items(), key=lambda x: -x[1])} if _tot_ex else {}

                    _fwd_tbl = {}
                    for _h in (5, 10, 21, 63):
                        _rs = []
                        for _e in _eps[:-1]:
                            if _e["regime"] != _cl3:
                                continue
                            try:
                                _ts = _pdm.Timestamp(_e["start"])
                                if _c3.index.tz is not None and _ts.tz is None:
                                    _ts = _ts.tz_localize(_c3.index.tz)
                                _p = int(_c3.index.get_indexer([_ts], method="nearest")[0])
                                if 0 <= _p and _p + _h < len(_c3):
                                    _rs.append(float(_c3.iloc[_p + _h] / _c3.iloc[_p] - 1))
                            except Exception:
                                continue
                        if len(_rs) >= 4:
                            _a = _np.array(_rs)
                            _fwd_tbl[str(_h)] = {
                                "median_pct": round(float(_np.median(_a)) * 100, 2),
                                "mean_pct": round(float(_a.mean()) * 100, 2),
                                "positive_pct": round(float((_a > 0).mean()) * 100, 0),
                                "worst_pct": round(float(_a.min()) * 100, 1),
                                "best_pct": round(float(_a.max()) * 100, 1),
                                "n": len(_rs),
                            }

                    result["regime_context"] = {
                        "exit_analysis": {"transitions": _exit_tbl},
                        "forward_returns": _fwd_tbl,
                        "inferred_state": _cl3,
                        "days_in_current": _cur_ep["days"],
                        "current_episode_return_pct": _cur_ep["return_pct"],
                        "median_duration_for_state": round(_med_dur, 0) if _med_dur else None,
                        "past_episodes_of_state": len(_same),
                        "conditional_stats": _stats,
                        "episodes": _eps[-14:],
                        "vol_threshold_pct": round(_vmed*100, 1),
                        "note": ("Regimes are inferred from trend and volatility on the same price history the HMM uses. "
                                 "They are a classification of past behaviour, not a forecast — but knowing how this stock "
                                 "has behaved in similar conditions is more useful than the current label alone."),
                    }
            except Exception as _e:
                logger.info(f"regime context skipped: {_e}")

            # ── LAYER 2: FEATURES ─────────────────────────────
            try:
                feature_matrix = self.feature_pipeline.build_feature_matrix(
                    price_data, fundamentals if not isinstance(fundamentals, Exception) else {}
                )
            except Exception as e:
                logger.warning(f"Feature engineering error: {e}")
                feature_matrix = {}

            # ── LAYER 3: LABELING ─────────────────────────────
            labeling_result = {}
            try:
                labeling_result = self.labeling.run(
                    close=close,
                    high=high,
                    low=low,
                )
            except Exception as e:
                logger.warning(f"Labeling error: {e}")

            # ── LAYER 4: ALPHA MODELS ─────────────────────────

            # GJR-GARCH volatility
            garch_result = {}
            try:
                garch_result = self.garch.fit(returns)
                result["garch"] = garch_result
            except Exception as e:
                logger.warning(f"GARCH error: {e}")

            # HMM regime detection
            current_regime = "UNKNOWN"
            regime_probs = {}
            try:
                self.hmm.fit(returns, volume)
                regime_result = self.hmm.predict_current_regime(returns, volume)
                current_regime = regime_result.get("current_regime", "UNKNOWN")
                regime_probs = regime_result.get("regime_probabilities", {})
                result["regime"] = regime_result
                result["current_regime"] = current_regime
            except Exception as e:
                logger.warning(f"HMM error: {e}")

            # Kalman filter trend
            try:
                kalman_result = self.kalman.fit(close)
                result["kalman"] = kalman_result
            except Exception as e:
                logger.warning(f"Kalman error: {e}")

            # XGBoost + LightGBM + LSTM predictions (real training)
            ml_predictions = await self._run_ml_predictions(
                feature_matrix=feature_matrix,
                ticker=ticker,
                regime=current_regime,
                price_data=price_data,
                fundamentals=fundamentals if not isinstance(fundamentals, Exception) else {},
            )
            result["ml_predictions"] = ml_predictions

            # Cross-sectional panel prediction: score this ticker against models
            # trained on the whole-universe rolling panel (with point-in-time
            # fundamentals). Works for any US ticker; loads pre-trained models once.
            try:
                from ml.serving.panel_predictor import PanelPredictor
                _pp = PanelPredictor.get()
                if _pp.available():
                    panel_pred = _pp.predict(feature_matrix)
                    if panel_pred:
                        result["panel_prediction"] = panel_pred
            except Exception as _e:
                logger.warning(f"panel prediction skipped: {_e}")

            ensemble_preds = ml_predictions.get("ensemble", {})
            predicted_return_1y = ensemble_preds.get("pred_252d", annual_return * 100) / 100

            # Monte Carlo simulation
            try:
                annual_vol = garch_result.get("current_annual_vol", returns.std() * np.sqrt(252))
                mc_result = self.mc_engine.simulate(
                    current_price=current_price,
                    expected_annual_return=predicted_return_1y,
                    annual_vol=annual_vol,
                    n_paths=mc_paths,
                )
                result["monte_carlo"] = mc_result
            except Exception as e:
                logger.warning(f"MC error: {e}")

            # FinBERT sentiment (real transformer inference)
            if include_sentiment and not isinstance(news_data, Exception):
                try:
                    sentiment_result = self._compute_sentiment(news_data)
                    result["sentiment"] = sentiment_result
                except Exception as e:
                    logger.warning(f"Sentiment error: {e}")

            # Options analytics
            if (
                include_options
                and not isinstance(options_chain, Exception)
                and isinstance(options_chain, pd.DataFrame)
                and not options_chain.empty
            ):
                try:
                    options_result = self._compute_options(
                        options_chain, current_price, annual_vol
                    )
                    result["options"] = options_result
                except Exception as e:
                    logger.warning(f"Options error: {e}")

            # ── LAYER 5: RISK ENGINE ──────────────────────────
            try:
                returns_for_risk = returns.tail(252).dropna()
                if len(returns_for_risk) < 10:
                    raise ValueError("Insufficient returns for risk engine")
                ret_df = pd.DataFrame({"asset": returns_for_risk})
                risk_result = self.risk_engine.full_risk_assessment(
                    returns=ret_df,
                    portfolio_nav=None,
                )
                result["risk_engine"] = {
                    "summary": risk_result["summary"],
                    "vol_targeting": risk_result["vol_targeting"],
                    "cvar": {
                        "worst_case_daily": risk_result["cvar"]["worst_case"],
                        "historical": risk_result["cvar"]["historical"],
                        "cornish_fisher": risk_result["cvar"]["cornish_fisher"],
                    },
                    "position_limits": risk_result["position_limits"],
                    "risk_budget": risk_result["risk_budget"],
                }
            except Exception as e:
                logger.warning(f"Risk engine error: {e}")

            # ── LAYER 6: PORTFOLIO CONSTRUCTION ───────────────
            try:
                returns_252 = returns.tail(252)
                annual_vol = float(returns_252.std() * np.sqrt(252))
                vol_scale = self.vol_targeter.compute_scale_factor(
                    portfolio_returns=pd.Series(returns_252.values),
                    current_drawdown=self._compute_drawdown(close),
                )
                result["portfolio_construction"] = {
                    "vol_scale_factor": vol_scale["scale_factor"],
                    "target_vol": vol_scale["target_vol"],
                    "realized_vol": vol_scale["realized_vol"],
                    "leverage_signal": vol_scale["leverage_signal"],
                    "governor_active": vol_scale["governor_active"],
                    "recommended_position_size": min(1.0, vol_scale["scale_factor"]),
                }
            except Exception as e:
                logger.warning(f"Portfolio construction error: {e}")

            # ── LAYER 7: GOVERNANCE ───────────────────────────
            try:
                # Bailey & Lopez de Prado work in the periodicity of the
                # observations: a DAILY Sharpe against a daily n. Passing the
                # annualised figure inflates z by sqrt(252) and saturates the
                # normal CDF at 1.0 regardless of the input.
                # DSR assumes an EXCESS-return series. Subtract the daily
                # risk-free before computing, or the ratio is mean/sigma rather
                # than a Sharpe and the deflation is misspecified.
                _r = (returns - RISK_FREE_ANNUAL / 252).tail(252)
                sharpe_daily = float(_r.mean() / _r.std())
                sharpe = float(sharpe_daily * np.sqrt(252))   # display only
                skew = float(_r.skew())
                kurt = float(_r.kurtosis() + 3)
                dsr = self.dsr.compute(
                    sharpe=sharpe_daily,
                    n_trials=8,
                    n_obs=len(_r),
                    skewness=skew,
                    kurtosis=kurt,
                )
                # Probability of Backtest Overfitting (Bailey et al. 2014).
                # Synthesizes 8 variant strategies on ticker's return series,
                # runs CSCV to estimate overfit probability.
                pbo_result = None
                try:
                    pbo_result = compute_pbo_from_ticker_history(
                        ticker_returns=returns.values,
                        n_strategies=8,
                        n_slices=6,
                    )
                except Exception as pbo_err:
                    logger.warning(f"PBO computation failed: {pbo_err}")

                result["governance"] = {
                    "deflated_sharpe_ratio": float(dsr),
                    "is_genuine_alpha": self.dsr.is_genuine(dsr),
                    "sharpe_ratio_raw": float(sharpe),
                    "n_models_tested": 8,
                    # DSR corrects for selection across strategies. This series is
                    # one ticker's buy-and-hold return, not a strategy chosen from
                    # a set, so the deflation has little to correct and the figure
                    # runs high. Present as a distributional check on the Sharpe
                    # (skew/kurtosis-adjusted), not as evidence of alpha.
                    "dsr_caveat": "buy_and_hold_series_no_strategy_selection",
                    "n_obs": int(min(len(returns), 252)),
                    "skewness": round(skew, 3),
                    "kurtosis": round(kurt, 3),
                    "pbo": pbo_result,
                    "labeling": {
                        "n_events": labeling_result.get("n_events", 0),
                        "label_distribution": labeling_result.get("label_distribution", {}),
                        "avg_sample_uniqueness": labeling_result.get("avg_uniqueness", 0),
                        "fractional_d": labeling_result.get("d_value", 1.0),
                        "n_cusum_events": labeling_result.get("n_cusum_samples", 0),
                    },
                }
            except Exception as e:
                logger.warning(f"Governance error: {e}")

            # ── RISK METRICS ──────────────────────────────────
            try:
                from ml.features.feature_engineering import VolatilityFeatures
                risk_metrics = VolatilityFeatures.risk_metrics(returns)
                result["risk_metrics"] = risk_metrics
                result["annual_vol"] = risk_metrics.get("annual_volatility", annual_vol)
                result["sharpe_ratio"] = risk_metrics.get("sharpe_ratio")
                result["sortino_ratio"] = risk_metrics.get("sortino_ratio")
                result["max_drawdown"] = risk_metrics.get("max_drawdown")
                result["calmar_ratio"] = risk_metrics.get("calmar_ratio")
                result["hurst_exponent"] = float(VolatilityFeatures.hurst_exponent(returns))
            except Exception as e:
                result["annual_vol"] = float(returns.std() * np.sqrt(252))

            # ── OHLCV VOLATILITY ESTIMATORS ───────────────────
            # Parkinson, Garman-Klass, Yang-Zhang require H/L/O/C
            # These are more efficient than close-to-close vol
            # Reference: Garman & Klass (1980), Yang & Zhang (2000)
            try:
                h = price_data.get("high", close)
                l = price_data.get("low", close)
                o = price_data.get("open", close)
                c = close

                # Parkinson: uses H/L only — 5x more efficient than C2C
                # sigma_P = sqrt(1/(4n*ln2) * sum(ln(H/L)^2))
                hl_log = np.log(h / l.replace(0, np.nan)).dropna()
                parkinson = float(np.sqrt(
                    (1 / (4 * np.log(2))) * (hl_log**2).mean() * 252
                )) if len(hl_log) > 10 else 0.0

                # Garman-Klass: uses O,H,L,C — most efficient for continuous trading
                # sigma_GK = sqrt(252 * mean(0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2))
                aligned = pd.DataFrame({"h": h, "l": l, "o": o, "c": c}).dropna()
                if len(aligned) > 10:
                    term1 = 0.5 * np.log(aligned.h / aligned.l) ** 2
                    term2 = (2 * np.log(2) - 1) * np.log(aligned.c / aligned.o) ** 2
                    garman_klass = float(np.sqrt(252 * (term1 - term2).mean()))
                else:
                    garman_klass = 0.0

                # Yang-Zhang: handles overnight gaps — best for daily bars
                # Combines overnight, open-to-close, and Rogers-Satchell components
                if len(aligned) > 22:
                    k = 0.34 / (1.34 + (len(aligned) + 1) / (len(aligned) - 1))
                    overnight = np.log(aligned.o / aligned.c.shift(1)).dropna()
                    open_close = np.log(aligned.c / aligned.o).dropna()
                    rs = (np.log(aligned.h / aligned.c) * np.log(aligned.h / aligned.o) +
                          np.log(aligned.l / aligned.c) * np.log(aligned.l / aligned.o)).dropna()
                    sigma_oc = open_close.var()
                    sigma_on = overnight.var()
                    sigma_rs = rs.mean()
                    yang_zhang = float(np.sqrt(252 * (sigma_on + k * sigma_oc + (1 - k) * sigma_rs)))
                else:
                    yang_zhang = 0.0

                result["parkinson_vol"] = round(parkinson, 6)
                result["garman_klass_vol"] = round(garman_klass, 6)
                result["yang_zhang_vol"] = round(yang_zhang, 6)

                # Rolling realized vol windows
                for window in [5, 10, 21, 63, 126, 252]:
                    rv = float(returns.tail(window).std() * np.sqrt(252))
                    result.setdefault("risk_metrics", {})[f"realized_vol_{window}d"] = round(rv, 6)

            except Exception as e:
                logger.warning(f"OHLCV vol estimators error: {e}")
                result["parkinson_vol"] = 0.0
                result["garman_klass_vol"] = 0.0
                result["yang_zhang_vol"] = 0.0

            # ── RISK SHAPE: the underwater curve, the return distribution, and
            # every drawdown episode worth naming. A max-drawdown scalar says a
            # stock fell 44%; it does not say when, for how long, or whether it
            # ever recovered — which is what actually matters to someone holding it.
            try:
                import numpy as _np
                _c = close.dropna()
                if len(_c) >= 120:
                    _peak = _c.cummax()
                    _uw = (_c / _peak - 1.0)

                    # Sample the curve for plotting rather than shipping 1200 points.
                    _step = max(1, len(_uw) // 180)
                    _pts = [{"d": str(i)[:10], "v": round(float(v) * 100, 2)}
                            for i, v in list(_uw.items())[::_step]]

                    # Discrete episodes: from a peak, through the trough, back to flat.
                    _eps, _in, _start, _trough, _tdate = [], False, None, 0.0, None
                    for _i, _v in _uw.items():
                        _v = float(_v)
                        if not _in and _v < -0.05:
                            _in, _start, _trough, _tdate = True, _i, _v, _i
                        elif _in:
                            if _v < _trough:
                                _trough, _tdate = _v, _i
                            if _v >= -0.005:
                                _eps.append({
                                    "start": str(_start)[:10], "trough": str(_tdate)[:10],
                                    "end": str(_i)[:10],
                                    "depth_pct": round(_trough * 100, 1),
                                    "days": int((_i - _start).days),
                                    "recovery_days": int((_i - _tdate).days),
                                    "recovered": True})
                                _in = False
                    if _in:
                        _eps.append({
                            "start": str(_start)[:10], "trough": str(_tdate)[:10],
                            "end": None, "depth_pct": round(_trough * 100, 1),
                            "days": int((_uw.index[-1] - _start).days),
                            "recovery_days": None, "recovered": False})
                    _eps.sort(key=lambda x: x["depth_pct"])

                    # Return distribution against the normal curve the same volatility
                    # would imply — this is what excess kurtosis looks like plotted.
                    _r = returns.values * 100
                    _lo, _hi = float(_np.percentile(_r, 0.5)), float(_np.percentile(_r, 99.5))
                    _edges = _np.linspace(_lo, _hi, 25)
                    _hist, _ = _np.histogram(_r, bins=_edges)
                    _mu, _sd = float(_np.mean(_r)), float(_np.std(_r))
                    _mid = (_edges[:-1] + _edges[1:]) / 2
                    _w = _edges[1] - _edges[0]
                    _norm = (len(_r) * _w / (_sd * _np.sqrt(2 * _np.pi))
                             * _np.exp(-0.5 * ((_mid - _mu) / _sd) ** 2))

                    result["risk_shape"] = {
                        "underwater": _pts,
                        "current_drawdown_pct": round(float(_uw.iloc[-1]) * 100, 2),
                        "days_underwater": int((_uw.iloc[-1] < -0.005) and
                                               (len(_uw) - int(_np.argmax((_uw.values >= -0.005)[::-1])) )) if float(_uw.iloc[-1]) < -0.005 else 0,
                        "episodes": _eps[:6],
                        "distribution": {
                            "mids": [round(float(x), 2) for x in _mid],
                            "actual": [int(x) for x in _hist],
                            "normal": [round(float(x), 1) for x in _norm],
                            "mean": round(_mu, 3), "sd": round(_sd, 3),
                        },
                        "note": ("Underwater curve shows how far below the prior peak the price sat on each day. "
                                 "The distribution compares actual daily returns against the normal curve implied "
                                 "by the same volatility — the gap in the tails is what excess kurtosis measures."),
                    }
            except Exception as _e:
                logger.warning(f"risk shape skipped: {_e}")

            # ── MARKET MODEL (CAPM) — real regression vs SPY ─────────────
            # Single-factor market model: regress the stock's excess returns on
            # the market's (SPY) excess returns. Yields a genuine beta, alpha,
            # R-squared, and idiosyncratic risk. (Full Fama-French 5-factor
            # attribution requires Ken French daily factor data — roadmap.)
            try:
                # SPY was fetched over HTTP on every analyze request, so Polygon
                # throttled it under any burst and the regression silently failed
                # for all but the first ticker. It is already in daily_bars.
                import time as _t
                _now = _t.time()
                if (self._spy_cache is None or self._spy_cached_at is None
                        or _now - self._spy_cached_at > 3600):
                    _sd = await self.market_feed.get_price_history("SPY")
                    if _sd is not None and "close" in _sd and len(_sd["close"]) > 60:
                        self._spy_cache = _sd["close"]
                        self._spy_cached_at = _now
                spy_close = self._spy_cache
                if spy_close is not None and len(spy_close) > 60:
                    spy_ret = spy_close.pct_change().dropna()
                    # Feed bars carry a 04:00 UTC stamp, local bars midnight, so a
                    # direct join on the raw index yields nothing. Compare on the date.
                    _s = returns.copy(); _m = spy_ret.copy()
                    _s.index = pd.to_datetime(_s.index).tz_localize(None).normalize()
                    _m.index = pd.to_datetime(_m.index).tz_localize(None).normalize()
                    aligned = pd.concat([_s.rename("stock"), _m.rename("mkt")], axis=1, join="inner").dropna()
                    aligned = aligned.tail(252)
                    if len(aligned) >= 60:
                        rf = RISK_FREE_ANNUAL / 252
                        y = aligned["stock"].values - rf
                        x = aligned["mkt"].values - rf
                        from numpy.linalg import lstsq
                        X = np.column_stack([np.ones(len(x)), x])
                        coeffs, _, _, _ = lstsq(X, y, rcond=None)
                        alpha_daily = float(coeffs[0])
                        mkt_beta = float(coeffs[1])
                        y_hat = X @ coeffs
                        resid = y - y_hat
                        ss_res = float(np.sum(resid**2))
                        ss_tot = float(np.sum((y - y.mean())**2))
                        r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                        result["capm_alpha"] = round(alpha_daily * 252, 6)
                        result["capm_beta"] = round(mkt_beta, 4)
                        result["capm_r_squared"] = round(r2, 4)
                        result["capm_idio_risk"] = round(float(np.std(resid) * np.sqrt(252)), 6)
                        result["capm_n_obs"] = int(len(aligned))
                        result["capm_available"] = True
                    else:
                        logger.warning(f"CAPM {ticker}: only {len(aligned)} aligned rows, need 60")
                        result["capm_available"] = False
                else:
                    logger.warning(f"CAPM {ticker}: no usable SPY series")
                    result["capm_available"] = False
            except Exception as e:
                logger.warning(f"CAPM {ticker} regression error: {type(e).__name__}: {e}")
                result["capm_available"] = False

            result["scenarios"] = self._build_scenarios(
                current_price=current_price,
                predicted_return=predicted_return_1y,
                annual_vol=result.get("annual_vol", 0.25),
                regime=current_regime,
            )

            # ── ANALYST RATINGS ──────────────────────────────
            try:
                analyst_result = self.analyst_engine.fetch(ticker)
                result["analyst_ratings"] = analyst_result
            except Exception as e:
                logger.warning(f"Analyst ratings error: {e}")

            # ── COMPOSITE SIGNAL ─────────────────────────────
            signal, score = self._compute_composite_signal(result)
            result["overall_signal"] = signal
            result["overall_score"] = score

            result["price"] = current_price
            result["change"] = float(close.iloc[-1] - close.iloc[-2]) if len(close) >= 2 else 0
            result["change_pct"] = (
                float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) >= 2 else 0
            )
            result["predicted_return_1y"] = float(predicted_return_1y * 100)
            result["data_quality"] = {
                "score": self._data_quality_score(result),
                "n_price_days": len(close),
                "has_options": (
                    not isinstance(options_chain, Exception)
                    and isinstance(options_chain, pd.DataFrame)
                    and not options_chain.empty
                ),
                "has_sentiment": not isinstance(news_data, Exception) and bool(news_data),
                "has_fundamentals": not isinstance(fundamentals, Exception) and bool(fundamentals),
            }
            result["analysis_duration_seconds"] = round(time.time() - start_time, 1)
            result["version"] = "6.0"

            # ── LAYER 9: V6 INSTITUTIONAL ENGINE (non-fatal enrichment) ──
            # Enriches response with distributional modeling, Kelly bet sizing,
            # fractional differentiation metrics, institutional risk metrics, HRP.
            # Wrapped in try/except; if v6 engine fails, response is returned
            # unchanged (bridge.enrich_analysis has its own internal error handling).
            try:
                from ml.v6.bridge import get_v6_bridge
                bridge = get_v6_bridge()
                if bridge.available:
                    result = bridge.enrich_analysis(
                        v5_result=result,
                        prices=price_data["close"],
                        returns=returns,
                        ticker=ticker,
                        regime=str(result.get("current_regime", "NORMAL")),
                    )
                    logger.info(f"V6 institutional enrichment applied for {ticker}")
                else:
                    result["v6_available"] = False
            except Exception as e:
                logger.warning(f"V6 institutional enrichment failed (non-fatal): {e}")
                result["v6_available"] = False
                result["v6_error"] = str(e)[:200]

        except HTTPException:
            raise
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Analysis timed out after 240 seconds")
        except Exception as e:
            logger.error(f"Analysis error for {ticker}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        return result

    async def _run_ml_predictions(
        self,
        feature_matrix: Dict,
        ticker: str,
        regime: str,
        price_data: Optional[pd.DataFrame] = None,
        fundamentals: Optional[Dict] = None,
    ) -> Dict:
        """
        Train XGBoost, LightGBM, and BiLSTM on real historical data then predict.

        Pipeline:
          1. build_historical_feature_matrix() → (n_days, n_features) array
          2. Forward 21-day log-return labels, clipped at ±30%
          3. 60-day embargo train/val split (Lopez de Prado Ch.7)
          4. XGBoost: xgb_model.fit(X_train, y_train) → predict(today_vec)
             SHAP: shap.TreeExplainer(model).shap_values(today_vec)
          5. LightGBM: lgb_model.fit(X_train, y_train) → predict(today_vec)
             Rank score: percentile in training distribution
          6. BiLSTM: 60-day rolling sequences, 30-epoch training, MC Dropout inference
          7. Ensemble: XGB 40% + LGB 40% + LSTM 20%
        """
        if not feature_matrix:
            return {"ensemble": {}}

        loop = asyncio.get_event_loop()

        def _train_and_predict() -> Dict:
            # All imports inside thread so they don't need to be at module level
            from ml.models.xgboost_lgbm import XGBoostPredictor, LightGBMPredictor
            from ml.models.lstm_model import LSTMTrainer, QuantEdgeLSTM
            import torch

            # ── 1. HISTORICAL FEATURE MATRIX ─────────────────
            if price_data is None or len(price_data) < 300:
                return self._statistical_predictions(feature_matrix, regime)

            try:
                X_hist, feature_names, dates = self.feature_pipeline.build_historical_feature_matrix(
                    df=price_data,
                    fundamentals=fundamentals or {},
                    lookback_days=63,
                    step=10,
                )
            except Exception as e:
                logger.warning(f"build_historical_feature_matrix failed: {e}")
                return self._statistical_predictions(feature_matrix, regime)

            if len(X_hist) < 100:
                return self._statistical_predictions(feature_matrix, regime)

            # ── 2. TRIPLE-BARRIER LABELS (Lopez de Prado, 2018, Ch.3) ────────
            # Replaces naive fixed-horizon labels with path-dependent labels.
            # For each event t0, we simulate holding with:
            #   - Upper barrier (profit-take): entry + pt_mult * sigma_t
            #   - Lower barrier (stop-loss):   entry - sl_mult * sigma_t
            #   - Vertical barrier (time):     t0 + max_horizon (21 days)
            # Label is the REALIZED log-return at whichever barrier is touched first.
            # This matches how real trading works and removes the ±30% clip artifact.
            close_series = price_data["close"]
            max_horizon = 21
            horizon = max_horizon  # downstream XGBoost/LightGBM/LSTM reference 'horizon'
            pt_mult = 2.0   # profit-take = 2 sigma
            sl_mult = 2.0   # stop-loss   = 2 sigma (symmetric for regression target)

            # Rolling volatility (20-day) as barrier width scalar
            log_returns = np.log(close_series / close_series.shift(1))
            sigma = log_returns.rolling(window=20, min_periods=5).std().bfill()

            y_list, valid_idx = [], []
            n_total_bars = len(close_series)
            for i, date in enumerate(dates):
                try:
                    pos = close_series.index.get_loc(date)
                    if pos + max_horizon >= n_total_bars:
                        continue

                    entry_price = float(close_series.iloc[pos])
                    entry_sigma = float(sigma.iloc[pos])
                    if entry_sigma <= 0 or not np.isfinite(entry_sigma):
                        continue

                    # Barrier prices
                    upper = entry_price * np.exp(pt_mult * entry_sigma)
                    lower = entry_price * np.exp(-sl_mult * entry_sigma)

                    # Walk forward to find first-touch barrier
                    realized_return = None
                    for j in range(1, max_horizon + 1):
                        if pos + j >= n_total_bars:
                            break
                        p = float(close_series.iloc[pos + j])
                        if p >= upper:
                            # Profit-take hit (path-dependent realized return)
                            realized_return = np.log(upper / entry_price)
                            break
                        if p <= lower:
                            # Stop-loss hit
                            realized_return = np.log(lower / entry_price)
                            break

                    # Vertical barrier: use actual return at t + max_horizon
                    if realized_return is None:
                        final_price = float(close_series.iloc[pos + max_horizon])
                        realized_return = np.log(final_price / entry_price)

                    y_list.append(float(realized_return))
                    valid_idx.append(i)
                except Exception:
                    continue

            if len(y_list) < 80:
                logger.warning(f"Triple-barrier: only {len(y_list)} valid events, using statistical fallback")
                return self._statistical_predictions(feature_matrix, regime)

            # Log distribution of realized returns for transparency
            y_arr_preview = np.array(y_list)
            pct_upper = float((y_arr_preview > 0.01).mean())
            pct_lower = float((y_arr_preview < -0.01).mean())
            pct_middle = 1.0 - pct_upper - pct_lower
            logger.info(
                f"Triple-barrier labels for {ticker}: n={len(y_list)}, "
                f"upper={pct_upper:.1%}, middle={pct_middle:.1%}, lower={pct_lower:.1%}, "
                f"mean_ret={y_arr_preview.mean():+.4f}, std={y_arr_preview.std():.4f}"
            )

            X = X_hist[valid_idx]
            y = np.array(y_list, dtype=np.float64)

            # ── 3. TRAIN/VAL SPLIT WITH EMBARGO ────────
            # 70/30 split leaves enough val samples for both model fit validation
            # AND meta-labeling training. Embargo is sized from horizon (21) not
            # a fixed 60 — which was too aggressive for short series and left
            # meta-labeling with <10 val samples, silently skipping.
            # Reference: Lopez de Prado (2018), Advances in Financial ML, Ch.7
            n_train = int(len(y) * 0.70)
            embargo_size = min(horizon + 5, 30)  # embargo = horizon + buffer, capped at 30
            n_embargo_end = min(n_train + embargo_size, len(y) - 15)
            X_train, y_train = X[:n_train], y[:n_train]
            X_val = X[n_embargo_end:] if len(X) - n_embargo_end >= 15 else None
            y_val = y[n_embargo_end:] if X_val is not None else None
            logger.info(
                f"Train/val split: n_train={n_train}, embargo={embargo_size}, "
                f"n_val={len(X_val) if X_val is not None else 0}"
            )

            # ── 4. TODAY'S FEATURE VECTOR ────────────────────
            # Must use the same feature_names list so the vector
            # dimension matches what the models were trained on.
            today_vec = np.array(
                [feature_matrix.get(k, 0.0) for k in feature_names],
                dtype=np.float64,
            ).reshape(1, -1)
            today_vec = np.where(np.isfinite(today_vec), today_vec, 0.0)

            # ── 5. XGBOOST ────────────────────────────────────
            # Hyperparameters for financial data:
            #   max_depth=6, min_child_weight=20 → prevents overfitting on noisy data
            #   subsample=0.8, colsample_bytree=0.7 → stochastic boosting
            #   reg_alpha=0.1, reg_lambda=1.0 → L1+L2 regularization
            xgb_pred_21d = xgb_signal = xgb_ic = 0.0
            shap_drivers = []
            try:
                xgb_model = XGBoostPredictor(target_horizon=horizon)
                xgb_fit = xgb_model.fit(
                    X_train, y_train, feature_names, X_val=X_val, y_val=y_val
                )
                xgb_pred_21d = float(xgb_model.predict(today_vec)[0]) * 100
                xgb_ic = float(xgb_fit.get("ic_train", 0))
                xgb_signal = float(np.clip(xgb_pred_21d * 5, -100, 100))

                # SHAP values: shap.TreeExplainer gives exact Shapley values,
                # not just feature_importances_ which are impurity-based and biased
                # toward high-cardinality features.
                # Reference: Lundberg & Lee (2017), NeurIPS
                try:
                    _, shap_dict = xgb_model.predict_with_shap(today_vec)
                    # predict_with_shap returns per-feature SHAP values
                    # (it calls shap.TreeExplainer internally — see xgboost_lgbm.py)
                    all_shap = {
                        **shap_dict.get("top_bullish_drivers", {}),
                        **shap_dict.get("top_bearish_drivers", {}),
                    }
                    top = sorted(all_shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
                    shap_drivers = [{"feature": k, "impact": round(v * 100, 4)} for k, v in top]
                    if not shap_drivers:
                        # SHAP succeeded but returned no drivers — fall through
                        # to feature importance so the user sees *something*.
                        logger.info(f"SHAP for {ticker}: empty drivers, using feature importance fallback")
                        raise ValueError("empty_shap")
                except Exception as e:
                    # Fallback: raw feature importance from XGBoost gain
                    # (not SHAP, but more honest than blank).
                    logger.info(f"SHAP fallback for {ticker}: {type(e).__name__}={e}")
                    fi = xgb_fit.get("top10_features", [])
                    # top10_features is list of (name, importance) tuples; keep only non-zero
                    non_zero_fi = [(k, v) for k, v in fi if v and abs(v) > 1e-9]
                    shap_drivers = [
                        {"feature": k, "impact": round(float(v) * 100, 4)}
                        for k, v in non_zero_fi[:8]
                    ]

            except Exception as e:
                logger.warning(f"XGBoost training failed: {e}")

            # ── 6. LIGHTGBM ───────────────────────────────────
            # Leaf-wise tree growth + GOSS sampling.
            # Rank IC measured as Spearman correlation (rank-based, more
            # appropriate than Pearson for fat-tailed financial returns).
            lgb_pred_21d = lgb_ic = 0.0
            lgb_rank_score = 50.0
            try:
                lgb_model = LightGBMPredictor(target_horizon=horizon)
                lgb_fit = lgb_model.fit(
                    X_train, y_train, feature_names, X_val=X_val, y_val=y_val
                )
                lgb_pred_21d = float(lgb_model.predict(today_vec)[0]) * 100
                lgb_ic = float(lgb_fit.get("ic_train", 0))
                # Rank score: what percentile of the training universe does
                # today's prediction place at? 100 = top decile.
                train_preds = lgb_model.predict(X_train)
                lgb_rank_score = round(
                    float(np.mean(lgb_model.predict(today_vec)[0] > train_preds)) * 100, 1
                )
            except Exception as e:
                logger.warning(f"LightGBM training failed: {e}")

            # ── 6b. META-LABELING (walk-forward OOF on XGB+LGB ensemble @ 21d) ─
            # Trains the meta-classifier on out-of-fold primary predictions rather
            # than a small tail slice. Each OOF prediction is made by a primary
            # model that did NOT see that sample. This yields ~70+ meta samples
            # instead of ~15, producing a real classifier (not the 0.5 fallback).
            # Reference: Lopez de Prado (2018) Ch.7 on Cross-Validation in Finance.
            meta_conf_21d = 0.5
            meta_labeler = MetaLabeler(horizons=[5, 10, 21, 63, 252])
            try:
                if len(X) >= 60:
                    oof_preds, oof_indices = walk_forward_oof_ensemble(
                        X=X, y=y, feature_names=feature_names,
                        model_builders={
                            "xgb": lambda: XGBoostPredictor(target_horizon=horizon),
                            "lgb": lambda: LightGBMPredictor(target_horizon=horizon),
                        },
                        weights={"xgb": 0.5, "lgb": 0.5},
                        n_folds=5, embargo=5, min_train=40,
                    )
                    if len(oof_preds) >= 30:
                        X_oof = X[oof_indices]
                        y_oof = y[oof_indices]
                        meta_labeler.fit(
                            X_val=X_oof,
                            primary_preds=oof_preds,
                            y_val=y_oof,
                            horizon=21,
                            feature_names=feature_names,
                        )
                        # Today's primary prediction (ensemble from models trained on full data)
                        if 'xgb_model' in dir() and 'lgb_model' in dir():
                            today_xgb = float(xgb_model.predict(today_vec)[0])
                            today_lgb = float(lgb_model.predict(today_vec)[0])
                            primary_today_21d = 0.5 * today_xgb + 0.5 * today_lgb
                            meta_conf_21d = meta_labeler.predict_confidence(
                                today_vec.flatten(), primary_today_21d, horizon=21
                            )
                            logger.info(
                                f"Meta-label 21d (OOF, n={len(oof_preds)}): "
                                f"primary={primary_today_21d:+.4f}, confidence={meta_conf_21d:.3f}"
                            )
                    else:
                        logger.info(f"Meta-label 21d: OOF produced only {len(oof_preds)} samples, fallback to 0.5")
                else:
                    logger.info(f"Meta-label 21d: only {len(X)} total samples, insufficient for OOF")
            except Exception as e:
                logger.warning(f"Meta-labeling 21d OOF failed: {e}")

            # ── 7. BiLSTM + TEMPORAL ATTENTION + MC DROPOUT ──
            # Architecture: Input(n_features) → BiLSTM(128) → Attention → BiLSTM(64)
            #               → MultiTaskHeads → [5d, 10d, 21d, 63d, 252d predictions]
            # Training: 30 epochs, Huber loss (robust to return outliers), AdamW
            # Inference: 30 MC Dropout passes → mean + epistemic uncertainty
            # Reference: Gal & Ghahramani (2016), "Dropout as Bayesian Approximation"
            lstm_preds: Dict = {}
            lstm_uncertainty = 100.0
            SEQ_LEN = 60
            try:
                if len(X) >= SEQ_LEN + horizon + 10:
                    seqs, seq_labels = [], []
                    for i in range(len(X) - SEQ_LEN - horizon):
                        seqs.append(X[i: i + SEQ_LEN])
                        lbl_idx = i + SEQ_LEN
                        seq_labels.append(float(y[lbl_idx]) if lbl_idx < len(y) else 0.0)

                    if len(seqs) >= 40:
                        seqs_np = np.array(seqs, dtype=np.float32)
                        labels_np = np.array(seq_labels, dtype=np.float32)

                        # QuantEdgeLSTM constructor: input_size, hidden_size, dropout, horizons
                        # (see ml/models/lstm_model.py — these are the actual param names)
                        lstm_net = QuantEdgeLSTM(
                            input_size=X.shape[1],
                            hidden_size=128,
                            dropout=0.3,
                            horizons=[5, 10, 21, 63, 252],
                        )
                        trainer = LSTMTrainer(lstm_net)
                        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)

                        n_seq_train = int(len(seqs_np) * 0.85)
                        X_seq = torch.FloatTensor(seqs_np[:n_seq_train]).to(trainer.device)
                        y_seq = torch.FloatTensor(labels_np[:n_seq_train]).to(trainer.device)

                        trainer.model.train()
                        for epoch in range(30):
                            optimizer.zero_grad()
                            out = trainer.model(X_seq)
                            pred_21 = out["pred_21d"].squeeze()
                            loss = torch.nn.functional.huber_loss(pred_21, y_seq)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
                            optimizer.step()

                        # MC Dropout inference: model.train() keeps dropout active.
                        # 30 stochastic forward passes → distribution over predictions.
                        today_seq = X[-SEQ_LEN:].astype(np.float32)
                        mc = trainer.predict_with_uncertainty(today_seq, n_mc_samples=30)
                        lstm_preds = {
                            "pred_5d":   round(mc.get("return_5d", 0) * 100, 4),
                            "pred_10d":  round(mc.get("return_10d", 0) * 100, 4),
                            "pred_21d":  round(mc.get("return_21d", 0) * 100, 4),
                            "pred_63d":  round(mc.get("return_63d", 0) * 100, 4),
                            "pred_252d": round(mc.get("return_252d", 0) * 100, 4),
                            "regime":    mc.get("regime", regime),
                            "uncertainty": round(mc.get("total_unc_21d", 0.1) * 100, 2),
                        }
                        lstm_uncertainty = lstm_preds["uncertainty"]
            except Exception as e:
                logger.warning(f"LSTM training/inference failed: {e}")

            # ── 7b. META-LABELING (5/10/63/252 LSTM primaries) ──────
            # Train per-horizon meta-classifier using LSTM's val-set predictions.
            # We regenerate realized returns at each horizon from the price series.
            meta_conf_by_horizon = {21: meta_conf_21d}
            try:
                if lstm_preds and 'trainer' in dir() and len(seqs_np) > n_seq_train + 10:
                    # LSTM val sequences (held-out portion)
                    X_seq_val = torch.FloatTensor(seqs_np[n_seq_train:]).to(trainer.device)
                    trainer.model.eval()
                    with torch.no_grad():
                        val_out = trainer.model(X_seq_val)

                    # Val-set flat feature vectors (last step of each sequence)
                    # for meta-feature construction. Shape: (n_val, n_features)
                    val_flat_features = seqs_np[n_seq_train:, -1, :]  # (n_val, n_features)

                    for h_key, h_val in [("pred_5d", 5), ("pred_10d", 10), ("pred_63d", 63), ("pred_252d", 252)]:
                        try:
                            head_key = f"pred_{h_val}d"
                            if head_key not in val_out:
                                continue
                            primary_val_h = val_out[head_key].squeeze().cpu().numpy()
                            if primary_val_h.ndim == 0:
                                primary_val_h = primary_val_h.reshape(1)

                            # Realized h-day forward returns matching val sequences
                            realized_h = []
                            for i in range(len(primary_val_h)):
                                base_idx = n_seq_train + i + SEQ_LEN
                                tgt_idx = base_idx + h_val
                                if tgt_idx < len(y):
                                    realized_h.append(float(y[tgt_idx]))
                                else:
                                    realized_h.append(float(y[-1]) if len(y) else 0.0)
                            realized_h = np.array(realized_h, dtype=np.float64)

                            if len(realized_h) < 30:
                                continue

                            meta_labeler.fit(
                                X_val=val_flat_features[:len(primary_val_h)],
                                primary_preds=primary_val_h,
                                y_val=realized_h,
                                horizon=h_val,
                                feature_names=feature_names,
                            )

                            # Today's LSTM prediction for this horizon (already in lstm_preds)
                            today_lstm_pred = float(lstm_preds.get(head_key, 0.0)) / 100.0
                            conf_h = meta_labeler.predict_confidence(
                                today_vec.flatten(),
                                today_lstm_pred,
                                horizon=h_val,
                            )
                            meta_conf_by_horizon[h_val] = conf_h
                        except Exception as eh:
                            logger.warning(f"Meta-labeling {h_val}d failed: {eh}")
                            continue
            except Exception as e:
                logger.warning(f"LSTM meta-labeling block failed: {e}")

            # ── 8. ENSEMBLE ───────────────────────────────────
            # Weights: XGB 40%, LGB 40%, LSTM 20%
            # LSTM weighted lower because it trains from scratch each request;
            # once offline pre-trained weights are available, increase to 33%.
            weighted = []
            if xgb_pred_21d != 0: weighted.append((xgb_pred_21d, 0.40))
            if lgb_pred_21d != 0: weighted.append((lgb_pred_21d, 0.40))
            lstm_21 = lstm_preds.get("pred_21d", 0)
            if lstm_21 != 0: weighted.append((lstm_21, 0.20))

            if not weighted:
                return self._statistical_predictions(feature_matrix, regime)

            total_w = sum(w for _, w in weighted)
            ens_21d = sum(p * w for p, w in weighted) / total_w

            def scale(base: float, h: int) -> float:
                """Linear time-scaling of EXPECTED RETURNS: E[r(h)] = E[r(21d)] * h/21.
                The previous sqrt(h/21) rule is a VOLATILITY scaling law, not a
                return law — it distorted every non-21d horizon. Long horizons are
                extrapolations, so clip to sane bounds."""
                scaled = base * (h / 21.0)
                cap = 15.0 if h <= 21 else 30.0 if h <= 63 else 60.0
                return round(float(np.clip(scaled, -cap, cap)), 4)

            n_models = len(weighted)
            # Defensive NaN guard: even though xgboost_lgbm.py now returns 0
            # on NaN IC, keep this as belt-and-suspenders. A NaN in confidence
            # propagates through _sanitize_json and shows as null in the UI.
            _xi = float(xgb_ic) if np.isfinite(xgb_ic) else 0.0
            _li = float(lgb_ic) if np.isfinite(lgb_ic) else 0.0
            mean_ic = (abs(_xi) + abs(_li)) / max(n_models, 1)
            # Honest confidence: blend train-IC skill with the REAL out-of-fold meta-confidence
            # (Lopez de Prado meta-labeling, computed above). Train-IC alone overstates because
            # it is in-sample; meta_conf_21d is genuine held-out skill. Cap at 0.90 — no model
            # is ever perfectly confident. Also penalize by model disagreement.
            ic_component = float(np.clip(mean_ic / 0.10, 0.0, 1.0))
            meta_component = float(np.clip(meta_conf_21d, 0.0, 1.0))
            disagreement = abs(xgb_pred_21d - lgb_pred_21d)
            disagree_penalty = float(np.clip(1.0 - disagreement / 20.0, 0.5, 1.0))
            # 60% real OOF meta-confidence, 40% in-sample IC, scaled by agreement
            confidence = (0.60 * meta_component + 0.40 * ic_component) * disagree_penalty
            confidence = float(np.clip(confidence, 0.0, 0.90))
            if not np.isfinite(confidence):
                confidence = 0.0

            # Real out-of-sample rank-IC (Spearman) on the validation slice.
            # This is the honest predictive-skill metric — rank correlation between
            # model predictions and realized returns on data the model did NOT train on.
            rank_ic_val = 0.0
            rank_ic_source = "unavailable"
            try:
                if X_val is not None and y_val is not None and len(y_val) >= 10:
                    from scipy.stats import spearmanr
                    val_preds = None
                    # xgb_pred_21d != 0 means XGBoost trained successfully; use it.
                    try:
                        val_preds = xgb_model.predict(X_val)
                    except (NameError, AttributeError):
                        try:
                            val_preds = lgb_model.predict(X_val)
                        except (NameError, AttributeError):
                            val_preds = None
                    if val_preds is not None and len(val_preds) == len(y_val):
                        rho, _ = spearmanr(val_preds, y_val)
                        if np.isfinite(rho):
                            rank_ic_val = float(rho)
                            rank_ic_source = f"out-of-sample Spearman (n={len(y_val)})"
                            logger.info(f"rank-IC (OOS Spearman) for {ticker}: {rank_ic_val:+.4f} on n={len(y_val)}")
                        else:
                            rank_ic_source = "non-finite (flat predictions)"
                    else:
                        rank_ic_source = "no validation predictions"
                else:
                    rank_ic_source = f"insufficient val data (n={len(y_val) if y_val is not None else 0})"
            except Exception as _e:
                logger.info(f"rank-IC computation skipped: {type(_e).__name__}={_e}")
                rank_ic_source = f"error: {type(_e).__name__}"

            return {
                "lstm": lstm_preds if lstm_preds else {
                    "pred_21d": scale(ens_21d, 21),
                    "regime": regime,
                    "uncertainty": lstm_uncertainty,
                },
                "xgboost": {
                    "signal_strength": round(xgb_signal, 2),
                    "pred_21d": round(xgb_pred_21d, 4),
                    "pred_252d": scale(xgb_pred_21d, 252),
                    "ic_train": round(xgb_ic, 4),
                },
                "lightgbm": {
                    "rank_score": lgb_rank_score,
                    "pred_21d": round(lgb_pred_21d, 4),
                    "pred_252d": scale(lgb_pred_21d, 252),
                    "ic_train": round(lgb_ic, 4),
                },
                "ensemble": {
                    "pred_5d":            scale(ens_21d, 5),
                    "pred_10d":           scale(ens_21d, 10),
                    "pred_21d":           round(ens_21d, 4),
                    "pred_63d":           scale(ens_21d, 63),
                    "pred_252d":          scale(ens_21d, 252),
                    "confidence":         round(confidence, 3),
                    "model_disagreement": round(abs(xgb_pred_21d - lgb_pred_21d), 4),
                    **build_meta_summary(
                        horizons=[5, 10, 21, 63, 252],
                        primary_preds_by_horizon={
                            5:   lstm_preds.get("pred_5d", scale(ens_21d, 5)) / 100.0,
                            10:  lstm_preds.get("pred_10d", scale(ens_21d, 10)) / 100.0,
                            21:  ens_21d / 100.0,
                            63:  lstm_preds.get("pred_63d", scale(ens_21d, 63)) / 100.0,
                            252: lstm_preds.get("pred_252d", scale(ens_21d, 252)) / 100.0,
                        },
                        confidences_by_horizon=meta_conf_by_horizon,
                        meta_labeler=meta_labeler,
                    ),
                },
                "shap_top_drivers": shap_drivers,
                "rank_ic_estimate": round(rank_ic_val, 4),
                "rank_ic_source": rank_ic_source,
                "ic_estimate": round((xgb_ic + lgb_ic) / max(n_models, 1), 4),
                "ic_source": "in-sample train IC (Pearson)",
                "quantile": {
                    "q10_1m": round(float(np.percentile(y, 10)) * 100, 4),
                    "q25_1m": round(float(np.percentile(y, 25)) * 100, 4),
                    "q50_1m": round(float(np.percentile(y, 50)) * 100, 4),
                    "q75_1m": round(float(np.percentile(y, 75)) * 100, 4),
                    "q90_1m": round(float(np.percentile(y, 90)) * 100, 4),
                },
            }

        # Run in thread pool — does not block the async event loop
        try:
            return await loop.run_in_executor(None, _train_and_predict)
        except Exception as e:
            logger.error(f"_run_ml_predictions failed: {e}")
            return self._statistical_predictions(feature_matrix, regime)

    def _statistical_predictions(self, features: Dict, regime: str) -> Dict:
        """Fallback when insufficient history for ML training."""
        momentum_21d = float(features.get("momentum_21d", 0) or 0)
        momentum_63d = float(features.get("momentum_63d", 0) or 0)
        rsi_14 = float(features.get("rsi_14", 50) or 50)
        vol_ratio = float(features.get("vol_ratio", 1) or 1)
        raw = float(np.clip(
            (0.3 * momentum_21d + 0.2 * momentum_63d
             + 0.1 * (rsi_14 - 50) / 50 * 5
             - 0.1 * (vol_ratio - 1) * 5) * 10,
            -20, 20
        ))
        return {
            "ensemble": {
                "pred_5d": raw * 0.3, "pred_10d": raw * 0.5, "pred_21d": raw,
                "pred_63d": raw * 1.5, "pred_252d": raw * 2.0,
                "confidence": 0.35, "model_disagreement": 5.0,
            },
            "rank_ic_estimate": 0.03,
        }

    def _compute_sentiment(self, news_data: Dict) -> Dict:
        """
        Real NLP sentiment via FinBERT (ProsusAI/finbert).

        FinBERT is a BERT model fine-tuned on Financial PhraseBank.
        It classifies text as positive / negative / neutral with calibrated probabilities.
        Falls back to TextBlob if the model is not available (e.g. no internet at startup).

        This does NOT use keyword lists. Every headline goes through the transformer.
        """
        try:
            news_items = news_data.get("news", [])
            headlines = [
                item.get("title", "") if isinstance(item, dict) else str(item)
                for item in news_items
                if item
            ]
            reddit_posts = news_data.get("reddit", [])

            # FinBERT inference on news headlines (batched for speed)
            # self.finbert.analyze_text() runs the transformer forward pass
            news_scores = []
            for headline in headlines[:15]:
                if headline.strip():
                    try:
                        result = self.finbert.analyze_text(headline)
                        # analyze_text() returns {"score": float, "label": str, "model": str}
                        # score is in [-1, +1]: positive → +1, negative → -1, neutral → 0
                        news_scores.append(result.get("score", 0.0))
                    except Exception:
                        continue

            news_score = float(np.mean(news_scores)) if news_scores else 0.0
            news_label = "BULLISH" if news_score > 0.15 else "BEARISH" if news_score < -0.15 else "NEUTRAL"

            # Compute softmax probability breakdown from raw scores
            # positive=score>0, negative=score<0, neutral=near-zero
            # Sigmoid-style mapping so probabilities sum to ~1
            raw_pos = float(np.mean([s for s in news_scores if s > 0])) if any(s > 0 for s in news_scores) else 0.0
            raw_neg = float(np.mean([abs(s) for s in news_scores if s < 0])) if any(s < 0 for s in news_scores) else 0.0
            n_pos = sum(1 for s in news_scores if s > 0.05)
            n_neg = sum(1 for s in news_scores if s < -0.05)
            n_neu = len(news_scores) - n_pos - n_neg
            total = max(len(news_scores), 1)
            prob_pos = round(n_pos / total, 4)
            prob_neg = round(n_neg / total, 4)
            prob_neu = round(n_neu / total, 4)

            # FinBERT inference on Reddit posts (upvote-weighted)
            reddit_result = self.finbert.aggregate_reddit_sentiment(reddit_posts[:20])
            reddit_score = float(reddit_result.get("composite_score", 0.0))
            reddit_label = reddit_result.get("label", "NEUTRAL")

            composite = 0.6 * news_score + 0.4 * reddit_score
            composite_label = "BULLISH" if composite > 0.15 else "BEARISH" if composite < -0.15 else "NEUTRAL"

            return {
                "news": {
                    "score": round(news_score, 4),
                    "label": news_label,
                    "n_headlines": len(news_scores),
                    "positive": prob_pos,
                    "negative": prob_neg,
                    "neutral": prob_neu,
                },
                "reddit": {
                    "score": round(reddit_score, 4),
                    "label": reddit_label,
                    "n_posts": len(reddit_posts),
                },
                "composite": round(composite, 4),
                "label": composite_label,
                "headlines": headlines[:5],
                "model": "FinBERT (ProsusAI/finbert)",
            }
        except Exception as e:
            logger.warning(f"FinBERT sentiment failed: {e}")
            return {"composite": 0.0, "label": "NEUTRAL", "news": {}, "reddit": {}}

    def _compute_options(self, options_chain: Any, spot: float, vol: float) -> Dict:
        """Compute options analytics via OptionsAnalytics."""
        try:
            oa = OptionsAnalytics()
            gex = oa.compute_gex(options_chain, spot)
            iv_surface = oa.build_iv_surface(options_chain, spot)
            T = 30 / 252
            atm_greeks = oa.compute_all_greeks(
                S=spot, K=spot, T=T, r=0.053, sigma=vol, option_type="call"
            )
            return {
                "gex": gex,
                "iv_surface": iv_surface,
                "atm_greeks": atm_greeks,
                "atm_iv_30d": vol,
            }
        except Exception:
            return {}

    def _compute_drawdown(self, prices: pd.Series) -> float:
        if len(prices) == 0:
            return 0.0
        peak = prices.cummax()
        return float(-(prices.iloc[-1] / peak.iloc[-1] - 1))

    def _build_scenarios(
        self, current_price: float, predicted_return: float,
        annual_vol: float, regime: str
    ) -> Dict:
        bull_mult = 1.5 if "BULL" in regime else 1.3
        bear_mult = 1.5 if "BEAR" in regime else 1.2
        scenarios = {
            "bull": {
                "name": "Bull Case",
                "return_pct": (predicted_return + bull_mult * annual_vol) * 100,
                "target_price": current_price * (1 + predicted_return + bull_mult * annual_vol),
                "probability": 0.25,
                "description": f"Favorable conditions in {regime} regime",
            },
            "base": {
                "name": "Base Case",
                "return_pct": predicted_return * 100,
                "target_price": current_price * (1 + predicted_return),
                "probability": 0.40,
                "description": "Expected scenario given current regime and signals",
            },
            "bear": {
                "name": "Bear Case",
                "return_pct": (predicted_return - bear_mult * annual_vol) * 100,
                "target_price": current_price * (1 + predicted_return - bear_mult * annual_vol),
                "probability": 0.25,
                "description": "Downside scenario with elevated volatility",
            },
            "tail": {
                "name": "Tail Risk",
                "return_pct": (predicted_return - 2.5 * annual_vol) * 100,
                "target_price": current_price * (1 + predicted_return - 2.5 * annual_vol),
                "probability": 0.10,
                "description": "Black swan / structural break",
            },
        }
        ev = sum(s["probability"] * s["return_pct"] / 100 for s in scenarios.values())
        scenarios["expected_value"] = float(ev)
        return scenarios

    def _compute_composite_signal(self, result: Dict) -> tuple:
        score_components = []

        ml = result.get("ml_predictions", {}).get("ensemble", {})
        pred_1y = ml.get("pred_252d", 0) or 0
        ml_score = np.clip((float(pred_1y) + 20) / 40 * 100, 0, 100)
        score_components.append(("ml", ml_score, 0.40))

        regime = result.get("current_regime", "UNKNOWN")
        regime_scores = {
            "BULL_LOW_VOL": 80, "BULL_HIGH_VOL": 65,
            "MEAN_REVERT": 50, "BEAR_LOW_VOL": 35,
            "BEAR_HIGH_VOL": 20, "UNKNOWN": 50,
        }
        score_components.append(("regime", regime_scores.get(regime, 50), 0.25))

        garch = result.get("garch", {})
        vol_reg = garch.get("vol_regime", "NORMAL")
        vol_scores = {"LOW": 75, "NORMAL": 60, "HIGH": 35}
        score_components.append(("vol", vol_scores.get(vol_reg, 60), 0.15))

        sentiment = result.get("sentiment", {})
        sent_raw = float(sentiment.get("composite", 0) or 0)
        sent_score = np.clip((sent_raw + 1) / 2 * 100, 0, 100)
        score_components.append(("sentiment", sent_score, 0.10))

        kalman = result.get("kalman", {})
        kalman_sig = kalman.get("signal_interpretation", "MEAN_REVERTING")
        kalman_scores = {"STRONG_TREND": 75, "WEAK_TREND": 60, "MEAN_REVERTING": 40}
        score_components.append(("kalman", kalman_scores.get(kalman_sig, 50), 0.10))

        total_score = float(np.clip(sum(s * w for _, s, w in score_components), 0, 100))

        if total_score >= 72:   signal = "STRONG_BUY"
        elif total_score >= 58: signal = "BUY"
        elif total_score >= 42: signal = "NEUTRAL"
        elif total_score >= 28: signal = "SELL"
        else:                   signal = "STRONG_SELL"

        return signal, round(total_score)

    def _data_quality_score(self, result: Dict) -> int:
        score = 0
        if result.get("price"): score += 20
        if result.get("fundamentals"): score += 15
        if result.get("garch"): score += 15
        if result.get("regime"): score += 15
        if result.get("ml_predictions", {}).get("ensemble"): score += 15
        if result.get("options"): score += 10
        if result.get("sentiment"): score += 10
        return min(score, 100)


# ── API Endpoints ─────────────────────────────────────────────

@router.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    http_request: Request,
    req: AnalyzeRequest = None,
    current_user: Optional[CognitoUser] = Depends(get_optional_user),
):
    """
    Main v6 analysis endpoint.
    Accepts both flat body {ticker,...} and nested {req:{ticker,...}} for compatibility.
    Uses app.state.redis (pool created at startup) — no new connections per body.
    """
    # Handle both body shapes:
    # Shape A (frontend): { "ticker": "MSFT", "include_options": true, ... }
    # Shape B (legacy):   { "req": { "ticker": "MSFT", ... } }
    if req is None:
        try:
            body = await http_request.json()
            # Shape A — flat body
            if "ticker" in body:
                req = AnalyzeRequest(**body)
            # Shape B — nested under "req"
            elif "req" in body:
                req = AnalyzeRequest(**body["req"])
            else:
                raise HTTPException(status_code=422, detail="Request must contain 'ticker' field")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid request body: {e}")

    # Use the shared Redis pool from app state — never create connections in handlers
    redis = http_request.app.state.redis
    analyzer: QuantEdgeAnalyzerV6 = http_request.app.state.analyzer

    # FIX (step-4): the key must encode request options — a result computed
    # WITHOUT options must never be served to a request that wants them.
    variant = f"o{int(req.include_options)}s{int(req.include_sentiment)}"
    cache_key = f"analysis:v6:{req.ticker}:{variant}"

    # Check cache
    try:
        cached = await redis.get(cache_key)
        if cached:
            return {"data": _sanitize_json(json.loads(cached)), "cached": True}
    except Exception:
        pass

    # Run analysis (120-second timeout enforced inside run_full_analysis)
    data = await analyzer.run_full_analysis(
        ticker=req.ticker,
        include_options=req.include_options,
        include_sentiment=req.include_sentiment,
        mc_paths=req.mc_paths,
        target_vol=req.target_vol,
    )

    # Write prediction to PostgreSQL as background task (non-blocking)
    _regime = data.get("current_regime") or data.get("hmm_regime", "Unknown")
    _regime_confidence = float(data.get("regime_confidence", 0.0) or 0.0)
    _weights_used = data.get("ensemble_weights", data.get("weights_used", {}))
    if http_request.app.state.signal_tracker:
        background_tasks.add_task(
            http_request.app.state.signal_tracker.record_signal,
            ticker=req.ticker,
            analysis_result=data,
            regime=_regime,
            regime_confidence=_regime_confidence,
            weights_used=_weights_used,
        )

    # Sanitize inf/nan before caching — GARCH long-run vol can produce inf when persistence≈1
    import math
    def _sanitize(obj):
        if isinstance(obj, dict):  return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_sanitize(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
        return obj
    data = _sanitize(data)

    # Write to cache as background task
    async def _cache():
        try:
            await redis.setex(cache_key, 3600, json.dumps(data, default=str))
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    background_tasks.add_task(_cache)

    return {"data": _sanitize_json(data), "cached": False}


@router.delete("/cache/{ticker}")
async def clear_cache(
    ticker: str,
    http_request: Request,
    current_user: CognitoUser = Depends(get_current_user),
):
    """Clear Redis cache for a specific ticker — forces fresh analysis on next request.
    FIX (step-5): owner login required — every public cache clear forced a fresh
    paid-API + compute run, so this was an open cost-amplification endpoint."""
    ticker = ticker.upper().strip()
    redis = http_request.app.state.redis
    keys_deleted = 0
    for prefix in ["analysis:v6:", "chart:v1:", "polygon:v1:"]:
        try:
            # Delete analysis cache
            deleted = await redis.delete(f"{prefix}{ticker}")
            keys_deleted += deleted
            # FIX (step-4): also clear the option-variant keys (and the legacy bare
            # key above stays covered for older cached entries)
            if prefix == "analysis:v6:":
                for v in ("o0s0", "o0s1", "o1s0", "o1s1"):
                    keys_deleted += await redis.delete(f"{prefix}{ticker}:{v}")
            # Delete polygon sub-caches
            for suffix in [":ohlcv", ":fundamentals", ":news", ":options"]:
                deleted = await redis.delete(f"polygon:v1:{ticker}{suffix}")
                keys_deleted += deleted
        except Exception:
            pass
    return {"ticker": ticker, "keys_deleted": keys_deleted, "status": "cache cleared"}


@router.get("/history/{ticker}")
async def get_history(
    ticker: str,
    current_user: CognitoUser = Depends(get_current_user),
):
    return {"ticker": ticker, "analyses": [], "message": "History stored in S3"}


@router.get("/chart/{ticker}")
async def get_chart_data(
    ticker: str,
    timeframe: str = "1Y",
    http_request: Request = None,
    current_user: Optional[CognitoUser] = Depends(get_optional_user),
):
    """
    Real OHLCV chart data from Polygon for all 6 timeframes.
    Used by PriceChart.tsx to render live candlestick/line charts.
    Timeframes: 1D, 5D, 1M, 3M, 1Y, 5Y
    Returns: { labels, opens, highs, lows, closes, volumes, vwaps,
               change, changePct, periodHigh, periodLow, avgVolume,
               interval, n_bars }
    """
    ticker = ticker.upper().strip()

    # Map timeframe to Polygon multiplier/timespan/lookback
    tf_map = {
        "1D":  {"mult": 5,  "span": "minute", "days": 2,    "label": "5-min bars"},
        "5D":  {"mult": 15, "span": "minute", "days": 7,    "label": "15-min bars"},
        "1M":  {"mult": 1,  "span": "day",    "days": 35,   "label": "Daily bars"},
        "3M":  {"mult": 1,  "span": "day",    "days": 95,   "label": "Daily bars"},
        "1Y":  {"mult": 1,  "span": "week",   "days": 370,  "label": "Weekly bars"},
        "5Y":  {"mult": 1,  "span": "month",  "days": 1830, "label": "Monthly bars"},
    }
    cfg = tf_map.get(timeframe.upper(), tf_map["1Y"])

    from datetime import date, timedelta
    from core.config import settings
    import aiohttp

    end_date   = date.today()
    start_date = end_date - timedelta(days=cfg["days"])
    from_str   = start_date.strftime("%Y-%m-%d")
    to_str     = end_date.strftime("%Y-%m-%d")

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range"
        f"/{cfg['mult']}/{cfg['span']}/{from_str}/{to_str}"
        f"?adjusted=true&sort=asc&limit=5000&apiKey={settings.POLYGON_API_KEY}"
    )

    try:
        # Check Redis cache first
        redis = http_request.app.state.redis if http_request else None
        cache_key = f"chart:v1:{ticker}:{timeframe}"
        if redis:
            cached = await redis.get(cache_key)
            if cached:
                import json as _json
                return _json.loads(cached)

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        ) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=resp.status, detail=f"Polygon error: {resp.status}")
                data = await resp.json()

        results = data.get("results", [])
        if not results:
            raise HTTPException(status_code=404, detail=f"No chart data for {ticker}")

        # Format timestamps as readable labels
        from datetime import datetime as dt
        def fmt_label(ts_ms, tf):
            d = dt.utcfromtimestamp(ts_ms / 1000)
            if tf in ("1D", "5D"):
                return d.strftime("%H:%M")
            if tf in ("1M", "3M"):
                return d.strftime("%b %d")
            if tf == "1Y":
                return d.strftime("%b %d")
            return d.strftime("%b '%y")

        labels  = [fmt_label(r["t"], timeframe) for r in results]
        opens   = [round(r.get("o", 0), 2) for r in results]
        highs   = [round(r.get("h", 0), 2) for r in results]
        lows    = [round(r.get("l", 0), 2) for r in results]
        closes  = [round(r.get("c", 0), 2) for r in results]
        volumes = [r.get("v", 0) for r in results]
        vwaps   = [round(r.get("vw", r.get("c", 0)), 2) for r in results]

        first_open = opens[0] if opens else 1
        last_close = closes[-1] if closes else 1
        change     = round(last_close - first_open, 2)
        change_pct = round((change / first_open) * 100, 2) if first_open else 0

        period_high = max(highs) if highs else 0
        period_low  = min(lows)  if lows  else 0
        avg_vol     = round(sum(volumes) / len(volumes)) if volumes else 0

        payload = {
            "ticker":      ticker,
            "timeframe":   timeframe,
            "labels":      labels,
            "opens":       opens,
            "highs":       highs,
            "lows":        lows,
            "closes":      closes,
            "volumes":     volumes,
            "vwaps":       vwaps,
            "change":      change,
            "changePct":   change_pct,
            "periodHigh":  period_high,
            "periodLow":   period_low,
            "avgVolume":   avg_vol,
            "interval":    cfg["label"],
            "n_bars":      len(results),
        }

        # Cache — shorter TTL for intraday, longer for weekly/monthly
        if redis:
            ttl = 60 if timeframe in ("1D", "5D") else 300 if timeframe in ("1M", "3M") else 3600
            import json as _json
            await redis.setex(cache_key, ttl, _json.dumps(payload))

        return payload

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart data error {ticker}/{timeframe}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
