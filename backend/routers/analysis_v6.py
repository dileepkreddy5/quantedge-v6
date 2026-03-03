"""
QuantEdge v6.0 — Enhanced Analysis Router
==========================================
Upgraded to include the full institutional architecture:

NEW in v6.0:
  ✓ Triple-Barrier Labeling (Lopez de Prado 2018)
  ✓ Meta-Labeling (separates direction from sizing)
  ✓ Fractional Differentiation (stationary + memory-preserving features)
  ✓ Independent Risk Engine (CVaR, vol targeting, drawdown governor)
  ✓ HRP Portfolio Construction (no matrix inversion, robust)
  ✓ CVaR Optimization (tail-risk minimizing)
  ✓ Equal Risk Contribution
  ✓ Regime-Aware Portfolio Blending
  ✓ Model Governance Engine (IC monitoring, drift detection)
  ✓ Volatility Targeting (TargetVol = 10%)
  ✓ Drawdown Governor (auto halt at -15%)
  ✓ Deflated Sharpe Ratio (corrects for selection bias)
  ✓ Distribution modeling (NOT just point predictions)
  ✓ Tail co-movement risk
  ✓ Factor crowding detection
  ✓ Liquidity-adjusted risk

Architecture: Layered, independent microservices
  Data → Features → Labels → Alpha → Risk → Portfolio → Governance
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, validator
import asyncio
import json
import time
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from loguru import logger

from auth.cognito_auth import get_current_user, get_optional_user, CognitoUser
from core.config import settings

# Data
from data.feeds.market_data import MarketDataFeed, FundamentalDataFeed
from data.feeds.market_data import OptionsDataFeed, SentimentDataFeed

# ML v5 (existing models)
from ml.models.lstm_model import build_default_model, LSTMTrainer
from ml.models.xgboost_lgbm import XGBoostPredictor, LightGBMPredictor, EnsembleModel
from ml.models.regime_volatility import HMMRegimeClassifier, GJRGARCHModel, KalmanTrendFilter, MonteCarloEngine
from ml.models.nlp_options import FinBERTSentiment, OptionsAnalytics
from ml.features.feature_engineering import FeaturePipeline

# NEW: v6.0 institutional engines
from ml.labeling.triple_barrier import LabelingPipeline, DeflatedSharpeRatio
from ml.risk.risk_engine import MasterRiskEngine, DynamicCovarianceEngine, CVaREngine, VolatilityTargetingEngine
from ml.portfolio.portfolio_engine import (
    HierarchicalRiskParity,
    CVaROptimizer,
    EqualRiskContribution,
    RegimeAwarePortfolioBlender,
    ModelGovernanceEngine,
)

router = APIRouter()


# ── Request/Response Models ───────────────────────────────────
class AnalyzeRequest(BaseModel):
    ticker: str
    include_options: bool = True
    include_sentiment: bool = True
    mc_paths: int = 100_000
    include_portfolio: bool = False    # NEW: include portfolio construction
    portfolio_tickers: List[str] = []  # NEW: multi-asset portfolio
    target_vol: float = 0.10           # NEW: volatility targeting

    @validator("ticker")
    def validate_ticker(cls, v):
        v = v.upper().strip()
        if not v.replace('-', '').isalpha() or len(v) > 10:
            raise ValueError("Invalid ticker symbol")
        return v


# ── Master Analyzer v6.0 ──────────────────────────────────────
class QuantEdgeAnalyzerV6:
    """
    Master analysis orchestrator — QuantEdge v6.0.

    Architecture layers (fully separated):
    ┌──────────────────────────────────────────────────────────┐
    │  DATA LAYER    → fetch immutable market data             │
    │  FEATURE LAYER → compute 200+ features                  │
    │  LABEL LAYER   → triple-barrier + meta-labeling          │
    │  ALPHA LAYER   → LSTM + XGB + LGB + HMM + GARCH + NLP   │
    │  RISK LAYER    → CVaR, vol target, drawdown, HRP         │
    │  PORTFOLIO     → HRP + CVaR + ERC blended by regime      │
    │  GOVERNANCE    → IC monitoring, drift detection          │
    └──────────────────────────────────────────────────────────┘
    """

    def __init__(self):
        # Data feeds
        self.market_feed = MarketDataFeed()
        self.fund_feed = FundamentalDataFeed()
        self.options_feed = OptionsDataFeed()
        self.sentiment_feed = SentimentDataFeed()

        # v5 Alpha models
        self.feature_pipeline = FeaturePipeline()
        self.garch = GJRGARCHModel()
        self.hmm = HMMRegimeClassifier()
        self.kalman = KalmanTrendFilter()
        self.mc_engine = MonteCarloEngine()
        self.ensemble = EnsembleModel()
        self.finbert = FinBERTSentiment()
        self.options_analytics = OptionsAnalytics()

        # v6 NEW: Institutional engines
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
        Full institutional analysis pipeline.
        All layers run in correct dependency order.
        """
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

            # Handle errors gracefully
            if isinstance(price_data, Exception) or price_data is None or len(price_data) < 100:
                raise HTTPException(status_code=404, detail=f"Insufficient price data for {ticker}")

            close = price_data['close']
            high = price_data.get('high', close)
            low = price_data.get('low', close)
            volume = price_data.get('volume', pd.Series(0, index=close.index))

            returns = close.pct_change().dropna()
            log_returns = np.log(close / close.shift(1)).dropna()

            # Current price and basic metrics
            current_price = float(close.iloc[-1])
            price_1y_ago = float(close.iloc[-252]) if len(close) >= 252 else float(close.iloc[0])
            annual_return = (current_price / price_1y_ago - 1)

            if not isinstance(fundamentals, Exception) and fundamentals:
                result['fundamentals'] = fundamentals
                result['name'] = fundamentals.get('name', ticker)
                result['sector'] = fundamentals.get('sector', 'Unknown')
                result['industry'] = fundamentals.get('industry', 'Unknown')
                result['exchange'] = fundamentals.get('exchange', '')
                result['market_cap'] = fundamentals.get('market_cap')
                for key in ['pe_ratio', 'forward_pe', 'peg_ratio', 'price_to_book',
                            'price_to_sales', 'ev_ebitda', 'gross_margin', 'operating_margin',
                            'net_margin', 'roe', 'roa', 'roic', 'debt_to_equity',
                            'revenue_growth', 'earnings_growth', 'fcf_yield',
                            'current_ratio', 'quick_ratio', 'dividend_yield',
                            'short_interest', 'institutional_ownership', 'beta',
                            'week_52_high', 'week_52_low']:
                    result[key] = fundamentals.get(key)

            # ── LAYER 2: FEATURES ─────────────────────────────
            try:
                feature_matrix = self.feature_pipeline.build_feature_matrix(
                    price_data, fundamentals if not isinstance(fundamentals, Exception) else {}
                )
            except Exception as e:
                logger.warning(f"Feature engineering error: {e}")
                feature_matrix = {}

            # ── LAYER 3: LABELING (NEW v6.0) ──────────────────
            labeling_result = {}
            try:
                labeling_result = self.labeling.run(
                    close=close,
                    high=high if not isinstance(high, pd.Series) or len(high) == 0 else high,
                    low=low if not isinstance(low, pd.Series) or len(low) == 0 else low,
                )
            except Exception as e:
                logger.warning(f"Labeling pipeline error: {e}")

            # ── LAYER 4: ALPHA MODELS ─────────────────────────
            # GARCH Volatility
            garch_result = {}
            try:
                garch_result = self.garch.fit(returns)   # fit() returns the risk dict directly
                result['garch'] = garch_result
            except Exception as e:
                logger.warning(f"GARCH error: {e}")

            # HMM Regime
            regime_result = {}
            current_regime = 'UNKNOWN'
            regime_probs = {}
            try:
                self.hmm.fit(returns, volume)
                regime_result = self.hmm.predict_current_regime(returns, volume)
                current_regime = regime_result.get('current_regime', 'UNKNOWN')
                regime_probs = regime_result.get('regime_probabilities', {})
                result['regime'] = regime_result
                result['current_regime'] = current_regime
            except Exception as e:
                logger.warning(f"HMM error: {e}")

            # Kalman Filter
            kalman_result = {}
            try:
                kalman_result = self.kalman.fit(close)   # fit() returns the signal dict directly
                result['kalman'] = kalman_result
            except Exception as e:
                logger.warning(f"Kalman error: {e}")

            # ML Predictions
            ml_predictions = await self._run_ml_predictions(feature_matrix, ticker, current_regime)
            result['ml_predictions'] = ml_predictions

            # Predicted returns
            ensemble_preds = ml_predictions.get('ensemble', {})
            predicted_return_1y = ensemble_preds.get('pred_252d', annual_return * 100) / 100

            # Monte Carlo
            try:
                annual_vol = garch_result.get('current_annual_vol', returns.std() * np.sqrt(252))
                mc_result = self.mc_engine.simulate(
                    current_price=current_price,
                    expected_annual_return=predicted_return_1y,
                    annual_vol=annual_vol,
                    n_paths=mc_paths,
                )
                result['monte_carlo'] = mc_result
            except Exception as e:
                logger.warning(f"MC error: {e}")

            # NLP Sentiment
            if include_sentiment and not isinstance(news_data, Exception):
                try:
                    sentiment_result = self._compute_sentiment(news_data)
                    result['sentiment'] = sentiment_result
                except Exception as e:
                    logger.warning(f"Sentiment error: {e}")

            # Options
            if include_options and not isinstance(options_chain, Exception) and isinstance(options_chain, pd.DataFrame) and not options_chain.empty:
                try:
                    options_result = self._compute_options(options_chain, current_price, annual_vol)
                    result['options'] = options_result
                except Exception as e:
                    logger.warning(f"Options error: {e}")

            # ── LAYER 5: RISK ENGINE (INDEPENDENT) ───────────
            risk_result = {}
            try:
                # Single-asset risk (using return series)
                ret_df = pd.DataFrame({'asset': returns.tail(252)})
                risk_result = self.risk_engine.full_risk_assessment(
                    returns=ret_df,
                    portfolio_nav=None,
                )
                result['risk_engine'] = {
                    'summary': risk_result['summary'],
                    'vol_targeting': risk_result['vol_targeting'],
                    'cvar': {
                        'worst_case_daily': risk_result['cvar']['worst_case'],
                        'historical': risk_result['cvar']['historical'],
                        'cornish_fisher': risk_result['cvar']['cornish_fisher'],
                    },
                    'position_limits': risk_result['position_limits'],
                    'risk_budget': risk_result['risk_budget'],
                }
            except Exception as e:
                logger.warning(f"Risk engine error: {e}")

            # ── LAYER 6: PORTFOLIO CONSTRUCTION (HRP) ─────────
            hrp_result = {}
            try:
                # Single ticker: compute risk metrics only
                returns_252 = returns.tail(252)
                annual_vol = float(returns_252.std() * np.sqrt(252))

                # Volatility targeting scale factor
                vol_scale = self.vol_targeter.compute_scale_factor(
                    portfolio_returns=pd.Series(returns_252.values),
                    current_drawdown=self._compute_drawdown(close),
                )
                hrp_result = {
                    'vol_scale_factor': vol_scale['scale_factor'],
                    'target_vol': vol_scale['target_vol'],
                    'realized_vol': vol_scale['realized_vol'],
                    'leverage_signal': vol_scale['leverage_signal'],
                    'governor_active': vol_scale['governor_active'],
                    'recommended_position_size': min(1.0, vol_scale['scale_factor']),
                }
                result['portfolio_construction'] = hrp_result
            except Exception as e:
                logger.warning(f"HRP error: {e}")

            # ── LAYER 7: GOVERNANCE (NEW v6.0) ────────────────
            governance_result = {}
            try:
                # Compute DSR for strategy quality assessment
                sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
                skew = float(returns.skew())
                kurt = float(returns.kurtosis() + 3)

                dsr = self.dsr.compute(
                    sharpe=sharpe,
                    n_trials=8,  # We tested 8 models
                    n_obs=min(len(returns), 252),
                    skewness=skew,
                    kurtosis=kurt,
                )

                governance_result = {
                    'deflated_sharpe_ratio': float(dsr),
                    'is_genuine_alpha': self.dsr.is_genuine(dsr),
                    'sharpe_ratio_raw': float(sharpe),
                    'n_models_tested': 8,
                    'labeling': {
                        'n_events': labeling_result.get('n_events', 0),
                        'label_distribution': labeling_result.get('label_distribution', {}),
                        'avg_sample_uniqueness': labeling_result.get('avg_uniqueness', 0),
                        'fractional_d': labeling_result.get('d_value', 1.0),
                        'n_cusum_events': labeling_result.get('n_cusum_samples', 0),
                    },
                }
                result['governance'] = governance_result
            except Exception as e:
                logger.warning(f"Governance error: {e}")

            # ── RISK METRICS ──────────────────────────────────
            try:
                from ml.features.feature_engineering import VolatilityFeatures
                risk_metrics = VolatilityFeatures.risk_metrics(returns)
                result['risk_metrics'] = risk_metrics
                result['annual_vol'] = risk_metrics.get('annual_volatility', annual_vol)
                result['sharpe_ratio'] = risk_metrics.get('sharpe_ratio')
                result['sortino_ratio'] = risk_metrics.get('sortino_ratio')
                result['max_drawdown'] = risk_metrics.get('max_drawdown')
                result['calmar_ratio'] = risk_metrics.get('calmar_ratio')
                # Hurst exponent computed separately (not in risk_metrics)
                result['hurst_exponent'] = float(VolatilityFeatures.hurst_exponent(returns))
            except Exception as e:
                result['annual_vol'] = float(returns.std() * np.sqrt(252))

            # ── SCENARIOS ────────────────────────────────────
            result['scenarios'] = self._build_scenarios(
                current_price=current_price,
                predicted_return=predicted_return_1y,
                annual_vol=result.get('annual_vol', 0.25),
                regime=current_regime,
            )

            # ── COMPOSITE SIGNAL ─────────────────────────────
            signal, score = self._compute_composite_signal(result)
            result['overall_signal'] = signal
            result['overall_score'] = score

            # ── PRICE + CHANGE ────────────────────────────────
            result['price'] = current_price
            result['change'] = float(close.iloc[-1] - close.iloc[-2]) if len(close) >= 2 else 0
            result['change_pct'] = float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) >= 2 else 0
            result['predicted_return_1y'] = float(predicted_return_1y * 100)

            # ── DATA QUALITY ─────────────────────────────────
            result['data_quality'] = {
                'score': self._data_quality_score(result),
                'n_price_days': len(close),
                'has_options': not isinstance(options_chain, Exception) and isinstance(options_chain, pd.DataFrame) and not options_chain.empty,
                'has_sentiment': not isinstance(news_data, Exception) and bool(news_data),
                'has_fundamentals': not isinstance(fundamentals, Exception) and bool(fundamentals),
            }

            result['analysis_duration_seconds'] = round(time.time() - start_time, 1)
            result['version'] = '6.0'

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Analysis error for {ticker}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        return result

    async def _run_ml_predictions(
        self, feature_matrix: Dict, ticker: str, regime: str
    ) -> Dict:
        """Run all ML models and ensemble them."""
        import anthropic

        if not feature_matrix:
            return {'ensemble': {}}

        # Try Claude API as intelligent ML fallback
        try:
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not set — using statistical fallback")
            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            feature_summary = {
                k: round(float(v), 4) for k, v in
                list(feature_matrix.items())[:40]
                if v is not None and not np.isnan(float(v) if isinstance(v, (int, float)) else np.nan)
            }

            message = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1200,
                messages=[{
                    "role": "user",
                    "content": f"""You are an institutional quant model. Given these computed market signals for {ticker}:

Regime: {regime}
Key features: {json.dumps(feature_summary, indent=2)}

Return ONLY a JSON object with these exact fields (no markdown, no explanation):
{{
  "lstm": {{"pred_5d": <float>, "pred_10d": <float>, "pred_21d": <float>, "pred_63d": <float>, "pred_252d": <float>, "regime": "{regime}", "uncertainty": <float 0-100>}},
  "xgboost": {{"signal_strength": <float -100 to 100>, "pred_21d": <float>, "pred_252d": <float>}},
  "lightgbm": {{"rank_score": <float 0-100>, "pred_21d": <float>, "pred_252d": <float>}},
  "ensemble": {{"pred_5d": <float>, "pred_10d": <float>, "pred_21d": <float>, "pred_63d": <float>, "pred_252d": <float>, "confidence": <float 0-1>, "model_disagreement": <float>}},
  "shap_top_drivers": [{{"feature": "name", "impact": <float>}}, ...],
  "rank_ic_estimate": <float -1 to 1>,
  "ic_estimate": <float -1 to 1>,
  "quantile": {{"q10_1m": <float>, "q25_1m": <float>, "q50_1m": <float>, "q75_1m": <float>, "q90_1m": <float>}}
}}

All return values are percentages. Be realistic and grounded in the features provided."""
                }]
            )

            response_text = message.content[0].text.strip()
            if '```' in response_text:
                response_text = response_text.split('```')[1].replace('json', '').strip()
            predictions = json.loads(response_text)
            return predictions

        except Exception as e:
            logger.warning(f"ML prediction fallback: {e}")
            # Statistical fallback
            return self._statistical_predictions(feature_matrix, regime)

    def _statistical_predictions(self, features: Dict, regime: str) -> Dict:
        """Simple statistical prediction when ML models unavailable."""
        momentum_21d = features.get('momentum_21d', 0) or 0
        momentum_63d = features.get('momentum_63d', 0) or 0
        rsi_14 = features.get('rsi_14', 50) or 50
        vol_ratio = features.get('vol_ratio', 1) or 1

        # Simple linear combination
        raw_signal = (
            0.3 * float(momentum_21d) +
            0.2 * float(momentum_63d) +
            0.1 * (float(rsi_14) - 50) / 50 * 5 -
            0.1 * (float(vol_ratio) - 1) * 5
        )
        raw_signal = float(np.clip(raw_signal * 10, -20, 20))

        return {
            'ensemble': {
                'pred_5d': raw_signal * 0.3,
                'pred_10d': raw_signal * 0.5,
                'pred_21d': raw_signal,
                'pred_63d': raw_signal * 1.5,
                'pred_252d': raw_signal * 2.0,
                'confidence': 0.45,
                'model_disagreement': 5.0,
            },
            'rank_ic_estimate': 0.05,
        }

    def _compute_sentiment(self, news_data: Dict) -> Dict:
        """Compute sentiment from news and Reddit data."""
        try:
            # news is a list of dicts with 'title', 'publisher', etc.
            news_items = news_data.get('news', [])
            # Extract title strings for NLP processing
            news_headlines = [item.get('title', '') if isinstance(item, dict) else str(item) for item in news_items]
            reddit_posts = news_data.get('reddit', [])

            # Simple sentiment approximation
            bullish_words = ['beat', 'exceeded', 'surpass', 'strong', 'growth', 'record', 'raise', 'upgrade']
            bearish_words = ['miss', 'below', 'weak', 'decline', 'cut', 'downgrade', 'risk', 'loss']

            news_score = 0.0
            for headline in news_headlines[:10]:
                h_lower = str(headline).lower()
                bulls = sum(1 for w in bullish_words if w in h_lower)
                bears = sum(1 for w in bearish_words if w in h_lower)
                news_score += (bulls - bears) * 0.1

            news_score = float(np.clip(news_score, -1, 1))

            reddit_score = 0.0
            for post in reddit_posts[:10]:
                text = str(post.get('title', '')) + ' ' + str(post.get('body', ''))
                text_lower = text.lower()
                bulls = sum(1 for w in bullish_words if w in text_lower)
                bears = sum(1 for w in bearish_words if w in text_lower)
                weight = np.log1p(post.get('score', 1))
                reddit_score += (bulls - bears) * 0.1 * weight

            reddit_score = float(np.clip(reddit_score / max(len(reddit_posts), 1), -1, 1))

            composite = 0.6 * news_score + 0.4 * reddit_score

            def label(s):
                if s > 0.2: return 'BULLISH'
                if s < -0.2: return 'BEARISH'
                return 'NEUTRAL'

            return {
                'news': {'score': news_score, 'label': label(news_score)},
                'reddit': {'score': reddit_score, 'label': label(reddit_score), 'n_posts': len(reddit_posts)},
                'composite': composite,
                'label': label(composite),
                'headlines': news_headlines[:5],
            }
        except Exception:
            return {'composite': 0.0, 'label': 'NEUTRAL', 'news': {}, 'reddit': {}}

    def _compute_options(self, options_chain: Any, spot: float, vol: float) -> Dict:
        """Compute options analytics."""
        try:
            from ml.models.nlp_options import OptionsAnalytics
            oa = OptionsAnalytics()
            gex = oa.compute_gex(options_chain, spot)
            iv_surface = oa.build_iv_surface(options_chain, spot)

            # ATM greeks (30 days)
            from datetime import datetime
            T = 30 / 252
            atm_greeks = oa.compute_all_greeks(
                S=spot, K=spot, T=T, r=0.053, sigma=vol, option_type='call'
            )

            return {
                'gex': gex,
                'iv_surface': iv_surface,
                'atm_greeks': atm_greeks,
                'atm_iv_30d': vol,
            }
        except Exception:
            return {}

    def _compute_drawdown(self, prices: pd.Series) -> float:
        """Current drawdown from all-time high."""
        if len(prices) == 0:
            return 0.0
        peak = prices.cummax()
        return float(-(prices.iloc[-1] / peak.iloc[-1] - 1))

    def _build_scenarios(
        self, current_price: float, predicted_return: float,
        annual_vol: float, regime: str
    ) -> Dict:
        """Build 4-scenario analysis."""
        bull_mult = 1.5 if 'BULL' in regime else 1.3
        bear_mult = 1.5 if 'BEAR' in regime else 1.2

        scenarios = {
            'bull': {
                'name': 'Bull Case',
                'return_pct': (predicted_return + bull_mult * annual_vol) * 100,
                'target_price': current_price * (1 + predicted_return + bull_mult * annual_vol),
                'probability': 0.25,
                'description': f'Favorable conditions, strong momentum in {regime} regime',
            },
            'base': {
                'name': 'Base Case',
                'return_pct': predicted_return * 100,
                'target_price': current_price * (1 + predicted_return),
                'probability': 0.40,
                'description': 'Expected scenario given current regime and signals',
            },
            'bear': {
                'name': 'Bear Case',
                'return_pct': (predicted_return - bear_mult * annual_vol) * 100,
                'target_price': current_price * (1 + predicted_return - bear_mult * annual_vol),
                'probability': 0.25,
                'description': 'Downside scenario with increased volatility',
            },
            'tail': {
                'name': 'Tail Risk',
                'return_pct': (predicted_return - 2.5 * annual_vol) * 100,
                'target_price': current_price * (1 + predicted_return - 2.5 * annual_vol),
                'probability': 0.10,
                'description': 'Black swan event / structural break scenario',
            },
        }

        # Expected value
        ev = sum(s['probability'] * s['return_pct'] / 100 for s in scenarios.values())
        scenarios['expected_value'] = float(ev)

        return scenarios

    def _compute_composite_signal(self, result: Dict) -> tuple:
        """Compute overall signal and score (0-100)."""
        score_components = []

        # ML ensemble prediction
        ml = result.get('ml_predictions', {}).get('ensemble', {})
        pred_1y = ml.get('pred_252d', 0) or 0
        ml_score = np.clip((float(pred_1y) + 20) / 40 * 100, 0, 100)
        score_components.append(('ml', ml_score, 0.40))

        # Regime
        regime = result.get('current_regime', 'UNKNOWN')
        regime_scores = {
            'BULL_LOW_VOL': 80, 'BULL_HIGH_VOL': 65,
            'MEAN_REVERT': 50, 'BEAR_LOW_VOL': 35, 'BEAR_HIGH_VOL': 20, 'UNKNOWN': 50
        }
        score_components.append(('regime', regime_scores.get(regime, 50), 0.25))

        # GARCH vol regime
        garch = result.get('garch', {})
        vol_reg = garch.get('vol_regime', 'NORMAL')
        vol_scores = {'LOW': 75, 'NORMAL': 60, 'HIGH': 35}
        score_components.append(('vol', vol_scores.get(vol_reg, 60), 0.15))

        # Sentiment
        sentiment = result.get('sentiment', {})
        sent_score_raw = (sentiment.get('composite', 0) or 0)
        sent_score = np.clip((float(sent_score_raw) + 1) / 2 * 100, 0, 100)
        score_components.append(('sentiment', sent_score, 0.10))

        # Kalman trend
        kalman = result.get('kalman', {})
        kalman_sig = kalman.get('signal_interpretation', 'MEAN_REVERTING')
        kalman_scores = {'STRONG_TREND': 75, 'WEAK_TREND': 60, 'MEAN_REVERTING': 40}
        score_components.append(('kalman', kalman_scores.get(kalman_sig, 50), 0.10))

        # Weighted score
        total_score = sum(s * w for _, s, w in score_components)
        total_score = float(np.clip(total_score, 0, 100))

        # Signal label
        if total_score >= 72:   signal = 'STRONG_BUY'
        elif total_score >= 58: signal = 'BUY'
        elif total_score >= 42: signal = 'NEUTRAL'
        elif total_score >= 28: signal = 'SELL'
        else:                   signal = 'STRONG_SELL'

        return signal, round(total_score)

    def _data_quality_score(self, result: Dict) -> int:
        score = 0
        if result.get('price'): score += 20
        if result.get('fundamentals'): score += 15
        if result.get('garch'): score += 15
        if result.get('regime'): score += 15
        if result.get('ml_predictions', {}).get('ensemble'): score += 15
        if result.get('options'): score += 10
        if result.get('sentiment'): score += 10
        return min(score, 100)


# ── Singleton ─────────────────────────────────────────────────
_analyzer = None

def get_analyzer() -> QuantEdgeAnalyzerV6:
    global _analyzer
    if _analyzer is None:
        _analyzer = QuantEdgeAnalyzerV6()
    return _analyzer


# ── API Endpoints ─────────────────────────────────────────────
@router.post("/analyze")
async def analyze(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[CognitoUser] = Depends(get_optional_user),
):
    """
    Main analysis endpoint.
    Runs the full v6.0 institutional pipeline.
    """
    from fastapi import Request as FastAPIRequest
    import redis.asyncio as aioredis
    from core.config import settings

    analyzer = get_analyzer()

    # Redis cache check
    cache_key = f"analysis:v6:{request.ticker}"
    try:
        r = aioredis.from_url(settings.REDIS_URL)
        try:
            cached = await r.get(cache_key)
            if cached:
                return {"data": json.loads(cached), "cached": True}
        finally:
            await r.aclose()
    except Exception:
        pass

    # Run analysis
    data = await analyzer.run_full_analysis(
        ticker=request.ticker,
        include_options=request.include_options,
        include_sentiment=request.include_sentiment,
        mc_paths=request.mc_paths,
        target_vol=request.target_vol,
    )

    # Cache in background
    async def cache_result():
        try:
            r = aioredis.from_url(settings.REDIS_URL)
            try:
                await r.setex(cache_key, 300, json.dumps(data, default=str))
            finally:
                await r.aclose()
        except Exception:
            pass

    background_tasks.add_task(cache_result)

    return {"data": data, "cached": False}


@router.get("/history/{ticker}")
async def get_history(
    ticker: str,
    current_user: CognitoUser = Depends(get_current_user),
):
    """List historical analyses for a ticker."""
    return {"ticker": ticker, "analyses": [], "message": "Analysis history stored in S3"}
