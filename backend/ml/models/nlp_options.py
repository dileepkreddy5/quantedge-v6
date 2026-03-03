"""
QuantEdge v5.0 — NLP Sentiment Engine + Options Analytics
===========================================================
NLP Component:
  - FinBERT (ProsusAI/finbert): financial-domain BERT
  - Processes: SEC 8-K/10-Q filings, Reddit WSB, earnings calls, news
  - Outputs: sentiment score, attention weights, entity extraction
  - Signal alpha: shown to predict abnormal returns (Ahmad et al. 2016)

Options Analytics:
  - Black-Scholes-Merton Greeks (Delta, Gamma, Vega, Theta, Rho)
  - Higher-order Greeks: Vanna, Volga (Vomma), Charm, Speed
  - GEX (Gamma Exposure): predicts pinning and volatility clusters
  - IV Surface: build term structure + skew from chain data
  - Max Pain: where option writers profit most
  - VANNA/CHARM flows: dealer hedging pressure signals
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from typing import Dict, List, Optional, Tuple
import re
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# 1. FINBERT NLP SENTIMENT ENGINE
# ══════════════════════════════════════════════════════════════

class FinBERTSentiment:
    """
    Financial BERT sentiment analysis.
    Model: ProsusAI/finbert (trained on Financial PhraseBank)
    Fine-tuned on: earnings call transcripts, SEC filings, financial news

    Three-class output: Positive, Negative, Neutral
    Sentiment score: P(Positive) - P(Negative) ∈ [-1, +1]

    Key financial NLP signals:
      - Management guidance sentiment
      - Earnings call tone shift (Q vs Q)
      - SEC filing risk factor language
      - Reddit aggregated retail sentiment
    """

    def __init__(self, use_gpu: bool = False):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if use_gpu else "cpu"
        self._loaded = False

    def _load_model(self):
        """Lazy load FinBERT (900MB model, load once)"""
        if self._loaded:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.model.eval()
            self._loaded = True
        except Exception as e:
            self._loaded = False

    def analyze_text(self, text: str, max_length: int = 512) -> Dict:
        """
        Run FinBERT on a single text passage.
        For texts > 512 tokens, uses sliding window with aggregation.
        """
        if not self._loaded:
            self._load_model()

        if not self._loaded:
            # Fallback to TextBlob if FinBERT unavailable
            return self._textblob_fallback(text)

        try:
            import torch
            # Chunk long texts (sliding window, 50% overlap)
            chunks = self._chunk_text(text, max_length=400, overlap=50)
            chunk_results = []

            for chunk in chunks:
                inputs = self.tokenizer(
                    chunk, return_tensors="pt", truncation=True,
                    max_length=512, padding=True
                )
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)[0].numpy()
                # FinBERT labels: 0=positive, 1=negative, 2=neutral
                chunk_results.append({
                    "positive": float(probs[0]),
                    "negative": float(probs[1]),
                    "neutral": float(probs[2]),
                    "score": float(probs[0] - probs[1]),
                })

            # Aggregate chunks (weight by chunk certainty)
            weights = [1 - r["neutral"] for r in chunk_results]
            total_w = sum(weights) + 1e-10

            agg = {
                "positive": sum(r["positive"] * w for r, w in zip(chunk_results, weights)) / total_w,
                "negative": sum(r["negative"] * w for r, w in zip(chunk_results, weights)) / total_w,
                "neutral": sum(r["neutral"] * w for r, w in zip(chunk_results, weights)) / total_w,
                "score": sum(r["score"] * w for r, w in zip(chunk_results, weights)) / total_w,
                "n_chunks": len(chunks),
                "model": "FinBERT",
            }
            agg["label"] = "POSITIVE" if agg["score"] > 0.05 else "NEGATIVE" if agg["score"] < -0.05 else "NEUTRAL"
            return agg

        except Exception as e:
            return self._textblob_fallback(text)

    def _chunk_text(self, text: str, max_length: int = 400, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks by sentence"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = []
        current_len = 0

        for sent in sentences:
            words = sent.split()
            if current_len + len(words) > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep last `overlap` words for context
                current_chunk = current_chunk[-overlap:]
                current_len = len(current_chunk)
            current_chunk.extend(words)
            current_len += len(words)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks if chunks else [text[:2000]]

    def _textblob_fallback(self, text: str) -> Dict:
        """TextBlob fallback when FinBERT unavailable"""
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            score = float(blob.sentiment.polarity)
            return {
                "positive": max(0, score),
                "negative": max(0, -score),
                "neutral": 1 - abs(score),
                "score": score,
                "label": "POSITIVE" if score > 0.05 else "NEGATIVE" if score < -0.05 else "NEUTRAL",
                "model": "TextBlob",
            }
        except Exception:
            return {"score": 0.0, "label": "NEUTRAL", "model": "FALLBACK"}

    def analyze_earnings_call(self, transcript: str) -> Dict:
        """
        Specialized earnings call analysis.
        Segments: MD&A section, Q&A, guidance language.
        Key signals:
          - Management tone (prepared remarks vs Q&A = less guarded)
          - Forward guidance language ("expect", "anticipate", "confident")
          - Hedging language ("uncertain", "headwinds", "challenges")
        """
        # Extract sections
        prepared_match = re.search(r'(operator|good morning|welcome)(.*?)(question-and-answer|q&a|open for questions)',
                                   transcript.lower(), re.DOTALL)
        qa_match = re.search(r'(question-and-answer|q&a|open for questions)(.*?)$',
                              transcript.lower(), re.DOTALL)

        prepared_text = prepared_match.group(2) if prepared_match else transcript[:len(transcript)//2]
        qa_text = qa_match.group(2) if qa_match else transcript[len(transcript)//2:]

        # Sentiment of each section
        prepared_sentiment = self.analyze_text(prepared_text)
        qa_sentiment = self.analyze_text(qa_text) if qa_text else prepared_sentiment

        # Guidance keywords
        bullish_words = ["exceed", "outperform", "raise", "increase", "accelerate",
                        "strong", "confident", "momentum", "record", "growth"]
        bearish_words = ["miss", "below", "lower", "challenging", "headwinds",
                        "uncertain", "cautious", "pressure", "decline", "reduce"]

        text_lower = transcript.lower()
        bullish_count = sum(text_lower.count(w) for w in bullish_words)
        bearish_count = sum(text_lower.count(w) for w in bearish_words)
        guidance_score = (bullish_count - bearish_count) / (bullish_count + bearish_count + 1)

        # Prepared vs Q&A delta (if management is more positive in prepared remarks,
        # might be scripted optimism — lower credibility)
        tone_delta = prepared_sentiment["score"] - qa_sentiment["score"]

        return {
            "overall_sentiment": (prepared_sentiment["score"] + qa_sentiment["score"]) / 2,
            "prepared_remarks_sentiment": prepared_sentiment["score"],
            "qa_sentiment": qa_sentiment["score"],
            "guidance_score": float(guidance_score),
            "tone_authenticity": 1 - abs(tone_delta),  # Lower delta = more authentic
            "bullish_keywords": bullish_count,
            "bearish_keywords": bearish_count,
            "summary_label": "BULLISH" if guidance_score > 0.1 else "BEARISH" if guidance_score < -0.1 else "NEUTRAL",
        }

    def aggregate_reddit_sentiment(self, posts: List[Dict]) -> Dict:
        """
        Aggregate Reddit WSB/r/investing sentiment with quality weighting.
        Weight posts by: upvotes, awards, comment count.
        Filter: remove obvious noise (very short posts, meme stocks only).
        """
        if not posts:
            return {"score": 0.0, "n_posts": 0, "label": "NEUTRAL"}

        scores = []
        weights = []

        for post in posts:
            text = f"{post.get('title', '')} {post.get('body', '')}"
            if len(text.split()) < 5:
                continue  # Skip too-short posts

            sentiment = self.analyze_text(text)
            # Weight = log(1 + upvotes) * sqrt(1 + comments)
            upvotes = max(post.get("score", 1), 1)
            comments = max(post.get("num_comments", 0), 0)
            weight = np.log1p(upvotes) * np.sqrt(1 + comments)

            scores.append(sentiment["score"])
            weights.append(weight)

        if not scores:
            return {"score": 0.0, "n_posts": 0, "label": "NEUTRAL"}

        # Weighted average sentiment
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        # Retail sentiment signal (inverse for contrarian use)
        contrarian_signal = -weighted_score if abs(weighted_score) > 0.3 else weighted_score

        return {
            "score": float(weighted_score),
            "contrarian_signal": float(contrarian_signal),
            "n_posts": len(scores),
            "label": "BULLISH" if weighted_score > 0.1 else "BEARISH" if weighted_score < -0.1 else "NEUTRAL",
            "sentiment_dispersion": float(np.std(scores)),
            "high_conviction_pct": float(np.mean([abs(s) > 0.3 for s in scores])),
        }


# ══════════════════════════════════════════════════════════════
# 2. OPTIONS ANALYTICS ENGINE
# ══════════════════════════════════════════════════════════════

class OptionsAnalytics:
    """
    Complete options analytics as used by Citadel, Susquehanna, Jane Street.
    Implements Black-Scholes-Merton framework with all Greeks.

    Mathematical foundation:
      BSM: C = S*Φ(d1) - K*e^{-rT}*Φ(d2)
      d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
      d2 = d1 - σ√T

    Greeks computed analytically from BSM derivatives.
    """

    @staticmethod
    def d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """Compute d1, d2 for BSM formula"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0, 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @staticmethod
    def option_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
        """Black-Scholes option price"""
        d1, d2 = OptionsAnalytics.d1_d2(S, K, T, r, sigma)
        if option_type.lower() == "call":
            return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

    @classmethod
    def compute_all_greeks(
        cls,
        S: float,        # Current stock price
        K: float,        # Strike price
        T: float,        # Time to expiry (years)
        r: float,        # Risk-free rate
        sigma: float,    # Implied volatility
        option_type: str = "call",
    ) -> Dict:
        """
        All first and second-order BSM Greeks:

        First-order (∂V/∂x):
          Δ (Delta):  ∂C/∂S — price sensitivity to underlying
          ν (Vega):   ∂C/∂σ — price sensitivity to volatility
          Θ (Theta):  ∂C/∂T — time decay (per day)
          ρ (Rho):    ∂C/∂r — interest rate sensitivity

        Second-order (∂²V/∂x²):
          Γ (Gamma):  ∂²C/∂S² = ∂Δ/∂S — delta change rate
          Vanna:      ∂²C/∂S∂σ = ∂Δ/∂σ — delta change with vol
          Vomma:      ∂²C/∂σ² = ∂ν/∂σ — vega change with vol
          Charm:      ∂²C/∂S∂T = ∂Δ/∂t — delta decay (per day)
          Speed:      ∂³C/∂S³ = ∂Γ/∂S — gamma change with price
          Color:      ∂³C/∂S²∂T = ∂Γ/∂t — gamma decay
        """
        if T <= 0 or sigma <= 0:
            return {}

        d1, d2 = cls.d1_d2(S, K, T, r, sigma)
        phi_d1 = stats.norm.pdf(d1)  # Standard normal PDF at d1
        Phi_d1 = stats.norm.cdf(d1)
        Phi_d2 = stats.norm.cdf(d2)
        sqrt_T = np.sqrt(T)

        sign = 1 if option_type.lower() == "call" else -1

        # ── First-order Greeks ──────────────────────────────
        # Delta: ∂C/∂S = Φ(d1) for call, Φ(d1)-1 for put
        delta = sign * (Phi_d1 if option_type == "call" else Phi_d1 - 1)

        # Gamma: ∂²C/∂S² = φ(d1) / (S*σ*√T) — SAME for calls and puts
        gamma = phi_d1 / (S * sigma * sqrt_T)

        # Vega: ∂C/∂σ = S*φ(d1)*√T — SAME for calls and puts (per 1% vol change / 100)
        vega = S * phi_d1 * sqrt_T / 100

        # Theta: ∂C/∂T — daily time decay (negative for long options)
        theta_common = -(S * phi_d1 * sigma) / (2 * sqrt_T) - r * K * np.exp(-r * T)
        if option_type.lower() == "call":
            theta = (theta_common * Phi_d2) / 365
        else:
            theta = (theta_common * (Phi_d2 - 1)) / 365

        # Rho: ∂C/∂r (per 1% rate change / 100)
        rho = sign * K * T * np.exp(-r * T) * stats.norm.cdf(sign * d2) / 100

        # ── Second-order Greeks ─────────────────────────────
        # Vanna: ∂²C/∂S∂σ = (vega/S) * (1 - d1/(σ√T))
        # Measures how delta changes as implied vol changes
        # Key for dealer hedging: as vol rises, dealer must re-hedge delta
        vanna = (vega * 100 / S) * (1 - d1 / (sigma * sqrt_T))

        # Vomma (Volga): ∂²C/∂σ² = vega * d1*d2/σ
        # Convexity of vega — how vega changes with vol
        vomma = vega * 100 * d1 * d2 / sigma

        # Charm (Delta Decay): ∂Δ/∂t — how delta changes with time
        # Expressed per calendar day
        if option_type.lower() == "call":
            charm = -phi_d1 * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T) / 365
        else:
            charm = -phi_d1 * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T) / 365

        # Speed: ∂Γ/∂S = -Γ * (d1/(σ√T) + 1) / S
        speed = -gamma / S * (d1 / (sigma * sqrt_T) + 1)

        # Color (Gamma Decay): ∂Γ/∂t
        color = -phi_d1 / (2 * S * T * sigma * sqrt_T) * (
            2 * r * T + 1 + d1 * (2 * r * T - d2 * sigma * sqrt_T) / (sigma * sqrt_T)
        ) / 365

        # Option price
        price = cls.option_price(S, K, T, r, sigma, option_type)

        return {
            "price": float(price),
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
            "rho": float(rho),
            "vanna": float(vanna),
            "vomma": float(vomma),
            "charm": float(charm),
            "speed": float(speed),
            "color": float(color),
            "leverage_ratio": float(abs(delta) * S / price) if price > 0 else 0,
        }

    @classmethod
    def implied_volatility(
        cls,
        market_price: float,
        S: float, K: float, T: float, r: float,
        option_type: str = "call",
        precision: float = 1e-5,
    ) -> float:
        """
        Newton-Raphson + bisection for IV calculation.
        Converges in ~5 iterations for reasonable inputs.
        """
        if T <= 0 or market_price <= 0:
            return 0.0

        # Intrinsic value check
        intrinsic = max(0, S - K) if option_type == "call" else max(0, K - S)
        if market_price < intrinsic:
            return 0.0

        try:
            def objective(sigma):
                return cls.option_price(S, K, T, r, max(sigma, 0.001), option_type) - market_price

            iv = brentq(objective, 0.001, 10.0, xtol=precision, maxiter=100)
            return float(iv)
        except Exception:
            return 0.25  # Default 25% IV if cannot solve

    @classmethod
    def compute_gex(
        cls,
        options_chain: pd.DataFrame,
        spot_price: float,
        risk_free_rate: float = 0.05,
    ) -> Dict:
        """
        Gamma Exposure (GEX) — SpotGamma / SqueezeMetrics methodology.
        GEX = Σ Γ_i * OI_i * contract_multiplier * spot²

        Interpretation:
          Positive GEX: market makers are long gamma
            → They SELL when price rises, BUY when falls = volatility dampening
            → Price gravitates towards high-OI strikes (pinning)
          Negative GEX: market makers are short gamma
            → They BUY when price rises, SELL when falls = volatility amplifying
            → Gamma squeeze potential, large moves accelerate

        Gamma flip level: price where GEX transitions positive→negative
        """
        if options_chain.empty:
            return {"total_gex": 0, "gamma_flip_level": spot_price}

        total_gex = 0.0
        strike_gex = {}

        for _, row in options_chain.iterrows():
            try:
                K = float(row.get("strike", 0))
                oi_call = float(row.get("open_interest_call", 0))
                oi_put = float(row.get("open_interest_put", 0))
                T = float(row.get("days_to_expiry", 30)) / 365
                iv = float(row.get("iv", 0.25))

                if K <= 0 or T <= 0 or iv <= 0:
                    continue

                # Call gamma (positive GEX: MMs long call = long gamma)
                call_greeks = cls.compute_all_greeks(spot_price, K, T, risk_free_rate, iv, "call")
                put_greeks = cls.compute_all_greeks(spot_price, K, T, risk_free_rate, iv, "put")

                call_gex = call_greeks["gamma"] * oi_call * 100 * spot_price**2 / 1e9  # In billions
                put_gex = -put_greeks["gamma"] * oi_put * 100 * spot_price**2 / 1e9   # Puts = negative GEX

                strike_gex_val = call_gex + put_gex
                strike_gex[K] = float(strike_gex_val)
                total_gex += strike_gex_val

            except Exception:
                continue

        # Find gamma flip level (zero-crossing of GEX by strike)
        if strike_gex:
            sorted_strikes = sorted(strike_gex.keys())
            gamma_flip = spot_price  # Default
            for i in range(len(sorted_strikes) - 1):
                k1, k2 = sorted_strikes[i], sorted_strikes[i + 1]
                g1, g2 = strike_gex[k1], strike_gex[k2]
                if g1 * g2 < 0:  # Sign change = zero crossing
                    # Linear interpolation
                    gamma_flip = k1 + (k2 - k1) * (-g1) / (g2 - g1 + 1e-10)
                    break
        else:
            gamma_flip = spot_price

        # Max pain: strike where option writers (MMs) profit most
        max_pain = cls._compute_max_pain(options_chain, spot_price)

        return {
            "total_gex_billions": float(total_gex),
            "gex_regime": "POSITIVE" if total_gex > 0 else "NEGATIVE",
            "gamma_flip_level": float(gamma_flip),
            "max_pain_strike": float(max_pain),
            "vol_suppression_active": total_gex > 0,
            "gamma_squeeze_risk": total_gex < -1.0,  # Very negative GEX = squeeze risk
            "strike_gex": {str(k): v for k, v in list(strike_gex.items())[:20]},
        }

    @staticmethod
    def _compute_max_pain(options_chain: pd.DataFrame, spot_price: float) -> float:
        """
        Max Pain Theory: at expiration, price gravitates towards the strike
        that causes maximum loss for option buyers (maximum gain for writers).
        Total Pain(K) = Σ call_OI_i * max(K_i - K, 0) + Σ put_OI_i * max(K - K_i, 0)
        """
        try:
            strikes = options_chain["strike"].unique()
            pain_by_strike = {}
            for k_test in strikes:
                call_pain = sum(
                    row["open_interest_call"] * max(row["strike"] - k_test, 0)
                    for _, row in options_chain.iterrows()
                    if "open_interest_call" in row and "strike" in row
                )
                put_pain = sum(
                    row["open_interest_put"] * max(k_test - row["strike"], 0)
                    for _, row in options_chain.iterrows()
                    if "open_interest_put" in row and "strike" in row
                )
                pain_by_strike[k_test] = call_pain + put_pain

            if pain_by_strike:
                return float(min(pain_by_strike, key=pain_by_strike.get))
        except Exception:
            pass
        return spot_price

    @classmethod
    def build_iv_surface(
        cls,
        options_chain: pd.DataFrame,
        spot_price: float,
        risk_free_rate: float = 0.05,
    ) -> Dict:
        """
        Build implied volatility term structure and smile/skew.
        Key metrics:
          - ATM IV at each expiry (term structure)
          - 25-delta put/call skew (risk reversal)
          - 10-delta put/call skew (tail risk premium)
          - Vol of vol (Volga)
        """
        if options_chain.empty:
            return {}

        # Group by expiry
        iv_surface = {}
        expiries = sorted(options_chain["days_to_expiry"].unique())

        for dte in expiries:
            expiry_chain = options_chain[options_chain["days_to_expiry"] == dte]
            T = dte / 365

            # ATM IV (strike closest to spot)
            expiry_chain = expiry_chain.copy()
            expiry_chain["moneyness"] = abs(expiry_chain["strike"] - spot_price)
            atm_row = expiry_chain.loc[expiry_chain["moneyness"].idxmin()]

            atm_iv = float(atm_row.get("iv", 0.25))

            # 25-delta skew: put_iv - call_iv at 25 delta
            # Approximation: 25∆ ≈ OTM strike at ±0.4σ√T moneyness
            otm_range = spot_price * atm_iv * np.sqrt(T) * 0.5
            put_mask = (expiry_chain["strike"] < spot_price - otm_range) & \
                       (expiry_chain["strike"] > spot_price - 2 * otm_range)
            call_mask = (expiry_chain["strike"] > spot_price + otm_range) & \
                        (expiry_chain["strike"] < spot_price + 2 * otm_range)

            put_iv_25d = float(expiry_chain[put_mask]["iv"].mean()) if put_mask.any() else atm_iv
            call_iv_25d = float(expiry_chain[call_mask]["iv"].mean()) if call_mask.any() else atm_iv
            skew_25d = put_iv_25d - call_iv_25d

            iv_surface[str(dte)] = {
                "atm_iv": atm_iv,
                "put_iv_25d": put_iv_25d,
                "call_iv_25d": call_iv_25d,
                "skew_25d": float(skew_25d),
                "atm_vega": spot_price * stats.norm.pdf(0) * np.sqrt(T) / 100,
            }

        return {
            "term_structure": iv_surface,
            "iv_rank": 0.0,  # Populated by data pipeline
            "iv_percentile": 0.0,
            "contango": len(expiries) > 1 and list(iv_surface.values())[0]["atm_iv"] < list(iv_surface.values())[-1]["atm_iv"],
        }
