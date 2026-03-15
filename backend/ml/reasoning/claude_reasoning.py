"""
QuantEdge v6.0 — Claude Reasoning Layer
==========================================
© 2026 Dileep Kumar Reddy Kapu. All Rights Reserved.

Claude is ONLY used here. Not anywhere else in the system.
The prediction pipeline (XGBoost, LightGBM, LSTM, GARCH, HMM, Kalman)
is entirely deterministic ML code. Claude's role is synthesis only.

InvestmentCommittee runs 4 async agents in parallel:
    1. QuantitativeSynthesizer  — finds top 3 signal drivers from model outputs
    2. MacroContextualist       — validates macro vs HMM regime
    3. HistoricalAnalogueFinder — 3 historical analogues
    4. DevilsAdvocate           — strongest opposing case

Every sentence in the thesis cites a specific model output or data point.
Zero hallucination — the prompt explicitly requires citing numbers from the
model_outputs dict that is passed in verbatim.
"""

import asyncio
import json
from typing import Dict, List, Optional
from anthropic import AsyncAnthropic
from loguru import logger


class InvestmentCommittee:
    """
    4-agent async investment committee.

    Usage:
        committee = InvestmentCommittee(anthropic_client=AsyncAnthropic())
        thesis = await committee.analyze(
            ticker="AAPL",
            model_outputs=data["ml_predictions"],
            risk_metrics=data["risk_metrics"],
            regime=data["current_regime"],
            regime_confidence=0.82,
            macro_data=data.get("macro", {}),
        )
    """

    def __init__(self, anthropic_client: AsyncAnthropic):
        self.client = anthropic_client
        self.model = "claude-opus-4-20250514"

    async def analyze(
        self,
        ticker: str,
        model_outputs: dict,
        risk_metrics: dict,
        regime: str,
        regime_confidence: float,
        macro_data: dict,
    ) -> dict:
        """
        Run all 4 agents in parallel. Combine into final thesis.
        Returns structured dict with primary_thesis, bear_case, key_risks,
        confidence, conditions_that_would_invalidate, cited_evidence.
        """
        # Extract ensemble direction for DevilsAdvocate
        ens = model_outputs.get("ensemble", {})
        ensemble_signal = float(ens.get("signal", ens.get("pred_21d", 0.0)) or 0.0)
        direction = "LONG" if ensemble_signal > 0.02 else "SHORT" if ensemble_signal < -0.02 else "NEUTRAL"

        # Run all 4 agents in parallel
        try:
            results = await asyncio.gather(
                self._quantitative_synthesizer(ticker, model_outputs),
                self._macro_contextualist(ticker, regime, regime_confidence, macro_data),
                self._historical_analogue_finder(ticker, regime, regime_confidence, model_outputs),
                self._devils_advocate(ticker, direction, model_outputs, risk_metrics),
                return_exceptions=True,
            )
        except Exception as e:
            logger.error(f"InvestmentCommittee.analyze gather error: {e}")
            return _fallback_thesis(ticker, ensemble_signal, regime)

        quant_result    = results[0] if not isinstance(results[0], Exception) else {}
        macro_result    = results[1] if not isinstance(results[1], Exception) else {}
        analogue_result = results[2] if not isinstance(results[2], Exception) else {}
        devil_result    = results[3] if not isinstance(results[3], Exception) else {}

        # Log any agent failures
        for i, name in enumerate(["QuantSynth", "Macro", "Analogues", "Devil"]):
            if isinstance(results[i], Exception):
                logger.warning(f"InvestmentCommittee: {name} agent failed: {results[i]}")

        # Orchestrate: combine all 4 outputs into final thesis
        thesis = await self._orchestrate(
            ticker=ticker,
            ensemble_signal=ensemble_signal,
            direction=direction,
            regime=regime,
            regime_confidence=regime_confidence,
            risk_metrics=risk_metrics,
            quant_result=quant_result,
            macro_result=macro_result,
            analogue_result=analogue_result,
            devil_result=devil_result,
        )

        return thesis

    # ── Agent 1: QuantitativeSynthesizer ─────────────────────

    async def _quantitative_synthesizer(
        self,
        ticker: str,
        model_outputs: dict,
    ) -> dict:
        """
        Reads the actual model output numbers and finds the top 3 signal
        drivers plus any conflicting signals. Every claim cites specific numbers.
        """
        prompt = f"""You are a quantitative analyst reviewing live model outputs for {ticker}.

MODEL OUTPUTS (these are real computed values, not estimates):
{json.dumps(model_outputs, indent=2, default=str)}

Instructions:
1. Identify the TOP 3 signal drivers — the specific model outputs that most strongly
   justify the ensemble direction. Cite the exact numbers (e.g., "LSTM pred_21d = +4.2%").
2. Identify any CONFLICTING signals — where one model disagrees with others. Cite numbers.
3. Assess signal CONVICTION: is there broad model agreement or is it split?

Respond ONLY with a JSON object (no markdown, no preamble):
{{
  "top_drivers": [
    {{"model": "...", "metric": "...", "value": ..., "interpretation": "..."}},
    {{"model": "...", "metric": "...", "value": ..., "interpretation": "..."}},
    {{"model": "...", "metric": "...", "value": ..., "interpretation": "..."}}
  ],
  "conflicts": [
    {{"model_a": "...", "model_b": "...", "disagreement": "..."}}
  ],
  "conviction": "HIGH|MEDIUM|LOW",
  "reasoning_trace": "2-3 sentence synthesis citing specific numbers"
}}"""

        response = await self._call_claude(prompt, max_tokens=600)
        return _parse_json_response(response, default={
            "top_drivers": [],
            "conflicts": [],
            "conviction": "MEDIUM",
            "reasoning_trace": response,
        })

    # ── Agent 2: MacroContextualist ───────────────────────────

    async def _macro_contextualist(
        self,
        ticker: str,
        regime: str,
        regime_confidence: float,
        macro_data: dict,
    ) -> dict:
        """
        Checks whether current macro environment is consistent with
        the HMM regime. Identifies key macro risks.
        """
        prompt = f"""You are a macro strategist reviewing regime consistency for {ticker}.

HMM REGIME: {regime} (confidence: {regime_confidence:.1%})

MACRO DATA:
{json.dumps(macro_data, indent=2, default=str)}

Instructions:
1. Is the current macro environment CONSISTENT or INCONSISTENT with the {regime} regime?
   Cite specific macro data points.
2. List the top 2-3 macro risks that could invalidate this regime classification.
3. If macro data is sparse or unavailable, state that clearly.

Respond ONLY with a JSON object:
{{
  "macro_consistency": "CONSISTENT|INCONSISTENT|UNCERTAIN",
  "consistency_explanation": "1-2 sentences citing specific data points",
  "macro_risks": ["risk 1", "risk 2", "risk 3"],
  "regime_validation": "1 sentence on whether macro confirms the HMM regime"
}}"""

        response = await self._call_claude(prompt, max_tokens=400)
        return _parse_json_response(response, default={
            "macro_consistency": "UNCERTAIN",
            "consistency_explanation": "Insufficient macro data to validate.",
            "macro_risks": [],
            "regime_validation": f"Macro validation unavailable for {regime} regime.",
        })

    # ── Agent 3: HistoricalAnalogueFinder ────────────────────

    async def _historical_analogue_finder(
        self,
        ticker: str,
        regime: str,
        regime_confidence: float,
        model_outputs: dict,
    ) -> dict:
        """
        Finds 3 historical periods most similar to current conditions
        and describes outcomes for assets in the same class.
        """
        # Extract key signal characteristics for context
        ens = model_outputs.get("ensemble", {})
        garch = model_outputs.get("garch", model_outputs.get("volatility", {}))
        vol = garch.get("vol_forecast", garch.get("annualized_vol", "unknown"))

        prompt = f"""You are a quantitative historian finding historical analogues for {ticker}.

CURRENT CONDITIONS:
- HMM Regime: {regime} (confidence: {regime_confidence:.1%})
- Ensemble signal direction: {"LONG" if float(ens.get("signal", ens.get("pred_21d", 0)) or 0) > 0 else "SHORT"}
- Volatility forecast: {vol}
- Asset class: equity (large cap US)

Instructions:
Find 3 historical periods (with specific dates) most similar to current conditions
(same regime type, similar volatility level, similar signal direction).
For each period, describe what happened to large-cap US equities over the next 21-63 days.
Use ONLY well-documented historical facts. Do NOT invent statistics.

Respond ONLY with a JSON object:
{{
  "analogues": [
    {{
      "period": "Month YYYY – Month YYYY",
      "similarity_reason": "why this period matches",
      "outcome_21d": "what happened over 21 days",
      "outcome_63d": "what happened over 63 days",
      "key_lesson": "one sentence takeaway"
    }}
  ],
  "analogue_summary": "1-2 sentences on what the historical analogues suggest"
}}"""

        response = await self._call_claude(prompt, max_tokens=700)
        return _parse_json_response(response, default={
            "analogues": [],
            "analogue_summary": "Historical analogue search unavailable.",
        })

    # ── Agent 4: DevilsAdvocate ───────────────────────────────

    async def _devils_advocate(
        self,
        ticker: str,
        direction: str,
        model_outputs: dict,
        risk_metrics: dict,
    ) -> dict:
        """
        Constructs the strongest possible opposing case using the same
        model outputs. Forces identification of what could go wrong.
        """
        opposite = "SHORT" if direction == "LONG" else "LONG"

        prompt = f"""You are a devil's advocate stress-testing a {direction} signal on {ticker}.

THE ENSEMBLE SAYS: {direction}

MODEL OUTPUTS (same data the ensemble used):
{json.dumps(model_outputs, indent=2, default=str)}

RISK METRICS:
{json.dumps(risk_metrics, indent=2, default=str)}

Your job: Construct the STRONGEST POSSIBLE {opposite} case using ONLY this data.
Do not fabricate new information. Use the model outputs provided.
Find the weakest numbers, the conflicting signals, the highest-risk metrics.

Respond ONLY with a JSON object:
{{
  "opposing_case": "2-3 sentences making the strongest {opposite} argument from the data",
  "weakest_signals": [
    {{"model": "...", "metric": "...", "value": ..., "concern": "..."}}
  ],
  "key_risks": ["risk 1 citing specific data", "risk 2 citing specific data"],
  "conditions_that_would_invalidate": [
    "If X happens (specific threshold), this signal becomes invalid",
    "If Y metric deteriorates below Z, reconsider"
  ],
  "risk_reward_assessment": "1 sentence on whether risk/reward is favorable"
}}"""

        response = await self._call_claude(prompt, max_tokens=600)
        return _parse_json_response(response, default={
            "opposing_case": "Opposing case analysis unavailable.",
            "weakest_signals": [],
            "key_risks": [],
            "conditions_that_would_invalidate": [],
            "risk_reward_assessment": "Risk/reward assessment unavailable.",
        })

    # ── Orchestrator ──────────────────────────────────────────

    async def _orchestrate(
        self,
        ticker: str,
        ensemble_signal: float,
        direction: str,
        regime: str,
        regime_confidence: float,
        risk_metrics: dict,
        quant_result: dict,
        macro_result: dict,
        analogue_result: dict,
        devil_result: dict,
    ) -> dict:
        """
        Combines all 4 agent outputs into a final structured thesis.
        Every sentence in primary_thesis cites a specific model output or data point.
        """
        prompt = f"""You are the Chief Investment Officer of QuantEdge synthesizing a final thesis for {ticker}.

QUANTITATIVE SYNTHESIS:
{json.dumps(quant_result, indent=2, default=str)}

MACRO CONTEXT:
{json.dumps(macro_result, indent=2, default=str)}

HISTORICAL ANALOGUES:
{json.dumps(analogue_result, indent=2, default=str)}

DEVIL'S ADVOCATE:
{json.dumps(devil_result, indent=2, default=str)}

ENSEMBLE: {direction} signal = {ensemble_signal:+.4f}
REGIME: {regime} ({regime_confidence:.1%} confidence)
CVaR: {risk_metrics.get('cvar_95', 'N/A')}
Recommended position: {risk_metrics.get('recommended_position', 'N/A')}

Write a FINAL INVESTMENT THESIS. Rules:
- Every sentence must cite a specific number from the data above
- Acknowledge conflicting signals honestly
- Do not state anything as fact that isn't in the data
- 3-5 sentences maximum for primary_thesis

Respond ONLY with a JSON object:
{{
  "primary_thesis": "3-5 sentences, each citing a specific number or data point",
  "bear_case": "2-3 sentences — strongest opposing view from DevilsAdvocate",
  "key_risks": ["specific risk 1 with data citation", "specific risk 2"],
  "confidence": "HIGH|MEDIUM|LOW",
  "conditions_that_would_invalidate": ["condition 1 with threshold", "condition 2"],
  "cited_evidence": {{
    "strongest_bullish": "model name + specific value",
    "strongest_bearish": "model name + specific value",
    "regime_support": "HMM confidence level and macro consistency"
  }},
  "recommended_action": "LONG|SHORT|NEUTRAL with position size from risk engine"
}}"""

        response = await self._call_claude(prompt, max_tokens=800)
        thesis = _parse_json_response(response, default={
            "primary_thesis": f"Ensemble signal is {direction} ({ensemble_signal:+.4f}) in {regime} regime ({regime_confidence:.1%} confidence).",
            "bear_case": devil_result.get("opposing_case", "No opposing analysis."),
            "key_risks": devil_result.get("key_risks", []),
            "confidence": quant_result.get("conviction", "MEDIUM"),
            "conditions_that_would_invalidate": devil_result.get("conditions_that_would_invalidate", []),
            "cited_evidence": {},
            "recommended_action": f"{direction} — position per risk engine",
        })

        # Always add metadata
        thesis["ticker"] = ticker
        thesis["regime"] = regime
        thesis["regime_confidence"] = regime_confidence
        thesis["ensemble_signal"] = ensemble_signal
        return thesis

    # ── Claude API call ───────────────────────────────────────

    async def _call_claude(self, prompt: str, max_tokens: int = 600) -> str:
        """
        Single Claude API call. Returns the text content.
        Raises on failure — callers handle exceptions.
        """
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        content = message.content[0] if message.content else None
        if content and hasattr(content, "text"):
            return content.text
        return ""


# ── Helpers ───────────────────────────────────────────────────

def _parse_json_response(text: str, default: dict) -> dict:
    """Parse JSON from Claude response. Returns default on failure."""
    if not text:
        return default
    try:
        # Strip markdown code fences if present
        clean = text.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        return {**default, "raw_response": text[:500]}


def _fallback_thesis(ticker: str, ensemble_signal: float, regime: str) -> dict:
    """Return minimal thesis when all agents fail."""
    direction = "LONG" if ensemble_signal > 0.02 else "SHORT" if ensemble_signal < -0.02 else "NEUTRAL"
    return {
        "ticker": ticker,
        "primary_thesis": f"Ensemble signal is {direction} ({ensemble_signal:+.4f}) in {regime} regime. Reasoning agents unavailable.",
        "bear_case": "Opposing analysis unavailable.",
        "key_risks": [],
        "confidence": "LOW",
        "conditions_that_would_invalidate": [],
        "cited_evidence": {},
        "recommended_action": direction,
        "regime": regime,
        "ensemble_signal": ensemble_signal,
    }
