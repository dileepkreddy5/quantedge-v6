# QuantEdge AI Agent — V1 Spec (approved 2026-09-27)

Role: explanation + orchestration only. Downstream of the engines; never a
signal source, never in a prediction path.

## Launch
- Owner-only for 7 days; "unlimited" = no message cap, still under the spend cap.
- Hard spend cap: $5/day (Redis counter; prices from current model pricing in
  config, not memory). At the cap: "Daily AI agent limit reached. Try again
  tomorrow." — no fallback call.
- Max 5 tool calls per message; bounded context (last 10 turns, client-side).

## Modes (labeled on every answer)
- QUANTEDGE: ticker named, or today/current/market/board/multibagger/rebound/
  ascent/pattern/filing/intel/signal/forecast/regime/risk, or "use QuantEdge".
  Every factual claim must come from a tool result of THIS turn. No external
  or remembered facts (e.g. "why did NVDA rise today" with no causal data ->
  "QuantEdge's data contains no verified explanation").
- GENERAL: concepts, math, ML, finance education. Model knowledge, labeled
  "GENERAL ANSWER — not from QuantEdge data". May not state any current
  price/return/company figure.
- MIXED questions answer in both labeled sections.

## Tools (internal calls to existing endpoints)
analyze_ticker, daily_brief, boards, pattern_analogs, pattern_evolution,
company_intel, system_status. Preferred order for company questions:
analyze -> intel -> patterns only if relevant -> system_status when
freshness matters. Don't call tools just because they exist.

## Rules
1. Numbers: automated post-check — every figure must appear in a tool
   payload; failures render a visible warning.
2. Qualitative claims: constrained by prompt + adversarial tests.
3. Missing/stale data stated plainly; stale CI != "no developments".
4. Forecast validation status always preserved ("not validated").
5. No recommendations, no web, no international markets.
6. Unknown ticker = analyze tool returned error/no data -> refuse, stop.

## Audit (per request)
request_id, timestamp, tools, endpoints, tickers, tool_result_hash, tokens.
Human-readable: "SOURCE: QuantEdge · analyze(NVDA), intel(NVDA)".

## Build order
backend /agent/chat -> provenance check -> Redis limits + spend cap ->
frontend panel -> 20+ adversarial tests (normal, CI, general, mixed,
unsupported fact, why-today trap, buy trap, jailbreak, unsupported source,
staleness, unknown company, tempting-number, tool-limit).

## V2 (not now)
Claim-level provenance: each claim mapped to a tool-result path.
