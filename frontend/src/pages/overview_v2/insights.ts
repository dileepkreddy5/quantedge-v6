// ============================================================
// QuantEdge v6.0 — Overview V2 Interpretation Layer
// Pure TypeScript — no React, no DOM. Just data → strings.
// ============================================================

export interface Insight {
  label: string;
  headline: string;
  explanation: string;
  sentiment: 'positive' | 'neutral' | 'negative' | 'warning';
}

const pct = (v: number | null | undefined, digits = 1): string =>
  v === null || v === undefined || isNaN(v) ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(digits)}%`;

const num = (v: number | null | undefined, digits = 2): string =>
  v === null || v === undefined || isNaN(v) ? '—' : v.toFixed(digits);

// ── Composite Score ──────────────────────────────────────────
export function interpretCompositeScore(score: number, _signal?: string): Insight {
  const s = Math.round(score);
  if (s >= 75) return {
    label: 'STRONG SIGNAL',
    headline: `Composite score of ${s}/100 indicates favorable alignment across momentum, quality, and regime factors.`,
    explanation: 'A 0-100 signal score combining momentum, quality, accumulation, and trend factors, weighted by the current market regime. Higher values indicate stronger multi-factor alignment.',
    sentiment: 'positive',
  };
  if (s >= 60) return {
    label: 'CONSTRUCTIVE',
    headline: `Composite score of ${s}/100 reflects a positive bias from the underlying factors, with some areas of weakness.`,
    explanation: 'A 0-100 signal score combining momentum, quality, accumulation, and trend factors, weighted by the current market regime.',
    sentiment: 'positive',
  };
  if (s >= 40) return {
    label: 'MIXED',
    headline: `Composite score of ${s}/100 suggests conflicting signals across factors. No clear directional bias.`,
    explanation: 'A 0-100 signal score combining momentum, quality, accumulation, and trend factors, weighted by the current market regime.',
    sentiment: 'neutral',
  };
  if (s >= 25) return {
    label: 'CAUTIOUS',
    headline: `Composite score of ${s}/100 indicates weakness across multiple factor categories.`,
    explanation: 'A 0-100 signal score combining momentum, quality, accumulation, and trend factors. Low values indicate factor deterioration.',
    sentiment: 'warning',
  };
  return {
    label: 'WEAK SIGNAL',
    headline: `Composite score of ${s}/100 shows broad deterioration in momentum, quality, or regime positioning.`,
    explanation: 'A 0-100 signal score combining momentum, quality, accumulation, and trend factors. Very low values indicate broad weakness.',
    sentiment: 'negative',
  };
}

// ── Sharpe Ratio ─────────────────────────────────────────────
export function interpretSharpe(sharpe: number): Insight {
  if (sharpe >= 2.0) return {
    label: 'EXCEPTIONAL',
    headline: `Sharpe ratio of ${num(sharpe)} is in the top decile of equity strategies, comparable to specialized quant funds.`,
    explanation: 'Return per unit of volatility. The S&P 500 historically averages around 0.5-0.7. Above 1.0 is good; above 2.0 is rare and usually indicates either real skill or unsustainable conditions.',
    sentiment: 'positive',
  };
  if (sharpe >= 1.0) return {
    label: 'STRONG',
    headline: `Sharpe ratio of ${num(sharpe)} meaningfully exceeds the S&P 500 baseline (~0.6), indicating returns well-compensated for risk.`,
    explanation: 'Return per unit of volatility. The S&P 500 historically averages around 0.5-0.7. Above 1.0 indicates returns are strong relative to the volatility taken.',
    sentiment: 'positive',
  };
  if (sharpe >= 0.5) return {
    label: 'ADEQUATE',
    headline: `Sharpe ratio of ${num(sharpe)} is consistent with broad market performance.`,
    explanation: 'Return per unit of volatility. Near-market Sharpe suggests no meaningful outperformance after accounting for risk.',
    sentiment: 'neutral',
  };
  if (sharpe >= 0) return {
    label: 'SUBPAR',
    headline: `Sharpe ratio of ${num(sharpe)} falls below market baseline — volatility exceeds return compensation.`,
    explanation: 'Return per unit of volatility. Low Sharpe indicates the asset carries volatility not adequately rewarded by returns.',
    sentiment: 'warning',
  };
  return {
    label: 'NEGATIVE',
    headline: `Sharpe ratio of ${num(sharpe)} is negative — returns fail to exceed the risk-free rate.`,
    explanation: 'When Sharpe is negative, the asset has underperformed even cash (T-bills) after accounting for volatility.',
    sentiment: 'negative',
  };
}

// ── Beta ─────────────────────────────────────────────────────
export function interpretBeta(beta: number): Insight {
  if (beta > 1.3) return {
    label: 'HIGH-BETA',
    headline: `Beta of ${num(beta)} indicates approximately ${((beta - 1) * 100).toFixed(0)}% more market sensitivity than the S&P 500.`,
    explanation: 'Beta measures how the asset moves relative to the market. A beta of 1.0 moves with the market; above 1 amplifies market moves; below 1 dampens them.',
    sentiment: 'warning',
  };
  if (beta > 0.7) return {
    label: 'MARKET-LIKE',
    headline: `Beta of ${num(beta)} indicates market-proportional sensitivity.`,
    explanation: 'Beta measures how the asset moves relative to the market. Values near 1.0 suggest the asset will move roughly in line with the S&P 500.',
    sentiment: 'neutral',
  };
  if (beta > 0.3) return {
    label: 'DEFENSIVE',
    headline: `Beta of ${num(beta)} reflects reduced market sensitivity — typical of defensive or low-vol names.`,
    explanation: 'Beta measures how the asset moves relative to the market. Low beta suggests the asset is less volatile than the broader market.',
    sentiment: 'positive',
  };
  return {
    label: 'LOW CORRELATION',
    headline: `Beta of ${num(beta)} suggests minimal linear correlation to market moves.`,
    explanation: 'Very low beta indicates the asset moves largely independent of the broader market.',
    sentiment: 'neutral',
  };
}

// ── Regime ───────────────────────────────────────────────────
export function interpretRegime(regime: string, confidence?: number): Insight {
  const r = (regime || '').toUpperCase();
  const conf = confidence != null ? ` (${(confidence * 100).toFixed(0)}% confidence)` : '';

  if (r.includes('BULL_LOW_VOL') || r.includes('BULL_QUIET')) return {
    label: r.replace(/_/g, ' '),
    headline: `Classified in low-volatility bullish regime${conf}. Historically favorable for trend-following strategies.`,
    explanation: 'The HMM identifies the current return/volatility pattern. Low-vol bull regimes typically persist for months and favor momentum-exposed positions.',
    sentiment: 'positive',
  };
  if (r.includes('BULL')) return {
    label: r.replace(/_/g, ' '),
    headline: `Classified in volatile bullish regime${conf}. Positive drift but with significant realized risk.`,
    explanation: 'Bullish with elevated volatility. Returns tend to be positive on average but day-to-day swings can be large.',
    sentiment: 'positive',
  };
  if (r.includes('BEAR')) return {
    label: r.replace(/_/g, ' '),
    headline: `Classified in bearish regime${conf}. Downside risk elevated, trend-following strategies under pressure.`,
    explanation: 'Bearish regime — the model detects a negative-return environment. Defensive positioning historically outperforms in these conditions.',
    sentiment: 'negative',
  };
  if (r.includes('MEAN_REVERT') || r.includes('RANGE')) return {
    label: r.replace(/_/g, ' '),
    headline: `Classified in mean-reverting regime${conf}. Momentum strategies underperform; fade-the-move approaches work better historically.`,
    explanation: 'Mean-reverting regimes are range-bound. Stocks tend to revert to their recent average rather than trend.',
    sentiment: 'warning',
  };
  return {
    label: r.replace(/_/g, ' ') || 'UNKNOWN',
    headline: `Regime classification available${conf}.`,
    explanation: 'The HMM identifies the current market regime from hidden state probabilities over returns and volatility.',
    sentiment: 'neutral',
  };
}

// ── Max Drawdown ─────────────────────────────────────────────
export function interpretMaxDrawdown(dd: number): Insight {
  const abs = Math.abs(dd);
  if (abs < 0.10) return {
    label: 'SHALLOW',
    headline: `Max drawdown of ${pct(dd * 100)} reflects limited historical stress.`,
    explanation: 'Max drawdown is the largest peak-to-trough decline over the measurement period. Single-digit drawdowns are unusual for equities.',
    sentiment: 'positive',
  };
  if (abs < 0.25) return {
    label: 'TYPICAL',
    headline: `Max drawdown of ${pct(dd * 100)} is within normal bounds for equity exposure.`,
    explanation: 'Max drawdown measures the worst historical peak-to-trough decline. Most large-caps show 20-30% drawdowns over multi-year windows.',
    sentiment: 'neutral',
  };
  if (abs < 0.40) return {
    label: 'ELEVATED',
    headline: `Max drawdown of ${pct(dd * 100)} is above typical equity ranges. Position sizing should account for this historical stress.`,
    explanation: 'Drawdowns of 30%+ are typical of higher-volatility names or stress periods.',
    sentiment: 'warning',
  };
  return {
    label: 'SEVERE',
    headline: `Max drawdown of ${pct(dd * 100)} indicates significant historical stress.`,
    explanation: 'Very large drawdowns are either rare stress events or indicators of fundamentally risky assets.',
    sentiment: 'negative',
  };
}

// ── Annual Vol ───────────────────────────────────────────────
export function interpretVol(annualVol: number): Insight {
  const v = annualVol * 100;
  if (v < 15) return {
    label: 'LOW VOL',
    headline: `Annualized volatility of ${v.toFixed(1)}% is below the S&P 500 baseline (~16%).`,
    explanation: 'Annualized volatility measures the typical magnitude of return swings. The S&P 500 averages 15-18% annually.',
    sentiment: 'positive',
  };
  if (v < 25) return {
    label: 'MARKET VOL',
    headline: `Annualized volatility of ${v.toFixed(1)}% aligns with broad equity market norms.`,
    explanation: 'Annualized volatility measures the typical magnitude of return swings. Broad market indexes run 15-20%.',
    sentiment: 'neutral',
  };
  if (v < 40) return {
    label: 'ELEVATED VOL',
    headline: `Annualized volatility of ${v.toFixed(1)}% is meaningfully above market norms.`,
    explanation: 'Elevated volatility is typical of high-growth, thematic, or speculative names. Position sizing should be reduced proportionally.',
    sentiment: 'warning',
  };
  return {
    label: 'HIGH VOL',
    headline: `Annualized volatility of ${v.toFixed(1)}% is extreme. Vol-targeted sizing will reduce position significantly.`,
    explanation: 'Very high volatility is typical of early-stage growth, distressed, or speculative assets.',
    sentiment: 'negative',
  };
}

// ── Fama-French Alpha ────────────────────────────────────────
export function interpretFFAlpha(alpha: number, r2?: number): Insight {
  const annual = alpha * 100 * 252;
  const r2Note = r2 != null ? ` (R² ${(r2 * 100).toFixed(0)}%)` : '';
  if (Math.abs(annual) < 1) return {
    label: 'NO ALPHA',
    headline: `Fama-French regression shows negligible alpha after factor adjustment${r2Note}.`,
    explanation: 'Fama-French alpha is the residual return after stripping out market, size, value, profitability, and investment exposures. Near-zero alpha suggests the return is fully explained by systematic factors.',
    sentiment: 'neutral',
  };
  if (annual > 3) return {
    label: 'POSITIVE ALPHA',
    headline: `Fama-French alpha estimated at ${pct(annual)} annualized — return exceeds what factor exposure predicts${r2Note}.`,
    explanation: 'Positive Fama-French alpha means the asset has delivered more return than its factor exposures would predict. Could reflect skill, security-specific catalysts, or an unmodeled risk factor.',
    sentiment: 'positive',
  };
  if (annual < -3) return {
    label: 'NEGATIVE ALPHA',
    headline: `Fama-French alpha of ${pct(annual)} annualized — underperforming factor-implied returns${r2Note}.`,
    explanation: 'Negative Fama-French alpha means the asset has underperformed what its factor exposures would predict.',
    sentiment: 'warning',
  };
  return {
    label: 'MILD ALPHA',
    headline: `Fama-French alpha of ${pct(annual)} annualized${r2Note}.`,
    explanation: 'Mild alpha — the asset has slightly deviated from what factor exposures predict. Not statistically unusual.',
    sentiment: 'neutral',
  };
}

// ── Valuation ────────────────────────────────────────────────
export function interpretPE(pe: number | null | undefined): Insight {
  if (pe == null || isNaN(pe) || pe < 0) return {
    label: 'N/A',
    headline: 'P/E ratio unavailable or negative (loss-making).',
    explanation: 'A negative P/E indicates current losses. A missing P/E typically reflects a lack of earnings history.',
    sentiment: 'neutral',
  };
  if (pe < 15) return {
    label: 'VALUE',
    headline: `P/E of ${num(pe)}x is below the S&P 500's typical 18-22x range.`,
    explanation: 'Price-to-earnings ratio. Lower P/E historically correlates with the value factor. Be cautious — low P/E can also reflect deteriorating fundamentals.',
    sentiment: 'positive',
  };
  if (pe < 25) return {
    label: 'MARKET-LIKE',
    headline: `P/E of ${num(pe)}x aligns with broad market valuation norms.`,
    explanation: 'Price-to-earnings ratio. Values around 15-25x are typical for mature, profitable large-caps.',
    sentiment: 'neutral',
  };
  if (pe < 40) return {
    label: 'GROWTH PREMIUM',
    headline: `P/E of ${num(pe)}x reflects growth premium pricing.`,
    explanation: 'Elevated P/E ratios are typical of growth names with expected earnings acceleration.',
    sentiment: 'neutral',
  };
  return {
    label: 'RICH',
    headline: `P/E of ${num(pe)}x is significantly elevated — prices strong future growth.`,
    explanation: 'Very high P/E ratios price significant future growth. Small earnings disappointments can lead to large price corrections.',
    sentiment: 'warning',
  };
}

// ── ML Consensus ─────────────────────────────────────────────
export function interpretMLConsensus(predictions: any): Insight {
  if (!predictions) return {
    label: 'N/A',
    headline: 'ML ensemble predictions unavailable.',
    explanation: 'ML predictions aggregate forecasts across horizons from the ensemble model.',
    sentiment: 'neutral',
  };
  const horizons = ['pred_5d', 'pred_10d', 'pred_21d', 'pred_63d', 'pred_252d'];
  const values = horizons.map(h => predictions[h]).filter(v => v != null && !isNaN(v));
  if (values.length === 0) return {
    label: 'N/A',
    headline: 'ML predictions unavailable across all horizons.',
    explanation: 'ML predictions aggregate forecasts across 1W/2W/1M/3M/1Y horizons from the ensemble model.',
    sentiment: 'neutral',
  };
  const avg = values.reduce((s, v) => s + v, 0) / values.length;
  const positive = values.filter(v => v > 0).length;

  if (avg > 5 && positive === values.length) return {
    label: 'UNANIMOUS UP',
    headline: `ML ensemble projects positive returns across all ${values.length} horizons (avg ${pct(avg)}).`,
    explanation: 'Ensemble consensus across 1W to 1Y horizons. Unanimous directional agreement strengthens signal but does not guarantee outcomes.',
    sentiment: 'positive',
  };
  if (avg > 0 && positive >= values.length - 1) return {
    label: 'CONSTRUCTIVE',
    headline: `ML ensemble leans positive across horizons (avg ${pct(avg)}, ${positive}/${values.length} positive).`,
    explanation: 'Ensemble consensus — most horizons forecast positive returns, with some divergence.',
    sentiment: 'positive',
  };
  if (positive < values.length / 3) return {
    label: 'NEGATIVE BIAS',
    headline: `ML ensemble leans negative (avg ${pct(avg)}, only ${positive}/${values.length} positive).`,
    explanation: 'Most forecast horizons project negative returns. Model may be detecting deteriorating momentum or regime shift.',
    sentiment: 'warning',
  };
  return {
    label: 'MIXED SIGNALS',
    headline: `ML ensemble shows no clear directional consensus (avg ${pct(avg)}, ${positive}/${values.length} positive).`,
    explanation: 'Mixed forecasts across horizons suggest conflicting signals. No strong conviction from the model.',
    sentiment: 'neutral',
  };
}

// ── THESIS BUILDER ───────────────────────────────────────────
export interface ThesisParagraphs {
  thesis: string;
  risk: string;
  forward: string;
}

export function buildThesis(data: any, ticker: string): ThesisParagraphs {
  const score = data.overall_score ?? 50;
  const signal = data.overall_signal ?? 'NEUTRAL';
  const regime = data.current_regime ?? '';
  const regimeConf = data.regime?.confidence ?? null;
  const sharpe = data.sharpe_ratio ?? 0;
  const annualVol = data.annual_vol ?? 0.2;
  const maxDD = data.max_drawdown ?? -0.2;
  const mlPreds = data.ml_predictions?.ensemble ?? {};
  const mc = data.monte_carlo ?? {};
  const positionPct = data.portfolio_construction?.recommended_position_pct ?? null;

  const compInsight = interpretCompositeScore(score, signal);
  const regimeInsight = interpretRegime(regime, regimeConf);
  const shortThesis = `${ticker} receives a composite signal of ${Math.round(score)}/100. ${compInsight.headline.replace(/^Composite score [^.]+\. /, '')} ${regimeInsight.headline}`;

  const volInsight = interpretVol(annualVol);
  const ddInsight = interpretMaxDrawdown(maxDD);
  const sharpeInsight = interpretSharpe(sharpe);
  const posSize = positionPct != null
    ? ` The volatility-targeted sizer recommends ${(positionPct * 100).toFixed(0)}% of capital allocation given realized volatility.`
    : '';
  const shortRisk = `${volInsight.headline} ${ddInsight.headline} ${sharpeInsight.headline}${posSize}`;

  const mlInsight = interpretMLConsensus(mlPreds);
  const mcStr = (mc.bull_case_price && mc.base_case_price && mc.bear_case_price)
    ? ` Monte Carlo simulation places the base case at $${mc.base_case_price.toFixed(2)}, with bull/bear bounds of $${mc.bull_case_price.toFixed(2)} and $${mc.bear_case_price.toFixed(2)}.`
    : '';
  const shortForward = `${mlInsight.headline}${mcStr} These projections are historical-simulation-derived and have not been validated in live forward-trading.`;

  return {
    thesis: shortThesis,
    risk: shortRisk,
    forward: shortForward,
  };
}
