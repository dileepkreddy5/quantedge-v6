// ============================================================
// QuantEdge v6.0 — Overview V2
// Bloomberg-style institutional layout with layered insights
// ============================================================
import React, { useState } from 'react';
import {
  Insight,
  interpretCompositeScore,
  interpretSharpe,
  interpretBeta,
  interpretRegime,
  interpretMaxDrawdown,
  interpretVol,
  interpretFFAlpha,
  interpretPE,
  interpretMLConsensus,
  buildThesis,
} from './insights';
import PriceChart from '../../components/charts/PriceChart';
import { ScenarioPanel } from '../../components/ui';

// ── Color tokens (match existing dashboard) ──────────────────
const COLORS = {
  bg:        '#0a0505',
  panel:     '#1a0f0a',
  panelAlt:  '#0f0805',
  border:    '#3a2920',
  borderLt:  '#2a1f18',
  text:      '#d4c4b0',
  textDim:   '#9d8b7a',
  textFaint: '#5a4838',
  amber:     '#daa520',
  amberSoft: '#b8891a',
  cyan:      '#5bcfd4',
  green:     '#22c55e',
  red:       '#ef4444',
  yellow:    '#f59e0b',
};

const sentimentColor = (s: Insight['sentiment']) => ({
  positive: COLORS.green,
  neutral:  COLORS.amber,
  warning:  COLORS.yellow,
  negative: COLORS.red,
}[s]);

const fontMono = "'Fira Code', 'JetBrains Mono', monospace";
const fontSans = "'Inter', system-ui, sans-serif";

// ── Info tooltip component ───────────────────────────────────
function InfoTip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <span
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onClick={() => setOpen(v => !v)}
      style={{
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        width: 14, height: 14, borderRadius: '50%',
        border: `1px solid ${COLORS.textFaint}`,
        color: COLORS.textFaint, fontSize: 9, fontFamily: fontMono,
        cursor: 'help', marginLeft: 6, userSelect: 'none',
        position: 'relative',
      }}
    >
      i
      {open && (
        <span style={{
          position: 'absolute', bottom: 'calc(100% + 6px)', right: 0,
          background: '#000', border: `1px solid ${COLORS.border}`,
          padding: '10px 12px', borderRadius: 4, width: 280,
          fontSize: 11, fontFamily: fontSans, color: COLORS.text,
          lineHeight: 1.5, zIndex: 10, whiteSpace: 'normal',
          textAlign: 'left', fontWeight: 400, letterSpacing: 0,
          boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
        }}>
          {text}
        </span>
      )}
    </span>
  );
}

// ── Section header ───────────────────────────────────────────
function SectionHeader({ label }: { label: string }) {
  return (
    <div style={{
      fontFamily: fontMono,
      fontSize: 9,
      letterSpacing: 3,
      color: COLORS.textFaint,
      marginBottom: 10,
      textTransform: 'uppercase',
      borderBottom: `1px solid ${COLORS.borderLt}`,
      paddingBottom: 6,
    }}>
      {label}
    </div>
  );
}

// ── Metric tile ──────────────────────────────────────────────
function MetricTile({ insight, valueOverride }: { insight: Insight; valueOverride?: string }) {
  const color = sentimentColor(insight.sentiment);
  return (
    <div style={{
      background: COLORS.panelAlt,
      border: `1px solid ${COLORS.borderLt}`,
      borderRadius: 4,
      padding: '14px 16px',
      display: 'flex', flexDirection: 'column', gap: 6,
      minHeight: 120,
    }}>
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
      }}>
        <span style={{
          fontFamily: fontMono, fontSize: 9, letterSpacing: 2,
          color: color, textTransform: 'uppercase',
        }}>
          {insight.label}
        </span>
        <InfoTip text={insight.explanation} />
      </div>
      {valueOverride && (
        <div style={{
          fontFamily: fontMono, fontSize: 20, fontWeight: 600, color: color,
          letterSpacing: 0.5, marginTop: 2,
        }}>
          {valueOverride}
        </div>
      )}
      <div style={{
        fontFamily: fontSans, fontSize: 11, color: COLORS.textDim,
        lineHeight: 1.5, marginTop: valueOverride ? 2 : 4,
      }}>
        {insight.headline}
      </div>
    </div>
  );
}

// ── Thesis paragraph block ───────────────────────────────────
function ThesisBlock({ label, text, accent }: { label: string; text: string; accent: string }) {
  return (
    <div style={{
      background: COLORS.panelAlt,
      borderLeft: `2px solid ${accent}`,
      padding: '14px 18px',
      marginBottom: 10,
    }}>
      <div style={{
        fontFamily: fontMono, fontSize: 9, letterSpacing: 3,
        color: accent, marginBottom: 8,
      }}>
        {label}
      </div>
      <div style={{
        fontFamily: fontSans, fontSize: 13, color: COLORS.text,
        lineHeight: 1.7, letterSpacing: 0.1,
      }}>
        {text}
      </div>
    </div>
  );
}

// ── Fama-French decomposition ────────────────────────────────
function FamaFrenchTable({ data }: { data: any }) {
  const rows = [
    { factor: 'MKT (Market)',     value: data.ff_mkt_beta, neutral: 1.0,
      desc: 'Exposure to broad market moves. 1.0 = moves with S&P; 0 = uncorrelated.' },
    { factor: 'SMB (Size)',       value: data.ff_smb,      neutral: 0.0,
      desc: 'Small-minus-big. Positive = small-cap tilt; negative = large-cap tilt.' },
    { factor: 'HML (Value)',      value: data.ff_hml,      neutral: 0.0,
      desc: 'High-minus-low book/market. Positive = value tilt; negative = growth tilt.' },
    { factor: 'RMW (Profit)',     value: data.ff_rmw,      neutral: 0.0,
      desc: 'Robust-minus-weak profitability. Positive = profitable firms; negative = weak.' },
    { factor: 'CMA (Investment)', value: data.ff_cma,      neutral: 0.0,
      desc: 'Conservative-minus-aggressive investment. Positive = low-capex; negative = high-capex.' },
    { factor: 'WML (Momentum)',   value: data.ff_wml,      neutral: 0.0,
      desc: 'Winners-minus-losers. Positive = riding momentum; negative = contrarian.' },
  ];

  const alphaInsight = data.ff_alpha != null
    ? interpretFFAlpha(data.ff_alpha, data.ff_r_squared)
    : null;

  return (
    <div>
      <SectionHeader label="FAMA–FRENCH 6-FACTOR DECOMPOSITION" />
      {alphaInsight && (
        <div style={{
          fontFamily: fontSans, fontSize: 12, color: COLORS.textDim,
          marginBottom: 12, lineHeight: 1.6,
        }}>
          {alphaInsight.headline}
        </div>
      )}
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8,
      }}>
        {rows.map(row => {
          const v = row.value;
          if (v == null || isNaN(v)) return null;
          const diff = v - row.neutral;
          const color = Math.abs(diff) < 0.1 ? COLORS.textDim
            : diff > 0 ? COLORS.green : COLORS.red;
          return (
            <div key={row.factor} style={{
              background: COLORS.panelAlt,
              border: `1px solid ${COLORS.borderLt}`,
              borderRadius: 4,
              padding: '10px 12px',
              display: 'flex', flexDirection: 'column', gap: 4,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{
                  fontFamily: fontMono, fontSize: 9, letterSpacing: 1.5,
                  color: COLORS.textFaint,
                }}>
                  {row.factor}
                </span>
                <InfoTip text={row.desc} />
              </div>
              <div style={{
                fontFamily: fontMono, fontSize: 15, fontWeight: 600, color: color,
              }}>
                {v >= 0 ? '+' : ''}{v.toFixed(3)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Main Overview V2 component ───────────────────────────────
export default function OverviewV2({
  data, ticker,
}: { data: any; ticker: string; onAnalyze?: (t: string) => void }) {
  if (!data) {
    return (
      <div style={{ padding: 40, textAlign: 'center', color: COLORS.textDim }}>
        No analysis data. Enter a ticker and click Analyze.
      </div>
    );
  }

  const thesis = buildThesis(data, ticker);
  const score = data.overall_score ?? 50;
  const scoreInsight = interpretCompositeScore(score, data.overall_signal);
  const scoreColor = sentimentColor(scoreInsight.sentiment);

  const price = data.price ?? 0;
  const changePct = data.change_pct ?? 0;
  const changeDollar = data.change ?? 0;
  const priceColor = changePct >= 0 ? COLORS.green : COLORS.red;

  // Build key metric insights
  const sharpeInsight = interpretSharpe(data.sharpe_ratio ?? 0);
  const betaInsight   = interpretBeta(data.beta ?? 1);
  const regimeInsight = interpretRegime(data.current_regime ?? '', data.regime?.confidence);
  const volInsight    = interpretVol(data.annual_vol ?? 0.2);
  const ddInsight     = interpretMaxDrawdown(data.max_drawdown ?? -0.2);
  const peInsight     = interpretPE(data.pe_ratio);
  const mlInsight     = interpretMLConsensus(data.ml_predictions?.ensemble);

  // Position sizing (from portfolio_construction)
  const positionPct = data.portfolio_construction?.recommended_position_pct ?? null;
  const positionInsight: Insight = positionPct != null ? {
    label: positionPct >= 0.4 ? 'FULL SIZE' : positionPct >= 0.2 ? 'MODERATE' : 'REDUCED',
    headline: `Vol-targeted sizer recommends ${(positionPct * 100).toFixed(0)}% of capital allocation.`,
    explanation: 'Position size derived from realized volatility. Higher vol assets are sized smaller to target a consistent portfolio-level volatility.',
    sentiment: 'neutral',
  } : {
    label: 'N/A',
    headline: 'Position sizing unavailable for this ticker.',
    explanation: 'Requires sufficient volatility history to compute.',
    sentiment: 'neutral',
  };

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', gap: 16,
      fontFamily: fontSans, color: COLORS.text,
    }}>

      {/* ── HERO ────────────────────────────────────────────── */}
      <div style={{
        background: COLORS.panel,
        border: `1px solid ${COLORS.border}`,
        borderRadius: 4,
        padding: '20px 24px',
        display: 'grid',
        gridTemplateColumns: '1fr auto auto',
        gap: 32, alignItems: 'center',
      }}>
        <div>
          <div style={{
            fontFamily: fontMono, fontSize: 9, letterSpacing: 3,
            color: COLORS.textFaint, marginBottom: 4,
          }}>
            {data.exchange ?? ''} · {data.sector ?? ''} · {data.industry ?? ''}
          </div>
          <div style={{
            fontFamily: fontMono, fontSize: 26, fontWeight: 700,
            color: COLORS.amber, letterSpacing: 2,
          }}>
            {ticker}
          </div>
          <div style={{
            fontFamily: fontSans, fontSize: 13, color: COLORS.textDim, marginTop: 2,
          }}>
            {data.name ?? ''}
          </div>
        </div>

        <div style={{ textAlign: 'right' }}>
          <div style={{
            fontFamily: fontMono, fontSize: 28, fontWeight: 700,
            color: COLORS.text,
          }}>
            ${price.toFixed(2)}
          </div>
          <div style={{
            fontFamily: fontMono, fontSize: 13, color: priceColor, marginTop: 2,
          }}>
            {changeDollar >= 0 ? '+' : ''}{changeDollar.toFixed(2)} ({changePct >= 0 ? '+' : ''}{changePct.toFixed(2)}%)
          </div>
        </div>

        <div style={{
          background: COLORS.panelAlt,
          border: `2px solid ${scoreColor}`,
          borderRadius: 4, padding: '14px 20px',
          textAlign: 'center', minWidth: 160,
        }}>
          <div style={{
            fontFamily: fontMono, fontSize: 8, letterSpacing: 3,
            color: COLORS.textFaint, marginBottom: 4,
          }}>
            COMPOSITE SIGNAL
          </div>
          <div style={{
            fontFamily: fontMono, fontSize: 32, fontWeight: 700,
            color: scoreColor, lineHeight: 1,
          }}>
            {Math.round(score)}
          </div>
          <div style={{
            fontFamily: fontMono, fontSize: 9, letterSpacing: 2,
            color: scoreColor, marginTop: 4,
          }}>
            {scoreInsight.label}
          </div>
        </div>
      </div>

      {/* ── THESIS PARAGRAPHS ───────────────────────────────── */}
      <div>
        <SectionHeader label="ANALYSIS NARRATIVE" />
        <ThesisBlock label="THESIS"        text={thesis.thesis}  accent={COLORS.amber} />
        <ThesisBlock label="RISK PROFILE"  text={thesis.risk}    accent={COLORS.cyan} />
        <ThesisBlock label="FORWARD VIEW"  text={thesis.forward} accent={COLORS.textDim} />
      </div>

      {/* ── KEY METRICS GRID ────────────────────────────────── */}
      <div>
        <SectionHeader label="KEY METRICS" />
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: 10,
        }}>
          <MetricTile insight={sharpeInsight} valueOverride={(data.sharpe_ratio ?? 0).toFixed(2)} />
          <MetricTile insight={betaInsight}   valueOverride={(data.beta ?? 1).toFixed(2)} />
          <MetricTile insight={regimeInsight} />
          <MetricTile insight={volInsight}    valueOverride={((data.annual_vol ?? 0) * 100).toFixed(1) + '%'} />
          <MetricTile insight={ddInsight}     valueOverride={((data.max_drawdown ?? 0) * 100).toFixed(1) + '%'} />
          <MetricTile insight={peInsight}     valueOverride={data.pe_ratio ? data.pe_ratio.toFixed(1) + 'x' : '—'} />
          <MetricTile insight={mlInsight} />
          <MetricTile insight={positionInsight} valueOverride={positionPct != null ? (positionPct * 100).toFixed(0) + '%' : '—'} />
        </div>
      </div>

      {/* ── FAMA-FRENCH ─────────────────────────────────────── */}
      <FamaFrenchTable data={data} />

      {/* ── PRICE CHART + SCENARIOS ─────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 12 }}>
        <div>
          <SectionHeader label="PRICE HISTORY" />
          <PriceChart ticker={ticker} data={data} />
        </div>
        <div>
          <SectionHeader label="MONTE CARLO SCENARIOS" />
          <ScenarioPanel data={data} compact />
        </div>
      </div>

      {/* ── DISCLAIMER ──────────────────────────────────────── */}
      <div style={{
        fontFamily: fontSans, fontSize: 10, color: COLORS.textFaint,
        lineHeight: 1.6, padding: '12px 0', borderTop: `1px solid ${COLORS.borderLt}`,
        textAlign: 'center',
      }}>
        Interpretations are model-derived and historical. No projection constitutes investment advice.
        Deflated Sharpe and out-of-sample validation should be reviewed before capital deployment.
      </div>
    </div>
  );
}
