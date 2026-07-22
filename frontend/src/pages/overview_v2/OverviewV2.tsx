// ============================================================
// QuantEdge v6.0 — Overview V2
// Bloomberg-style institutional layout with layered insights
// ============================================================
import React, { useState, useEffect } from 'react';
import { api } from '../../auth/authStore';
import {
  Insight,
  interpretCompositeScore,
  interpretSharpe,
  interpretBeta,
  interpretRegime,
  interpretMaxDrawdown,
  interpretVol,
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

// ── CAPM market exposure ─────────────────────────────────────
// Single-factor regression against SPY. Not Fama-French: there is no
// SMB/HML/RMW/CMA in this stack, so nothing here claims one.
function CapmPanel({ data }: { data: any }) {
  if (data.capm_available !== true) {
    return (
      <div>
        <SectionHeader label="MARKET EXPOSURE — CAPM VS SPY" />
        <div style={{ fontFamily: fontSans, fontSize: 12, color: COLORS.textDim }}>
          Not computed — requires 60 overlapping sessions with SPY.
        </div>
      </div>
    );
  }
  const n = data.capm_n_obs;
  const rows = [
    { k: 'BETA', v: data.capm_beta?.toFixed(2),
      d: 'Slope against SPY excess returns. 1.0 moves with the market.' },
    { k: 'ALPHA (ANN.)', v: data.capm_alpha != null ? `${(data.capm_alpha * 100).toFixed(2)}%` : null,
      d: 'Annualised intercept. Return not explained by market exposure.' },
    { k: 'R-SQUARED', v: data.capm_r_squared != null ? `${(data.capm_r_squared * 100).toFixed(0)}%` : null,
      d: 'Share of price movement the market explains. The rest is company-specific.' },
    { k: 'IDIO VOL', v: data.capm_idio_risk != null ? `${(data.capm_idio_risk * 100).toFixed(1)}%` : null,
      d: 'Annualised volatility of the regression residual.' },
  ];
  return (
    <div>
      <SectionHeader label="MARKET EXPOSURE — CAPM VS SPY" />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 8 }}>
        {rows.map(r => r.v == null ? null : (
          <div key={r.k} style={{
            background: COLORS.panelAlt, border: `1px solid ${COLORS.borderLt}`,
            borderRadius: 4, padding: '10px 12px',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontFamily: fontMono, fontSize: 9, letterSpacing: 1.5, color: COLORS.textFaint }}>
                {r.k}
              </span>
              <InfoTip text={r.d} />
            </div>
            <div style={{ fontFamily: fontMono, fontSize: 15, fontWeight: 600, color: COLORS.text, marginTop: 4 }}>
              {r.v}
            </div>
          </div>
        ))}
      </div>
      <div style={{ fontFamily: fontMono, fontSize: 9, color: COLORS.textFaint, marginTop: 8, letterSpacing: 1 }}>
        {n} ALIGNED SESSIONS · RF 5.3% · SINGLE-FACTOR, NOT FAMA-FRENCH
      </div>
    </div>
  );
}

// ── Main Overview V2 component ───────────────────────────────

// ═══════════════════════════════════════════════════════════════════════════
// HONESTY PANEL — Deflated Sharpe + PBO
// ═══════════════════════════════════════════════════════════════════════════
function HonestyPanel({ data }: { data: any }) {
  const gov = data?.governance;
  if (!gov) return null;

  const dsr: number | null = typeof gov.deflated_sharpe_ratio === 'number'
    ? gov.deflated_sharpe_ratio : null;
  const sharpeRaw: number | null = typeof gov.sharpe_ratio_raw === 'number'
    ? gov.sharpe_ratio_raw : null;
  const isGenuine: boolean = gov.is_genuine_alpha === true;

  const pbo = gov.pbo;
  const pboValue: number | null = pbo && typeof pbo.pbo === 'number' ? pbo.pbo : null;
  const pboInterp: string = pbo?.interpretation ?? '';
  const pboIsOverfit: boolean = pbo?.is_overfit === true;

  const COLORS = {
    bg:        '#1a0f0a',
    border:    '#3a2920',
    text:      '#d4c4b0',
    textDim:   '#9d8b7a',
    amber:     '#daa520',
    green:     '#22c55e',
    red:       '#ef4444',
    yellow:    '#f59e0b',
  };

  const dsrColor = dsr == null ? COLORS.textDim
    : dsr >= 0.95 ? COLORS.green
    : dsr >= 0.70 ? COLORS.amber
    : dsr >= 0.50 ? COLORS.yellow
    : COLORS.red;

  const pboColor = pboValue == null ? COLORS.textDim
    : pboValue < 0.10 ? COLORS.green
    : pboValue < 0.30 ? COLORS.amber
    : pboValue < 0.50 ? COLORS.yellow
    : COLORS.red;

  const dsrPct = dsr != null ? (dsr * 100).toFixed(1) + '%' : '—';
  const pboPct = pboValue != null ? (pboValue * 100).toFixed(1) + '%' : '—';

  return (
    <div style={{
      background: COLORS.bg,
      border: `1px solid ${COLORS.border}`,
      borderRadius: 4,
      padding: 16,
      marginTop: 12,
    }}>
      <div style={{
        fontFamily: "'Fira Code', monospace", fontSize: 10, letterSpacing: 2,
        color: COLORS.textDim, marginBottom: 12,
      }}>
        HONESTY — IS THIS SIGNAL REAL?
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {/* DSR */}
        <div>
          <div style={{ fontSize: 11, color: COLORS.textDim, marginBottom: 4 }}>
            DEFLATED SHARPE RATIO
          </div>
          <div style={{ fontSize: 24, color: dsrColor, fontWeight: 600 }}>
            {dsrPct}
          </div>
          <div style={{ fontSize: 11, color: COLORS.text, marginTop: 4, lineHeight: 1.5 }}>
            {dsr == null ? 'Not available.' : isGenuine
              ? 'Passes threshold for genuine alpha after selection-bias correction.'
              : `Below 95% genuine-alpha threshold. Raw Sharpe ${sharpeRaw?.toFixed(2) ?? '—'} may be inflated by multiple-testing bias.`}
          </div>
          <div style={{ fontSize: 10, color: COLORS.textDim, marginTop: 6 }}>
            Bailey & López de Prado (2014). Corrects Sharpe for N strategies tested.
          </div>
        </div>

        {/* PBO */}
        <div>
          <div style={{ fontSize: 11, color: COLORS.textDim, marginBottom: 4 }}>
            PROBABILITY OF BACKTEST OVERFITTING
          </div>
          <div style={{ fontSize: 24, color: pboColor, fontWeight: 600 }}>
            {pboPct}
          </div>
          <div style={{ fontSize: 11, color: COLORS.text, marginTop: 4, lineHeight: 1.5 }}>
            {pboValue == null ? (pboInterp || 'Not available.') : pboInterp}
          </div>
          <div style={{ fontSize: 10, color: COLORS.textDim, marginTop: 6 }}>
            Bailey et al. (2014). CSCV across {pbo?.n_strategies ?? 0} variants × {pbo?.n_slices ?? 0} slices.
          </div>
        </div>
      </div>

      {(pboIsOverfit || (dsr != null && !isGenuine)) && (
        <div style={{
          marginTop: 12,
          padding: '8px 12px',
          background: '#2d1410',
          border: `1px solid ${COLORS.red}`,
          borderRadius: 3,
          fontSize: 11,
          color: COLORS.text,
          lineHeight: 1.5,
        }}>
          <strong style={{ color: COLORS.red }}>⚠ Caution:</strong>{' '}
          The honesty metrics suggest the apparent edge may not generalize out-of-sample.
          Consider paper trading before deploying capital.
        </div>
      )}
    </div>
  );
}


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

  // ── 16-module conviction decomposition (the deep score) ──
  const [conv, setConv] = useState<any>(null);
  useEffect(() => {
    if (!ticker) return;
    api.get(`/api/v7/conviction/${ticker}`).then(r => setConv(r.data?.data || null)).catch(() => setConv(null));
  }, [ticker]);

  const thesis = buildThesis(data, ticker, conv);
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
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
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

      {/* ── MARKET EXPOSURE (CAPM vs SPY) ───────────────────── */}
      <CapmPanel data={data} />

      {/* ── HONESTY (Deflated Sharpe + PBO) ─────────────────── */}
      <HonestyPanel data={data} />

      {/* ── PRICE CHART + SCENARIOS ─────────────────────────── */}
      <div className="qe-chart-split" style={{ display: 'grid', gap: 12 }}>
        <div>
          <SectionHeader label="PRICE HISTORY" />
          <PriceChart ticker={ticker} data={data} />
        </div>
        <div>
          <SectionHeader label="MONTE CARLO SCENARIOS" />
          <ScenarioPanel data={data} compact />
        </div>
      </div>

      {/* ── CONVICTION DECOMPOSITION (16-module) ────────────── */}
      {conv && conv.modules && conv.conviction_score != null && (() => {
        const mods = [...conv.modules].filter((m:any) => m.score != null);
        // weighted contribution of each module to the final score
        const totalW = mods.reduce((s:number,m:any)=>s+(m.weight||0),0) || 100;
        const withContrib = mods.map((m:any)=>({ ...m,
          contrib: (m.score * (m.weight||0)) / totalW,
          label: (m.label||'').replace(' Intelligence',''),
        })).sort((a:any,b:any)=>b.contrib-a.contrib);
        const maxContrib = Math.max(...withContrib.map((m:any)=>m.contrib), 1);
        const vColor = (s:number) => s>=70 ? COLORS.green : s>=50 ? COLORS.amber : COLORS.red;
        const top = withContrib.slice(0,3).map((m:any)=>m.label).join(', ');
        const drags = withContrib.filter((m:any)=>m.score<45).map((m:any)=>m.label);
        const cScore = conv.conviction_score;
        const cCol = conv.verdict?.includes('BUY') ? COLORS.green : conv.verdict?.includes('SELL') ? COLORS.red : COLORS.amber;
        return (
          <div>
            <SectionHeader label={`CONVICTION DECOMPOSITION — WHY ${cScore.toFixed(0)}`} />
            <div style={{ display:'flex', alignItems:'baseline', gap:12, marginBottom:10, flexWrap:'wrap' }}>
              <span style={{ fontFamily:"'Fira Code',monospace", fontSize:32, fontWeight:800, color:cCol, lineHeight:1 }}>{cScore.toFixed(0)}</span>
              <span style={{ fontFamily:"'Fira Code',monospace", fontSize:14, fontWeight:700, color:cCol }}>{(conv.verdict||'').replace('_',' ')}</span>
              <span style={{ fontFamily:"'Outfit',sans-serif", fontSize:12, color:COLORS.textDim, flex:1, minWidth:200 }}>
                16-module weighted conviction · driven by {top}{drags.length ? ` · held back by ${drags.join(', ')}` : ''}
              </span>
            </div>
            <div style={{ display:'flex', flexDirection:'column', gap:5 }}>
              {withContrib.map((m:any)=>(
                <div key={m.id} style={{ display:'grid', gridTemplateColumns:'150px 1fr 60px 44px', alignItems:'center', gap:10 }}>
                  <span style={{ fontFamily:"'Outfit',sans-serif", fontSize:11, color:COLORS.text }}>
                    {m.label}<span style={{ color:COLORS.textDim, fontSize:9 }}> · {m.weight}%</span>
                  </span>
                  <div style={{ height:16, background:'#1a1512', borderRadius:3, position:'relative', overflow:'hidden' }}>
                    <div style={{ position:'absolute', left:0, top:0, bottom:0,
                      width:`${(m.contrib/maxContrib)*100}%`, background:vColor(m.score), opacity:0.55, borderRadius:3 }} />
                  </div>
                  <span style={{ fontFamily:"'Fira Code',monospace", fontSize:11, color:vColor(m.score), fontWeight:700, textAlign:'right' }}>
                    {m.score.toFixed(1)}
                  </span>
                  <span style={{ fontFamily:"'Fira Code',monospace", fontSize:10, color:COLORS.textDim, textAlign:'right' }}>
                    +{m.contrib.toFixed(1)}
                  </span>
                </div>
              ))}
            </div>
            <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10, color:COLORS.textDim, marginTop:8, lineHeight:1.5 }}>
              Each module scores 0–100 on its dimension; contribution = score × weight. The final conviction is the weighted sum across all 16 institutional dimensions.
            </div>
          </div>
        );
      })()}


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
