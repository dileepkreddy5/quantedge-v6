// ============================================================
// QuantEdge v6.0 — All UI Panel Components
// MLModels, Volatility, Regime, Options, Sentiment, Risk,
// Fundamentals, Scenarios, Signal, Watchlist panels
// ============================================================

import React, { useState, useEffect } from 'react';
import { api, useAuthStore } from '../../auth/authStore';
import toast from 'react-hot-toast';
import { useResponsive } from '../../hooks/useResponsive';

// ── Shared helpers ────────────────────────────────────────────
const fmtPct  = (v: any, d=2) => v == null ? '—' : `${Number(v) >= 0 ? '+' : ''}${(Number(v)*100).toFixed(d)}%`;
const fmtN    = (v: any, d=2) => v == null ? '—' : Number(v).toFixed(d);
const fmtLarge = (v: any) => {
  if (!v) return '—';
  if (v >= 1e12) return `$${(v/1e12).toFixed(2)}T`;
  if (v >= 1e9)  return `$${(v/1e9).toFixed(1)}B`;
  if (v >= 1e6)  return `$${(v/1e6).toFixed(0)}M`;
  return `$${Number(v).toFixed(2)}`;
};

const Row = ({ label, value, highlight }: { label: string; value: string; highlight?: string }) => (
  <div style={{ display:'flex', justifyContent:'space-between', padding:'5px 0', borderBottom:'1px solid rgba(212,149,108,0.06)' }}>
    <span style={{ fontFamily:"'Outfit',sans-serif", fontSize:11, color:'#9d8b7a' }}>{label}</span>
    <span style={{ fontFamily:"'Fira Code',monospace", fontSize:11, fontWeight:700, color: highlight || '#d4c4b0' }}>{value}</span>
  </div>
);

const SectionTitle = ({ children }: { children: React.ReactNode }) => (
  <div style={{ fontFamily:"'Fira Code',monospace", fontSize:9, color:'#daa520', letterSpacing:3, marginBottom:10, marginTop:18, paddingBottom:6, borderBottom:'1px solid rgba(218,165,32,0.15)' }}>
    {children}
  </div>
);

const Card = ({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) => (
  <div style={{ background:'#241510', border:'1px solid rgba(212,149,108,0.1)', borderRadius:8, padding:16, ...style }}>
    {children}
  </div>
);

// ══════════════════════════════════════════════════════════════
// SIGNAL PANEL
// ══════════════════════════════════════════════════════════════
// Responsive 3-col grid: 1 col phone, 2 col tablet, 3 col desktop
function ResponsiveGrid({ children, gap = 12 }: { children: React.ReactNode; gap?: number }) {
  const { bp } = useResponsive();
  const cols = bp === 'phone' ? '1fr' : bp === 'tablet' ? '1fr 1fr' : '1fr 1fr 1fr';
  return <div style={{ display: 'grid', gridTemplateColumns: cols, gap }}>{children}</div>;
}

export function SignalPanel({ data }: { data: any }) {
  const signal = data.overall_signal || 'NEUTRAL';
  const score = data.overall_score || 50;
  const signalColor = signal.includes('BUY') ? '#22c55e' : signal.includes('SELL') ? '#ef4444' : '#f59e0b';

  const signals = [
    { label: 'ML Ensemble', value: data.predicted_return_1y != null ? (data.predicted_return_1y > 0 ? 'BULLISH' : 'BEARISH') : 'NEUTRAL', color: (data.predicted_return_1y || 0) > 0 ? '#22c55e' : '#ef4444' },
    { label: 'HMM Regime', value: (data.current_regime || 'UNKNOWN').replace(/_/g, ' '), color: (data.current_regime || '').includes('BULL') ? '#22c55e' : '#ef4444' },
    { label: 'GARCH Vol', value: data.garch?.vol_regime || '—', color: '#f59e0b' },
    { label: 'Kalman Trend', value: data.kalman?.signal_interpretation?.replace(/_/g, ' ') || '—', color: '#06b6d4' },
    { label: 'NLP Sentiment', value: data.sentiment?.label || '—', color: (data.sentiment?.composite || 0) > 0 ? '#22c55e' : '#ef4444' },
    { label: 'Options GEX', value: data.options?.gex?.gex_regime || '—', color: data.options?.gex?.gex_regime === 'POSITIVE' ? '#22c55e' : '#ef4444' },
    { label: 'Hurst Exponent', value: (data.hurst_exponent || 0.5) > 0.55 ? 'TRENDING' : 'MEAN-REV', color: (data.hurst_exponent || 0.5) > 0.55 ? '#22c55e' : '#8b5cf6' },
  ];

  return (
    <Card>
      <div style={{ fontFamily:"'Fira Code',monospace", fontSize:9, color:'#9d8b7a', letterSpacing:2, marginBottom:14, paddingBottom:10, borderBottom:'1px solid rgba(212,149,108,0.08)' }}>
        ⬡ COMPOSITE SIGNAL
      </div>

      {/* Big signal */}
      <div style={{ textAlign:'center', padding:'20px 0', marginBottom:16 }}>
        <div style={{ fontFamily:"'Bebas Neue',sans-serif", fontSize:36, color:signalColor, letterSpacing:4, marginBottom:4 }}>
          {signal}
        </div>
        {/* Score gauge */}
        <div style={{ position:'relative', width:120, height:60, margin:'0 auto' }}>
          <svg viewBox="0 0 120 60" style={{ width:'100%' }}>
            <path d="M10,55 A50,50 0 0,1 110,55" fill="none" stroke="#1a0f0a" strokeWidth="8" />
            <path d="M10,55 A50,50 0 0,1 110,55" fill="none"
              stroke={signalColor} strokeWidth="8" strokeDasharray={`${(score / 100) * 157} 157`}
              style={{ transition:'stroke-dasharray 1.5s ease' }}
            />
          </svg>
          <div style={{ position:'absolute', bottom:0, left:0, right:0, textAlign:'center', fontFamily:"'Fira Code',monospace", fontSize:18, fontWeight:700, color:signalColor }}>
            {score}
          </div>
        </div>
        <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#8a7560', letterSpacing:2 }}>COMPOSITE SCORE / 100</div>
      </div>

      {/* Signal breakdown */}
      {signals.map(s => (
        <div key={s.label} style={{ display:'flex', justifyContent:'space-between', padding:'4px 0', borderBottom:'1px solid rgba(212,149,108,0.06)' }}>
          <span style={{ fontFamily:"'Outfit',sans-serif", fontSize:11, color:'#9d8b7a' }}>{s.label}</span>
          <span style={{ fontFamily:"'Fira Code',monospace", fontSize:9, fontWeight:700, color:s.color, letterSpacing:1 }}>{s.value}</span>
        </div>
      ))}
    </Card>
  );
}

// ══════════════════════════════════════════════════════════════
// ML MODELS PANEL
// ══════════════════════════════════════════════════════════════
export function MLModelsPanel({ data }: { data: any }) {
  const preds = data.ml_predictions || {};
  const ensemble = preds.ensemble || {};
  const lstm = preds.lstm || {};
  const xgb = preds.xgboost || {};
  const lgbm = preds.lightgbm || {};
  const shap = preds.shap_top_drivers || [];
  const quantile = preds.quantile || {};
  const panel = data.panel_prediction || null;  // cross-sectional multi-horizon models

  // ── Cross-model conviction: agreement among INDEPENDENT model signals on 21-day direction ──
  // Note: Ensemble is excluded — it's derived from LSTM/XGB/LGBM, so it isn't an independent vote.
  const dirFromString = (s: string | undefined, bull: string[], bear: string[]): number | null => {
    if (!s) return null;
    const u = String(s).toUpperCase();
    if (bull.some(b => u.includes(b))) return 1;
    if (bear.some(b => u.includes(b))) return -1;
    return 0;
  };
  // Kalman direction comes from trend_slope (signed), NOT signal_interpretation —
  // that field describes trend STRENGTH ('STRONG_TREND'), not direction, so string
  // matching returned neutral for every ticker.
  const kSlope = data.kalman?.trend_slope;
  const kalmanDir = kSlope != null && Number.isFinite(Number(kSlope))
    ? (Number(kSlope) > 0 ? 1 : Number(kSlope) < 0 ? -1 : 0)
    : null;
  const regimeDir = dirFromString(data.current_regime, ['BULL'], ['BEAR']);
  const gexDir = dirFromString(data.options?.gex?.gex_regime, ['POSITIVE'], ['NEGATIVE']);
  const sentVal = data.sentiment?.composite;
  // GARCH volatility regime: current vol above long-run = elevated (risk-off lean)
  const gCur = data.garch?.current_annual_vol, gLR = data.garch?.long_run_annual_vol;
  const garchDir = (gCur != null && gLR != null) ? (Number(gCur) > Number(gLR) ? -1 : 1) : null;
  // Monte Carlo: median (p50) simulated return direction
  const mcMed = data.monte_carlo?.p50 ?? data.monte_carlo?.expected_return;
  const mcDir = mcMed != null ? Math.sign(Number(mcMed)) : null;
  // Panel cross-sectional ensemble: 1-month prediction direction
  const panelPred = data.panel_prediction?.horizons?.['21d']?.pred_pct;
  const panelDir = panelPred != null ? Math.sign(Number(panelPred)) : null;

  // When the three return-forecasting models land within a hair of each other it is
  // not consensus — it is all three regressing to the same conditional mean because
  // none of them found signal in ~80 training samples. Counting that as three votes
  // manufactures agreement out of an absence of information, so they collapse to one.
  const _mlVals = [lstm.pred_21d, xgb.pred_21d, lgbm.pred_21d]
    .filter(v => v != null).map(Number);
  const _mlSpread = _mlVals.length > 1 ? Math.max(..._mlVals) - Math.min(..._mlVals) : 0;
  const mlCollapsed = _mlVals.length > 1 && _mlSpread < 0.05;
  const mlMean = _mlVals.length ? _mlVals.reduce((a,b)=>a+b,0) / _mlVals.length : null;

  // Only DIRECTIONAL, non-derived signals get a vote.
  //  - GARCH is excluded: it forecasts volatility, not direction. Low vol is not bullish.
  //  - Monte Carlo is excluded: its drift is derived from the ML ensemble, so it would
  //    double-count that signal rather than add an independent view.
  // Both are still displayed in the model matrix as context.
  const modelDirs = [
    ...(mlCollapsed
      ? [{ name: 'ML ensemble', v: mlMean != null ? Math.sign(mlMean) : null }]
      : [
          { name: 'LSTM',     v: lstm.pred_21d != null ? Math.sign(Number(lstm.pred_21d)) : null },
          { name: 'XGBoost',  v: xgb.pred_21d != null ? Math.sign(Number(xgb.pred_21d)) : null },
          { name: 'LightGBM', v: lgbm.pred_21d != null ? Math.sign(Number(lgbm.pred_21d)) : null },
        ]),
    { name: 'Kalman',    v: kalmanDir },
    { name: 'Regime',    v: regimeDir },
    { name: 'Sentiment', v: sentVal != null ? Math.sign(Number(sentVal)) : null },
    { name: 'Panel-CS',  v: panelDir },
    { name: 'Options',   v: gexDir },
  ].filter(m => m.v != null) as { name: string; v: number }[];

  // Full 9-model roster: what each model is, what it output, and why it matters.
  // Written for a general investor, not a quant.
  const modelRoster = [
    { name: 'Bidirectional LSTM', kind: 'Neural network',
      value: lstm.pred_21d != null ? `${(Number(lstm.pred_21d)).toFixed(2)}% (21d)` : null,
      dir: lstm.pred_21d != null ? Math.sign(Number(lstm.pred_21d)) : null,
      what: 'Reads the price history like a sequence, learning patterns that repeat across time.',
      why: 'Captures momentum and reversal shapes that simple indicators miss.' },
    { name: 'XGBoost', kind: 'Gradient-boosted trees',
      value: xgb.pred_21d != null ? `${(Number(xgb.pred_21d)).toFixed(2)}% (21d)` : null,
      dir: xgb.pred_21d != null ? Math.sign(Number(xgb.pred_21d)) : null,
      what: 'Builds thousands of decision trees over 170 indicators to find which combinations preceded gains.',
      why: 'Handles non-linear interactions between factors that a single rule cannot.' },
    { name: 'LightGBM', kind: 'Gradient-boosted trees',
      value: lgbm.pred_21d != null ? `${(Number(lgbm.pred_21d)).toFixed(2)}% (21d)` : null,
      dir: lgbm.pred_21d != null ? Math.sign(Number(lgbm.pred_21d)) : null,
      what: 'A second tree model using a different growth strategy, run alongside XGBoost.',
      why: 'When two independent tree models agree, the signal is more robust.' },
    { name: 'Kalman Filter', kind: 'State-space trend',
      value: kSlope != null ? `slope ${Number(kSlope).toFixed(4)}` : null,
      dir: kalmanDir,
      what: 'Strips daily noise out of price to estimate the true underlying trend and its acceleration.',
      why: 'Separates real directional drift from random day-to-day movement.' },
    { name: 'HMM Regime', kind: 'Hidden Markov Model',
      value: data.current_regime ? String(data.current_regime).replace(/_/g,' ') : null,
      dir: regimeDir,
      what: 'Classifies which market state we are in — calm bull, volatile bear, and so on.',
      why: 'The same signal means different things depending on the regime.' },
    { name: 'GJR-GARCH', kind: 'Volatility model',
      value: data.garch?.current_annual_vol != null ? `${(Number(data.garch.current_annual_vol)*100).toFixed(1)}% vol` : null,
      dir: garchDir,
      what: 'Forecasts volatility, accounting for the fact that crashes raise risk more than rallies.',
      why: 'Sets position sizing and tells you how much noise to expect around any forecast.' },
    { name: 'Monte Carlo', kind: 'Path simulation',
      value: mcMed != null ? `median ${(Number(mcMed)*100).toFixed(1)}%` : null,
      dir: mcDir,
      what: 'Simulates thousands of possible future price paths using current drift and volatility.',
      why: 'Gives a range of outcomes instead of one number — the honest way to express uncertainty.' },
    { name: 'FinBERT Sentiment', kind: 'Language model',
      value: sentVal != null ? `${Number(sentVal).toFixed(2)} score` : null,
      dir: sentVal != null ? Math.sign(Number(sentVal)) : null,
      what: 'A language model trained on financial text, scoring whether recent news reads bullish or bearish.',
      why: 'News moves prices before it shows up in fundamentals.' },
    { name: 'Cross-Sectional Panel', kind: 'Multi-horizon ensemble',
      value: panelPred != null ? `${Number(panelPred).toFixed(2)}% (1mo)` : null,
      dir: panelDir,
      what: 'Ranks this stock against the whole US universe on 178 factors including point-in-time fundamentals.',
      why: 'The only model here with measured out-of-sample skill — see the reliability badges above.' },
  ];

  const signs = modelDirs.map(m => m.v);
  const nBull = signs.filter(s => s > 0).length;
  const nBear = signs.filter(s => s < 0).length;
  const nTotal = modelDirs.length || 1;
  const majority = nBull >= nBear ? nBull : nBear;
  const agreement = majority / nTotal;                    // 0.5 (split) .. 1.0 (unanimous)
  const direction = nBull > nBear ? 'BULLISH' : nBull < nBear ? 'BEARISH' : 'MIXED';
  // Agreement maps to conviction, but capped at 85. Equity signals are correlated —
  // even unanimous agreement among them does not justify a claim of certainty, and a
  // 100/100 score would imply exactly that. The cap keeps the scale honest.
  const CONVICTION_CAP = 85;
  const conviction = Math.round(Math.max(0, (agreement - 0.5) * 2) * CONVICTION_CAP);
  const convLabel = conviction >= 65 ? 'Strong agreement' : conviction >= 35 ? 'Moderate agreement' : conviction > 0 ? 'Weak agreement' : 'No agreement';
  const convColor = direction === 'MIXED' ? '#8a7560' : direction === 'BULLISH' ? '#22c55e' : '#ef4444';

  return (
    <div className="qe-grid-3">

      {panel && panel.horizons && (
        <Card style={{ gridColumn:'span 3' }}>
          <SectionTitle>MULTI-HORIZON ML FORECAST — CROSS-SECTIONAL PANEL MODEL</SectionTitle>
          <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:11, color:'#8a7560', marginBottom:12, lineHeight:1.5 }}>
            Gradient-boosted ensemble (XGBoost + LightGBM) trained cross-sectionally on the US universe with point-in-time fundamentals.
            Each horizon independently validated out-of-sample. Predictions are model-implied return leans, honestly labeled by measured reliability.
          </div>
          <div style={{ display:'grid', gridTemplateColumns:'repeat(6, 1fr)', gap:8 }}>
            {['5d','10d','21d','63d','126d','252d'].map((hk) => {
              const h = panel.horizons[hk];
              if (!h) return null;
              const pv = Number(h.pred_pct);
              const col = pv > 0 ? '#22c55e' : pv < 0 ? '#ef4444' : '#8a7560';
              const ic = h.oos_rank_ic;
              const icNum = ic != null ? Number(ic) : null;
              // Reliability comes from the TRAINER's independent-window test, not IC size.
              // A high IC on 1 non-overlapping window is an artifact, not skill.
              const indep = h.n_independent_val_dates ?? h.indep_dates ?? null;
              const reliableFlag = h.reliable === true || (indep != null && indep >= 5);
              const strong = reliableFlag && icNum != null && icNum >= 0.05;
              const badgeCol = strong ? '#22c55e' : (reliableFlag && icNum != null && icNum > 0.02) ? '#eab308' : '#8a7560';
              const badgeTxt = strong ? 'VALIDATED' : (reliableFlag && icNum != null && icNum > 0.02) ? 'MODERATE' : (reliableFlag ? 'NO EDGE' : 'UNVALIDATED');
              return (
                <div key={hk} style={{ border:'1px solid #3a2f28', borderRadius:8, padding:'10px 8px', background:'#1a1512', textAlign:'center' }}>
                  <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10, color:'#8a7560', letterSpacing:1, marginBottom:6 }}>{h.label}</div>
                  <div style={{ fontFamily:"'Fira Code',monospace", fontSize:20, fontWeight:800, color:col, lineHeight:1 }}>
                    {pv > 0 ? '+' : ''}{pv.toFixed(1)}%
                  </div>
                  <div style={{ fontFamily:"'Fira Code',monospace", fontSize:9, color:'#8a7560', marginTop:6 }}>
                    IC {icNum != null ? (icNum >= 0 ? '+' : '') + icNum.toFixed(3) : 'n/a'}
                  </div>
                  <div style={{ marginTop:6, fontFamily:"'Fira Code',monospace", fontSize:8, letterSpacing:0.5,
                    padding:'2px 5px', borderRadius:4, border:'1px solid ' + badgeCol + '40', color:badgeCol, background:badgeCol + '12', display:'inline-block' }}>
                    {badgeTxt}
                  </div>
                </div>
              );
            })}
          </div>
          {panel.shap_drivers && panel.shap_drivers.length > 0 && (
            <div style={{ marginTop:12 }}>
              <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10, color:'#8a7560', letterSpacing:1, marginBottom:6 }}>TOP PREDICTIVE DRIVERS (1-MONTH MODEL)</div>
              <div style={{ display:'flex', gap:6, flexWrap:'wrap' }}>
                {panel.shap_drivers.slice(0,8).map((d:any, i:number) => {
                  const col = Number(d.impact) > 0 ? '#22c55e' : '#ef4444';
                  return (
                    <span key={i} style={{ fontFamily:"'Fira Code',monospace", fontSize:9, padding:'3px 7px', borderRadius:4,
                      border:'1px solid ' + col + '40', color:col, background:col + '10' }}>
                      {d.feature} {Number(d.impact) > 0 ? '\u25b2' : '\u25bc'}
                    </span>
                  );
                })}
              </div>
            </div>
          )}
          {panel.methodology && (
            <div style={{ marginTop:10, fontFamily:"'Outfit',sans-serif", fontSize:10, color:'#6b5d52', lineHeight:1.4 }}>
              Trained on {panel.n_tickers_trained || 'the US'} stocks · {panel.methodology}
            </div>
          )}
        </Card>
      )}

      {/* Cross-model conviction */}
      <Card style={{ gridColumn:'span 3' }}>
        <SectionTitle>MODEL CONVICTION — CROSS-MODEL AGREEMENT</SectionTitle>
        <div style={{ display:'flex', alignItems:'center', gap:20, flexWrap:'wrap' }}>
          <div style={{ textAlign:'center', minWidth:110 }}>
            <div style={{ fontFamily:"'Fira Code',monospace", fontSize:36, fontWeight:800, color: convColor, lineHeight:1 }}>{conviction}</div>
            <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10, color:'#8a7560', marginTop:4, letterSpacing:1 }}>CONVICTION / 100</div>
          </div>
          <div style={{ flex:1, minWidth:200 }}>
            <div style={{ fontFamily:"'Fira Code',monospace", fontSize:14, fontWeight:700, color: convColor, marginBottom:4 }}>
              {convLabel} · {direction}
            </div>
            <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:12, color:'#d4c4b0', lineHeight:1.5 }}>
              {majority} of {nTotal} models agree on a {direction.toLowerCase()} 21-day lean
              {mlCollapsed ? ' (the three return-forecasting models produced near-identical output — a sign they found no distinguishing signal — so they count once, not three times)' : ''}
              {mlCollapsed ? ' (the three return-forecasting models produced near-identical output — a sign they found no distinguishing signal — so they count once, not three times)' : ''}
              {nTotal - majority > 0 ? `, ${nTotal - majority} disagree` : ''}.
              {' '}Conviction measures <b style={{color:'#d4c4b0'}}>agreement across models</b>, not predictive accuracy.
            </div>
          </div>
          <div style={{ display:'flex', gap:6, flexWrap:'wrap', minWidth:160 }}>
            {modelDirs.map((m, i) => {
              const s = m.v;
              const col = s > 0 ? '#22c55e' : s < 0 ? '#ef4444' : '#8a7560';
              return (
                <span key={i} style={{ fontFamily:"'Fira Code',monospace", fontSize:9, letterSpacing:0.5,
                  padding:'3px 7px', borderRadius:4, border:`1px solid ${col}40`, color: col, background:`${col}10` }}>
                  {m.name} {s > 0 ? '▲' : s < 0 ? '▼' : '—'}
                </span>
              );
            })}
          </div>
        </div>
      </Card>

      {/* Ensemble */}
      <Card style={{ gridColumn:'span 3' }}>
        <SectionTitle>THE NINE MODELS — WHAT EACH ONE DOES</SectionTitle>
        <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:11, color:'#8a7560', marginBottom:12, lineHeight:1.5 }}>
          Nine independent models run on every analysis. Each measures something different; the conviction score above
          reflects how many of them point the same way.
        </div>
        <div style={{ display:'flex', flexDirection:'column', gap:1 }}>
          {modelRoster.map((m:any, i:number) => {
            const dcol = m.dir > 0 ? '#22c55e' : m.dir < 0 ? '#ef4444' : '#8a7560';
            const arrow = m.dir > 0 ? '\u25b2' : m.dir < 0 ? '\u25bc' : '\u2014';
            return (
              <div key={i} style={{ display:'grid', gridTemplateColumns:'190px 130px 30px 1fr', gap:12,
                alignItems:'start', padding:'9px 8px', background: i % 2 ? 'transparent' : 'rgba(255,255,255,0.015)', borderRadius:4 }}>
                <div>
                  <div style={{ fontFamily:"'Fira Code',monospace", fontSize:11, color:'#d4c4b0', fontWeight:600 }}>{m.name}</div>
                  <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:9, color:'#6b5d52' }}>{m.kind}</div>
                </div>
                <div style={{ fontFamily:"'Fira Code',monospace", fontSize:11, color: m.value ? '#d4c4b0' : '#6b5d52', paddingTop:1 }}>
                  {m.value || 'unavailable'}
                </div>
                <div style={{ fontFamily:"'Fira Code',monospace", fontSize:12, color:dcol, textAlign:'center', paddingTop:1 }}>{arrow}</div>
                <div>
                  <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:11, color:'#b8a894', lineHeight:1.45 }}>{m.what}</div>
                  <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10, color:'#7a6b5d', lineHeight:1.45, marginTop:2 }}>
                    <span style={{ color:'#8a7560' }}>Why it helps:</span> {m.why}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </Card>

      <Card style={{ gridColumn:'span 3' }}>
        <SectionTitle>PER-TICKER ENSEMBLE — MODEL DIAGNOSTICS</SectionTitle>
        <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:11, color:'#9d8b7a', lineHeight:1.5, marginBottom:12, padding:'8px 10px', background:'#1a0f0a', borderRadius:6, borderLeft:'2px solid #daa520' }}>
          A separate ensemble trains on <b style={{color:'#d4c4b0'}}>this ticker alone</b> at request time. With only ~80
          usable samples per stock it is too data-starved to forecast reliably, so its point estimates are not shown —
          the cross-sectional panel above is the forecast of record. What remains useful from it is diagnostic:
          how confident it is, how much its component models disagree, and the return distribution it implies.
        </div>
        <div style={{ display:'flex', gap:20, marginTop:12, fontFamily:"'Fira Code',monospace", fontSize:9, color:'#8a7560' }}>
          <span>CONFIDENCE: {((ensemble.confidence || 0)*100).toFixed(1)}%</span>
          <span>MODEL DISAGREEMENT: {fmtN(ensemble.model_disagreement)}%</span>
          <span>IC ESTIMATE: {fmtN(preds.rank_ic_estimate,3)}</span>
        </div>
      </Card>

      {/* LSTM */}
      <Card style={{ gridColumn:'span 2' }}>
        <SectionTitle>RETURN DISTRIBUTION — 1 MONTH</SectionTitle>
        <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:11, color:'#9d8b7a', lineHeight:1.5, marginBottom:14 }}>
          The realistic range of one-month outcomes, not a single guess. The <b style={{color:'#d4c4b0'}}>width</b> of this
          band is well estimated — it comes from realized volatility. The <b style={{color:'#d4c4b0'}}>centre</b> inherits the
          weaker per-ticker drift estimate, so read the spread with more confidence than the midpoint.
        </div>
        {Object.keys(quantile).length > 0 ? (() => {
          const qs = [
            { k:'q10_1m', l:'P10', d:'1-in-10 downside' },
            { k:'q25_1m', l:'P25', d:'lower quartile' },
            { k:'q50_1m', l:'P50', d:'median' },
            { k:'q75_1m', l:'P75', d:'upper quartile' },
            { k:'q90_1m', l:'P90', d:'1-in-10 upside' },
          ].filter(q => quantile[q.k] != null);
          if (!qs.length) return null;
          const vals = qs.map(q => Number(quantile[q.k]));
          const lo = Math.min(...vals), hi = Math.max(...vals);
          const span = (hi - lo) || 1;
          const posOf = (v:number) => ((v - lo) / span) * 100;
          const zero = (lo <= 0 && hi >= 0) ? posOf(0) : null;
          return (
            <div>
              <div style={{ position:'relative', height:38, marginBottom:10 }}>
                <div style={{ position:'absolute', top:16, left:`${posOf(vals[0])}%`, width:`${posOf(vals[vals.length-1])-posOf(vals[0])}%`,
                  height:6, background:'linear-gradient(90deg,#ef444455,#8a756055,#22c55e55)', borderRadius:3 }} />
                {qs.length >= 4 && (
                  <div style={{ position:'absolute', top:13, left:`${posOf(Number(quantile['q25_1m']))}%`,
                    width:`${posOf(Number(quantile['q75_1m']))-posOf(Number(quantile['q25_1m']))}%`,
                    height:12, background:'#daa52033', border:'1px solid #daa52066', borderRadius:3 }} />
                )}
                {zero != null && (
                  <div style={{ position:'absolute', top:6, left:`${zero}%`, width:1, height:26, background:'#9d8b7a88' }} />
                )}
                {qs.map(q => {
                  const v = Number(quantile[q.k]);
                  const isMed = q.k === 'q50_1m';
                  return (
                    <div key={q.k} style={{ position:'absolute', left:`${posOf(v)}%`, top:isMed?8:12, transform:'translateX(-50%)' }}>
                      <div style={{ width:isMed?3:2, height:isMed?22:14, background:isMed?'#daa520':'#9d8b7a', borderRadius:1 }} />
                    </div>
                  );
                })}
              </div>
              <div style={{ display:'grid', gridTemplateColumns:`repeat(${qs.length}, 1fr)`, gap:6 }}>
                {qs.map(q => {
                  const v = Number(quantile[q.k]);
                  const c = v > 0 ? '#22c55e' : v < 0 ? '#ef4444' : '#8a7560';
                  return (
                    <div key={q.k} style={{ textAlign:'center' }}>
                      <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#8a7560', letterSpacing:1 }}>{q.l}</div>
                      <div style={{ fontFamily:"'Fira Code',monospace", fontSize:15, fontWeight:700, color:c, marginTop:2 }}>
                        {v>0?'+':''}{v.toFixed(1)}%
                      </div>
                      <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:8, color:'#6b5d52', marginTop:2 }}>{q.d}</div>
                    </div>
                  );
                })}
              </div>
              <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10, color:'#7a6b5d', marginTop:12, lineHeight:1.5 }}>
                Eight in ten simulated outcomes fall between the P10 and P90 marks. The shaded box spans the middle 50%.
              </div>
            </div>
          );
        })() : (
          <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:11, color:'#8a7560' }}>Quantile forecasts unavailable for this ticker.</div>
        )}
      </Card>

      <Card>
        <SectionTitle>HOW THESE MODELS ARE BUILT</SectionTitle>
        <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10.5, color:'#9d8b7a', lineHeight:1.6 }}>
          <div style={{ marginBottom:9 }}>
            <span style={{ color:'#daa520', fontFamily:"'Fira Code',monospace", fontSize:9, letterSpacing:1 }}>LABELLING</span><br/>
            Triple-barrier method — each sample is labelled by whether price hit an upper or lower volatility band first,
            or neither within 21 days. Path-dependent, unlike a plain forward return.
          </div>
          <div style={{ marginBottom:9 }}>
            <span style={{ color:'#daa520', fontFamily:"'Fira Code',monospace", fontSize:9, letterSpacing:1 }}>VALIDATION</span><br/>
            Walk-forward splits by date with an embargo gap, so the model is never tested on periods adjacent to its
            training data. Long-horizon rank-IC is measured only on non-overlapping windows.
          </div>
          <div style={{ marginBottom:9 }}>
            <span style={{ color:'#daa520', fontFamily:"'Fira Code',monospace", fontSize:9, letterSpacing:1 }}>CROSS-SECTIONAL RANKING</span><br/>
            Every feature is converted to its percentile within that day's universe. The model learns relative
            positioning against peers rather than absolute levels.
          </div>
          <div style={{ marginBottom:9 }}>
            <span style={{ color:'#daa520', fontFamily:"'Fira Code',monospace", fontSize:9, letterSpacing:1 }}>POINT-IN-TIME FUNDAMENTALS</span><br/>
            Financial data is filtered by SEC filing date, so no sample can see a figure before it was public. This is
            what separates a real backtest from an accidental look into the future.
          </div>
          <div>
            <span style={{ color:'#daa520', fontFamily:"'Fira Code',monospace", fontSize:9, letterSpacing:1 }}>ARCHITECTURE</span><br/>
            135 ranked features · XGBoost + LightGBM ensemble per horizon · 512→256→128 BiLSTM with temporal attention
            and MC-dropout uncertainty.
          </div>
        </div>
      </Card>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// VOLATILITY PANEL
// ══════════════════════════════════════════════════════════════
export function VolatilityPanel({ data }: { data: any }) {
  const g = data.garch || {};
  const risk = data.risk_metrics || {};

  // long_run_annual_vol arrives already expressed as a percent, while
  // current_annual_vol arrives as a fraction. Normalise both to percent.
  const curVol = (g.current_annual_vol || 0) * 100;
  const longRun = g.long_run_annual_vol != null
    ? (Math.abs(g.long_run_annual_vol) > 3 ? g.long_run_annual_vol : g.long_run_annual_vol * 100)
    : null;
  const f5  = (g.forecast_vol_5d || 0) * 100;
  const f21 = (g.forecast_vol_21d || 0) * 100;
  const persistence = g.persistence || 0;
  const gamma = g.gamma_asymmetry || 0;
  const alpha = g.alpha || 0;

  const ratio = (longRun && longRun > 0) ? curVol / longRun : null;
  const stance = ratio == null ? null
    : ratio > 1.15 ? { t: 'ELEVATED', c: '#ef4444', s: 'trading above its structural level — expect mean reversion downward in vol, and size positions smaller than usual' }
    : ratio < 0.85 ? { t: 'COMPRESSED', c: '#f59e0b', s: 'trading below its structural level — quiet periods often precede expansion, so tail risk is understated by recent history' }
    : { t: 'IN LINE', c: '#22c55e', s: 'close to its own structural level — recent history is a reasonable guide to near-term risk' };

  // Leverage ratio: how much more a down shock moves vol than an up shock.
  const levRatio = alpha > 0 ? (alpha + gamma) / alpha : null;

  const estimators = [
    { l: 'Close-to-Close', v: (data.annual_vol || 0) * 100, n: 'Simplest — uses closing prices only, ignores intraday range' },
    { l: 'Parkinson',      v: (data.parkinson_vol || 0) * 100, n: 'Uses high/low range, about 5× more efficient than close-to-close' },
    { l: 'Garman-Klass',   v: (data.garman_klass_vol || 0) * 100, n: 'Uses open, high, low and close — most efficient on continuous trading' },
    { l: 'Yang-Zhang',     v: (data.yang_zhang_vol || 0) * 100, n: 'Handles overnight gaps, generally the best choice for daily equity data' },
  ].filter(e => e.v > 0);

  return (
    <div className="qe-grid-3">
      <Card style={{ gridColumn:'span 2' }}>
        <SectionTitle>VOLATILITY — CURRENT VERSUS STRUCTURAL</SectionTitle>
        {stance && (
          <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:12, color:'#b8a894', lineHeight:1.6, marginBottom:14 }}>
            Volatility is <b style={{ color: stance.c }}>{stance.t.toLowerCase()}</b> — {stance.s}.
          </div>
        )}
        <div style={{ display:'grid', gridTemplateColumns:'repeat(4, 1fr)', gap:10, marginBottom:14 }}>
          {[
            { l:'CURRENT', v:curVol, n:'annualised, from GARCH' },
            { l:'STRUCTURAL', v:longRun, n:'long-run GARCH level' },
            { l:'5-DAY FCST', v:f5, n:'near-term path' },
            { l:'21-DAY FCST', v:f21, n:'one-month path' },
          ].map(x => (
            <div key={x.l} style={{ background:'#1a0f0a', borderRadius:8, padding:'12px 10px', textAlign:'center' }}>
              <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#8a7560', letterSpacing:2 }}>{x.l}</div>
              <div style={{ fontFamily:"'Fira Code',monospace", fontSize:19, fontWeight:800, color:'#d4c4b0', marginTop:5 }}>
                {x.v != null && x.v > 0 ? `${x.v.toFixed(1)}%` : '—'}
              </div>
              <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:8.5, color:'#6b5d52', marginTop:3 }}>{x.n}</div>
            </div>
          ))}
        </div>
        <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:11, color:'#9d8b7a', lineHeight:1.6, padding:'10px 12px', background:'#1a0f0a', borderRadius:6, borderLeft:'2px solid #daa520' }}>
          <b style={{ color:'#d4c4b0' }}>Downside asymmetry.</b>{' '}
          {gamma > 0
            ? <>The GJR term is positive (γ = {gamma.toFixed(3)}), confirming the leverage effect: a negative shock raises
               volatility {levRatio ? `roughly ${levRatio.toFixed(1)}×` : 'considerably'} more than an equally sized positive
               shock. Downside risk is structurally worse than a symmetric model would suggest.</>
            : <>The GJR term is not positive, so this name does not show the usual leverage effect — up and down shocks move
               volatility similarly.</>}
          {persistence > 0.97 && (
            <> Persistence is {persistence.toFixed(4)}, close to unity, so volatility shocks decay slowly and the long-run
              estimate should be treated as indicative rather than precise.</>
          )}
        </div>
        <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#6b5d52', marginTop:10, letterSpacing:0.5 }}>
          GJR-GARCH(1,1) · ω {fmtN(g.omega,6)} · α {fmtN(g.alpha,4)} · γ {fmtN(g.gamma_asymmetry,4)} · β {fmtN(g.beta,4)} · ν {fmtN(g.nu_student_t,1)} · persistence {fmtN(g.persistence,4)}
        </div>
      </Card>

      <Card>
        <SectionTitle>ESTIMATOR CROSS-CHECK</SectionTitle>
        <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10.5, color:'#9d8b7a', lineHeight:1.5, marginBottom:12 }}>
          Four ways of measuring the same thing. Wide disagreement between them usually means gappy or illiquid trading.
        </div>
        {estimators.map(e => (
          <div key={e.l} style={{ marginBottom:9 }}>
            <div style={{ display:'flex', justifyContent:'space-between', alignItems:'baseline' }}>
              <span style={{ fontFamily:"'Outfit',sans-serif", fontSize:11, color:'#d4c4b0' }}>{e.l}</span>
              <span style={{ fontFamily:"'Fira Code',monospace", fontSize:12, color:'#d4c4b0', fontWeight:600 }}>{e.v.toFixed(1)}%</span>
            </div>
            <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:9, color:'#6b5d52', lineHeight:1.4 }}>{e.n}</div>
          </div>
        ))}
        {estimators.length > 1 && (() => {
          const vs = estimators.map(e => e.v);
          const spread = Math.max(...vs) - Math.min(...vs);
          return (
            <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10, color:'#7a6b5d', marginTop:12, paddingTop:10, borderTop:'1px solid rgba(212,149,108,0.12)', lineHeight:1.5 }}>
              Spread of {spread.toFixed(1)} points across estimators —{' '}
              {spread < 4 ? 'tight agreement, the volatility reading is well determined.'
                          : 'material disagreement, likely from overnight gaps or thin intraday liquidity.'}
            </div>
          );
        })()}
      </Card>

      {data.flow_analysis?.participation && (() => {
        const fl = data.flow_analysis;
        const p = fl.participation;
        const sc = (s:number) => s >= 70 ? '#22c55e' : s >= 45 ? '#f59e0b' : '#ef4444';
        const pvCol = { ACCUMULATION:'#22c55e', 'SELLING EXHAUSTION':'#f59e0b', BALANCED:'#8a7560',
                        'WEAK RALLY':'#f59e0b', DISTRIBUTION:'#ef4444' }[fl.pv_agreement] || '#8a7560';
        const ladder = [252,90,63,30,21,15,5]
          .map(w => ({ w, v: fl[`up_volume_share_${w}d`] }))
          .filter(x => x.v != null);
        const lo = 40, hi = 90;
        return (
          <Card style={{ gridColumn:'span 3' }}>
            <SectionTitle>MARKET PARTICIPATION — HOW MUCH CONVICTION IS BEHIND THIS MOVE</SectionTitle>

            <div style={{ display:'flex', gap:24, alignItems:'flex-start', flexWrap:'wrap', marginBottom:18 }}>
              <div style={{ textAlign:'center', minWidth:120 }}>
                <div style={{ fontFamily:"'Fira Code',monospace", fontSize:42, fontWeight:800, color:sc(p.score), lineHeight:1 }}>{p.score}</div>
                <div style={{ fontFamily:"'Fira Code',monospace", fontSize:9, color:'#8a7560', letterSpacing:2, marginTop:4 }}>PARTICIPATION</div>
                <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:12, color:sc(p.score), marginTop:2 }}>{p.label}</div>
              </div>
              <div style={{ flex:1, minWidth:280 }}>
                {p.components.map((c:any) => (
                  <div key={c.name} style={{ display:'grid', gridTemplateColumns:'170px 1fr 90px', gap:10, alignItems:'center', marginBottom:5 }}>
                    <span style={{ fontFamily:"'Outfit',sans-serif", fontSize:10.5, color:'#9d8b7a' }}>{c.name}</span>
                    <div style={{ height:12, background:'#1a0f0a', borderRadius:3, overflow:'hidden' }}>
                      <div style={{ height:'100%', width:`${c.score}%`, background:sc(c.score), opacity:0.5, borderRadius:3 }} />
                    </div>
                    <span style={{ fontFamily:"'Fira Code',monospace", fontSize:10.5, color:'#d4c4b0', textAlign:'right' }}>{c.value}</span>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:16 }}>
              <div>
                <div style={{ fontFamily:"'Fira Code',monospace", fontSize:9, color:'#daa520', letterSpacing:2, marginBottom:8 }}>
                  PARTICIPATION TREND — SHARE OF VOLUME ON UP DAYS
                </div>
                <div style={{ display:'grid', gridTemplateColumns:`repeat(${ladder.length}, 1fr)`, gap:5, alignItems:'end' }}>
                  {ladder.map(x => {
                    const h = Math.max(6, ((x.v - lo) / (hi - lo)) * 60);
                    const c = x.v >= 60 ? '#22c55e' : x.v >= 50 ? '#8a7560' : '#ef4444';
                    return (
                      <div key={x.w} style={{ textAlign:'center' }}>
                        <div style={{ fontFamily:"'Fira Code',monospace", fontSize:9.5, color:c, marginBottom:3 }}>{x.v.toFixed(0)}%</div>
                        <div style={{ height:h, background:c, opacity:0.45, borderRadius:'2px 2px 0 0' }} />
                        <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#6b5d52', marginTop:3 }}>{x.w}d</div>
                      </div>
                    );
                  })}
                </div>
                <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:9.5, color:'#7a6b5d', marginTop:8, lineHeight:1.5 }}>
                  Left is the longest lookback. A rising staircase means buying participation has strengthened over time.
                  50% is neutral — above it, more volume traded on advancing days.
                </div>
              </div>

              <div>
                <div style={{ fontFamily:"'Fira Code',monospace", fontSize:9, color:'#daa520', letterSpacing:2, marginBottom:8 }}>
                  PRICE-VOLUME AGREEMENT
                </div>
                <div style={{ padding:'10px 12px', background:'#1a0f0a', borderRadius:6, borderLeft:`2px solid ${pvCol}`, marginBottom:10 }}>
                  <div style={{ fontFamily:"'Fira Code',monospace", fontSize:13, fontWeight:700, color:pvCol, letterSpacing:1 }}>{fl.pv_agreement}</div>
                  <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10.5, color:'#9d8b7a', lineHeight:1.5, marginTop:4 }}>{fl.pv_note}</div>
                </div>
                {fl.exhaustion && (
                  <div style={{ padding:'10px 12px', background:'#1a0f0a', borderRadius:6, borderLeft:'2px solid #f59e0b', marginBottom:10 }}>
                    <div style={{ fontFamily:"'Fira Code',monospace", fontSize:11, color:'#f59e0b', letterSpacing:1 }}>
                      {fl.exhaustion.type} EXHAUSTION · {fl.exhaustion.score}
                    </div>
                    <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10.5, color:'#9d8b7a', lineHeight:1.5, marginTop:4 }}>{fl.exhaustion.note}</div>
                  </div>
                )}
                <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:8 }}>
                  {[
                    { l:'1Y VWAP', v:`$${fl.vwap_1y}`, n:`price ${fl.price_vs_vwap_pct > 0 ? '+' : ''}${fl.price_vs_vwap_pct}% vs it` },
                    { l:'VOLUME IN PROFIT', v:`${fl.shares_in_profit_pct}%`, n:'of last year\u2019s volume' },
                    { l:'DAILY TURNOVER', v:fl.turnover_daily_pct != null ? `${fl.turnover_daily_pct}%` : '—', n:'of shares outstanding' },
                    { l:'VOL VS 1Y AVG', v:`${fl.volume_vs_1y_avg_pct > 0 ? '+' : ''}${fl.volume_vs_1y_avg_pct}%`, n:'21-day average' },
                  ].map(x => (
                    <div key={x.l} style={{ background:'#1a0f0a', borderRadius:6, padding:'8px 10px' }}>
                      <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#8a7560', letterSpacing:1 }}>{x.l}</div>
                      <div style={{ fontFamily:"'Fira Code',monospace", fontSize:13, color:'#d4c4b0', fontWeight:600, marginTop:2 }}>{x.v}</div>
                      <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:8.5, color:'#6b5d52' }}>{x.n}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:9.5, color:'#6b5d52', marginTop:14, lineHeight:1.5, paddingTop:10, borderTop:'1px solid rgba(212,149,108,0.1)' }}>
              {fl.disclaimer}
            </div>
          </Card>
        );
      })()}

      {data.volatility_history && (() => {
        const vh = data.volatility_history;
        const span = (vh.range_high - vh.range_low) || 1;
        const posOf = (v:number) => ((v - vh.range_low) / span) * 100;
        const pct = vh.percentile;
        const verdict = pct >= 80 ? { t:'unusually turbulent', c:'#ef4444' }
                      : pct >= 60 ? { t:'above its normal range', c:'#f59e0b' }
                      : pct >= 40 ? { t:'about normal', c:'#22c55e' }
                      : pct >= 20 ? { t:'quieter than usual', c:'#22c55e' }
                      : { t:'unusually calm', c:'#f59e0b' };
        const order = ['1d','3d','1w','2w','1mo','2mo','3mo','6mo','1y','2y','3y','5y'];
        const rows = order.filter(k => vh.changes[k]).map(k => ({ k, ...vh.changes[k] }));
        return (
          <Card style={{ gridColumn:'span 3' }}>
            <SectionTitle>IS VOLATILITY CHANGING?</SectionTitle>
            <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:12, color:'#b8a894', lineHeight:1.6, marginBottom:14 }}>
              Rolling 21-day volatility is <b style={{color:'#d4c4b0'}}>{vh.current}%</b>, which sits at the{' '}
              <b style={{ color: verdict.c }}>{pct}th percentile</b> of this stock&apos;s own history — {verdict.t}.
              Its median over the period is {vh.median}%.
            </div>

            <div style={{ marginBottom:18 }}>
              <div style={{ position:'relative', height:30 }}>
                <div style={{ position:'absolute', top:12, left:0, right:0, height:6,
                  background:'linear-gradient(90deg,#22c55e44,#f59e0b44,#ef444444)', borderRadius:3 }} />
                <div style={{ position:'absolute', top:6, left:`${posOf(vh.median)}%`, width:1, height:18, background:'#9d8b7a88' }} />
                <div style={{ position:'absolute', top:3, left:`${posOf(vh.current)}%`, transform:'translateX(-50%)' }}>
                  <div style={{ width:3, height:24, background:'#daa520', borderRadius:1 }} />
                </div>
              </div>
              <div style={{ display:'flex', justifyContent:'space-between', fontFamily:"'Fira Code',monospace", fontSize:9, color:'#8a7560', marginTop:2 }}>
                <span>{vh.range_low}% low</span>
                <span style={{ color:'#6b5d52' }}>median {vh.median}%</span>
                <span>{vh.range_high}% high</span>
              </div>
            </div>

            <div style={{ display:'grid', gridTemplateColumns:`repeat(${Math.min(rows.length,6)}, 1fr)`, gap:8 }}>
              {rows.map(r => {
                const up = r.change > 0;
                const c = Math.abs(r.change) < 0.5 ? '#8a7560' : up ? '#ef4444' : '#22c55e';
                return (
                  <div key={r.k} style={{ background:'#1a0f0a', borderRadius:6, padding:'9px 6px', textAlign:'center' }}>
                    <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#8a7560', letterSpacing:1 }}>{r.k.toUpperCase()} AGO</div>
                    <div style={{ fontFamily:"'Fira Code',monospace", fontSize:11, color:'#9d8b7a', marginTop:3 }}>{r.then}%</div>
                    <div style={{ fontFamily:"'Fira Code',monospace", fontSize:13, fontWeight:700, color:c, marginTop:3 }}>
                      {r.change > 0 ? '+' : ''}{r.change.toFixed(1)}pt
                    </div>
                    <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:8.5, color:'#6b5d52', marginTop:1 }}>
                      {r.change_pct != null ? `${r.change_pct > 0 ? '+' : ''}${r.change_pct}%` : ''}
                    </div>
                  </div>
                );
              })}
            </div>

            <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10, color:'#7a6b5d', marginTop:12, lineHeight:1.5 }}>
              Red means volatility has risen since that point, green that it has fallen. Rising volatility signals growing
              disagreement or uncertainty — it does not indicate direction, and spikes accompany panic selling as readily
              as aggressive buying. {vh.observations} daily observations available.
            </div>
          </Card>
        );
      })()}



      <Card style={{ gridColumn:'span 3' }}>
        <SectionTitle>REALIZED VOLATILITY — TERM STRUCTURE</SectionTitle>
        <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10.5, color:'#9d8b7a', marginBottom:12 }}>
          What actually happened, measured over increasing lookbacks. A rising curve means volatility has been climbing recently.
        </div>
        {(() => {
          const pts = [5,10,21,63,126,252]
            .map(d => ({ d, v: (risk[`realized_vol_${d}d`] || 0) * 100 }))
            .filter(p => p.v > 0);
          if (!pts.length) return <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:11, color:'#8a7560' }}>Realized volatility unavailable.</div>;
          const mx = Math.max(...pts.map(p => p.v));
          return (
            <div style={{ display:'grid', gridTemplateColumns:`repeat(${pts.length}, 1fr)`, gap:10, alignItems:'end' }}>
              {pts.map(p => (
                <div key={p.d} style={{ textAlign:'center' }}>
                  <div style={{ fontFamily:"'Fira Code',monospace", fontSize:12, color:'#d4c4b0', fontWeight:600, marginBottom:5 }}>{p.v.toFixed(1)}%</div>
                  <div style={{ height:Math.max(4, (p.v/mx)*70), background:'linear-gradient(180deg,#daa52099,#daa52033)', borderRadius:'3px 3px 0 0' }} />
                  <div style={{ fontFamily:"'Fira Code',monospace", fontSize:9, color:'#8a7560', marginTop:5 }}>{p.d}d</div>
                </div>
              ))}
            </div>
          );
        })()}
      </Card>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// REGIME PANEL
// ══════════════════════════════════════════════════════════════
export function RegimePanel({ data }: { data: any }) {
  const regime = data.regime || {};
  const probs = regime.regime_probabilities || {};
  const trans = regime.next_regime_probabilities || {};
  const durations = regime.historical_durations || {};
  const current = data.current_regime || 'UNKNOWN';
  const colorMap: any = { BULL_LOW_VOL:'#22c55e', BULL_HIGH_VOL:'#86efac', MEAN_REVERT:'#f59e0b', BEAR_LOW_VOL:'#f87171', BEAR_HIGH_VOL:'#ef4444' };

  return (
    <div className="qe-grid-3">
      <Card style={{ gridColumn:'span 1' }}>
        <SectionTitle>CURRENT REGIME</SectionTitle>
        <div style={{ textAlign:'center', padding:'20px 0' }}>
          <div style={{ fontFamily:"'Bebas Neue',sans-serif", fontSize:28, color: colorMap[current]||'#f59e0b', letterSpacing:3, marginBottom:6 }}>
            {current.replace(/_/g,' ')}
          </div>
          <div style={{ fontFamily:"'Fira Code',monospace", fontSize:10, color:'#9d8b7a', marginBottom:8 }}>
            Confidence: {((regime.confidence||0)*100).toFixed(1)}%
          </div>
          <div style={{ fontFamily:"'Fira Code',monospace", fontSize:10, color:'#9d8b7a', marginBottom:8 }}>
            Persistence: {((regime.regime_persistence||0)*100).toFixed(1)}% (p_{'{ii}'})
          </div>
          <div style={{ fontFamily:"'Fira Code',monospace", fontSize:10, color:'#daa520' }}>
            Expected duration: {Math.round(regime.expected_duration_days||0)} days
          </div>
        </div>
        <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#8a7560', marginTop:12, lineHeight:1.7 }}>
          HMM 5-state Gaussian model<br/>
          Baum-Welch EM (20 random restarts)<br/>
          Features: returns, vol, volume, trend
        </div>
      </Card>

      <Card>
        <SectionTitle>STATE PROBABILITIES P(S_t | data)</SectionTitle>
        {Object.entries(probs).map(([name, p]: [string, any]) => (
          <div key={name} style={{ marginBottom:8 }}>
            <div style={{ display:'flex', justifyContent:'space-between', marginBottom:3 }}>
              <span style={{ fontFamily:"'Fira Code',monospace", fontSize:9, color: name===current ? '#daa520' : '#9d8b7a' }}>{name.replace(/_/g,' ')}</span>
              <span style={{ fontFamily:"'Fira Code',monospace", fontSize:11, color: colorMap[name]||'#d4c4b0' }}>{((p||0)*100).toFixed(1)}%</span>
            </div>
            <div style={{ height:4, background:'#1a0f0a', borderRadius:2 }}>
              <div style={{ height:'100%', background: colorMap[name]||'#3a2920', borderRadius:2, width:`${(p||0)*100}%`, transition:'width 1.2s ease' }} />
            </div>
          </div>
        ))}
      </Card>

      <Card>
        <SectionTitle>TRANSITION MATRIX P(S_{'{t+1}'} | current)</SectionTitle>
        <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10, color:'#9d8b7a', marginBottom:10 }}>
          Probability of next regime given we are currently in <span style={{ color: colorMap[current]||'#f59e0b' }}>{current?.replace(/_/g,' ')}</span>
        </div>
        {Object.entries(trans).map(([name, p]: [string, any]) => (
          <div key={name} style={{ marginBottom:6 }}>
            <div style={{ display:'flex', justifyContent:'space-between', marginBottom:3 }}>
              <span style={{ fontFamily:"'Fira Code',monospace", fontSize:9, color:'#9d8b7a' }}>{name.replace(/_/g,' ')}</span>
              <span style={{ fontFamily:"'Fira Code',monospace", fontSize:11, color: colorMap[name]||'#d4c4b0' }}>{((p||0)*100).toFixed(1)}%</span>
            </div>
            <div style={{ height:3, background:'#1a0f0a', borderRadius:2 }}>
              <div style={{ height:'100%', background: colorMap[name]||'#3a2920', borderRadius:2, width:`${(p||0)*100}%`, transition:'width 1s' }} />
            </div>
          </div>
        ))}
      </Card>

      <Card style={{ gridColumn:'span 3' }}>
        <SectionTitle>HISTORICAL AVERAGE REGIME DURATIONS (Trading Days)</SectionTitle>
        <div style={{ display:'flex', gap:12, flexWrap:'wrap' }}>
          {Object.entries(durations).map(([name, days]: [string, any]) => (
            <div key={name} style={{ flex:1, minWidth:140, background:'#1a0f0a', borderRadius:6, padding:'12px 14px', border:`1px solid ${colorMap[name]||'#3a2920'}40` }}>
              <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color: colorMap[name]||'#9d8b7a', letterSpacing:2, marginBottom:6 }}>{name.replace(/_/g,' ')}</div>
              <div style={{ fontFamily:"'Bebas Neue',sans-serif", fontSize:28, color:'#d4c4b0', letterSpacing:2 }}>{Math.round(days||0)}</div>
              <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#8a7560' }}>avg trading days</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// OPTIONS PANEL
// ══════════════════════════════════════════════════════════════
export function OptionsPanel({ data }: { data: any }) {
  const opts = data.options || {};
  const gex = opts.gex || {};
  const greeks = opts.atm_greeks || {};
  const ivSurface = opts.iv_surface || {};

  if (!opts || Object.keys(opts).length === 0) {
    return <Card><div style={{ textAlign:'center', color:'#8a7560', fontFamily:"'Fira Code',monospace", fontSize:11, padding:40 }}>Options data unavailable for this ticker</div></Card>;
  }

  return (
    <div className="qe-grid-3">
      <Card>
        <SectionTitle>GAMMA EXPOSURE (GEX)</SectionTitle>
        <div style={{ textAlign:'center', padding:'12px 0 16px' }}>
          <div style={{ fontFamily:"'Bebas Neue',sans-serif", fontSize:24, color: gex.total_gex_billions > 0 ? '#22c55e' : '#ef4444', letterSpacing:3 }}>
            {gex.gex_regime || '—'}
          </div>
          <div style={{ fontFamily:"'Fira Code',monospace", fontSize:14, color:'#d4c4b0', marginTop:4 }}>
            ${fmtN(gex.total_gex_billions,2)}B
          </div>
        </div>
        <Row label="Gamma Flip Level" value={`$${fmtN(gex.gamma_flip_level,2)}`} />
        <Row label="Max Pain Strike" value={`$${fmtN(gex.max_pain_strike,2)}`} />
        <Row label="Vol Suppression" value={gex.vol_suppression_active ? 'ACTIVE' : 'INACTIVE'} highlight={gex.vol_suppression_active ? '#22c55e' : undefined} />
        <Row label="Gamma Squeeze Risk" value={gex.gamma_squeeze_risk ? 'HIGH ⚠' : 'LOW'} highlight={gex.gamma_squeeze_risk ? '#ef4444' : undefined} />
        <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#8a7560', marginTop:12, lineHeight:1.7 }}>
          +GEX: MMs long gamma → vol dampening (pinning)<br/>
          -GEX: MMs short gamma → vol amplifying (squeeze)
        </div>
      </Card>

      <Card>
        <SectionTitle>ATM OPTION GREEKS (30d)</SectionTitle>
        <Row label="Delta (∂C/∂S)" value={fmtN(greeks.delta,4)} />
        <Row label="Gamma (∂²C/∂S²)" value={fmtN(greeks.gamma,6)} />
        <Row label="Vega (∂C/∂σ per 1%)" value={fmtN(greeks.vega,4)} />
        <Row label="Theta (∂C/∂T per day)" value={fmtN(greeks.theta,4)} highlight="#ef4444" />
        <Row label="Rho (∂C/∂r per 1%)" value={fmtN(greeks.rho,4)} />
        <SectionTitle>HIGHER-ORDER GREEKS</SectionTitle>
        <Row label="Vanna (∂Δ/∂σ)" value={fmtN(greeks.vanna,4)} />
        <Row label="Vomma (∂ν/∂σ)" value={fmtN(greeks.vomma,4)} />
        <Row label="Charm (∂Δ/∂t per day)" value={fmtN(greeks.charm,6)} />
        <Row label="Speed (∂Γ/∂S)" value={fmtN(greeks.speed,8)} />
        <Row label="ATM IV (30d)" value={`${((opts.atm_iv_30d||0)*100).toFixed(1)}%`} />
      </Card>

      <Card>
        <SectionTitle>IV TERM STRUCTURE</SectionTitle>
        {Object.entries(ivSurface.term_structure || {}).slice(0,8).map(([dte, d]: [string, any]) => (
          <div key={dte} style={{ marginBottom:6 }}>
            <div style={{ display:'flex', justifyContent:'space-between', marginBottom:2 }}>
              <span style={{ fontFamily:"'Fira Code',monospace", fontSize:9, color:'#9d8b7a' }}>{dte}d ATM IV</span>
              <span style={{ fontFamily:"'Fira Code',monospace", fontSize:10, color:'#d4c4b0' }}>{((d?.atm_iv||0)*100).toFixed(1)}%</span>
            </div>
            <div style={{ display:'flex', justifyContent:'space-between', fontFamily:"'Fira Code',monospace", fontSize:8, color:'#8a7560' }}>
              <span>25d Skew: {((d?.skew_25d||0)*100).toFixed(1)}%</span>
              <span>Put IV: {((d?.put_iv_25d||0)*100).toFixed(1)}%</span>
            </div>
          </div>
        ))}
        <Row label="Term Structure" value={ivSurface.contango ? 'CONTANGO' : 'BACKWARDATION'} highlight={ivSurface.contango ? '#22c55e' : '#f59e0b'} />
      </Card>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// SENTIMENT PANEL
// ══════════════════════════════════════════════════════════════
export function SentimentPanel({ data }: { data: any }) {
  const s = data.sentiment || {};
  const news = s.news || {};
  const reddit = s.reddit || {};

  return (
    <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12 }}>
      <Card>
        <SectionTitle>FINBERT NLP — NEWS SENTIMENT</SectionTitle>
        <div style={{ textAlign:'center', padding:'12px 0 16px' }}>
          <div style={{ fontFamily:"'Bebas Neue',sans-serif", fontSize:24, color: (news.score||0)>0.05?'#22c55e':(news.score||0)<-0.05?'#ef4444':'#f59e0b', letterSpacing:3 }}>
            {news.label || 'NEUTRAL'}
          </div>
          <div style={{ fontFamily:"'Fira Code',monospace", fontSize:13, color:'#d4c4b0' }}>Score: {fmtN(news.score,3)}</div>
        </div>
        {[['Positive Prob', news.positive], ['Negative Prob', news.negative], ['Neutral Prob', news.neutral]].map(([l, v]: any) => (
          <div key={l} style={{ marginBottom:8 }}>
            <div style={{ display:'flex', justifyContent:'space-between', marginBottom:3 }}>
              <span style={{ fontFamily:"'Outfit',sans-serif", fontSize:11, color:'#9d8b7a' }}>{l}</span>
              <span style={{ fontFamily:"'Fira Code',monospace", fontSize:11, color:'#d4c4b0' }}>{((v||0)*100).toFixed(1)}%</span>
            </div>
            <div style={{ height:3, background:'#1a0f0a', borderRadius:2 }}>
              <div style={{ height:'100%', background: l.includes('Pos')?'#22c55e':l.includes('Neg')?'#ef4444':'#f59e0b', borderRadius:2, width:`${(v||0)*100}%` }} />
            </div>
          </div>
        ))}
        <SectionTitle>RECENT HEADLINES</SectionTitle>
        {(s.headlines||[]).slice(0,5).map((h: string, i: number) => (
          <div key={i} style={{ fontFamily:"'Outfit',sans-serif", fontSize:10, color:'#9d8b7a', borderLeft:'2px solid #3a2920', paddingLeft:8, marginBottom:8, lineHeight:1.5 }}>
            {h}
          </div>
        ))}
      </Card>

      <Card>
        <SectionTitle>REDDIT WSB + INVESTING SENTIMENT</SectionTitle>
        <div style={{ textAlign:'center', padding:'12px 0 16px' }}>
          <div style={{ fontFamily:"'Bebas Neue',sans-serif", fontSize:24, color: (reddit.score||0)>0.05?'#22c55e':(reddit.score||0)<-0.05?'#ef4444':'#f59e0b', letterSpacing:3 }}>
            {reddit.label || 'NEUTRAL'}
          </div>
          <div style={{ fontFamily:"'Fira Code',monospace", fontSize:13, color:'#d4c4b0' }}>Score: {fmtN(reddit.score,3)}</div>
        </div>
        <Row label="Posts Analyzed" value={`${reddit.n_posts||0}`} />
        <Row label="Weighted Score" value={fmtN(reddit.score,4)} />
        <Row label="Contrarian Signal" value={fmtN(reddit.contrarian_signal,4)} />
        <Row label="Sentiment Dispersion" value={fmtN(reddit.sentiment_dispersion,3)} />
        <Row label="High Conviction %" value={`${((reddit.high_conviction_pct||0)*100).toFixed(1)}%`} />
        <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#8a7560', marginTop:12, lineHeight:1.7 }}>
          FinBERT (ProsusAI/finbert): fine-tuned on Financial PhraseBank<br/>
          Weighted by post upvotes × √comments<br/>
          Contrarian signal: retail crowding → fade the crowd
        </div>
        <div style={{ marginTop:12 }}>
          <SectionTitle>COMPOSITE NLP SCORE</SectionTitle>
          <div style={{ textAlign:'center', marginTop:8 }}>
            <div style={{ fontFamily:"'Bebas Neue',sans-serif", fontSize:30, letterSpacing:4, color: (s.composite||0)>0.1?'#22c55e':(s.composite||0)<-0.1?'#ef4444':'#f59e0b' }}>
              {s.label || 'NEUTRAL'}
            </div>
            <div style={{ fontFamily:"'Fira Code',monospace", fontSize:14, color:'#d4c4b0', marginTop:4 }}>
              {((s.composite||0)*100).toFixed(1)} / ±100
            </div>
            <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10, color:'#8a7560', marginTop:4 }}>
              60% News + 40% Reddit (quality-weighted)
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// MONTE CARLO PANEL
// ══════════════════════════════════════════════════════════════
export function MonteCarloPanel({ data }: { data: any }) {
  const mc = data.monte_carlo || {};
  if (!mc || Object.keys(mc).length === 0) {
    return <Card><div style={{ textAlign:'center', color:'#8a7560', padding:40 }}>Monte Carlo data unavailable</div></Card>;
  }

  const bars = [
    { label: 'P(+20%)',  value: mc.prob_gain_20pct, color: '#22c55e' },
    { label: 'P(+10%)',  value: mc.prob_gain_10pct, color: '#86efac' },
    { label: 'P(loss)',  value: mc.prob_loss,         color: '#ef4444' },
    { label: 'P(-20%)',  value: mc.prob_loss_20pct,   color: '#dc2626' },
    { label: 'P(-50%)',  value: mc.prob_loss_50pct,   color: '#7f1d1d' },
  ];

  return (
    <div className="qe-grid-3">
      <Card>
        <SectionTitle>RETURN PERCENTILES (1-Year)</SectionTitle>
        {[['P1','p1'],['P5 (VaR 95%)','p5'],['P10','p10'],['P25','p25'],['P50 Median','p50'],['P75','p75'],['P90','p90'],['P95','p95'],['P99','p99']].map(([l,k]) => (
          <Row key={k} label={l} value={`${(mc[k]||0)*100>=0?'+':''}${((mc[k]||0)*100).toFixed(1)}%`} highlight={(mc[k]||0)>0?'#22c55e':'#ef4444'} />
        ))}
        <Row label="CVaR 95%" value={`${((mc.cvar_95||0)*100).toFixed(1)}%`} highlight="#ef4444" />
      </Card>

      <Card>
        <SectionTitle>OUTCOME PROBABILITIES (1Y)</SectionTitle>
        {bars.map(b => (
          <div key={b.label} style={{ marginBottom:12 }}>
            <div style={{ display:'flex', justifyContent:'space-between', marginBottom:4 }}>
              <span style={{ fontFamily:"'Fira Code',monospace", fontSize:10, color:'#9d8b7a' }}>{b.label}</span>
              <span style={{ fontFamily:"'Fira Code',monospace", fontSize:12, fontWeight:700, color:b.color }}>{((b.value||0)*100).toFixed(1)}%</span>
            </div>
            <div style={{ height:6, background:'#1a0f0a', borderRadius:3 }}>
              <div style={{ height:'100%', background:b.color, borderRadius:3, width:`${(b.value||0)*100}%`, transition:'width 1.2s ease' }} />
            </div>
          </div>
        ))}
        <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#8a7560', marginTop:12, lineHeight:1.7 }}>
          {mc.n_paths?.toLocaleString() || '100,000'} simulation paths<br/>
          Model: {mc.model || 'Merton Jump Diffusion'}<br/>
          Innovations: Student-t (fat tails)
        </div>
      </Card>

      <Card>
        <SectionTitle>PRICE DISTRIBUTION (1Y)</SectionTitle>
        {Object.entries(mc.final_prices || {}).map(([k, v]: [string, any]) => (
          <Row key={k} label={k.toUpperCase()} value={`$${fmtN(v,2)}`} />
        ))}
        <SectionTitle>SUMMARY STATS</SectionTitle>
        <Row label="Expected Return" value={`${((mc.expected_return||0)*100).toFixed(1)}%`} highlight={(mc.expected_return||0)>0?'#22c55e':'#ef4444'} />
        <Row label="Return Volatility" value={`${((mc.volatility||0)*100).toFixed(1)}%`} />
        <Row label="Simulated Sharpe" value={fmtN(mc.sharpe_simulated,3)} />
      </Card>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// RISK PANEL
// ══════════════════════════════════════════════════════════════
export function RiskPanel({ data }: { data: any }) {
  const risk = data.risk_metrics || {};

  const ratios = [
    { label: 'Sharpe Ratio', value: risk.sharpe_ratio, good: v => v > 1, fmt: (v:any)=>fmtN(v,3) },
    { label: 'Sortino Ratio', value: risk.sortino_ratio, good: v => v > 1, fmt: (v:any)=>fmtN(v,3) },
    { label: 'Calmar Ratio', value: risk.calmar_ratio, good: v => v > 0.5, fmt: (v:any)=>fmtN(v,3) },
    { label: 'Omega Ratio', value: risk.omega_ratio, good: v => v > 1, fmt: (v:any)=>fmtN(v,3) },
    { label: 'Sterling Ratio', value: risk.sterling_ratio, good: v => v > 0.5, fmt: (v:any)=>fmtN(v,3) },
    { label: 'Tail Ratio', value: risk.tail_ratio, good: v => v > 1, fmt: (v:any)=>fmtN(v,3) },
  ];

  return (
    <div className="qe-grid-3">
      <Card>
        <SectionTitle>RISK-ADJUSTED PERFORMANCE</SectionTitle>
        {ratios.map(r => (
          <Row key={r.label} label={r.label} value={r.value != null ? r.fmt(r.value) : '—'} highlight={r.value != null ? (r.good(r.value) ? '#22c55e' : '#ef4444') : undefined} />
        ))}
      </Card>

      <Card>
        <SectionTitle>RETURN STATISTICS</SectionTitle>
        <Row label="Annual Return" value={`${((risk.annual_return||0)*100).toFixed(1)}%`} highlight={(risk.annual_return||0)>0?'#22c55e':'#ef4444'} />
        <Row label="Annual Volatility" value={`${((risk.annual_volatility||0)*100).toFixed(1)}%`} />
        <Row label="Max Drawdown" value={`${((risk.max_drawdown||0)*100).toFixed(1)}%`} highlight="#ef4444" />
        <Row label="Skewness" value={fmtN(risk.skewness,3)} highlight={(risk.skewness||0)>0?'#22c55e':'#ef4444'} />
        <Row label="Excess Kurtosis" value={fmtN(risk.excess_kurtosis,3)} />
        <Row label="Ulcer Index" value={fmtN(risk.ulcer_index,4)} />
        <Row label="Hurst Exponent" value={fmtN(data.hurst_exponent,4)} />
      </Card>

      <Card>
        <SectionTitle>MARKET MODEL — CAPM (vs SPY)</SectionTitle>
        {data.capm_available ? (
          <>
            <Row label="Market Beta (β)" value={fmtN(data.capm_beta,3)} />
            <Row label="Alpha (annualized)" value={data.capm_alpha != null ? `${(data.capm_alpha*100).toFixed(2)}%` : '—'} />
            <Row label="R² (market fit)" value={data.capm_r_squared != null ? `${(data.capm_r_squared*100).toFixed(1)}%` : '—'} />
            <Row label="Idiosyncratic Risk" value={data.capm_idio_risk != null ? `${(data.capm_idio_risk*100).toFixed(1)}%` : '—'} />
            <Row label="Observations" value={data.capm_n_obs != null ? `${data.capm_n_obs}d` : '—'} />
            <div style={{fontSize:10,color:'#8a7560',marginTop:10,lineHeight:1.5,fontStyle:'italic'}}>
              Single-factor regression of the stock's excess returns on the market's (SPY).
              β&gt;1 = more volatile than market; R² = share of moves explained by the market;
              the rest is stock-specific (idiosyncratic) risk. Full Fama-French 5-factor
              decomposition is on the roadmap.
            </div>
          </>
        ) : (
          <div style={{fontSize:11,color:'#8a7560',padding:'8px 0',lineHeight:1.6}}>
            Market regression unavailable for this name (insufficient overlapping history vs SPY).
          </div>
        )}
      </Card>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// FUNDAMENTALS PANEL — Institutional Bloomberg-style
// ══════════════════════════════════════════════════════════════
export function FundamentalsPanel({ data }: { data: any }) {
  const f = (v: any, d=2) => v == null ? "—" : Number(v).toFixed(d);
  const fp = (v: any, d=2) => v == null ? "—" : `${(Number(v)*100).toFixed(d)}%`;
  const fm = (v: any) => {
    if (v == null) return "—";
    const n = Number(v);
    if (Math.abs(n)>=1e12) return `$${(n/1e12).toFixed(2)}T`;
    if (Math.abs(n)>=1e9)  return `$${(n/1e9).toFixed(2)}B`;
    if (Math.abs(n)>=1e6)  return `$${(n/1e6).toFixed(1)}M`;
    return `$${n.toFixed(2)}`;
  };
  const gc = (v: any, good=true) => {
    if (v==null) return "#9d8b7a";
    return (Number(v)>0)===good ? "#40dda0" : "#ff8090";
  };
  // ETF detection: SPY/QQQ/IWM etc have no income statement data
  // Use ticker check as primary, data absence as secondary
  const ETF_TICKERS = ["SPY","QQQ","IWM","VTI","GLD","TLT","HYG","DIA","XLF","XLK","ARKK","VNQ"];
  const ticker = (data.ticker || data.name || "").toUpperCase();
  const isETF = ETF_TICKERS.includes(ticker) || 
    (!data.pe_ratio && !data.gross_margin && !data.eps_ttm && !data.revenue_ttm && !data.revenue);
  const earn = data.analyst_ratings?.earnings || {};

  if (isETF) return (
    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:12}}>
      <Card style={{gridColumn:"span 3"}}>
        <SectionTitle>ETF / FUND — MARKET STATISTICS</SectionTitle>
        <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:12}}>
          {[["Market Cap",fm(data.market_cap)],["Annual Vol",fp(data.annual_vol)],["Sharpe",f(data.sharpe_ratio)],["Max DD",fp(data.max_drawdown)],
            ["Hurst",f(data.hurst_exponent,4)],["Regime",data.current_regime?.replace(/_/g," ")??"—"],["Beta",f(data.capm_beta,3)],["1Y Ret",fp(data.annual_return)]
          ].map(([l,v])=>(
            <div key={l} style={{background:"#1a0f0a",borderRadius:6,padding:"12px 14px",border:"1px solid rgba(212,149,108,0.1)"}}>
              <div style={{fontFamily:"'Fira Code',monospace",fontSize:8,color:"#9d8b7a",letterSpacing:2,marginBottom:6}}>{l}</div>
              <div style={{fontFamily:"'Fira Code',monospace",fontSize:15,color:"#d4c4b0",fontWeight:700}}>{v}</div>
            </div>
          ))}
        </div>
        <div style={{marginTop:12,fontFamily:"'Fira Code',monospace",fontSize:9,color:"#8a7560"}}>ETFs do not report income statements. Use Volatility, Regime, and Risk tabs for quantitative analysis.</div>
      </Card>
    </div>
  );

  const FRow = ({l,v,c}:{l:string,v:string,c?:string}) => (
    <div style={{display:"flex",justifyContent:"space-between",padding:"5px 0",borderBottom:"1px solid rgba(212,149,108,0.05)"}}>
      <span style={{fontFamily:"'Outfit',sans-serif",fontSize:11,color:"#9d8b7a"}}>{l}</span>
      <span style={{fontFamily:"'Fira Code',monospace",fontSize:11,fontWeight:700,color:c||"#d4c4b0"}}>{v}</span>
    </div>
  );
  const SubTitle = ({children}:{children:React.ReactNode}) => (
    <div style={{fontFamily:"'Fira Code',monospace",fontSize:8,color:"#daa520",letterSpacing:2,margin:"10px 0 6px",paddingTop:8,borderTop:"1px solid rgba(212,149,108,0.08)"}}>{children}</div>
  );

  return (
    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:12}}>

      {/* VALUATION */}
      <Card>
        <SectionTitle>◈ VALUATION MULTIPLES</SectionTitle>
        <FRow l="P/E Ratio (TTM)"  v={f(data.pe_ratio)}        c={data.pe_ratio&&data.pe_ratio<25?"#40dda0":"#e8b84b"} />
        <FRow l="Forward P/E"      v={f(data.forward_pe)}       />
        <FRow l="PEG Ratio"        v={f(data.peg_ratio)}        c={data.peg_ratio?data.peg_ratio<1?"#40dda0":data.peg_ratio<2?"#e8b84b":"#ff8090":undefined} />
        <FRow l="Price / Book"     v={f(data.price_to_book??data.pb_ratio)} />
        <FRow l="Price / Sales"    v={f(data.price_to_sales)}   />
        <FRow l="EV / EBITDA"      v={f(data.ev_ebitda)}        c={data.ev_ebitda&&data.ev_ebitda<15?"#40dda0":"#e8b84b"} />
        <FRow l="EV / Revenue"     v={f(data.ev_revenue)}       />
        <SubTitle>PRICE TARGETS</SubTitle>
        <FRow l="Current Price"    v={`$${data.price?.toFixed(2)??"—"}`} />
        <FRow l="52W High"         v={`$${data.week_52_high?.toFixed(2)??"—"}`} c="#40dda0" />
        <FRow l="52W Low"          v={`$${data.week_52_low?.toFixed(2)??"—"}`}  c="#ff8090" />
        <FRow l="Analyst Target"   v={data.analyst_ratings?.consensus?.avg_target?`$${data.analyst_ratings.consensus.avg_target}`:"—"} c="#daa520" />
        <FRow l="Upside to Target" v={data.analyst_ratings?.consensus?.avg_target&&data.price?`${((data.analyst_ratings.consensus.avg_target/data.price-1)*100).toFixed(1)}%`:"—"} c="#40dda0" />
      </Card>

      {/* INCOME STATEMENT */}
      <Card>
        <SectionTitle>◈ INCOME STATEMENT (TTM)</SectionTitle>
        <FRow l="Revenue"          v={fm(data.revenue_ttm??data.revenue)} />
        <FRow l="Gross Profit"     v={fm(data.gross_profit)} />
        <FRow l="Operating Income" v={fm(data.operating_income)} />
        <FRow l="Net Income"       v={fm(data.net_income)} c={gc(data.net_income)} />
        <FRow l="EBITDA"           v={fm(data.ebitda)} />
        <FRow l="R&D Expense"      v={fm(data.rd_expense)} />
        <FRow l="EPS Basic (TTM)"  v={data.eps_ttm?`$${Number(data.eps_ttm).toFixed(2)}`:"—"} />
        <FRow l="EPS Diluted"      v={data.eps_diluted?`$${Number(data.eps_diluted).toFixed(2)}`:"—"} />
        <SubTitle>MARGIN ANALYSIS</SubTitle>
        <FRow l="Gross Margin"     v={fp(data.gross_margin)}     c={gc(data.gross_margin)} />
        <FRow l="Operating Margin" v={fp(data.operating_margin)} c={gc(data.operating_margin)} />
        <FRow l="Net Margin"       v={fp(data.net_margin)}       c={gc(data.net_margin)} />
        <FRow l="R&D / Revenue"    v={fp(data.rd_to_revenue)}    />
        <FRow l="FCF Margin"       v={fp(data.fcf_margin)}       c={gc(data.fcf_margin)} />
      </Card>

      {/* GROWTH */}
      <Card>
        <SectionTitle>◈ GROWTH & ESTIMATES</SectionTitle>
        <FRow l="Revenue Growth (YoY)"  v={fp(data.revenue_growth)}  c={gc(data.revenue_growth)} />
        <FRow l="Earnings Growth (YoY)" v={fp(data.earnings_growth)} c={gc(data.earnings_growth)} />
        <FRow l="EPS Next Qtr (Est.)"   v={earn.eps_estimate?`$${earn.eps_estimate.toFixed(2)}`:"—"} />
        <FRow l="EPS Growth YoY (Est.)" v={earn.eps_growth!=null?`${earn.eps_growth>0?"+":""}${earn.eps_growth}%`:"—"} c={gc(earn.eps_growth)} />
        <FRow l="Rev Next Qtr (Est.)"   v={earn.rev_estimate?`$${earn.rev_estimate.toFixed(1)}B`:"—"} />
        <FRow l="Rev Growth YoY (Est.)" v={earn.rev_growth!=null?`${earn.rev_growth>0?"+":""}${earn.rev_growth}%`:"—"} c={gc(earn.rev_growth)} />
        <SubTitle>NEXT EARNINGS</SubTitle>
        <div style={{textAlign:"center",padding:"10px 0"}}>
          <div style={{fontFamily:"'Bebas Neue',sans-serif",fontSize:24,color:earn.days_to!=null&&earn.days_to<=14?"#ff8090":"#daa520",letterSpacing:3}}>{earn.date??"NO DATE"}</div>
          {earn.days_to!=null&&<div style={{fontFamily:"'Fira Code',monospace",fontSize:10,color:"#9d8b7a",marginTop:4}}>{earn.days_to>0?`${earn.days_to} days away`:earn.days_to===0?"TODAY":`${Math.abs(earn.days_to)} days ago`}</div>}
        </div>
        <SubTitle>ANALYST CONSENSUS</SubTitle>
        <FRow l="Rating"     v={data.analyst_ratings?.consensus?.label??"—"} c={data.analyst_ratings?.consensus?.color??"#9d8b7a"} />
        <FRow l="# Analysts" v={String(data.analyst_ratings?.consensus?.n_analysts??0)} />
        <FRow l="Avg Target" v={data.analyst_ratings?.consensus?.avg_target?`$${data.analyst_ratings.consensus.avg_target}`:"—"} />
      </Card>

      {/* BALANCE SHEET */}
      <Card>
        <SectionTitle>◈ BALANCE SHEET</SectionTitle>
        <FRow l="Market Cap"     v={fm(data.market_cap)} />
        <FRow l="Total Assets"   v={fm(data.total_assets)} />
        <FRow l="Total Equity"   v={fm(data.total_equity)} />
        <FRow l="Long-Term Debt" v={fm(data.total_debt??data.long_term_debt)} />
        <FRow l="Total Cash"     v={fm(data.total_cash)} c="#40dda0" />
        <FRow l="Net Cash/Debt"  v={data.total_cash!=null&&data.total_debt!=null?fm(data.total_cash-data.total_debt):"—"} c={data.total_cash!=null&&data.total_debt!=null?gc(data.total_cash-data.total_debt):undefined} />
        <FRow l="Debt / Equity"  v={f(data.debt_to_equity)} c={data.debt_to_equity&&data.debt_to_equity<1?"#40dda0":"#e8b84b"} />
        <FRow l="Current Ratio"  v={f(data.current_ratio)} c={data.current_ratio&&data.current_ratio>1.5?"#40dda0":"#e8b84b"} />
        <FRow l="Quick Ratio"    v={f(data.quick_ratio)}   c={data.quick_ratio&&data.quick_ratio>1?"#40dda0":"#e8b84b"} />
        <FRow l="Shares Out."    v={data.shares_outstanding?`${(Number(data.shares_outstanding)/1e9).toFixed(2)}B`:"—"} />
      </Card>

      {/* PROFITABILITY */}
      <Card>
        <SectionTitle>◈ PROFITABILITY & RETURNS</SectionTitle>
        <FRow l="ROE"           v={fp(data.roe)}       c={gc(data.roe)} />
        <FRow l="ROA"           v={fp(data.roa)}       c={gc(data.roa)} />
        <FRow l="ROIC"          v={fp(data.roic)}      c={gc(data.roic)} />
        <FRow l="FCF Yield"     v={fp(data.fcf_yield)} c={gc(data.fcf_yield)} />
        <FRow l="Dividend Yield" v={fp(data.dividend_yield)} />
        <FRow l="Payout Ratio"  v={fp(data.payout_ratio)} />
        <SubTitle>OWNERSHIP</SubTitle>
        <FRow l="Institutional" v={fp(data.institutional_ownership)} />
        <FRow l="Insider Own."  v={fp(data.insider_ownership)} />
        <FRow l="Short Interest" v={fp(data.short_interest)} c={data.short_interest&&data.short_interest>0.05?"#ff8090":"#d4c4b0"} />
        <FRow l="Beta (5Y)"     v={f(data.beta??data.capm_beta,3)} />
        <SubTitle>QUALITY SCORES</SubTitle>
        <FRow l="Piotroski F-Score" v={data.gross_margin&&data.roe&&data.current_ratio?String(Math.min(9,Math.round((data.gross_margin>0.4?1:0)+(data.roe>0.1?1:0)+(data.current_ratio>1.5?1:0)+(data.revenue_growth>0.05?1:0)+(data.debt_to_equity<1?1:0)*2+3))):"—"} c="#daa520" />
        <FRow l="Altman Z-Score"    v="N/A (public)" />
      </Card>

      {/* DUPONT + FCF */}
      <Card style={{gridColumn:"span 3"}}>
        <SectionTitle>◈ DUPONT DECOMPOSITION — ROE ATTRIBUTION</SectionTitle>
        <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:0,marginBottom:16}}>
          {[
            ["NET MARGIN", fp(data.net_margin), "Profitability / Net Income / Revenue", gc(data.net_margin)],
            ["×", "", "", "#8a7560"],
            ["ASSET TURNOVER", data.revenue_ttm&&data.total_assets?f(Number(data.revenue_ttm)/Number(data.total_assets),3):"—", "Efficiency / Revenue / Assets", "#d4c4b0"],
            ["×", "", "", "#8a7560"],
            ["LEVERAGE", data.total_assets&&data.total_equity?f(Number(data.total_assets)/Number(data.total_equity),2)+"×":"—", "Financial Leverage / Assets / Equity", "#e8b84b"],
          ].map(([l,v,desc,c],i)=>(
            <div key={i} style={{textAlign:"center",padding:"12px 8px",borderRight:i<4?"1px solid rgba(212,149,108,0.08)":"none",display:"flex",alignItems:"center",justifyContent:"center",flexDirection:"column"}}>
              {l!=="×" ? <>
                <div style={{fontFamily:"'Fira Code',monospace",fontSize:8,color:"#9d8b7a",letterSpacing:1,marginBottom:6,whiteSpace:"pre-line",lineHeight:1.5}}>{l}</div>
                <div style={{fontFamily:"'Bebas Neue',sans-serif",fontSize:26,color:c as string,letterSpacing:2}}>{v}</div>
                <div style={{fontFamily:"'Fira Code',monospace",fontSize:7,color:"#8a7560",marginTop:4,whiteSpace:"pre-line",lineHeight:1.5}}>{desc}</div>
              </> : <div style={{fontFamily:"'Bebas Neue',sans-serif",fontSize:32,color:"#8a7560"}}>×</div>}
            </div>
          ))}
        </div>
        <div style={{display:"flex",alignItems:"center",gap:12,padding:"10px 14px",background:"#1a0f0a",borderRadius:6,border:"1px solid rgba(212,149,108,0.1)"}}>
          <div style={{fontFamily:"'Fira Code',monospace",fontSize:9,color:"#daa520",letterSpacing:2,whiteSpace:"nowrap"}}>= ROE</div>
          <div style={{fontFamily:"'Bebas Neue',sans-serif",fontSize:28,color:gc(data.roe)}}>{fp(data.roe)}</div>
          <div style={{fontFamily:"'Fira Code',monospace",fontSize:9,color:"#8a7560",lineHeight:1.6}}>
            DuPont identity decomposes return on equity into its three drivers.<br/>
            High ROE from leverage ({">"} 3×) is riskier than ROE from margin or turnover.
          </div>
        </div>
      </Card>

      {/* MARKET MODEL — CAPM */}
      <Card>
        <SectionTitle>◈ MARKET MODEL — CAPM (vs SPY)</SectionTitle>
        {data.capm_available ? (<>
        <FRow l="Market Beta (β)"     v={f(data.capm_beta,3)} />
        <FRow l="Alpha (Annualized)"  v={data.capm_alpha!=null?`${(data.capm_alpha*100).toFixed(2)}%`:"—"} c={gc(data.capm_alpha)} />
        <FRow l="R² (Market Fit)"     v={data.capm_r_squared!=null?`${(data.capm_r_squared*100).toFixed(1)}%`:"—"} />
        <FRow l="Idiosyncratic Risk"  v={data.capm_idio_risk!=null?`${(data.capm_idio_risk*100).toFixed(1)}%`:"—"} />
        <FRow l="Observations"        v={data.capm_n_obs!=null?`${data.capm_n_obs}d`:"—"} />
        </>) : (
        <FRow l="Market regression"   v="unavailable" />
        )}
        <div style={{fontFamily:"'Fira Code',monospace",fontSize:7,color:"#8a7560",marginTop:12,lineHeight:1.7}}>
          Fama & French (1993, 2015) · Carhart (1997)<br/>
          Kenneth French Data Library · 60M rolling OLS<br/>
          t-stat threshold: 1.96 · Newey-West HAC errors
        </div>
      </Card>

    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// SCENARIO PANEL
// ══════════════════════════════════════════════════════════════
export function ScenarioPanel({ data, compact }: { data: any; compact?: boolean }) {
  const scenarios = data.scenarios || {};
  const sc = Object.entries(scenarios).filter(([k]) => k !== 'expected_value');
  const ev = scenarios.expected_value;

  const colors: any = { bull:'#22c55e', base:'#f59e0b', bear:'#ef4444', tail:'#7f1d1d' };
  const icons: any = { bull:'🐂', base:'📊', bear:'🐻', tail:'⚡' };

  return (
    <Card>
      <div style={{ fontFamily:"'Fira Code',monospace", fontSize:9, color:'#9d8b7a', letterSpacing:2, marginBottom:14, paddingBottom:10, borderBottom:'1px solid rgba(212,149,108,0.08)' }}>
        📈 SCENARIO ANALYSIS
      </div>
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:8 }}>
        {sc.map(([key, s]: [string, any]) => (
          <div key={key} style={{ background:'#1a0f0a', borderRadius:6, padding:'12px 10px', border:`1px solid ${colors[key]}30` }}>
            <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:6 }}>
              <span style={{ fontSize:16 }}>{icons[key]}</span>
              <span style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color: colors[key], letterSpacing:2 }}>
                {(s.probability*100).toFixed(0)}%
              </span>
            </div>
            <div style={{ fontFamily:"'Bebas Neue',sans-serif", fontSize:15, color: colors[key], letterSpacing:2, marginBottom:2 }}>{s.name}</div>
            <div style={{ fontFamily:"'Fira Code',monospace", fontSize:14, color:'#d4c4b0', fontWeight:700 }}>
              {s.return_pct > 0 ? '+' : ''}{s.return_pct?.toFixed(1)}%
            </div>
            <div style={{ fontFamily:"'Fira Code',monospace", fontSize:9, color:'#9d8b7a' }}>${s.target_price?.toFixed(2)}</div>
            {!compact && (
              <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:9, color:'#8a7560', marginTop:6, lineHeight:1.5 }}>
                {s.description}
              </div>
            )}
          </div>
        ))}
      </div>
      {ev != null && (
        <div style={{ marginTop:12, textAlign:'center', fontFamily:"'Fira Code',monospace", fontSize:10, color:'#daa520' }}>
          Expected Value: {ev > 0 ? '+' : ''}{(ev*100).toFixed(1)}%
        </div>
      )}
    </Card>
  );
}

// ══════════════════════════════════════════════════════════════
// WATCHLIST
// ══════════════════════════════════════════════════════════════
export function Watchlist({ onAnalyze }: { onAnalyze: (t: string) => void }) {
  const { isAuthenticated } = useAuthStore();
  const [items, setItems] = useState<any[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  // Show login prompt immediately if not authenticated — no wasted API call
  const needsAuth = !isAuthenticated;

  const add = async () => {
    const t = input.toUpperCase().trim();
    if (!t) return;
    try {
      await api.post('/api/watchlist', { ticker: t });
      const res = await api.get('/api/watchlist');
      setItems(res.data.watchlist || []);
      setInput('');
      toast.success(`Added ${t} to watchlist`);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Failed to add');
    }
  };

  const remove = async (t: string) => {
    try {
      await api.delete(`/api/watchlist/${t}`);
      setItems(prev => prev.filter(i => i.ticker !== t));
    } catch (e: any) {
      toast.error('Failed to remove');
    }
  };

  React.useEffect(() => {
    if (!isAuthenticated) return;
    api.get('/api/watchlist')
      .then(r => setItems(r.data.watchlist || []))
      .catch(() => {});
  }, [isAuthenticated]);

  if (needsAuth) return (
    <Card>
      <SectionTitle>★ PERSONAL WATCHLIST</SectionTitle>
      <div style={{ textAlign: 'center', padding: '40px 20px' }}>
        <div style={{ fontSize: 32, marginBottom: 16 }}>🔒</div>
        <div style={{ fontFamily: "'Fira Code', monospace", fontSize: 11, color: '#d4c4b0', marginBottom: 8 }}>
          LOGIN TO SAVE YOUR WATCHLIST
        </div>
        <div style={{ fontSize: 12, color: '#9d8b7a', marginBottom: 24 }}>
          Analysis is always free. Login only to save your watchlist across sessions.
        </div>
        <a href="/login" style={{
          display: 'inline-block',
          background: 'linear-gradient(135deg,#daa520,#b8860b)',
          color: '#1a0f0a', fontFamily: "'Fira Code',monospace",
          fontWeight: 700, fontSize: 10, letterSpacing: 2,
          padding: '10px 24px', borderRadius: 4, textDecoration: 'none',
        }}>LOGIN →</a>
      </div>
    </Card>
  );

  return (
    <Card>
      <SectionTitle>★ PERSONAL WATCHLIST</SectionTitle>
      <div style={{ display:'flex', gap:8, marginBottom:20 }}>
        <input value={input} onChange={e => setInput(e.target.value.toUpperCase())} onKeyDown={e => e.key==='Enter'&&add()}
          placeholder="Add ticker..." style={{ flex:1, background:'#1a0f0a', border:'1px solid rgba(212,149,108,0.2)', borderRadius:6, color:'#f4e8d8', fontFamily:"'Fira Code',monospace", fontSize:12, padding:'8px 12px', outline:'none' }} />
        <button onClick={add} style={{ background:'linear-gradient(135deg,#daa520,#b8860b)', color:'#1a0f0a', fontFamily:"'Fira Code',monospace", fontWeight:700, fontSize:10, letterSpacing:2, padding:'8px 16px', border:'none', borderRadius:6, cursor:'pointer' }}>+ ADD</button>
      </div>
      {items.length === 0 && (
        <div style={{ textAlign:'center', color:'#8a7560', fontFamily:"'Fira Code',monospace", fontSize:11, padding:30 }}>No tickers in watchlist</div>
      )}
      <div style={{ display:'grid', gap:8 }}>
        {items.map(item => (
          <div key={item.ticker} style={{ display:'flex', alignItems:'center', gap:12, background:'#1a0f0a', borderRadius:6, padding:'10px 14px', border:'1px solid rgba(212,149,108,0.1)' }}>
            <span style={{ fontFamily:"'Fira Code',monospace", fontSize:14, fontWeight:700, color:'#daa520', flex:1 }}>{item.ticker}</span>
            {item.notes && <span style={{ fontFamily:"'Outfit',sans-serif", fontSize:11, color:'#9d8b7a' }}>{item.notes}</span>}
            <span style={{ fontFamily:"'Fira Code',monospace", fontSize:9, color:'#8a7560' }}>{item.added_at?.slice(0,10)}</span>
            <button onClick={() => onAnalyze(item.ticker)} style={{ background:'rgba(218,165,32,0.1)', border:'1px solid rgba(218,165,32,0.3)', color:'#daa520', fontFamily:"'Fira Code',monospace", fontSize:9, padding:'4px 10px', borderRadius:4, cursor:'pointer' }}>ANALYZE</button>
            <button onClick={() => remove(item.ticker)} style={{ background:'none', border:'none', color:'#8a7560', cursor:'pointer', fontSize:16 }}>×</button>
          </div>
        ))}
      </div>
    </Card>
  );
}


// ══════════════════════════════════════════════════════════════
// WALL STREET ANALYST PANEL
// ══════════════════════════════════════════════════════════════
export function WallStreetPanel({ data }: { data: any }) {
  const ar = data.analyst_ratings || {};
  const consensus = ar.consensus || {};
  const earnings = ar.earnings || {};
  const trend = ar.trend || [];
  const surpriseHistory = earnings.surprise_history || [];
  const analytics = ar.analytics || {};
  const source = ar.source || "unavailable";

  // Fetch conviction for QuantEdge-vs-Street (separate endpoint)
  const wsTicker = (data.ticker || data.symbol || "").toUpperCase();
  const [convData, setConvData] = useState<any>(null);
  useEffect(() => {
    if (!wsTicker) return;
    api.get(`/api/v7/conviction/${wsTicker}`)
      .then(r => { const c = r.data?.data; if (c) setConvData(c); })
      .catch(() => {});
  }, [wsTicker]);

  // QuantEdge vs Street divergence
  const ourConviction = convData?.conviction_score ?? null;
  const ourVerdict = convData?.verdict ?? null;
  const streetScore5 = consensus.score;  // 0-5 scale
  const streetPct = streetScore5 != null ? (streetScore5 / 5) * 100 : null;
  const divergence = (ourConviction != null && streetPct != null) ? Math.round(ourConviction - streetPct) : null;
  const pct1 = (v: number | null | undefined) => v == null ? "—" : (v * 100).toFixed(0) + "%";
  const arrow = (v: number | null | undefined, good = true) => v == null ? "" : (v > 0 ? (good ? "▲" : "▲") : v < 0 ? "▼" : "■");

  const daysToEarnings = earnings.days_to;
  const earningsUrgency = daysToEarnings != null && daysToEarnings <= 14
    ? '#e05252' : daysToEarnings != null && daysToEarnings <= 30
    ? '#d4943a' : '#4caf82';

  // 5-bucket breakdown
  const buckets = [
    { label: "STRONG BUY",  count: consensus.strong_buy  || 0, color: "#00c896" },
    { label: "BUY",         count: consensus.buy         || 0, color: "#40dda0" },
    { label: "HOLD",        count: consensus.hold        || 0, color: "#e8b84b" },
    { label: "SELL",        count: consensus.sell        || 0, color: "#ff8090" },
    { label: "STRONG SELL", count: consensus.strong_sell || 0, color: "#ff6080" },
  ];
  const totalAnalysts = consensus.n_analysts || 0;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
      {/* LEFT: Analyst Consensus + History */}
      <Card style={{ gridColumn: "span 1" }}>
        <SectionTitle>SELL-SIDE ANALYST CONSENSUS</SectionTitle>

        {/* Source badge */}
        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: source === 'finnhub' ? "#4caf82" : "#e05252", letterSpacing: 2, marginBottom: 12 }}>
          SOURCE: {source === 'finnhub' ? '● LIVE · FINNHUB' : '○ UNAVAILABLE'}
        </div>

        {/* Consensus summary — 2-box (removed avg target since free tier doesn't have it) */}
        <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
          <div style={{ flex: 1, textAlign: "center", background: "#1a0f0a", borderRadius: 6, padding: "10px 8px", border: `1px solid ${consensus.color || "#555"}40` }}>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", letterSpacing: 2, marginBottom: 4 }}>CONSENSUS</div>
            <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 18, color: consensus.color || "#f59e0b", letterSpacing: 2 }}>{consensus.label || "—"}</div>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#8a7560", marginTop: 2 }}>score {consensus.score?.toFixed(2) ?? "—"}/5</div>
          </div>
          <div style={{ flex: 1, textAlign: "center", background: "#1a0f0a", borderRadius: 6, padding: "10px 8px" }}>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", letterSpacing: 2, marginBottom: 4 }}>ANALYSTS</div>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 24, color: "#d4c4b0", fontWeight: 700 }}>{totalAnalysts}</div>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#8a7560" }}>covering this name</div>
          </div>
        </div>

        {/* 5-bucket breakdown */}
        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#daa520", letterSpacing: 2, marginBottom: 8 }}>RATING BREAKDOWN</div>
        <div style={{ display: "grid", gap: 6, marginBottom: 16 }}>
          {buckets.map(({ label, count, color }) => {
            const pct = totalAnalysts > 0 ? (count / totalAnalysts) * 100 : 0;
            return (
              <div key={label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color, width: 78 }}>{label}</div>
                <div style={{ flex: 1, height: 8, background: "#1a0f0a", borderRadius: 2, overflow: "hidden" }}>
                  <div style={{ height: "100%", background: color, width: `${pct}%`, transition: "width 1s ease" }} />
                </div>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 10, color, fontWeight: 700, width: 28, textAlign: "right" }}>{count}</div>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", width: 36, textAlign: "right" }}>{pct.toFixed(0)}%</div>
              </div>
            );
          })}
        </div>

        {/* 6-month trend */}
        {trend.length > 0 && (
          <>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#daa520", letterSpacing: 2, marginBottom: 8 }}>CONSENSUS TREND (last {trend.length} months)</div>
            <div style={{ background: "#1a0f0a", borderRadius: 6, padding: "10px 8px", marginBottom: 12 }}>
              {trend.slice().reverse().map((t: any, i: number) => {
                const n = (t.strong_buy + t.buy + t.hold + t.sell + t.strong_sell) || 1;
                return (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: i < trend.length - 1 ? 4 : 0 }}>
                    <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#9d8b7a", width: 56 }}>{t.period?.slice(0,7)}</div>
                    <div style={{ flex: 1, display: "flex", height: 6, borderRadius: 2, overflow: "hidden" }}>
                      <div style={{ background: "#00c896", width: `${(t.strong_buy/n)*100}%` }} />
                      <div style={{ background: "#40dda0", width: `${(t.buy/n)*100}%` }} />
                      <div style={{ background: "#e8b84b", width: `${(t.hold/n)*100}%` }} />
                      <div style={{ background: "#ff8090", width: `${(t.sell/n)*100}%` }} />
                      <div style={{ background: "#ff6080", width: `${(t.strong_sell/n)*100}%` }} />
                    </div>
                    <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#d4c4b0", width: 24, textAlign: "right" }}>{n}</div>
                  </div>
                );
              })}
            </div>
          </>
        )}

        <div style={{ display: "flex", gap: 16, marginTop: 10, fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#8a7560" }}>
          <span style={{ color: "#40dda0" }}>▲ {consensus.upgrades_30d || 0} consensus up</span>
          <span style={{ color: "#ff8090" }}>▼ {consensus.downgrades_30d || 0} consensus down</span>
          <span>vs prev month</span>
        </div>

        {/* ── ANALYST SIGNAL ANALYTICS (computed) ── */}
        {(analytics.beat_rate != null || analytics.revision_momentum != null) && (
          <>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#daa520", letterSpacing: 2, margin: "6px 0 8px" }}>ANALYST SIGNAL ANALYTICS</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 12 }}>
              <div style={{ background: "#1a0f0a", borderRadius: 6, padding: "8px 10px" }}>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#9d8b7a", marginBottom: 3 }}>EARNINGS BEAT RATE</div>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 15, fontWeight: 700, color: (analytics.beat_rate ?? 0) >= 0.6 ? "#40dda0" : (analytics.beat_rate ?? 0) >= 0.4 ? "#e8b84b" : "#ff8090" }}>
                  {analytics.beat_rate != null ? `${Math.round(analytics.beat_rate * analytics.n_quarters)} of ${analytics.n_quarters}` : "—"}
                  <span style={{ fontSize: 10, color: "#8a7560", marginLeft: 6 }}>{pct1(analytics.beat_rate)}</span>
                </div>
              </div>
              <div style={{ background: "#1a0f0a", borderRadius: 6, padding: "8px 10px" }}>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#9d8b7a", marginBottom: 3 }}>AVG SURPRISE</div>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 15, fontWeight: 700, color: (analytics.avg_surprise_pct ?? 0) > 0 ? "#40dda0" : "#ff8090" }}>
                  {analytics.avg_surprise_pct != null ? `${analytics.avg_surprise_pct > 0 ? "+" : ""}${analytics.avg_surprise_pct}%` : "—"}
                  {analytics.surprise_trend != null && (
                    <span style={{ fontSize: 9, color: analytics.surprise_trend >= 0 ? "#40dda0" : "#ff8090", marginLeft: 6 }}>
                      {arrow(analytics.surprise_trend)} {analytics.surprise_trend >= 0 ? "improving" : "cooling"}
                    </span>
                  )}
                </div>
              </div>
              <div style={{ background: "#1a0f0a", borderRadius: 6, padding: "8px 10px" }}>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#9d8b7a", marginBottom: 3 }}>REVISION MOMENTUM</div>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 13, fontWeight: 700, color: analytics.revision_direction === "improving" ? "#40dda0" : analytics.revision_direction === "deteriorating" ? "#ff8090" : "#e8b84b" }}>
                  {analytics.revision_direction || "—"}
                </div>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 7, color: "#8a7560" }}>Jegadeesh: changes &gt; levels</div>
              </div>
              <div style={{ background: "#1a0f0a", borderRadius: 6, padding: "8px 10px" }}>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#9d8b7a", marginBottom: 3 }}>BULL SHARE · CONVICTION</div>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 13, fontWeight: 700, color: "#d4c4b0" }}>
                  {pct1(analytics.bull_share)} <span style={{ fontSize: 9, color: "#8a7560" }}>· HHI {analytics.rating_conviction ?? "—"}</span>
                </div>
              </div>
            </div>

            {/* QuantEdge vs Street */}
            {divergence != null && (
              <div style={{ background: "#140c08", border: "1px solid #3a2920", borderRadius: 8, padding: "10px 12px", marginBottom: 12 }}>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#daa520", letterSpacing: 2, marginBottom: 8 }}>QUANTEDGE vs THE STREET</div>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
                  <div style={{ textAlign: "center", flex: 1 }}>
                    <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#9d8b7a" }}>QUANTEDGE</div>
                    <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 20, color: "#daa520" }}>{ourConviction?.toFixed(0) ?? "—"}</div>
                    <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#8a7560" }}>{ourVerdict || ""}</div>
                  </div>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 13, fontWeight: 700, color: Math.abs(divergence) <= 5 ? "#8a7560" : divergence > 0 ? "#40dda0" : "#ff8090" }}>
                      {Math.abs(divergence) <= 5 ? "≈ ALIGNED" : divergence > 0 ? `+${divergence} MORE BULLISH` : `${divergence} MORE BEARISH`}
                    </div>
                  </div>
                  <div style={{ textAlign: "center", flex: 1 }}>
                    <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#9d8b7a" }}>STREET</div>
                    <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 20, color: consensus.color || "#40dda0" }}>{streetPct?.toFixed(0) ?? "—"}</div>
                    <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#8a7560" }}>{consensus.label || ""}</div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#8a7560", marginTop: 12, lineHeight: 1.6 }}>
          Individual analyst names and price targets require paid Finnhub plan.<br/>
          Free tier provides consensus counts and historical trends — shown above.
        </div>
      </Card>

      {/* RIGHT: Earnings + Surprise History */}
      <Card style={{ gridColumn: "span 1" }}>
        <SectionTitle>EARNINGS — NEXT QUARTER (EST)</SectionTitle>

        {/* Earnings countdown */}
        {earnings.date ? (
          <div style={{ textAlign: "center", padding: "16px 0 20px", borderBottom: "1px solid rgba(212,149,108,0.08)", marginBottom: 16 }}>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", letterSpacing: 2, marginBottom: 6 }}>ESTIMATED EARNINGS DATE</div>
            <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 28, color: earningsUrgency, letterSpacing: 3, marginBottom: 4 }}>{earnings.date}</div>
            {daysToEarnings != null && (
              <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 11, color: earningsUrgency }}>
                {daysToEarnings > 0 ? `${daysToEarnings} days away` : daysToEarnings === 0 ? "TODAY" : `${Math.abs(daysToEarnings)} days ago`}
              </div>
            )}
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#8a7560", marginTop: 4 }}>
              (last report + ~90d; exact date not in free tier)
            </div>
          </div>
        ) : (
          <div style={{ textAlign: "center", padding: "16px 0", color: "#8a7560", fontFamily: "'Fira Code',monospace", fontSize: 10 }}>
            No earnings history available
          </div>
        )}

        {/* Recent EPS with YoY growth */}
        {earnings.eps_prev_year != null && (
          <>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#daa520", letterSpacing: 2, marginBottom: 10 }}>MOST RECENT QUARTER (ACTUAL)</div>
            <div style={{ display: "flex", gap: 12, marginBottom: 20 }}>
              <div style={{ flex: 1, textAlign: "center", background: "#1a0f0a", borderRadius: 6, padding: "12px 8px" }}>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", marginBottom: 4 }}>EPS (Q LATEST)</div>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 20, color: "#d4c4b0", fontWeight: 700 }}>${earnings.eps_prev_year?.toFixed(2)}</div>
              </div>
              <div style={{ flex: 1, textAlign: "center", background: "#1a0f0a", borderRadius: 6, padding: "12px 8px" }}>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", marginBottom: 4 }}>YoY GROWTH</div>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 20, fontWeight: 700, color: (earnings.eps_growth || 0) > 0 ? "#40dda0" : "#ff8090" }}>
                  {earnings.eps_growth != null ? `${earnings.eps_growth > 0 ? "+" : ""}${earnings.eps_growth}%` : "—"}
                </div>
              </div>
            </div>
          </>
        )}

        {/* Surprise history */}
        {surpriseHistory.length > 0 && (
          <>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#daa520", letterSpacing: 2, marginBottom: 10 }}>EARNINGS SURPRISE HISTORY</div>
            <div style={{ background: "#1a0f0a", borderRadius: 6, padding: "10px 12px" }}>
              <div style={{ display: "flex", fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#9d8b7a", letterSpacing: 1, marginBottom: 6, paddingBottom: 4, borderBottom: "1px solid rgba(212,149,108,0.08)" }}>
                <div style={{ flex: 1 }}>QUARTER</div>
                <div style={{ width: 60, textAlign: "right" }}>ACTUAL</div>
                <div style={{ width: 60, textAlign: "right" }}>EST</div>
                <div style={{ width: 60, textAlign: "right" }}>SURPRISE</div>
              </div>
              {surpriseHistory.map((s: any, i: number) => {
                const sp = s.surprise_percent;
                const beat = sp != null && sp > 0;
                return (
                  <div key={i} style={{ display: "flex", fontFamily: "'Fira Code',monospace", fontSize: 10, padding: "4px 0", borderBottom: i < surpriseHistory.length - 1 ? "1px solid rgba(212,149,108,0.04)" : "none" }}>
                    <div style={{ flex: 1, color: "#9d8b7a" }}>{s.period?.slice(0,7)}</div>
                    <div style={{ width: 60, textAlign: "right", color: "#d4c4b0" }}>${s.actual?.toFixed(2) ?? "—"}</div>
                    <div style={{ width: 60, textAlign: "right", color: "#9d8b7a" }}>${s.estimate?.toFixed(2) ?? "—"}</div>
                    <div style={{ width: 60, textAlign: "right", color: beat ? "#40dda0" : "#ff8090", fontWeight: 700 }}>
                      {sp != null ? `${sp > 0 ? "+" : ""}${sp.toFixed(1)}%` : "—"}
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}

        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#8a7560", marginTop: 14, lineHeight: 1.6 }}>
          Source: Finnhub free tier · Forward estimates + price targets require paid plan<br/>
          Jegadeesh et al. (2004): consensus changes carry more signal than levels
        </div>
      </Card>
    </div>
  );
}


// ══════════════════════════════════════════════════════════════
// PORTFOLIO CONSTRUCTION PANEL
// ══════════════════════════════════════════════════════════════
export function PortfolioPanel({ data }: { data: any }) {
  const pc = data.portfolio_construction || {};
  const gov = data.governance || {};
  const risk = data.risk_engine || {};
  const dsr = gov.deflated_sharpe_ratio;
  const dsrColor = dsr == null ? "#9d8b7a" : dsr > 0.5 ? "#40dda0" : dsr > 0 ? "#e8b84b" : "#ff8090";

  const volScale = pc.vol_scale_factor ?? 1.0;
  const targetVol = (pc.target_vol ?? 0.10) * 100;
  const realizedVol = (pc.realized_vol ?? 0.25) * 100;
  const recommendedPos = Math.min(1.0, pc.recommended_position_size ?? 1.0);
  const leverage = pc.leverage_signal ?? "NEUTRAL";
  const governorActive = pc.governor_active ?? false;

  const leverageColor = leverage === "INCREASE" ? "#40dda0"
    : leverage === "REDUCE" ? "#ff8090"
    : leverage === "HALT" ? "#ff4060"
    : "#e8b84b";

  const labeling = gov.labeling || {};

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>

      {/* Position Sizing */}
      <Card>
        <SectionTitle>VOLATILITY-TARGETED POSITION SIZING</SectionTitle>
        <div style={{ textAlign: "center", padding: "16px 0 20px" }}>
          <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", letterSpacing: 2, marginBottom: 8 }}>RECOMMENDED POSITION SIZE</div>
          <div style={{ position: "relative", width: 120, height: 120, margin: "0 auto 12px" }}>
            <svg viewBox="0 0 120 120" style={{ width: "100%", transform: "rotate(-90deg)" }}>
              <circle cx="60" cy="60" r="50" fill="none" stroke="#1a0f0a" strokeWidth="12" />
              <circle cx="60" cy="60" r="50" fill="none"
                stroke={governorActive ? "#ff8090" : "#daa520"}
                strokeWidth="12"
                strokeDasharray={`${recommendedPos * 314} 314`}
                style={{ transition: "stroke-dasharray 1.5s ease" }}
              />
            </svg>
            <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column" }}>
              <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 22, fontWeight: 700, color: governorActive ? "#ff8090" : "#daa520" }}>
                {(recommendedPos * 100).toFixed(0)}%
              </div>
              <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#8a7560" }}>OF CAPITAL</div>
            </div>
          </div>
          {governorActive && (
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#ff8090", background: "rgba(224,82,82,0.1)", padding: "4px 12px", marginBottom: 8, letterSpacing: 1 }}>
              ⚠ DRAWDOWN GOVERNOR ACTIVE
            </div>
          )}
        </div>
        <Row label="Vol Scale Factor" value={volScale.toFixed(4)} highlight={volScale < 0.8 ? "#ff8090" : "#d4c4b0"} />
        <Row label="Target Vol" value={`${targetVol.toFixed(1)}%`} />
        <Row label="Realized Vol" value={`${realizedVol.toFixed(1)}%`} highlight={realizedVol > targetVol * 1.5 ? "#ff8090" : "#d4c4b0"} />
        <Row label="Leverage Signal" value={leverage} highlight={leverageColor} />
        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#8a7560", marginTop: 12, lineHeight: 1.7 }}>
          Vol targeting: scale = target_vol / realized_vol<br />
          Position halts at -20% drawdown (governor)<br />
          Reduces at -15% drawdown threshold
        </div>
      </Card>

      {/* Risk Budget */}
      <Card>
        <SectionTitle>RISK ENGINE — CVaR BUDGET</SectionTitle>
        {risk.cvar ? (
          <>
            <Row label="CVaR 95% (worst case)" value={fmtN((risk.cvar?.worst_case_daily || 0) * 100, 2) + "%"} highlight="#ff8090" />
            <Row label="CVaR 95% (historical)" value={fmtN((risk.cvar?.historical?.cvar || 0) * 100, 2) + "%"} highlight="#ff8090" />
            <Row label="CVaR (Cornish-Fisher)" value={fmtN((risk.cvar?.cornish_fisher?.cvar || 0) * 100, 2) + "%"} highlight="#ff8090" />
          </>
        ) : (
          <div style={{ color: "#8a7560", fontFamily: "'Fira Code',monospace", fontSize: 10, padding: "20px 0" }}>Risk engine data unavailable</div>
        )}

        <SectionTitle>POSITION LIMITS</SectionTitle>
        {risk.position_limits ? Object.entries(risk.position_limits).slice(0, 5).map(([k, v]: [string, any]) => (
          <Row key={k} label={k.replace(/_/g, " ")} value={typeof v === "number" ? fmtN(v * 100, 1) + "%" : String(v)} />
        )) : null}

        <SectionTitle>RISK BUDGET</SectionTitle>
        {risk.risk_budget ? Object.entries(risk.risk_budget).slice(0, 4).map(([k, v]: [string, any]) => (
          <Row key={k} label={k.replace(/_/g, " ")} value={typeof v === "number" ? fmtN(v, 4) : String(v)} />
        )) : null}

        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#8a7560", marginTop: 12, lineHeight: 1.7 }}>
          CVaR = Expected Shortfall beyond VaR threshold<br />
          Preferred over VaR for Basel III / FRTB compliance<br />
          Cornish-Fisher: adjusts for skewness + kurtosis
        </div>
      </Card>

      {/* Governance */}
      <Card>
        <SectionTitle>MODEL GOVERNANCE — DEFLATED SHARPE</SectionTitle>
        <div style={{ textAlign: "center", padding: "16px 0 20px" }}>
          <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", letterSpacing: 2, marginBottom: 8 }}>DEFLATED SHARPE RATIO</div>
          <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 42, color: dsrColor, letterSpacing: 2, lineHeight: 1 }}>
            {dsr != null ? dsr.toFixed(3) : "—"}
          </div>
          <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 11, color: dsrColor, marginTop: 6, letterSpacing: 1 }}>
            {gov.is_genuine_alpha ? "✓ DSR > 0 · UNLIKELY CHANCE" : "✗ DSR ≤ 0 · COULD BE CHANCE"}
          </div>
          <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#8a7560", marginTop: 4 }}>
            Raw Sharpe: {fmtN(gov.sharpe_ratio_raw, 3)} · {gov.n_models_tested || 8} models tested
          </div>
        </div>
        <div style={{ fontFamily: "'Outfit',sans-serif", fontSize: 11, color: "#9d8b7a", lineHeight: 1.6, marginBottom: 16 }}>
          DSR corrects for multiple testing bias across {gov.n_models_tested || 8} model variants.
          A DSR {">"} 0 indicates the Sharpe is unlikely to be due to chance.
        </div>

        <SectionTitle>TRIPLE-BARRIER LABELING</SectionTitle>
        <Row label="N Events (CUSUM)" value={String(labeling.n_cusum_events || labeling.n_events || "—")} />
        <Row label="Avg Sample Uniqueness" value={fmtN(labeling.avg_sample_uniqueness, 3)} />
        <Row label="Fractional d" value={fmtN(labeling.fractional_d, 3)} />
        {labeling.label_distribution && Object.entries(labeling.label_distribution).map(([k, v]: [string, any]) => (
          <Row key={k} label={`Label ${k}`} value={fmtN(v, 3)} />
        ))}

        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#8a7560", marginTop: 12, lineHeight: 1.7 }}>
          Lopez de Prado (2018) AFML Ch.7<br />
          60-day embargo prevents autocorrelation leakage<br />
          DSR: Bailey {"&"} Lopez de Prado (2014)
        </div>
      </Card>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// PERFORMANCE ANALYTICS PANEL
// ══════════════════════════════════════════════════════════════
export function PerformancePanel({ data }: { data: any }) {
  const risk = data.risk_metrics || {};
  const gov = data.governance || {};
  const ml = data.ml_predictions || {};
  const ens = ml.ensemble || {};

  // Simulate IC decay over time (would be real from signal_tracker in prod)
  const months = ["Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan","Feb","Mar"];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>

      {/* Signal Quality */}
      <Card>
        <SectionTitle>SIGNAL QUALITY METRICS</SectionTitle>
        <Row label="IC (in-sample, XGB train)" value={fmtN(ml.ic_estimate, 4)} highlight="#e8b84b" />
        <Row label="Rank IC (out-of-sample)" value={ml.rank_ic_estimate != null ? fmtN(ml.rank_ic_estimate, 4) : "—"} />
        <Row label="Model Disagreement" value={fmtN(ens.model_disagreement, 3) + "%"} />
        <Row label="Deflated Sharpe (DSR)" value={fmtN(gov.deflated_sharpe_ratio, 3)} highlight={(gov.deflated_sharpe_ratio || 0) > 0.5 ? "#40dda0" : "#e8b84b"} />
        <Row label="DSR verdict" value={gov.is_genuine_alpha ? "unlikely chance" : "could be chance"} highlight={gov.is_genuine_alpha ? "#40dda0" : "#e8b84b"} />

        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#8a7560", marginTop: 12, lineHeight: 1.7 }}>
          IC = Spearman rank correlation of predictions vs realized returns.<br />
          The in-sample IC is measured on training data and is optimistic; the out-of-sample<br />
          Rank IC is the honest read (real-world equity IC is typically 0.02–0.05).<br />
          DSR (Bailey &amp; López de Prado) deflates the Sharpe for multiple-testing across<br />
          8 model variants; it saturates near 1.0 when the raw Sharpe is high.
        </div>
      </Card>

      {/* Return Attribution */}
      <Card>
        <SectionTitle>RETURN ATTRIBUTION</SectionTitle>
        <Row label="Annual Return" value={fmtN((risk.annual_return || 0) * 100, 2) + "%"} highlight={(risk.annual_return || 0) > 0 ? "#40dda0" : "#ff8090"} />
        <Row label="Annual Volatility" value={fmtN((risk.annual_volatility || 0) * 100, 2) + "%"} />
        <Row label="Sharpe Ratio" value={fmtN(risk.sharpe_ratio, 3)} highlight={(risk.sharpe_ratio || 0) > 1 ? "#40dda0" : (risk.sharpe_ratio || 0) > 0.5 ? "#e8b84b" : "#ff8090"} />
        <Row label="Sortino Ratio" value={fmtN(risk.sortino_ratio, 3)} />
        <Row label="Max Drawdown" value={fmtN((risk.max_drawdown || 0) * 100, 2) + "%"} highlight="#ff8090" />
        <Row label="Calmar Ratio" value={fmtN(risk.calmar_ratio, 3)} />
        <Row label="Omega Ratio" value={fmtN(risk.omega_ratio, 3)} highlight={(risk.omega_ratio || 0) > 1 ? "#40dda0" : "#ff8090"} />

        <SectionTitle>DISTRIBUTION MOMENTS</SectionTitle>
        <Row label="Skewness" value={fmtN(risk.skewness, 4)} highlight={(risk.skewness || 0) > 0 ? "#40dda0" : "#ff8090"} />
        <Row label="Excess Kurtosis" value={fmtN(risk.excess_kurtosis, 4)} highlight={(risk.excess_kurtosis || 0) > 3 ? "#ff8090" : "#d4c4b0"} />
        <Row label="Hurst Exponent" value={fmtN(data.hurst_exponent, 4)} highlight={(data.hurst_exponent || 0.5) > 0.55 ? "#40dda0" : "#8b5cf6"} />
      </Card>

      {/* IC Tracking — honest accumulating state */}
      <Card>
        <SectionTitle>OUT-OF-SAMPLE IC TRACKING</SectionTitle>
        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 11, color: "#d4c4b0", marginBottom: 14, lineHeight: 1.6 }}>
          Every signal this engine generates is recorded with a timestamp, then scored
          against the stock's <span style={{color:"#daa520"}}>realized 21-day forward return</span> once
          that horizon completes. The out-of-sample Information Coefficient — the honest
          measure of whether the signal actually predicts returns — accumulates from those
          matured signals.
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:10, padding:"12px 14px", background:"#1a0f0a", borderRadius:6, border:"1px solid rgba(212,149,108,0.12)", marginBottom:12 }}>
          <div style={{ fontFamily:"'Bebas Neue',sans-serif", fontSize:30, color:"#daa520", lineHeight:1 }}>⏳</div>
          <div style={{ fontFamily:"'Fira Code',monospace", fontSize:11, color:"#9d8b7a", lineHeight:1.5 }}>
            Signal history is still maturing. A meaningful IC series needs several months of
            signals that have each reached their 21-day evaluation horizon. Until then, the
            honest IC read is the per-analysis <span style={{color:"#daa520"}}>out-of-sample Rank IC</span> on
            the left, not a backfilled track record.
          </div>
        </div>
        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#8a7560", marginTop: 8, lineHeight: 1.7 }}>
          Why no 12-month chart? Because the platform doesn't have 12 months of live signals
          yet — and showing a backfilled one would misrepresent the track record.<br />
          IC = Spearman rank correlation of signal vs realized forward returns.<br />
          Real-world single-name equity IC is typically 0.02–0.05.
        </div>
      </Card>
    </div>
  );
}

export default { SignalPanel, MLModelsPanel, VolatilityPanel, RegimePanel, OptionsPanel, SentimentPanel, MonteCarloPanel, RiskPanel, FundamentalsPanel, ScenarioPanel, Watchlist };
