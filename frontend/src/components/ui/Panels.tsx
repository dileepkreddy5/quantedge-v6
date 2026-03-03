// ============================================================
// QuantEdge v6.0 — All UI Panel Components
// MLModels, Volatility, Regime, Options, Sentiment, Risk,
// Fundamentals, Scenarios, Signal, Watchlist panels
// ============================================================

import React, { useState } from 'react';
import { api, useAuthStore } from '../../auth/authStore';
import toast from 'react-hot-toast';

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
        <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#4a3428', letterSpacing:2 }}>COMPOSITE SCORE / 100</div>
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

  return (
    <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:12 }}>
      {/* Ensemble */}
      <Card style={{ gridColumn:'span 3' }}>
        <SectionTitle>ENSEMBLE MODEL — RETURN FORECASTS</SectionTitle>
        <div style={{ display:'flex', gap:8, flexWrap:'wrap' }}>
          {[{l:'1W',k:'pred_5d'},{l:'2W',k:'pred_10d'},{l:'1M',k:'pred_21d'},{l:'3M',k:'pred_63d'},{l:'1Y',k:'pred_252d'}].map(h => {
            const v = ensemble[h.k];
            const c = v == null ? '#4a3428' : v > 0 ? '#22c55e' : '#ef4444';
            return (
              <div key={h.k} style={{ flex:1, minWidth:80, background:'#1a0f0a', borderRadius:8, padding:'14px 10px', textAlign:'center', border:`1px solid ${c}30` }}>
                <div style={{ fontFamily:"'Fira Code',monospace", fontSize:9, color:'#9d8b7a', letterSpacing:2, marginBottom:6 }}>{h.l}</div>
                <div style={{ fontFamily:"'Fira Code',monospace", fontSize:20, fontWeight:800, color:c }}>
                  {v != null ? `${v>0?'+':''}${v.toFixed(1)}%` : '—'}
                </div>
                <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:9, color:'#4a3428', marginTop:4 }}>forecast</div>
              </div>
            );
          })}
        </div>
        <div style={{ display:'flex', gap:20, marginTop:12, fontFamily:"'Fira Code',monospace", fontSize:9, color:'#4a3428' }}>
          <span>CONFIDENCE: {((ensemble.confidence || 0)*100).toFixed(1)}%</span>
          <span>MODEL DISAGREEMENT: {fmtN(ensemble.model_disagreement)}%</span>
          <span>IC ESTIMATE: {fmtN(preds.rank_ic_estimate,3)}</span>
        </div>
      </Card>

      {/* LSTM */}
      <Card>
        <SectionTitle>BIDIRECTIONAL LSTM</SectionTitle>
        <div style={{ marginBottom:8 }}>
          <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10, color:'#9d8b7a', marginBottom:4 }}>Architecture: 512→256→128 BiLSTM + Temporal Attention</div>
          <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10, color:'#9d8b7a' }}>MC Dropout uncertainty quantification</div>
        </div>
        {[{l:'5d Return',v:lstm.pred_5d,suf:'%'},{l:'21d Return',v:lstm.pred_21d,suf:'%'},{l:'252d Return',v:lstm.pred_252d,suf:'%'},{l:'Regime',v:lstm.regime},{l:'Uncertainty',v:lstm.uncertainty,suf:'%'}].map(r => (
          <Row key={r.l} label={r.l} value={r.v != null ? `${typeof r.v==='number'&&r.v>0?'+':''}${typeof r.v==='number'?r.v.toFixed(1):''}${r.suf||''}${typeof r.v==='string'?r.v:''}` : '—'} />
        ))}
      </Card>

      {/* XGBoost */}
      <Card>
        <SectionTitle>XGBOOST CROSS-SECTIONAL</SectionTitle>
        <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10, color:'#9d8b7a', marginBottom:8 }}>120 features · SHAP explanations · Walk-forward CV</div>
        {[{l:'Signal Strength',v:xgb.signal_strength,suf:'%'},{l:'21d Alpha',v:xgb.pred_21d,suf:'%'},{l:'252d Alpha',v:xgb.pred_252d,suf:'%'},{l:'IC Estimate',v:preds.ic_estimate}].map(r => (
          <Row key={r.l} label={r.l} value={r.v != null ? `${typeof r.v==='number'&&r.v>0?'+':''}${typeof r.v==='number'?r.v.toFixed(2):''}${r.suf||''}` : '—'} />
        ))}
        {shap.slice(0,3).length > 0 && (
          <>
            <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#daa520', letterSpacing:2, marginTop:10, marginBottom:6 }}>TOP SHAP DRIVERS</div>
            {shap.slice(0,3).map((s: any, i: number) => (
              <div key={i} style={{ display:'flex', justifyContent:'space-between', fontSize:10, padding:'3px 0', color:'#9d8b7a' }}>
                <span style={{ fontFamily:"'Fira Code',monospace", fontSize:9 }}>{s.feature}</span>
                <span style={{ color: s.impact > 0 ? '#22c55e' : '#ef4444', fontFamily:"'Fira Code',monospace" }}>{s.impact > 0 ? '+' : ''}{s.impact?.toFixed(3)}</span>
              </div>
            ))}
          </>
        )}
      </Card>

      {/* LightGBM */}
      <Card>
        <SectionTitle>LIGHTGBM + QUANTILE FORECASTS</SectionTitle>
        <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10, color:'#9d8b7a', marginBottom:8 }}>Leaf-wise growth · Cross-sectional ranking</div>
        {[{l:'Rank Score',v:lgbm.rank_score,suf:'%'},{l:'21d Pred',v:lgbm.pred_21d,suf:'%'},{l:'252d Pred',v:lgbm.pred_252d,suf:'%'}].map(r => (
          <Row key={r.l} label={r.l} value={r.v != null ? `${r.v>0?'+':''}${r.v.toFixed(1)}${r.suf||''}` : '—'} />
        ))}
        {Object.keys(quantile).length > 0 && (
          <>
            <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#daa520', letterSpacing:2, marginTop:10, marginBottom:6 }}>1M RETURN QUANTILES</div>
            {[['q10_1m','P10'],['q25_1m','P25'],['q50_1m','P50 (Median)'],['q75_1m','P75'],['q90_1m','P90']].map(([k,l]) => (
              <Row key={k} label={l} value={quantile[k] != null ? `${quantile[k]>0?'+':''}${quantile[k].toFixed(1)}%` : '—'} />
            ))}
          </>
        )}
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
  const k = data.kalman || {};

  return (
    <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:12 }}>
      <Card style={{ gridColumn:'span 1' }}>
        <SectionTitle>GJR-GARCH(1,1) PARAMETERS</SectionTitle>
        <Row label="ω (omega)" value={fmtN(g.omega,6)} />
        <Row label="α (arch)" value={fmtN(g.alpha,4)} />
        <Row label="γ (asymmetry)" value={fmtN(g.gamma_asymmetry,4)} highlight={g.gamma_asymmetry > 0 ? '#ef4444' : undefined} />
        <Row label="β (garch)" value={fmtN(g.beta,4)} />
        <Row label="ν Student-t df" value={fmtN(g.nu_student_t,1)} />
        <Row label="Persistence α+β" value={fmtN(g.persistence,4)} highlight={(g.persistence||0) > 0.95 ? '#f59e0b' : undefined} />
        <Row label="Leverage Effect" value={g.leverage_effect ? '✓ YES' : 'NO'} highlight={g.leverage_effect ? '#ef4444' : undefined} />
      </Card>

      <Card>
        <SectionTitle>VOLATILITY FORECASTS (Annualized)</SectionTitle>
        <Row label="Current Daily Vol" value={fmtN((g.current_daily_vol||0)*100,3)+'%'} />
        <Row label="Annual Vol" value={fmtN((g.current_annual_vol||0)*100,1)+'%'} />
        <Row label="5d Forecast" value={fmtN((g.forecast_vol_5d||0)*100,1)+'%'} />
        <Row label="21d Forecast" value={fmtN((g.forecast_vol_21d||0)*100,1)+'%'} />
        <Row label="Long-Run Vol" value={fmtN((g.long_run_annual_vol||0)*100,1)+'%'} />
        <Row label="Vol Regime" value={g.vol_regime||'—'} />
        <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#4a3428', marginTop:12, lineHeight:1.6 }}>
          GJR-GARCH: γ&gt;0 means bad news (negative returns) increases<br/>
          volatility more than good news — the "leverage effect"
        </div>
      </Card>

      <Card>
        <SectionTitle>VaR & CVaR (Student-t)</SectionTitle>
        <Row label="VaR 95% (1d)" value={fmtN((g.var_95_daily||0)*100,2)+'%'} highlight="#ef4444" />
        <Row label="VaR 99% (1d)" value={fmtN((g.var_99_daily||0)*100,2)+'%'} highlight="#ef4444" />
        <Row label="CVaR 95% (1d)" value={fmtN((g.cvar_95_daily||0)*100,2)+'%'} highlight="#ef4444" />
        <Row label="CVaR 99% (1d)" value={fmtN((g.cvar_99_daily||0)*100,2)+'%'} highlight="#ef4444" />
        <Row label="Max Drawdown" value={fmtN((risk.max_drawdown||0)*100,1)+'%'} highlight="#ef4444" />
        <Row label="Hurst Exponent" value={fmtN(data.hurst_exponent,3)} highlight={(data.hurst_exponent||0.5)>0.55?'#22c55e':'#8b5cf6'} />
        <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#4a3428', marginTop:12, lineHeight:1.6 }}>
          CVaR = Expected loss given VaR is breached<br/>
          More conservative than VaR (Basel III preferred measure)
        </div>
      </Card>

      <Card>
        <SectionTitle>KALMAN FILTER — TREND DECOMPOSITION</SectionTitle>
        <Row label="SNR (Signal/Noise)" value={fmtN(k.snr,4)} highlight={(k.snr||0)>0.05?'#22c55e':undefined} />
        <Row label="R² (Trend Fit)" value={fmtN(k.r_squared,4)} />
        <Row label="Process Noise Q" value={fmtN(k.process_noise_Q,6)} />
        <Row label="Observation Noise R" value={fmtN(k.observation_noise_R,4)} />
        <Row label="Kalman Gain" value={fmtN(k.current_kalman_gain,4)} />
        <Row label="Trend Slope" value={fmtN(k.trend_slope,4)} highlight={(k.trend_slope||0)>0?'#22c55e':'#ef4444'} />
        <Row label="Signal Type" value={k.signal_interpretation?.replace(/_/g,' ')||'—'} />
      </Card>

      <Card>
        <SectionTitle>REALIZED VOLATILITY (Historical)</SectionTitle>
        {[5,10,21,63,126,252].map(d => (
          <Row key={d} label={`${d}d Realized Vol`} value={fmtN((data.risk_metrics?.[`realized_vol_${d}d`]||0)*100,1)+'%'} />
        ))}
      </Card>

      <Card>
        <SectionTitle>VOLATILITY ESTIMATORS</SectionTitle>
        <Row label="Close-to-Close" value={fmtN((data.annual_vol||0)*100,1)+'%'} />
        <Row label="Parkinson" value={fmtN((data.parkinson_vol||0)*100,1)+'%'} />
        <Row label="Garman-Klass" value={fmtN((data.garman_klass_vol||0)*100,1)+'%'} />
        <Row label="Yang-Zhang" value={fmtN((data.yang_zhang_vol||0)*100,1)+'%'} />
        <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#4a3428', marginTop:12, lineHeight:1.7 }}>
          Parkinson: uses H/L only (5× efficient)<br/>
          Garman-Klass: uses O,H,L,C (most efficient)<br/>
          Yang-Zhang: handles overnight gaps (best for daily)
        </div>
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
    <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:12 }}>
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
        <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#4a3428', marginTop:12, lineHeight:1.7 }}>
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
              <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#4a3428' }}>avg trading days</div>
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
    return <Card><div style={{ textAlign:'center', color:'#4a3428', fontFamily:"'Fira Code',monospace", fontSize:11, padding:40 }}>Options data unavailable for this ticker</div></Card>;
  }

  return (
    <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:12 }}>
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
        <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#4a3428', marginTop:12, lineHeight:1.7 }}>
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
            <div style={{ display:'flex', justifyContent:'space-between', fontFamily:"'Fira Code',monospace", fontSize:8, color:'#4a3428' }}>
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
        <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#4a3428', marginTop:12, lineHeight:1.7 }}>
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
            <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:10, color:'#4a3428', marginTop:4 }}>
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
    return <Card><div style={{ textAlign:'center', color:'#4a3428', padding:40 }}>Monte Carlo data unavailable</div></Card>;
  }

  const bars = [
    { label: 'P(+20%)',  value: mc.prob_gain_20pct, color: '#22c55e' },
    { label: 'P(+10%)',  value: mc.prob_gain_10pct, color: '#86efac' },
    { label: 'P(loss)',  value: mc.prob_loss,         color: '#ef4444' },
    { label: 'P(-20%)',  value: mc.prob_loss_20pct,   color: '#dc2626' },
    { label: 'P(-50%)',  value: mc.prob_loss_50pct,   color: '#7f1d1d' },
  ];

  return (
    <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:12 }}>
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
        <div style={{ fontFamily:"'Fira Code',monospace", fontSize:8, color:'#4a3428', marginTop:12, lineHeight:1.7 }}>
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
  const ff = data.ff_alpha != null ? data : {};

  const ratios = [
    { label: 'Sharpe Ratio', value: risk.sharpe_ratio, good: v => v > 1, fmt: (v:any)=>fmtN(v,3) },
    { label: 'Sortino Ratio', value: risk.sortino_ratio, good: v => v > 1, fmt: (v:any)=>fmtN(v,3) },
    { label: 'Calmar Ratio', value: risk.calmar_ratio, good: v => v > 0.5, fmt: (v:any)=>fmtN(v,3) },
    { label: 'Omega Ratio', value: risk.omega_ratio, good: v => v > 1, fmt: (v:any)=>fmtN(v,3) },
    { label: 'Sterling Ratio', value: risk.sterling_ratio, good: v => v > 0.5, fmt: (v:any)=>fmtN(v,3) },
    { label: 'Tail Ratio', value: risk.tail_ratio, good: v => v > 1, fmt: (v:any)=>fmtN(v,3) },
  ];

  return (
    <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:12 }}>
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
        <SectionTitle>FACTOR EXPOSURES (Fama-French)</SectionTitle>
        {[['Alpha (annualized)','ff_alpha'],['MKT Beta','ff_mkt_beta'],['SMB (Size)','ff_smb'],['HML (Value)','ff_hml'],['RMW (Profitability)','ff_rmw'],['CMA (Investment)','ff_cma'],['WML (Momentum)','ff_wml'],['R²','ff_r_squared'],['Idio Risk','ff_idio_risk']].map(([l,k]) => (
          <Row key={k} label={l} value={data[k] != null ? fmtN(data[k],4) : '—'} />
        ))}
      </Card>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// FUNDAMENTALS PANEL
// ══════════════════════════════════════════════════════════════
export function FundamentalsPanel({ data }: { data: any }) {
  const sections = [
    { title: 'VALUATION', items: [['P/E Ratio','pe_ratio'],['Forward P/E','forward_pe'],['PEG Ratio','peg_ratio'],['P/B Ratio','price_to_book'],['P/S Ratio','price_to_sales'],['EV/EBITDA','ev_ebitda'],['EV/Revenue','ev_revenue']] },
    { title: 'PROFITABILITY', items: [['Gross Margin','gross_margin',true],['Operating Margin','operating_margin',true],['Net Margin','net_margin',true],['FCF Margin','fcf_margin',true],['ROE','roe',true],['ROA','roa',true],['ROIC','roic',true]] },
    { title: 'GROWTH', items: [['Revenue Growth','revenue_growth',true],['Earnings Growth','earnings_growth',true],['EPS (TTM)','eps_ttm'],['EPS Forward','eps_forward'],['Revenue TTM','revenue_ttm']] },
    { title: 'BALANCE SHEET', items: [['Market Cap','market_cap'],['Total Debt','total_debt'],['Total Cash','total_cash'],['Debt/Equity','debt_to_equity'],['Current Ratio','current_ratio'],['Quick Ratio','quick_ratio']] },
    { title: 'OWNERSHIP', items: [['Institutional Own.','institutional_ownership',true],['Insider Own.','insider_ownership',true],['Short Interest','short_interest',true],['Float Shares','float_shares'],['Shares Short','shares_short']] },
    { title: 'QUALITY', items: [['FCF Yield','fcf_yield',true],['Dividend Yield','dividend_yield',true],['Payout Ratio','payout_ratio',true],['Beta','beta'],['52W High','week_52_high'],['52W Low','week_52_low']] },
  ];

  return (
    <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:12 }}>
      {sections.map(sec => (
        <Card key={sec.title}>
          <SectionTitle>{sec.title}</SectionTitle>
          {sec.items.map(([label, key, isPct]: any) => {
            const v = data[key] ?? data[`fund_${key}`];
            const display = v == null ? '—'
              : isPct ? `${(Number(v)*100).toFixed(2)}%`
              : key.includes('cap') || key.includes('debt') || key.includes('cash') || key.includes('revenue') || key.includes('ebitda') || key.includes('shares')
                ? fmtLarge(v)
                : fmtN(v, 2);
            return <Row key={key} label={label} value={display} />;
          })}
        </Card>
      ))}
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
              <div style={{ fontFamily:"'Outfit',sans-serif", fontSize:9, color:'#4a3428', marginTop:6, lineHeight:1.5 }}>
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
        <div style={{ textAlign:'center', color:'#4a3428', fontFamily:"'Fira Code',monospace", fontSize:11, padding:30 }}>No tickers in watchlist</div>
      )}
      <div style={{ display:'grid', gap:8 }}>
        {items.map(item => (
          <div key={item.ticker} style={{ display:'flex', alignItems:'center', gap:12, background:'#1a0f0a', borderRadius:6, padding:'10px 14px', border:'1px solid rgba(212,149,108,0.1)' }}>
            <span style={{ fontFamily:"'Fira Code',monospace", fontSize:14, fontWeight:700, color:'#daa520', flex:1 }}>{item.ticker}</span>
            {item.notes && <span style={{ fontFamily:"'Outfit',sans-serif", fontSize:11, color:'#9d8b7a' }}>{item.notes}</span>}
            <span style={{ fontFamily:"'Fira Code',monospace", fontSize:9, color:'#4a3428' }}>{item.added_at?.slice(0,10)}</span>
            <button onClick={() => onAnalyze(item.ticker)} style={{ background:'rgba(218,165,32,0.1)', border:'1px solid rgba(218,165,32,0.3)', color:'#daa520', fontFamily:"'Fira Code',monospace", fontSize:9, padding:'4px 10px', borderRadius:4, cursor:'pointer' }}>ANALYZE</button>
            <button onClick={() => remove(item.ticker)} style={{ background:'none', border:'none', color:'#4a3428', cursor:'pointer', fontSize:16 }}>×</button>
          </div>
        ))}
      </div>
    </Card>
  );
}

export default { SignalPanel, MLModelsPanel, VolatilityPanel, RegimePanel, OptionsPanel, SentimentPanel, MonteCarloPanel, RiskPanel, FundamentalsPanel, ScenarioPanel, Watchlist };
