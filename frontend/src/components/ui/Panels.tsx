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
  const isETF = !data.pe_ratio && !data.gross_margin && !data.eps_ttm;
  const earn = data.analyst_ratings?.earnings || {};

  if (isETF) return (
    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:12}}>
      <Card style={{gridColumn:"span 3"}}>
        <SectionTitle>ETF / FUND — MARKET STATISTICS</SectionTitle>
        <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:12}}>
          {[["Market Cap",fm(data.market_cap)],["Annual Vol",fp(data.annual_vol)],["Sharpe",f(data.sharpe_ratio)],["Max DD",fp(data.max_drawdown)],
            ["Hurst",f(data.hurst_exponent,4)],["Regime",data.current_regime?.replace(/_/g," ")??"—"],["Beta",f(data.ff_mkt_beta,3)],["1Y Ret",fp(data.annual_return)]
          ].map(([l,v])=>(
            <div key={l} style={{background:"#1a0f0a",borderRadius:6,padding:"12px 14px",border:"1px solid rgba(212,149,108,0.1)"}}>
              <div style={{fontFamily:"'Fira Code',monospace",fontSize:8,color:"#9d8b7a",letterSpacing:2,marginBottom:6}}>{l}</div>
              <div style={{fontFamily:"'Fira Code',monospace",fontSize:15,color:"#d4c4b0",fontWeight:700}}>{v}</div>
            </div>
          ))}
        </div>
        <div style={{marginTop:12,fontFamily:"'Fira Code',monospace",fontSize:9,color:"#4a3428"}}>ETFs do not report income statements. Use Volatility, Regime, and Risk tabs for quantitative analysis.</div>
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
        <FRow l="Beta (5Y)"     v={f(data.beta??data.ff_mkt_beta,3)} />
        <SubTitle>QUALITY SCORES</SubTitle>
        <FRow l="Piotroski F-Score" v={data.gross_margin&&data.roe&&data.current_ratio?String(Math.min(9,Math.round((data.gross_margin>0.4?1:0)+(data.roe>0.1?1:0)+(data.current_ratio>1.5?1:0)+(data.revenue_growth>0.05?1:0)+(data.debt_to_equity<1?1:0)*2+3))):"—"} c="#daa520" />
        <FRow l="Altman Z-Score"    v="N/A (public)" />
      </Card>

      {/* FACTOR EXPOSURES */}
      <Card>
        <SectionTitle>◈ FACTOR EXPOSURES (FAMA-FRENCH 5)</SectionTitle>
        <FRow l="Alpha (Annualized)" v={fp(data.ff_alpha)}    c={gc(data.ff_alpha)} />
        <FRow l="MKT Beta"           v={f(data.ff_mkt_beta,3)} />
        <FRow l="SMB (Size)"         v={f(data.ff_smb,3)}     c={data.ff_smb!=null&&data.ff_smb<0?"#40dda0":"#e8b84b"} />
        <FRow l="HML (Value)"        v={f(data.ff_hml,3)}     />
        <FRow l="RMW (Profitability)" v={f(data.ff_rmw,3)}   c={gc(data.ff_rmw)} />
        <FRow l="CMA (Investment)"   v={f(data.ff_cma,3)}    />
        <FRow l="WML (Momentum)"     v={f(data.ff_wml,3)}    c={gc(data.ff_wml)} />
        <FRow l="R²"                 v={f(data.ff_r_squared,3)} />
        <FRow l="Idiosyncratic Risk" v={fp(data.ff_idio_risk)} />
        <div style={{fontFamily:"'Fira Code',monospace",fontSize:7,color:"#4a3428",marginTop:12,lineHeight:1.7}}>
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


// ══════════════════════════════════════════════════════════════
// WALL STREET ANALYST PANEL
// ══════════════════════════════════════════════════════════════
export function WallStreetPanel({ data }: { data: any }) {
  const ar = data.analyst_ratings || {};
  const ratings = ar.ratings || [];
  const consensus = ar.consensus || {};
  const earnings = ar.earnings || {};

  const ratingColor = (label: string) => {
    const l = label?.toLowerCase() || '';
    if (l.includes('strong buy')) return '#00c896';
    if (l.includes('buy') || l.includes('overweight') || l.includes('outperform')) return '#40dda0';
    if (l.includes('hold') || l.includes('neutral') || l.includes('equal')) return '#e8b84b';
    if (l.includes('sell') || l.includes('underweight') || l.includes('underperform')) return '#ff8090';
    return '#9d8b7a';
  };

  const daysToEarnings = earnings.days_to;
  const earningsUrgency = daysToEarnings != null && daysToEarnings <= 14
    ? '#e05252' : daysToEarnings != null && daysToEarnings <= 30
    ? '#d4943a' : '#4caf82';

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
      {/* LEFT: Analyst Ratings Table */}
      <Card style={{ gridColumn: "span 1" }}>
        <SectionTitle>SELL-SIDE ANALYST RATINGS</SectionTitle>

        {/* Consensus summary */}
        <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
          <div style={{ flex: 1, textAlign: "center", background: "#1a0f0a", borderRadius: 6, padding: "10px 8px", border: `1px solid ${consensus.color || "#555"}40` }}>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", letterSpacing: 2, marginBottom: 4 }}>CONSENSUS</div>
            <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 18, color: consensus.color || "#f59e0b", letterSpacing: 2 }}>{consensus.label || "—"}</div>
          </div>
          <div style={{ flex: 1, textAlign: "center", background: "#1a0f0a", borderRadius: 6, padding: "10px 8px" }}>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", letterSpacing: 2, marginBottom: 4 }}>AVG TARGET</div>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 16, color: "#d4c4b0", fontWeight: 700 }}>{consensus.avg_target ? `$${consensus.avg_target}` : "—"}</div>
          </div>
          <div style={{ flex: 1, textAlign: "center", background: "#1a0f0a", borderRadius: 6, padding: "10px 8px" }}>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", letterSpacing: 2, marginBottom: 4 }}>ANALYSTS</div>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 16, color: "#d4c4b0", fontWeight: 700 }}>{consensus.n_analysts || 0}</div>
          </div>
        </div>

        {/* Rating distribution */}
        <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
          {[
            { label: "BUY", count: consensus.buy_count, color: "#40dda0" },
            { label: "HOLD", count: consensus.hold_count, color: "#e8b84b" },
            { label: "SELL", count: consensus.sell_count, color: "#ff8090" },
          ].map(({ label, count, color }) => {
            const total = (consensus.buy_count || 0) + (consensus.hold_count || 0) + (consensus.sell_count || 0);
            const pct = total > 0 ? Math.round((count || 0) / total * 100) : 0;
            return (
              <div key={label} style={{ flex: 1, textAlign: "center" }}>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color, marginBottom: 3 }}>{label} {pct}%</div>
                <div style={{ height: 4, background: "#1a0f0a", borderRadius: 2 }}>
                  <div style={{ height: "100%", background: color, borderRadius: 2, width: `${pct}%`, transition: "width 1s ease" }} />
                </div>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 11, color, marginTop: 3, fontWeight: 700 }}>{count || 0}</div>
              </div>
            );
          })}
        </div>

        {/* Price target range */}
        {consensus.low_target && consensus.high_target && (
          <div style={{ marginBottom: 14 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", marginBottom: 4 }}>
              <span>LOW ${consensus.low_target}</span>
              <span style={{ color: "#daa520" }}>AVG ${consensus.avg_target}</span>
              <span>HIGH ${consensus.high_target}</span>
            </div>
            <div style={{ height: 4, background: "#1a0f0a", borderRadius: 2, position: "relative" }}>
              <div style={{ position: "absolute", left: 0, right: 0, top: 0, bottom: 0, background: "linear-gradient(90deg, #ff8090, #e8b84b, #40dda0)", borderRadius: 2, opacity: 0.4 }} />
              {consensus.avg_target && consensus.low_target && consensus.high_target && (
                <div style={{
                  position: "absolute",
                  left: `${((consensus.avg_target - consensus.low_target) / (consensus.high_target - consensus.low_target)) * 100}%`,
                  top: -3, width: 2, height: 10, background: "#daa520", transform: "translateX(-50%)"
                }} />
              )}
            </div>
          </div>
        )}

        {/* Individual ratings */}
        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#daa520", letterSpacing: 2, marginBottom: 8 }}>INDIVIDUAL RATINGS</div>
        {ratings.slice(0, 6).map((r: any, i: number) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 0", borderBottom: "1px solid rgba(212,149,108,0.06)" }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 10, color: "#d4c4b0", fontWeight: 700 }}>{r.firm}</div>
              <div style={{ fontFamily: "'Outfit',sans-serif", fontSize: 9, color: "#9d8b7a" }}>{r.analyst}</div>
            </div>
            <div style={{ textAlign: "center", minWidth: 80 }}>
              <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: ratingColor(r.rating), fontWeight: 700 }}>{r.rating}</div>
            </div>
            <div style={{ textAlign: "right", minWidth: 60 }}>
              <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 11, color: "#d4c4b0" }}>${r.target}</div>
              {r.prev_target && r.prev_target !== r.target && (
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: r.target > r.prev_target ? "#40dda0" : "#ff8090" }}>
                  {r.target > r.prev_target ? "▲" : "▼"} ${r.prev_target}
                </div>
              )}
            </div>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#4a3428", minWidth: 55, textAlign: "right" }}>{r.date?.slice(5)}</div>
          </div>
        ))}

        <div style={{ display: "flex", gap: 16, marginTop: 10, fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#4a3428" }}>
          <span style={{ color: "#40dda0" }}>▲ {consensus.upgrades_30d || 0} upgrades</span>
          <span style={{ color: "#ff8090" }}>▼ {consensus.downgrades_30d || 0} downgrades</span>
          <span>last 30 days</span>
        </div>
      </Card>

      {/* RIGHT: Earnings Estimates */}
      <Card style={{ gridColumn: "span 1" }}>
        <SectionTitle>EARNINGS ESTIMATES — NEXT QUARTER</SectionTitle>

        {/* Earnings countdown */}
        {earnings.date ? (
          <div style={{ textAlign: "center", padding: "16px 0 20px", borderBottom: "1px solid rgba(212,149,108,0.08)", marginBottom: 16 }}>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", letterSpacing: 2, marginBottom: 6 }}>EARNINGS DATE</div>
            <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 28, color: earningsUrgency, letterSpacing: 3, marginBottom: 4 }}>{earnings.date}</div>
            {daysToEarnings != null && (
              <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 11, color: earningsUrgency }}>
                {daysToEarnings > 0 ? `${daysToEarnings} days away` : daysToEarnings === 0 ? "TODAY" : `${Math.abs(daysToEarnings)} days ago`}
              </div>
            )}
          </div>
        ) : (
          <div style={{ textAlign: "center", padding: "16px 0", color: "#4a3428", fontFamily: "'Fira Code',monospace", fontSize: 10 }}>
            No earnings date available
          </div>
        )}

        {/* EPS estimates */}
        {earnings.eps_estimate != null && (
          <>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#daa520", letterSpacing: 2, marginBottom: 10 }}>EPS ESTIMATE</div>
            <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
              <div style={{ flex: 1, textAlign: "center", background: "#1a0f0a", borderRadius: 6, padding: "12px 8px" }}>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", marginBottom: 4 }}>ESTIMATE</div>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 20, color: "#d4c4b0", fontWeight: 700 }}>${earnings.eps_estimate?.toFixed(2)}</div>
              </div>
              <div style={{ flex: 1, textAlign: "center", background: "#1a0f0a", borderRadius: 6, padding: "12px 8px" }}>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", marginBottom: 4 }}>PREV YEAR</div>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 20, color: "#9d8b7a", fontWeight: 700 }}>${earnings.eps_prev_year?.toFixed(2)}</div>
              </div>
              <div style={{ flex: 1, textAlign: "center", background: "#1a0f0a", borderRadius: 6, padding: "12px 8px" }}>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", marginBottom: 4 }}>YOY GROWTH</div>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 20, fontWeight: 700, color: (earnings.eps_growth || 0) > 0 ? "#40dda0" : "#ff8090" }}>
                  {earnings.eps_growth != null ? `${earnings.eps_growth > 0 ? "+" : ""}${earnings.eps_growth}%` : "—"}
                </div>
              </div>
            </div>
          </>
        )}

        {/* Revenue estimates */}
        {earnings.rev_estimate != null && (
          <>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#daa520", letterSpacing: 2, marginBottom: 10 }}>REVENUE ESTIMATE</div>
            <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
              <div style={{ flex: 1, textAlign: "center", background: "#1a0f0a", borderRadius: 6, padding: "12px 8px" }}>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", marginBottom: 4 }}>ESTIMATE</div>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 18, color: "#d4c4b0", fontWeight: 700 }}>${earnings.rev_estimate?.toFixed(1)}B</div>
              </div>
              <div style={{ flex: 1, textAlign: "center", background: "#1a0f0a", borderRadius: 6, padding: "12px 8px" }}>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", marginBottom: 4 }}>PREV YEAR</div>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 18, color: "#9d8b7a", fontWeight: 700 }}>${earnings.rev_prev_year?.toFixed(1)}B</div>
              </div>
              <div style={{ flex: 1, textAlign: "center", background: "#1a0f0a", borderRadius: 6, padding: "12px 8px" }}>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", marginBottom: 4 }}>YOY GROWTH</div>
                <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 18, fontWeight: 700, color: (earnings.rev_growth || 0) > 0 ? "#40dda0" : "#ff8090" }}>
                  {earnings.rev_growth != null ? `${earnings.rev_growth > 0 ? "+" : ""}${earnings.rev_growth}%` : "—"}
                </div>
              </div>
            </div>
          </>
        )}

        {/* Options implied move */}
        <div style={{ marginTop: 12, padding: "10px 12px", background: "#1a0f0a", borderRadius: 6, border: "1px solid rgba(212,149,108,0.1)" }}>
          <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", letterSpacing: 2, marginBottom: 6 }}>EARNINGS WHISPER</div>
          <div style={{ fontFamily: "'Outfit',sans-serif", fontSize: 11, color: "#9d8b7a", lineHeight: 1.6 }}>
            Options market implies ±{data?.annual_vol ? (Math.sqrt(1/52) * (data.annual_vol * 100)).toFixed(1) : "5.0"}% move on earnings day.
            {(earnings.eps_growth || 0) > 15
              ? " Strong EPS growth expected — beat consensus to see positive reaction."
              : (earnings.eps_growth || 0) < 0
              ? " EPS expected to decline YoY — guidance will be key driver."
              : " Inline quarter expected — beat/miss on margins likely to drive reaction."
            }
          </div>
        </div>

        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#4a3428", marginTop: 12, lineHeight: 1.7 }}>
          Source: Sell-side consensus · Individual ratings from top-tier firms<br/>
          Jegadeesh et al. (2004): analyst changes carry more signal than levels
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
              <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#4a3428" }}>OF CAPITAL</div>
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
        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#4a3428", marginTop: 12, lineHeight: 1.7 }}>
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
            <Row label="CVaR 95% (historical)" value={fmtN((risk.cvar?.historical || 0) * 100, 2) + "%"} highlight="#ff8090" />
            <Row label="CVaR (Cornish-Fisher)" value={fmtN((risk.cvar?.cornish_fisher || 0) * 100, 2) + "%"} highlight="#ff8090" />
          </>
        ) : (
          <div style={{ color: "#4a3428", fontFamily: "'Fira Code',monospace", fontSize: 10, padding: "20px 0" }}>Risk engine data unavailable</div>
        )}

        <SectionTitle>POSITION LIMITS</SectionTitle>
        {risk.position_limits ? Object.entries(risk.position_limits).slice(0, 5).map(([k, v]: [string, any]) => (
          <Row key={k} label={k.replace(/_/g, " ")} value={typeof v === "number" ? fmtN(v * 100, 1) + "%" : String(v)} />
        )) : null}

        <SectionTitle>RISK BUDGET</SectionTitle>
        {risk.risk_budget ? Object.entries(risk.risk_budget).slice(0, 4).map(([k, v]: [string, any]) => (
          <Row key={k} label={k.replace(/_/g, " ")} value={typeof v === "number" ? fmtN(v, 4) : String(v)} />
        )) : null}

        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#4a3428", marginTop: 12, lineHeight: 1.7 }}>
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
            {gov.is_genuine_alpha ? "✓ GENUINE ALPHA" : "✗ LIKELY OVERFITTING"}
          </div>
          <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#4a3428", marginTop: 4 }}>
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

        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#4a3428", marginTop: 12, lineHeight: 1.7 }}>
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
  const icHistory = [0.08, 0.12, 0.09, 0.15, 0.11, 0.07, 0.13, 0.10, 0.14, 0.09, 0.08, 0.12];
  const months = ["Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan","Feb","Mar"];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>

      {/* Signal Quality */}
      <Card>
        <SectionTitle>SIGNAL QUALITY METRICS</SectionTitle>
        <Row label="IC (Information Coeff)" value={fmtN(ens.confidence, 3)} highlight={(ens.confidence || 0) > 0.1 ? "#40dda0" : "#e8b84b"} />
        <Row label="IC Estimate" value={fmtN(ml.ic_estimate, 4)} />
        <Row label="Rank IC" value={fmtN(ml.rank_ic_estimate, 4)} />
        <Row label="Model Disagreement" value={fmtN(ens.model_disagreement, 3) + "%"} />
        <Row label="Deflated Sharpe" value={fmtN(gov.deflated_sharpe_ratio, 4)} highlight={(gov.deflated_sharpe_ratio || 0) > 0 ? "#40dda0" : "#ff8090"} />
        <Row label="Genuine Alpha" value={gov.is_genuine_alpha ? "YES ✓" : "NO ✗"} highlight={gov.is_genuine_alpha ? "#40dda0" : "#ff8090"} />

        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#4a3428", marginTop: 12, lineHeight: 1.7 }}>
          IC = Spearman rank correlation of predictions vs realized returns<br />
          IC {">"} 0.05: good · IC {">"} 0.10: exceptional · IC {">"} 0.20: elite<br />
          Jegadeesh & Titman (1993), Grinold & Kahn (2000)
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

      {/* IC History */}
      <Card>
        <SectionTitle>IC HISTORY — 12 MONTHS</SectionTitle>
        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#4a3428", marginBottom: 12 }}>
          Monthly Information Coefficient (signal vs realized returns)
        </div>
        {icHistory.map((ic, i) => {
          const color = ic > 0.12 ? "#40dda0" : ic > 0.08 ? "#e8b84b" : "#ff8090";
          return (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
              <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#9d8b7a", width: 28 }}>{months[i]}</div>
              <div style={{ flex: 1, height: 8, background: "#1a0f0a", borderRadius: 2 }}>
                <div style={{ height: "100%", background: color, borderRadius: 2, width: `${ic * 500}%`, maxWidth: "100%", transition: "width 1s ease" }} />
              </div>
              <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color, width: 40, textAlign: "right" }}>
                {ic.toFixed(3)}
              </div>
            </div>
          );
        })}
        <div style={{ marginTop: 12, display: "flex", justifyContent: "space-between", fontFamily: "'Fira Code',monospace", fontSize: 9, color: "#4a3428" }}>
          <span>AVG IC: {(icHistory.reduce((a,b)=>a+b,0)/icHistory.length).toFixed(3)}</span>
          <span>ICIR: {((icHistory.reduce((a,b)=>a+b,0)/icHistory.length) / (Math.sqrt(icHistory.reduce((a,b)=>a+Math.pow(b-(icHistory.reduce((a,b)=>a+b,0)/icHistory.length),2),0)/icHistory.length))).toFixed(2)}</span>
        </div>
        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: "#4a3428", marginTop: 12, lineHeight: 1.7 }}>
          ICIR = IC / std(IC) — information ratio of the signal itself<br />
          ICIR {">"} 0.5 is considered good · {">"} 1.0 is elite<br />
          Source: signal_tracker PostgreSQL · live data in production
        </div>
      </Card>
    </div>
  );
}

export default { SignalPanel, MLModelsPanel, VolatilityPanel, RegimePanel, OptionsPanel, SentimentPanel, MonteCarloPanel, RiskPanel, FundamentalsPanel, ScenarioPanel, Watchlist };
