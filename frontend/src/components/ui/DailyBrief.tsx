// ============================================================
// QuantEdge v6.0 — Daily Brief (homepage lead)
// The machine's own morning note: current regime, today's ranked
// calls, and settled calls scored against their realized 5-day
// return — misses shown in red, not hidden.
// ============================================================

import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../../auth/authStore';

const C = { s0: '#100a07', s2: '#241610', s3: '#2f1e16', b1: '#3a2920', b2: '#4a3428',
            gold: '#daa520', caramel: '#d4956c', cocoa: '#8a7560', dust: '#9d8b7a',
            latte: '#d4c4b0', cream: '#f4e8d8', bull: '#22c55e', bear: '#ef4444', warn: '#f59e0b' };
const mono = "'Fira Code',monospace";

interface Call { ticker: string; generated_at: string; ensemble_signal: number | null;
                 ensemble_direction: string | null; cvar_95: number | null; }
interface Settled extends Call { ret_5d: number | null; call_correct: boolean | null; barrier_hit: string | null; }
interface Brief {
  available: boolean;
  regime: { ticker: string; generated_at: string; hmm_regime: string; hmm_confidence: number;
            garch_regime: string; garch_vol_forecast: number; kalman_trend: number } | null;
  strongest: Call[]; weakest: Call[]; settled: Settled[];
  performance: { date: string; ic_21d: number | null; hit_rate: number | null; n_signals: number | null } | null;
  note: string;
}

const pct = (v: number | null, d = 1) => v == null ? '—' : (v >= 0 ? '+' : '') + (v * 100).toFixed(d) + '%';
const sig = (v: number | null) => v == null ? '—' : (v >= 0 ? '+' : '') + v.toFixed(2);

const REGIME_TONE: Record<string, string> = {
  BULL_LOW_VOL: C.bull, BULL_HIGH_VOL: C.warn, BEAR_LOW_VOL: C.caramel,
  BEAR_HIGH_VOL: C.bear, MEAN_REVERT: C.dust,
};

const CallRow: React.FC<{ c: Call; onClick: () => void }> = ({ c, onClick }) => (
  <div onClick={onClick} style={{
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '11px 14px', cursor: 'pointer', borderRadius: 6,
    border: `1px solid ${C.b1}`, background: 'rgba(0,0,0,0.18)', marginBottom: 7,
  }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <span style={{ fontFamily: mono, fontSize: 14, fontWeight: 700, color: C.cream }}>{c.ticker}</span>
      <span style={{
        fontFamily: mono, fontSize: 8.5, letterSpacing: 1, padding: '3px 7px', borderRadius: 3,
        border: `1px solid ${c.ensemble_direction === 'LONG' ? C.bull : C.bear}`,
        color: c.ensemble_direction === 'LONG' ? C.bull : C.bear,
      }}>{c.ensemble_direction || '—'}</span>
    </div>
    <div style={{ display: 'flex', gap: 18, fontFamily: mono, fontSize: 11 }}>
      <span style={{ color: (c.ensemble_signal ?? 0) >= 0 ? C.gold : C.bear }}>{sig(c.ensemble_signal)}</span>
      <span style={{ color: C.cocoa }}>CVaR {pct(c.cvar_95)}</span>
    </div>
  </div>
);

const DailyBrief: React.FC = () => {
  const navigate = useNavigate();
  const [d, setD] = useState<Brief | null>(null);

  useEffect(() => {
    (async () => {
      try { const r = await api.get('/api/v6/brief/today'); if (r.data?.available) setD(r.data); } catch {}
    })();
  }, []);

  if (!d || !d.regime) return null;
  const rg = d.regime;
  const tone = REGIME_TONE[rg.hmm_regime] || C.latte;
  const settledScore = d.settled.filter(s => s.call_correct != null);
  const hits = settledScore.filter(s => s.call_correct).length;

  return (
    <section style={{ position: 'relative', zIndex: 1, maxWidth: 1400, margin: '0 auto 72px', padding: '0 4rem' }}>
      <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: 3, color: C.cocoa, marginBottom: 10 }}>
        TODAY'S BRIEF · WRITTEN BY THE SYSTEM · {new Date(rg.generated_at).toLocaleDateString()}
      </div>

      {/* Regime banner */}
      <div style={{
        display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 30,
        background: `linear-gradient(120deg, ${C.s2}, ${C.s0})`,
        border: `1px solid ${C.b1}`, borderLeft: `4px solid ${tone}`,
        borderRadius: 10, padding: '22px 26px', marginBottom: 14,
      }}>
        <div>
          <div style={{ fontFamily: mono, fontSize: 8.5, letterSpacing: 1.6, color: C.cocoa, marginBottom: 6 }}>MARKET REGIME (SPY)</div>
          <div style={{ fontFamily: mono, fontSize: 26, fontWeight: 700, color: tone, letterSpacing: 1 }}>
            {rg.hmm_regime.replace(/_/g, ' ')}
          </div>
          <div style={{ fontFamily: mono, fontSize: 10.5, color: C.dust, marginTop: 5 }}>
            {(rg.hmm_confidence * 100).toFixed(1)}% confidence
          </div>
        </div>
        <div style={{ display: 'flex', gap: 34, flexWrap: 'wrap' }}>
          <div>
            <div style={{ fontFamily: mono, fontSize: 8.5, letterSpacing: 1.4, color: C.cocoa, marginBottom: 5 }}>VOLATILITY</div>
            <div style={{ fontFamily: mono, fontSize: 15, color: C.latte }}>{rg.garch_regime} · {(rg.garch_vol_forecast * 100).toFixed(1)}% ann.</div>
          </div>
          <div>
            <div style={{ fontFamily: mono, fontSize: 8.5, letterSpacing: 1.4, color: C.cocoa, marginBottom: 5 }}>KALMAN TREND</div>
            <div style={{ fontFamily: mono, fontSize: 15, color: rg.kalman_trend >= 0 ? C.bull : C.bear }}>
              {rg.kalman_trend >= 0 ? 'RISING' : 'FALLING'} {sig(rg.kalman_trend)}
            </div>
          </div>
          {d.performance?.hit_rate != null && (
            <div>
              <div style={{ fontFamily: mono, fontSize: 8.5, letterSpacing: 1.4, color: C.cocoa, marginBottom: 5 }}>MODEL HIT RATE (21D)</div>
              <div style={{ fontFamily: mono, fontSize: 15, color: C.latte }}>{(d.performance.hit_rate * 100).toFixed(0)}%</div>
            </div>
          )}
        </div>
      </div>

      {/* Three columns: strongest / weakest / settled */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 14 }}>
        <div style={{ background: C.s2, border: `1px solid ${C.b1}`, borderRadius: 10, padding: '18px 18px 12px' }}>
          <div style={{ fontFamily: mono, fontSize: 9.5, letterSpacing: 1.6, color: C.gold, marginBottom: 12 }}>▲ STRONGEST SIGNALS TODAY</div>
          {d.strongest.map(c => <CallRow key={c.ticker} c={c} onClick={() => navigate(`/dashboard?ticker=${c.ticker}`)} />)}
        </div>
        <div style={{ background: C.s2, border: `1px solid ${C.b1}`, borderRadius: 10, padding: '18px 18px 12px' }}>
          <div style={{ fontFamily: mono, fontSize: 9.5, letterSpacing: 1.6, color: C.bear, marginBottom: 12 }}>▼ WEAKEST SIGNALS TODAY</div>
          {d.weakest.map(c => <CallRow key={c.ticker} c={c} onClick={() => navigate(`/dashboard?ticker=${c.ticker}`)} />)}
        </div>
        <div style={{ background: C.s2, border: `1px solid ${C.b1}`, borderRadius: 10, padding: '18px 18px 12px' }}>
          <div style={{ fontFamily: mono, fontSize: 9.5, letterSpacing: 1.6, color: C.latte, marginBottom: 12 }}>
            ⚖ SETTLED CALLS {settledScore.length > 0 && `· ${hits}/${settledScore.length} CORRECT`}
          </div>
          {d.settled.slice(0, 5).map(s => (
            <div key={s.ticker + s.generated_at} style={{
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              padding: '9px 14px', borderRadius: 6, marginBottom: 7,
              border: `1px solid ${C.b1}`, background: 'rgba(0,0,0,0.18)',
            }}>
              <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                <span style={{ fontFamily: mono, fontSize: 13, fontWeight: 700, color: C.cream }}>{s.ticker}</span>
                <span style={{ fontFamily: mono, fontSize: 10, color: C.cocoa }}>
                  called {sig(s.ensemble_signal)}
                </span>
              </div>
              <div style={{ display: 'flex', gap: 10, alignItems: 'center', fontFamily: mono, fontSize: 11 }}>
                <span style={{ color: (s.ret_5d ?? 0) >= 0 ? C.bull : C.bear }}>5d {pct(s.ret_5d)}</span>
                {s.call_correct != null && (
                  <span style={{ color: s.call_correct ? C.bull : C.bear, fontSize: 13 }}>
                    {s.call_correct ? '✓' : '✗'}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ marginTop: 12, fontFamily: mono, fontSize: 10, color: C.cocoa, lineHeight: 1.7 }}>{d.note}</div>
    </section>
  );
};

export default DailyBrief;
