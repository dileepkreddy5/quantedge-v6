// ============================================================
// QuantEdge v6.0 — Panel Skill Matrix (homepage centerpiece)
// Publishes the panel ensemble's own out-of-sample skill, including
// when that skill is nil. Reads GET /api/v6/ml/panel/skill.
// Renders nothing if the panel has never been trained.
// ============================================================

import React, { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

interface Horizon {
  horizon_days: number; label: string;
  ic_all_dates: number | null; ic_xgboost: number | null; ic_lightgbm: number | null;
  ic_t_stat: number | null; ic_hit_rate: number | null;
  n_scoring_dates: number | null; n_independent_windows: number | null;
  ic_independent: number | null; independent_measurable: boolean | null;
  hac_stable: boolean | null; hac_lag_obs: number | null;
  reliable: boolean | null; confidence_note: string | null;
}
interface Skill {
  trained_at: string; panel: string | null;
  n_tickers: number; n_features: number; split_date: string;
  horizons: Horizon[]; any_reliable: boolean; method: string; disclaimer: string;
}

const C = {
  panel: '#1a0f0a', panel2: '#241610', border: '#3a2920', border2: '#4a3428',
  gold: '#daa520', caramel: '#d4956c', cocoa: '#8a7560', dust: '#9d8b7a',
  latte: '#d4c4b0', cream: '#f4e8d8', bull: '#22c55e', bear: '#ef4444', warn: '#f59e0b',
};

// IC is a correlation: ±0.15 is a wide band for cross-sectional equity ranking.
const IC_SCALE = 0.15;

const fmtIC = (v: number | null) => v == null ? '—' : (v >= 0 ? '+' : '') + v.toFixed(4);
const fmtT  = (v: number | null) => v == null ? '—' : (v >= 0 ? '+' : '') + v.toFixed(2);

const ago = (iso: string) => {
  const h = (Date.now() - new Date(iso).getTime()) / 36e5;
  if (h < 1) return `${Math.max(1, Math.round(h * 60))}m ago`;
  if (h < 48) return `${Math.round(h)}h ago`;
  return `${Math.round(h / 24)}d ago`;
};

/** Diverging bar centred on zero: red left, gold right. */
const ICBar: React.FC<{ ic: number | null }> = ({ ic }) => {
  const v = ic ?? 0;
  const pct = Math.min(Math.abs(v) / IC_SCALE, 1) * 50;
  const pos = v >= 0;
  return (
    <div style={{ position: 'relative', height: 22, background: 'rgba(0,0,0,0.28)',
                  borderRadius: 3, overflow: 'hidden', minWidth: 120 }}>
      <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0,
                    width: 1, background: C.border2, zIndex: 2 }} />
      <div style={{
        position: 'absolute', top: 3, bottom: 3,
        left: pos ? '50%' : `${50 - pct}%`, width: `${pct}%`,
        background: pos
          ? `linear-gradient(90deg, rgba(218,165,32,0.45), ${C.gold})`
          : `linear-gradient(90deg, ${C.bear}, rgba(239,68,68,0.45))`,
        borderRadius: 2, transition: 'all .4s ease',
      }} />
    </div>
  );
};

const PanelSkillMatrix: React.FC = () => {
  const [d, setD] = useState<Skill | null>(null);
  const [open, setOpen] = useState<number | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const r = await api.get('/api/v6/ml/panel/skill');
        if (r.data?.horizons?.length) setD(r.data);
      } catch { /* never trained — render nothing */ }
    })();
  }, []);

  if (!d) return null;

  const th: React.CSSProperties = {
    textAlign: 'left', padding: '9px 12px', fontFamily: "'Fira Code',monospace",
    fontSize: 9, letterSpacing: 1.4, color: C.cocoa, fontWeight: 500,
    borderBottom: `1px solid ${C.border}`, whiteSpace: 'nowrap',
  };
  const td: React.CSSProperties = {
    padding: '13px 12px', fontFamily: "'Fira Code',monospace", fontSize: 12,
    color: C.latte, borderBottom: `1px solid rgba(58,41,32,0.55)`, whiteSpace: 'nowrap',
  };

  return (
    <section style={{ position: 'relative', zIndex: 1, maxWidth: 1400, margin: '0 auto 64px', padding: '0 4rem' }}>
      <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 10,
                    letterSpacing: 3, color: C.cocoa, marginBottom: 10 }}>
        MEASURED SKILL — NOT MARKETING
      </div>
      <h2 style={{ fontSize: 38, fontWeight: 800, color: C.cream, margin: '0 0 14px',
                   letterSpacing: -0.5 }}>
        WHAT THE MODELS ACTUALLY SCORE
      </h2>
      <p style={{ color: C.dust, fontSize: 15, lineHeight: 1.75, maxWidth: 780, marginBottom: 26 }}>
        Every horizon publishes its own held-out rank information coefficient and the
        Newey-West t-statistic computed on that same sample. Nothing here is a
        backtest return. When a model ranks backwards on data it never saw, the table
        says so.
      </p>

      <div style={{ background: C.panel, border: `1px solid ${C.border}`,
                    borderRadius: 10, overflow: 'hidden' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 26, padding: '16px 20px',
                      borderBottom: `1px solid ${C.border}`, background: C.panel2 }}>
          {[
            ['PANEL', d.panel ? String(d.n_tickers) + ' tickers' : '—'],
            ['FEATURES', String(d.n_features)],
            ['TRAIN/TEST SPLIT', d.split_date],
            ['LAST RETRAIN', ago(d.trained_at)],
            ['VALIDATED HORIZONS', d.any_reliable ? 'see table' : '0 of ' + d.horizons.length],
          ].map(([k, v]) => (
            <div key={k}>
              <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8.5,
                            letterSpacing: 1.4, color: C.cocoa, marginBottom: 4 }}>{k}</div>
              <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 13,
                            color: k === 'VALIDATED HORIZONS' && !d.any_reliable ? C.warn : C.cream }}>{v}</div>
            </div>
          ))}
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: 900 }}>
            <thead>
              <tr>
                <th style={th}>HORIZON</th>
                <th style={{ ...th, minWidth: 150 }}>RANK-IC (ALL DATES)</th>
                <th style={th}>IC</th>
                <th style={th}>NW t</th>
                <th style={th}>HIT RATE</th>
                <th style={th}>SCORING DATES</th>
                <th style={th}>INDEP. WINDOWS</th>
                <th style={th}>XGB / LGB</th>
                <th style={th}>VERDICT</th>
              </tr>
            </thead>
            <tbody>
              {d.horizons.map(h => {
                const isOpen = open === h.horizon_days;
                return (
                  <React.Fragment key={h.horizon_days}>
                    <tr onClick={() => setOpen(isOpen ? null : h.horizon_days)}
                        style={{ cursor: 'pointer', background: isOpen ? 'rgba(218,165,32,0.04)' : 'transparent' }}>
                      <td style={{ ...td, color: C.cream, fontWeight: 700 }}>
                        {h.label}
                        <span style={{ color: C.cocoa, fontSize: 10 }}> · {h.horizon_days}d</span>
                      </td>
                      <td style={{ ...td, width: 170 }}><ICBar ic={h.ic_all_dates} /></td>
                      <td style={{ ...td, color: (h.ic_all_dates ?? 0) >= 0 ? C.gold : C.bear }}>
                        {fmtIC(h.ic_all_dates)}
                      </td>
                      <td style={{ ...td, color: Math.abs(h.ic_t_stat ?? 0) >= 2 ? C.gold : C.dust }}>
                        {fmtT(h.ic_t_stat)}
                      </td>
                      <td style={td}>{h.ic_hit_rate == null ? '—' : (h.ic_hit_rate * 100).toFixed(1) + '%'}</td>
                      <td style={td}>{h.n_scoring_dates ?? '—'}</td>
                      <td style={td}>
                        <span style={{ color: h.independent_measurable ? C.latte : C.cocoa }}>
                          {h.n_independent_windows ?? '—'}w
                        </span>
                        <span style={{ color: C.cocoa, fontSize: 10.5 }}>
                          {' '}{h.independent_measurable ? fmtIC(h.ic_independent) : '· n/a'}
                        </span>
                      </td>
                      <td style={{ ...td, fontSize: 10.5, color: C.cocoa }}>
                        {fmtIC(h.ic_xgboost)} / {fmtIC(h.ic_lightgbm)}
                      </td>
                      <td style={td}>
                        <span style={{
                          fontFamily: "'Fira Code',monospace", fontSize: 9, letterSpacing: 1,
                          padding: '4px 9px', borderRadius: 3,
                          border: `1px solid ${h.reliable ? C.gold : C.border2}`,
                          color: h.reliable ? C.gold : C.dust,
                          background: h.reliable ? 'rgba(218,165,32,0.10)' : 'transparent',
                        }}>
                          {h.reliable ? 'VALIDATED' : 'NOT VALIDATED'}
                        </span>
                      </td>
                    </tr>
                    {isOpen && h.confidence_note && (
                      <tr>
                        <td colSpan={9} style={{
                          padding: '0 12px 15px', borderBottom: `1px solid rgba(58,41,32,0.55)`,
                          background: 'rgba(218,165,32,0.04)',
                        }}>
                          <div style={{
                            color: C.latte, fontSize: 12.5, lineHeight: 1.75, maxWidth: 900,
                            borderLeft: `2px solid ${C.caramel}`, paddingLeft: 14,
                          }}>
                            {h.confidence_note}
                            {h.hac_stable === false && (
                              <div style={{ color: C.warn, marginTop: 7, fontSize: 11.5 }}>
                                HAC correction unstable at lag {h.hac_lag_obs} — this horizon
                                cannot be measured on the available validation window.
                              </div>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>

        <div style={{ padding: '15px 20px', borderTop: `1px solid ${C.border}`,
                      background: C.panel2, color: C.cocoa, fontSize: 11.5, lineHeight: 1.7 }}>
          {d.method} <span style={{ color: C.dust }}>{d.disclaimer}</span>
        </div>
      </div>
    </section>
  );
};

export default PanelSkillMatrix;
