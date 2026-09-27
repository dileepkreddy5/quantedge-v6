import VolumeTracker from './VolumeTracker';
import PatternChart from './PatternChart';
import SituationMode from './SituationMode';
import FormationsDeep from './FormationsDeep';
import EvolutionMode from './EvolutionMode';
import AnalogsDeep from './AnalogsDeep';
// ============================================================
// QuantEdge v6.0 — Pattern Lab tab
// Historical analogs of the ticker's current trajectory: overlay
// chart, outcome distributions vs base rate, volume/regime splits.
// Distributions, never predictions — the caveat renders on-panel.
// ============================================================

import React, { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

const C = { s0: '#100a07', s2: '#241610', b1: '#3a2920', b2: '#4a3428',
            gold: '#daa520', caramel: '#d4956c', cocoa: '#8a7560', dust: '#9d8b7a',
            latte: '#d4c4b0', cream: '#f4e8d8', bull: '#22c55e', bear: '#ef4444', warn: '#f59e0b' };
const mono = "'Fira Code',monospace";

interface Dist { n: number; positive_pct: number; median_pct: number; mean_pct: number; p10_pct: number; p90_pct: number; }
interface Analog { ticker: string; start: string; end?: string | null; duration_sessions?: number;
                   regime?: string; volume_slope?: number; similarity_pct: number; trajectory: number[];
                   fwd: Record<string, number | null>; }
interface Result {
  ticker: string; as_of: string; window_days: number; episodes: number;
  distributions: Record<string, Dist>; base_rates: Record<string, Dist>;
  excess_vs_spy?: Record<string, Dist | null>;
  method?: Record<string, string>; episode_date_range?: [string, string];
  splits: { volume_slope: Record<string, Dist | null>; regime: Record<string, Dist | null> };
  analogs: Analog[]; query_trajectory: number[]; caveat: string;
}

/** SVG overlay: query trajectory bold gold, analogs faint. */


// ── Formations mode ──────────────────────────────────────────
interface FormStats { positive_pct: number; median_pct: number; p25_pct: number; p75_pct: number; }
interface Formation { occurrences: number; raw_detections: number; median_duration: number;
                      breakout_up_pct: number | null; fwd20: FormStats | null;
                      examples: { ticker: string; start: string; end: string; duration: number;
                                  breakout_up: boolean; fwd_5d: number; fwd_20d: number; fwd_60d: number }[]; }
interface FormArt { generated: string; method: string; universe: number;
                    formations: Record<string, Formation>; }

const FORM_LABEL: Record<string, string> = {
  head_shoulders: 'HEAD & SHOULDERS', inv_head_shoulders: 'INV. HEAD & SHOULDERS',
  double_top: 'DOUBLE TOP', double_bottom: 'DOUBLE BOTTOM',
  triple_top: 'TRIPLE TOP', triple_bottom: 'TRIPLE BOTTOM',
  ascending_triangle: 'ASCENDING TRIANGLE', descending_triangle: 'DESCENDING TRIANGLE',
  symmetrical_triangle: 'SYMMETRICAL TRIANGLE', rectangle: 'RECTANGLE',
  rising_wedge: 'RISING WEDGE', falling_wedge: 'FALLING WEDGE',
};

// ── Conditions mode (momentum / extremes / volatility lenses) ─
interface CellStats { n: number; positive_pct: number; median_pct: number; p25_pct: number; p75_pct: number; }
interface CondRes { ticker: string; generated: string; samples: number; note: string;
                    base: Record<string, CellStats | null>;
                    conditions: Record<string, { value: number; quintile: number;
                                                 cell: Record<string, any> | null }>; }

const COND_SETS: Record<string, { title: string; keys: string[]; fmt: (v: number) => string }> = {
  momentum: { title: 'MOMENTUM & REVERSAL — WHERE THIS TICKER SITS, AND WHAT FOLLOWED HISTORICALLY',
              keys: ['mom_20d', 'mom_60d', 'mom_120d', 'mom_252d'],
              fmt: v => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%` },
  extremes: { title: 'PRICE EXTREMES — DISTANCE FROM 52-WEEK HIGH',
              keys: ['dist_52w_high'], fmt: v => `${(v * 100).toFixed(1)}%` },
  volatility: { title: 'VOLATILITY — 21D REALIZED VS OWN HISTORY',
                keys: ['vol_21d_pctile'], fmt: v => `${(v * 100).toFixed(0)}th pctile` },
};
const COND_LABEL: Record<string, string> = {
  mom_20d: '20-DAY MOMENTUM', mom_60d: '60-DAY MOMENTUM',
  mom_120d: '120-DAY MOMENTUM', mom_252d: '252-DAY MOMENTUM',
  dist_52w_high: 'VS 52-WEEK HIGH', vol_21d_pctile: 'VOLATILITY PERCENTILE',
};

const ConditionsMode: React.FC<{ ticker: string; set: string }> = ({ ticker, set }) => {
  const [d, setD] = useState<CondRes | null>(null);
  const [err, setErr] = useState('');
  useEffect(() => {
    let dead = false;
    (async () => {
      setD(null); setErr('');
      try { const r = await api.get(`/api/v6/patterns/conditions/${ticker}`); if (!dead) setD(r.data); }
      catch (e: any) { if (!dead) setErr(e?.response?.data?.detail || 'condition scan unavailable'); }
    })();
    return () => { dead = true; };
  }, [ticker]);
  if (err) return <div style={{ fontFamily: mono, fontSize: 11, color: C.warn }}>{err}</div>;
  if (!d) return <div style={{ fontFamily: mono, fontSize: 11, color: C.dust }}>placing {ticker} against 403,362 historical samples…</div>;
  const cfg = COND_SETS[set];
  return (
    <div>
      <div style={{ fontFamily: mono, fontSize: 9, letterSpacing: 1.5, color: C.cocoa, marginBottom: 14 }}>
        {cfg.title}
      </div>
      <div style={{ display: 'grid', gap: 12 }}>
        {cfg.keys.map(k => {
          const c = d.conditions[k];
          if (!c) return null;
          return (
            <div key={k} style={{ background: C.s2, border: `1px solid ${C.b1}`, borderRadius: 10, padding: 18,
                                  display: 'grid', gridTemplateColumns: '200px 1fr', gap: 24, alignItems: 'center' }}>
              <div>
                <div style={{ fontFamily: mono, fontSize: 9, letterSpacing: 1.4, color: C.cocoa, marginBottom: 6 }}>{COND_LABEL[k]}</div>
                <div style={{ fontFamily: mono, fontSize: 22, fontWeight: 700, color: C.cream }}>{cfg.fmt(c.value)}</div>
                <div style={{ fontFamily: mono, fontSize: 10, color: C.gold, marginTop: 4 }}>QUINTILE {c.quintile} OF 5</div>
              </div>
              <div>
                {(['fwd_5d', 'fwd_20d', 'fwd_60d'] as const).map(h => {
                  const cell: CellStats | null = c.cell?.[h] ?? null;
                  const base = d.base[h];
                  const edge = cell && base ? cell.positive_pct - base.positive_pct : null;
                  return (
                    <div key={h} style={{ display: 'flex', gap: 16, alignItems: 'baseline',
                                          fontFamily: mono, fontSize: 11, padding: '4px 0' }}>
                      <span style={{ color: C.cocoa, fontSize: 9, width: 40 }}>+{h.slice(4)}</span>
                      {cell ? (<>
                        <span style={{ color: cell.positive_pct >= 50 ? C.bull : C.bear }}>
                          {cell.positive_pct}% pos</span>
                        <span style={{ color: C.dust }}>med {cell.median_pct >= 0 ? '+' : ''}{cell.median_pct}%</span>
                        <span style={{ color: edge == null ? C.cocoa : Math.abs(edge) < 2 ? C.cocoa : edge > 0 ? C.bull : C.bear, fontSize: 10 }}>
                          {edge == null ? '' : `${edge >= 0 ? '+' : ''}${edge.toFixed(1)} vs base`}</span>
                        <span style={{ color: C.cocoa, fontSize: 9 }}>n={cell.n.toLocaleString()}</span>
                      </>) : <span style={{ color: C.cocoa }}>INSUFFICIENT SAMPLES</span>}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
      <div style={{ marginTop: 12, fontFamily: mono, fontSize: 10, color: C.cocoa, lineHeight: 1.7 }}>
        {d.note} Scan of {d.samples.toLocaleString()} ticker-dates, generated {d.generated}.
      </div>
    </div>
  );
};

// ── Shell ─────────────────────────────────────────────────────
const MODES = [
  { id: 'chart', label: 'PATTERN CHART' },
  { id: 'situation', label: 'SITUATION REPORT' },
  { id: 'volume', label: 'VOLUME TRACKER' },
  { id: 'analogs', label: 'HISTORICAL ANALOGS' },
  { id: 'formations', label: 'FORMATION LIBRARY' },
  { id: 'conditions', label: 'STATE LENSES' },
  { id: 'evolution', label: 'PATTERN EVOLUTION' },
];
const LENSES = [['momentum','MOMENTUM & REVERSAL'],['extremes','PRICE EXTREMES'],['volatility','VOLATILITY STATE']];

const PatternLab: React.FC<{ ticker: string }> = ({ ticker }) => {
  const [mode, setMode] = useState('chart');
  const [lens, setLens] = useState('momentum');
  return (
    <div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 }}>
        {MODES.map(m => (
          <button key={m.id} onClick={() => setMode(m.id)} style={{
            fontFamily: mono, fontSize: 9.5, letterSpacing: 1.5, padding: '8px 14px',
            background: mode === m.id ? 'rgba(218,165,32,0.1)' : 'none',
            border: `1px solid ${mode === m.id ? C.gold : C.b1}`, borderRadius: 4,
            color: mode === m.id ? C.gold : C.dust, cursor: 'pointer',
          }}>{m.label}</button>
        ))}
      </div>
      {mode === 'chart' && <PatternChart ticker={ticker} />}
      {mode === 'situation' && <SituationMode ticker={ticker} />}
      {mode === 'volume' && <VolumeTracker ticker={ticker} />}
      {mode === 'analogs' && <AnalogsDeep ticker={ticker} />}
      {mode === 'formations' && <FormationsDeep />}
      {mode === 'evolution' && <EvolutionMode ticker={ticker} />}
      {mode === 'conditions' && (<div>
        <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
          {LENSES.map(([id, label]) => (
            <button key={id} onClick={() => setLens(id)} style={{
              fontFamily: mono, fontSize: 9, letterSpacing: 1.3, padding: '6px 12px',
              background: lens === id ? 'rgba(218,165,32,0.1)' : 'none',
              border: `1px solid ${lens === id ? C.gold : C.b1}`, borderRadius: 4,
              color: lens === id ? C.gold : C.dust, cursor: 'pointer',
            }}>{label}</button>))}
        </div>
        <ConditionsMode ticker={ticker} set={lens} />
      </div>)}
    </div>
  );
};

export default PatternLab;
