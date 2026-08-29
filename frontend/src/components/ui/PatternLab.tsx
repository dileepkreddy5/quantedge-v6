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
const Overlay: React.FC<{ q: number[]; analogs: Analog[] }> = ({ q, analogs }) => {
  const W = 640, H = 240, PAD = 10;
  const all = [...q, ...analogs.flatMap(a => a.trajectory)];
  const mn = Math.min(...all), mx = Math.max(...all);
  const sx = (i: number, n: number) => PAD + (i / (n - 1)) * (W - 2 * PAD);
  const sy = (v: number) => H - PAD - ((v - mn) / (mx - mn || 1)) * (H - 2 * PAD);
  const path = (t: number[]) => t.map((v, i) => `${i ? 'L' : 'M'}${sx(i, t.length).toFixed(1)},${sy(v).toFixed(1)}`).join('');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', background: 'rgba(0,0,0,0.25)', borderRadius: 8 }}>
      {analogs.map((a, k) => (
        <path key={k} d={path(a.trajectory)} fill="none" stroke={C.cocoa}
              strokeWidth={1} opacity={0.28} />
      ))}
      <path d={path(q)} fill="none" stroke={C.gold} strokeWidth={2.5} />
    </svg>
  );
};

const DistRow: React.FC<{ label: string; d: Dist; base?: Dist | null; ex?: Dist | null }> = ({ label, d, base, ex }) => {
  const edge = base ? d.positive_pct - base.positive_pct : null;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '54px 1fr auto auto auto auto', gap: 14,
                  alignItems: 'center', padding: '10px 12px', borderBottom: `1px solid rgba(58,41,32,0.5)` }}>
      <span style={{ fontFamily: mono, fontSize: 12, color: C.cream, fontWeight: 700 }}>{label}</span>
      <div style={{ position: 'relative', height: 16, background: 'rgba(0,0,0,0.3)', borderRadius: 3 }}>
        {/* p10..p90 band with median tick */}
        {(() => {
          const lo = -30, hi = 30;
          const x = (v: number) => Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100));
          return (<>
            <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: C.b2 }} />
            <div style={{ position: 'absolute', top: 4, bottom: 4, left: `${x(d.p10_pct)}%`,
                          width: `${x(d.p90_pct) - x(d.p10_pct)}%`,
                          background: 'linear-gradient(90deg, rgba(239,68,68,0.35), rgba(34,197,94,0.35))', borderRadius: 2 }} />
            <div style={{ position: 'absolute', top: 1, bottom: 1, left: `${x(d.median_pct)}%`,
                          width: 2, background: C.gold }} />
          </>);
        })()}
      </div>
      <span style={{ fontFamily: mono, fontSize: 11.5, color: d.positive_pct >= 50 ? C.bull : C.bear }}>
        {d.positive_pct}% pos
      </span>
      <span style={{ fontFamily: mono, fontSize: 10.5, color: C.dust }}>med {d.median_pct >= 0 ? '+' : ''}{d.median_pct}%</span>
      <span style={{ fontFamily: mono, fontSize: 10, color: edge == null ? C.cocoa : Math.abs(edge) < 3 ? C.cocoa : edge > 0 ? C.bull : C.bear }}>
        {edge == null ? '' : `${edge >= 0 ? '+' : ''}${edge.toFixed(1)} vs base`}
      </span>
      <span style={{ fontFamily: mono, fontSize: 10, color: ex == null ? C.cocoa : ex.positive_pct >= 50 ? C.bull : C.bear }}>
        {ex == null ? '' : `${ex.positive_pct}% beat SPY · med ${ex.median_pct >= 0 ? '+' : ''}${ex.median_pct}%`}
      </span>
    </div>
  );
};

const AnalogsMode: React.FC<{ ticker: string }> = ({ ticker }) => {
  const [w, setW] = useState<20 | 60>(60);
  const [res, setRes] = useState<Result | null>(null);
  const [err, setErr] = useState<string>('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let dead = false;
    (async () => {
      setBusy(true); setErr(''); setRes(null);
      try {
        const r = await api.get(`/api/v6/patterns/analogs/${ticker}?window=${w}`);
        if (!dead) setRes(r.data);
      } catch (e: any) {
        if (!dead) setErr(e?.response?.data?.detail || 'pattern query failed');
      } finally { if (!dead) setBusy(false); }
    })();
    return () => { dead = true; };
  }, [ticker, w]);

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: 2.5, color: C.cocoa }}>
          PATTERN LAB — HISTORICAL ANALOGS
        </div>
        {([20, 60] as const).map(x => (
          <button key={x} onClick={() => setW(x)} style={{
            fontFamily: mono, fontSize: 9.5, letterSpacing: 1.2, padding: '6px 12px',
            background: w === x ? 'rgba(218,165,32,0.1)' : 'none',
            border: `1px solid ${w === x ? C.gold : C.b1}`, borderRadius: 4,
            color: w === x ? C.gold : C.dust, cursor: 'pointer',
          }}>{x}-DAY SHAPE</button>
        ))}
      </div>

      {busy && <div style={{ fontFamily: mono, fontSize: 11, color: C.dust }}>searching {w === 20 ? '508,493' : '487,072'} historical windows…</div>}
      {err && <div style={{ fontFamily: mono, fontSize: 11, color: C.warn }}>{err}</div>}

      {res && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(300px, 1.4fr) minmax(280px, 1fr)', gap: 14 }}>
            <div style={{ background: C.s2, border: `1px solid ${C.b1}`, borderRadius: 10, padding: 18 }}>
              <div style={{ fontFamily: mono, fontSize: 9, letterSpacing: 1.5, color: C.cocoa, marginBottom: 10 }}>
                {res.ticker} LAST {res.window_days} SESSIONS (GOLD) VS {res.analogs.length} CLOSEST EPISODES
              </div>
              <Overlay q={res.query_trajectory} analogs={res.analogs} />
              <div style={{ marginTop: 14, overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: mono, fontSize: 10 }}>
                  <thead><tr>
                    {['TICKER','FORMED','SIM','REGIME','VOL SLOPE','+5D','+20D','+60D'].map(h => (
                      <th key={h} style={{ textAlign: 'left', padding: '5px 8px', color: C.cocoa,
                                           letterSpacing: 1, fontSize: 8.5, fontWeight: 500,
                                           borderBottom: `1px solid ${C.b1}` }}>{h}</th>
                    ))}
                  </tr></thead>
                  <tbody>
                    {res.analogs.slice(0, 10).map(a => (
                      <tr key={a.ticker + a.start}>
                        <td style={{ padding: '6px 8px', color: C.cream, fontWeight: 700 }}>{a.ticker}</td>
                        <td style={{ padding: '6px 8px', color: C.dust }}>{a.start}{a.end ? ` → ${a.end}` : ''}</td>
                        <td style={{ padding: '6px 8px', color: C.gold }}>{a.similarity_pct}%</td>
                        <td style={{ padding: '6px 8px', color: C.latte, fontSize: 9 }}>{(a.regime || '—').replace(/_/g, ' ')}</td>
                        <td style={{ padding: '6px 8px', color: (a.volume_slope ?? 0) >= 0 ? C.bull : C.bear }}>
                          {a.volume_slope != null ? (a.volume_slope >= 0 ? '+' : '') + a.volume_slope.toFixed(2) : '—'}</td>
                        {(['5d','20d','60d'] as const).map(h => (
                          <td key={h} style={{ padding: '6px 8px',
                                               color: a.fwd[h] == null ? C.cocoa : (a.fwd[h]! >= 0 ? C.bull : C.bear) }}>
                            {a.fwd[h] == null ? '—' : `${a.fwd[h]! >= 0 ? '+' : ''}${a.fwd[h]}%`}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div style={{ background: C.s2, border: `1px solid ${C.b1}`, borderRadius: 10, padding: '14px 6px 6px' }}>
              <div style={{ fontFamily: mono, fontSize: 9, letterSpacing: 1.5, color: C.cocoa, padding: '0 12px 8px' }}>
                WHAT FOLLOWED — {res.episodes} NON-OVERLAPPING EPISODES
              </div>
              {Object.entries(res.distributions).map(([h, d]) => d && (
                <DistRow key={h} label={`+${h}`} d={d}
                         base={res.base_rates[h.replace('d', '')] || (res.base_rates as any)[parseInt(h)]}
                         ex={res.excess_vs_spy?.[h] ?? null} />
              ))}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 14, marginTop: 14 }}>
            <div style={{ background: C.s2, border: `1px solid ${C.b1}`, borderRadius: 10, padding: 16 }}>
              <div style={{ fontFamily: mono, fontSize: 9, letterSpacing: 1.5, color: C.cocoa, marginBottom: 10 }}>
                BY VOLUME DURING FORMATION (+20d)
              </div>
              {Object.entries(res.splits.volume_slope).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '7px 4px',
                                      fontFamily: mono, fontSize: 11.5 }}>
                  <span style={{ color: C.latte }}>{k.toUpperCase()} VOLUME</span>
                  <span style={{ color: v ? (v.positive_pct >= 50 ? C.bull : C.bear) : C.cocoa }}>
                    {v ? `${v.positive_pct}% pos · n=${v.n}` : 'insufficient episodes'}
                  </span>
                </div>
              ))}
            </div>
            <div style={{ background: C.s2, border: `1px solid ${C.b1}`, borderRadius: 10, padding: 16 }}>
              <div style={{ fontFamily: mono, fontSize: 9, letterSpacing: 1.5, color: C.cocoa, marginBottom: 10 }}>
                BY MARKET REGIME AT FORMATION (+20d)
              </div>
              {Object.entries(res.splits.regime).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '7px 4px',
                                      fontFamily: mono, fontSize: 11.5 }}>
                  <span style={{ color: C.latte }}>{k.replace(/_/g, ' ')}</span>
                  <span style={{ color: v ? (v.positive_pct >= 50 ? C.bull : C.bear) : C.cocoa }}>
                    {v ? `${v.positive_pct}% pos · n=${v.n}` : 'insufficient episodes'}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div style={{ marginTop: 12, fontFamily: mono, fontSize: 10, color: C.cocoa, lineHeight: 1.7 }}>
            {res.method && (
              <div style={{ marginBottom: 6 }}>
                METHOD: {res.method.normalization} · {res.method.stage1} → {res.method.stage2} · {res.method.dedup}.
                {res.episode_date_range && ` Episodes span ${res.episode_date_range[0]} → ${res.episode_date_range[1]}.`}
              </div>
            )}
            {res.caveat}
          </div>
        </>
      )}
    </div>
  );
};




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

const FormationsMode: React.FC = () => {
  const [art, setArt] = useState<FormArt | null>(null);
  const [sel, setSel] = useState<string>('double_top');
  const [err, setErr] = useState('');
  useEffect(() => {
    (async () => {
      try { const r = await api.get('/api/v6/patterns/formations'); setArt(r.data); }
      catch (e: any) { setErr(e?.response?.data?.detail || 'formation scan unavailable'); }
    })();
  }, []);
  if (err) return <div style={{ fontFamily: mono, fontSize: 11, color: C.warn }}>{err}</div>;
  if (!art) return <div style={{ fontFamily: mono, fontSize: 11, color: C.dust }}>loading formation library…</div>;
  const f = art.formations[sel];
  return (
    <div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 16 }}>
        {Object.keys(art.formations).sort().map(k => (
          <button key={k} onClick={() => setSel(k)} style={{
            fontFamily: mono, fontSize: 8.5, letterSpacing: 1, padding: '6px 10px',
            background: sel === k ? 'rgba(218,165,32,0.1)' : 'none',
            border: `1px solid ${sel === k ? C.gold : C.b1}`, borderRadius: 4,
            color: sel === k ? C.gold : C.dust, cursor: 'pointer',
          }}>{FORM_LABEL[k] || k.toUpperCase()} · {art.formations[k].occurrences}</button>
        ))}
      </div>
      {f && (
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(240px,1fr) minmax(300px,1.6fr)', gap: 14 }}>
          <div style={{ background: C.s2, border: `1px solid ${C.b1}`, borderRadius: 10, padding: 18 }}>
            <div style={{ fontFamily: mono, fontSize: 9, letterSpacing: 1.5, color: C.cocoa, marginBottom: 12 }}>
              {FORM_LABEL[sel]} — MEASURED, 2021-2026
            </div>
            {[['NON-OVERLAPPING OCCURRENCES', String(f.occurrences)],
              ['RAW DETECTIONS', String(f.raw_detections)],
              ['MEDIAN FORMATION LENGTH', `${f.median_duration} sessions`],
              ['BROKE OUT UPWARD', f.breakout_up_pct != null ? `${f.breakout_up_pct}%` : '—'],
            ].map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between',
                                    padding: '7px 0', fontFamily: mono, fontSize: 11 }}>
                <span style={{ color: C.cocoa, fontSize: 9, letterSpacing: 1 }}>{k}</span>
                <span style={{ color: C.latte }}>{v}</span>
              </div>
            ))}
            <div style={{ marginTop: 12, paddingTop: 12, borderTop: `1px solid ${C.b1}` }}>
              <div style={{ fontFamily: mono, fontSize: 9, letterSpacing: 1.5, color: C.cocoa, marginBottom: 8 }}>
                +20D AFTER CONFIRMATION
              </div>
              {f.fwd20 ? (
                <div style={{ fontFamily: mono, fontSize: 12, color: C.latte, lineHeight: 2 }}>
                  <div style={{ color: f.fwd20.positive_pct >= 50 ? C.bull : C.bear, fontSize: 18, fontWeight: 700 }}>
                    {f.fwd20.positive_pct}% positive
                  </div>
                  median {f.fwd20.median_pct >= 0 ? '+' : ''}{f.fwd20.median_pct}% ·
                  p25/p75 {f.fwd20.p25_pct}% / {f.fwd20.p75_pct}%
                </div>
              ) : <div style={{ fontFamily: mono, fontSize: 11, color: C.cocoa }}>INSUFFICIENT EPISODES</div>}
            </div>
          </div>
          <div style={{ background: C.s2, border: `1px solid ${C.b1}`, borderRadius: 10, padding: '14px 10px' }}>
            <div style={{ fontFamily: mono, fontSize: 9, letterSpacing: 1.5, color: C.cocoa, padding: '0 8px 8px' }}>
              MOST RECENT OCCURRENCES
            </div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: mono, fontSize: 10 }}>
              <thead><tr>{['TICKER','FORMED','LEN','BREAK','+5D','+20D','+60D'].map(h => (
                <th key={h} style={{ textAlign: 'left', padding: '5px 8px', color: C.cocoa, fontSize: 8.5,
                                     letterSpacing: 1, fontWeight: 500, borderBottom: `1px solid ${C.b1}` }}>{h}</th>
              ))}</tr></thead>
              <tbody>{f.examples.map(e => (
                <tr key={e.ticker + e.end}>
                  <td style={{ padding: '6px 8px', color: C.cream, fontWeight: 700 }}>{e.ticker}</td>
                  <td style={{ padding: '6px 8px', color: C.dust }}>{e.start} → {e.end}</td>
                  <td style={{ padding: '6px 8px', color: C.dust }}>{e.duration}d</td>
                  <td style={{ padding: '6px 8px', color: e.breakout_up ? C.bull : C.bear }}>{e.breakout_up ? '▲' : '▼'}</td>
                  {[e.fwd_5d, e.fwd_20d, e.fwd_60d].map((v, i) => (
                    <td key={i} style={{ padding: '6px 8px', color: v >= 0 ? C.bull : C.bear }}>
                      {v >= 0 ? '+' : ''}{v}%</td>
                  ))}
                </tr>
              ))}</tbody>
            </table>
          </div>
        </div>
      )}
      <div style={{ marginTop: 12, fontFamily: mono, fontSize: 10, color: C.cocoa, lineHeight: 1.7 }}>
        METHOD: {art.method} · Universe: {art.universe} tickers.
      </div>
    </div>
  );
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
  { id: 'analogs', label: 'HISTORICAL ANALOGS' },
  { id: 'formations', label: 'CLASSICAL FORMATIONS' },
  { id: 'momentum', label: 'MOMENTUM & REVERSAL' },
  { id: 'extremes', label: 'PRICE EXTREMES' },
  { id: 'volatility', label: 'VOLATILITY' },
];

const PatternLab: React.FC<{ ticker: string }> = ({ ticker }) => {
  const [mode, setMode] = useState('analogs');
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
      {mode === 'analogs' && <AnalogsMode ticker={ticker} />}
      {mode === 'formations' && <FormationsMode />}
      {(mode === 'momentum' || mode === 'extremes' || mode === 'volatility') &&
        <ConditionsMode ticker={ticker} set={mode} />}
    </div>
  );
};

export default PatternLab;
