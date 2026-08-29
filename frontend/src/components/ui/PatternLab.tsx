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
interface Analog { ticker: string; start: string; similarity_pct: number; trajectory: number[];
                   fwd: Record<string, number | null>; }
interface Result {
  ticker: string; as_of: string; window_days: number; episodes: number;
  distributions: Record<string, Dist>; base_rates: Record<string, Dist>;
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

const DistRow: React.FC<{ label: string; d: Dist; base?: Dist | null }> = ({ label, d, base }) => {
  const edge = base ? d.positive_pct - base.positive_pct : null;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '54px 1fr auto auto auto', gap: 14,
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
    </div>
  );
};

const PatternLab: React.FC<{ ticker: string }> = ({ ticker }) => {
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
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 12 }}>
                {res.analogs.slice(0, 8).map(a => (
                  <span key={a.ticker + a.start} title={`fwd 20d: ${a.fwd['20d'] ?? '—'}%`}
                        style={{ fontFamily: mono, fontSize: 9, padding: '4px 8px',
                                 border: `1px solid ${C.b1}`, borderRadius: 3, color: C.latte }}>
                    {a.ticker} '{a.start.slice(2, 7)} · {a.similarity_pct}%
                  </span>
                ))}
              </div>
            </div>

            <div style={{ background: C.s2, border: `1px solid ${C.b1}`, borderRadius: 10, padding: '14px 6px 6px' }}>
              <div style={{ fontFamily: mono, fontSize: 9, letterSpacing: 1.5, color: C.cocoa, padding: '0 12px 8px' }}>
                WHAT FOLLOWED — {res.episodes} NON-OVERLAPPING EPISODES
              </div>
              {Object.entries(res.distributions).map(([h, d]) => d && (
                <DistRow key={h} label={`+${h}`} d={d}
                         base={res.base_rates[h.replace('d', '')] || (res.base_rates as any)[parseInt(h)]} />
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
            {res.caveat}
          </div>
        </>
      )}
    </div>
  );
};

export default PatternLab;
