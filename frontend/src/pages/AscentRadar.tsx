// ============================================================
// QuantEdge v6.0 — Ascent Radar Page
// Board of US companies climbing toward larger-cap tiers on
// sustained strength + persistent volume. Discovery, not advice.
// ============================================================

import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../auth/authStore';
import toast from 'react-hot-toast';

const C = {
  bg: '#0a0505', panel: '#1a0f0a', panel2: '#1f130d', border: '#3a2920',
  border2: '#8a7560', gold: '#daa520', goldDim: '#b8860b', text: '#d4c4b0',
  textDim: '#9d8b7a', green: '#22c55e', orange: '#f59e0b', red: '#ef4444',
};

interface Delta { rank_change: number; score_change: number; }
interface Row {
  rank: number; ticker: string; name: string; sector: string;
  ascent_score: number; strength_score: number; volume_score: number;
  tier_score: number; high_score: number; tier: string; market_cap: number | null;
  flags: string[]; delta_3d: Delta | null; delta_1w: Delta | null;
  delta_1m: Delta | null; first_seen: string | null; is_new: boolean;
}
interface Board { scan_time: string | null; rows: Row[]; history_available: boolean; }

const scoreColor = (v: number) => {
  if (v >= 75) return C.green;
  if (v >= 55) return C.goldDim;
  if (v >= 45) return C.textDim;
  if (v >= 30) return C.orange;
  return C.red;
};
const fmtCap = (c: number | null) => {
  if (!c) return '—';
  if (c >= 1e12) return `$${(c / 1e12).toFixed(2)}T`;
  if (c >= 1e9) return `$${(c / 1e9).toFixed(1)}B`;
  if (c >= 1e6) return `$${(c / 1e6).toFixed(0)}M`;
  return `$${c}`;
};
const fmtDate = (iso: string | null) => {
  if (!iso) return '—';
  try { return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }); }
  catch { return '—'; }
};

const DeltaChip: React.FC<{ d: Delta | null; historyOn: boolean }> = ({ d, historyOn }) => {
  if (!historyOn) return <span style={{ color: C.textDim, fontSize: 12 }}>—</span>;
  if (!d) return <span style={{ color: C.textDim, fontSize: 12 }}>new</span>;
  const rc = d.rank_change;
  if (rc === 0) return <span style={{ color: C.textDim, fontSize: 12 }}>—</span>;
  const up = rc > 0;
  return <span style={{ color: up ? C.green : C.red, fontSize: 12, fontWeight: 600 }}>{up ? '▲' : '▼'} {Math.abs(rc)}</span>;
};

const th: React.CSSProperties = { padding: '8px 10px', fontWeight: 600, fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5, whiteSpace: 'nowrap' };
const td: React.CSSProperties = { padding: '9px 10px', whiteSpace: 'nowrap' };
const newBadge: React.CSSProperties = { marginLeft: 6, fontSize: 9, fontWeight: 700, color: C.bg, background: C.gold, padding: '1px 5px', borderRadius: 3, verticalAlign: 'middle' };
const flagChip: React.CSSProperties = { fontSize: 11, color: C.text, background: C.panel, border: `1px solid ${C.border2}`, borderRadius: 4, padding: '3px 8px' };
const btnStyle = (primary: boolean): React.CSSProperties => ({
  background: primary ? C.gold : 'transparent', color: primary ? C.bg : C.text,
  border: `1px solid ${primary ? C.gold : C.border2}`, borderRadius: 6,
  padding: '7px 12px', fontSize: 13, cursor: 'pointer', fontWeight: 600,
});

const SubScore: React.FC<{ label: string; v: number }> = ({ label, v }) => (
  <div style={{ minWidth: 90 }}>
    <div style={{ fontSize: 10, color: C.textDim, textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</div>
    <div style={{ fontSize: 16, fontWeight: 700, color: scoreColor(v) }}>{v?.toFixed(0) ?? '—'}</div>
  </div>
);

const AscentRadar: React.FC = () => {
  const navigate = useNavigate();
  const [board, setBoard] = useState<Board | null>(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.get('/api/v6/ascent/board');
      setBoard(res.data);
    } catch (e: any) {
      if (e?.response?.status === 503) toast.error('Ascent Radar is warming up — the first scan runs shortly.');
      else toast.error('Could not load the Ascent Radar.');
      setBoard({ scan_time: null, rows: [], history_available: false });
    } finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  const downloadCsv = () => {
    const base = (api.defaults.baseURL || '').replace(/\/$/, '');
    window.open(`${base}/api/v6/ascent/board.csv`, '_blank');
  };

  const rows = board?.rows || [];
  const historyOn = !!board?.history_available;

  return (
    <div style={{ minHeight: '100vh', background: C.bg, color: C.text, padding: '16px' }}>
      <div style={{ maxWidth: 1100, margin: '0 auto' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 8 }}>
          <div>
            <h1 style={{ margin: 0, fontSize: 26, color: C.gold, letterSpacing: 1 }}>ASCENT RADAR</h1>
            <div style={{ color: C.textDim, fontSize: 13, marginTop: 4 }}>Companies climbing toward larger-cap tiers on sustained strength &amp; volume</div>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button onClick={() => navigate('/')} style={btnStyle(false)}>← Home</button>
            <button onClick={load} style={btnStyle(false)}>↻ Refresh</button>
            <button onClick={downloadCsv} style={btnStyle(true)}>⤓ CSV</button>
          </div>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'center', padding: '8px 0 14px', color: C.textDim, fontSize: 12, borderBottom: `1px solid ${C.border}` }}>
          <span>Last updated: <b style={{ color: C.text }}>{board?.scan_time ? new Date(board.scan_time).toLocaleString() : '—'}</b></span>
          <span>Updates hourly during market hours + end of day.</span>
          {!historyOn && <span style={{ color: C.orange }}>Building history — 3-day / 1-week / 1-month movement fills in as scans accumulate.</span>}
        </div>

        <div style={{ fontSize: 11, color: C.textDim, margin: '10px 0 16px', fontStyle: 'italic' }}>
          Discovery tool. A high ascent score means a stock is climbing and worth investigating — it is not a prediction or investment advice.
        </div>

        {loading && <div style={{ color: C.textDim, padding: 40, textAlign: 'center' }}>Loading the board…</div>}

        {!loading && rows.length === 0 && (
          <div style={{ color: C.textDim, padding: 40, textAlign: 'center' }}>
            The first scan hasn't completed yet. The board populates after the next scheduled scan.
          </div>
        )}

        {!loading && rows.length > 0 && (
          <div className="ascent-table-wrap" style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ color: C.textDim, textAlign: 'left' }}>
                  <th style={th}>#</th><th style={th}>Ticker</th><th style={th}>Company</th><th style={th}>Sector</th>
                  <th style={{ ...th, textAlign: 'right' as const }}>Ascent</th>
                  <th style={{ ...th, textAlign: 'center' as const }}>Tier</th>
                  <th style={{ ...th, textAlign: 'center' as const }}>3d</th>
                  <th style={{ ...th, textAlign: 'center' as const }}>1w</th>
                  <th style={{ ...th, textAlign: 'center' as const }}>1m</th>
                  <th style={{ ...th, textAlign: 'right' as const }}>Mkt Cap</th>
                  <th style={{ ...th, textAlign: 'center' as const }}>Since</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <React.Fragment key={r.ticker}>
                    <tr onClick={() => setExpanded(expanded === r.ticker ? null : r.ticker)}
                        style={{ borderTop: `1px solid ${C.border}`, cursor: 'pointer', background: r.is_new ? 'rgba(218,165,32,0.06)' : 'transparent' }}>
                      <td style={td}>{r.rank}</td>
                      <td style={{ ...td, fontWeight: 700, color: C.gold }}>{r.ticker}{r.is_new && <span style={newBadge}>NEW</span>}</td>
                      <td style={{ ...td, color: C.text, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.name || '—'}</td>
                      <td style={{ ...td, color: C.textDim, fontSize: 11, maxWidth: 130, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.sector || '—'}</td>
                      <td style={{ ...td, textAlign: 'right', fontWeight: 700, color: scoreColor(r.ascent_score) }}>{r.ascent_score.toFixed(1)}</td>
                      <td style={{ ...td, textAlign: 'center', textTransform: 'capitalize' }}>{r.tier}</td>
                      <td style={{ ...td, textAlign: 'center' }}><DeltaChip d={r.delta_3d} historyOn={historyOn} /></td>
                      <td style={{ ...td, textAlign: 'center' }}><DeltaChip d={r.delta_1w} historyOn={historyOn} /></td>
                      <td style={{ ...td, textAlign: 'center' }}><DeltaChip d={r.delta_1m} historyOn={historyOn} /></td>
                      <td style={{ ...td, textAlign: 'right', color: C.textDim }}>{fmtCap(r.market_cap)}</td>
                      <td style={{ ...td, textAlign: 'center', color: C.textDim, fontSize: 11 }}>{fmtDate(r.first_seen)}</td>
                    </tr>
                    {expanded === r.ticker && (
                      <tr style={{ background: C.panel2 }}>
                        <td colSpan={11} style={{ padding: '10px 14px' }}>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 18, alignItems: 'center' }}>
                            <SubScore label="Strength" v={r.strength_score} />
                            <SubScore label="Volume" v={r.volume_score} />
                            <SubScore label="Tier room" v={r.tier_score} />
                            <SubScore label="Near high" v={r.high_score} />
                          </div>
                          {r.flags.length > 0 && (
                            <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                              {r.flags.map((f, i) => (<span key={i} style={flagChip}>{f}</span>))}
                            </div>
                          )}
                          <button onClick={() => navigate(`/dashboard?ticker=${r.ticker}`)} style={{ ...btnStyle(true), marginTop: 10 }}>Full analysis →</button>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <style>{`
        @media (max-width: 680px) {
          .ascent-table-wrap table { font-size: 12px; }
          .ascent-table-wrap th:nth-child(3), .ascent-table-wrap td:nth-child(3),
          .ascent-table-wrap th:nth-child(4), .ascent-table-wrap td:nth-child(4) { display: none; }
        }
      `}</style>
    </div>
  );
};

export default AscentRadar;
