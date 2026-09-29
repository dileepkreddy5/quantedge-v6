// Valuation & peers: what the price assumes, valuation vs its own history and vs real peers,
// each valuation method with its assumption, price vs peers, and the full model (feeds the score).
import React, { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
const C = { s1: '#1c130e', s2: '#241610', b1: '#3a2920', gold: '#daa520', cocoa: '#8a7560', dust: '#9d8b7a', latte: '#d4c4b0', cream: '#f4e8d8', up: '#22c55e', dn: '#ef4444', amber: '#e0ad3a' };
const mono = "'Fira Code',monospace";
const pc = (v: any) => v == null ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(0)}%`;
const x = (v: any) => v == null ? '—' : `${v.toFixed(1)}×`;
const card = { background: C.s1, border: `1px solid ${C.b1}`, borderRadius: 10, padding: 18, marginBottom: 16 };
const H: React.FC<{ t: string; sub?: string }> = ({ t, sub }) => <div style={{ display: 'flex', gap: 10, alignItems: 'baseline', marginBottom: 12, flexWrap: 'wrap' }}>
  <span style={{ fontFamily: mono, fontSize: 10, letterSpacing: 2, color: C.gold }}>{t}</span>{sub && <span style={{ fontFamily: mono, fontSize: 10, color: C.cocoa }}>{sub}</span>}</div>;

const Band: React.FC<{ label: string; b: any; since?: string }> = ({ label, b, since }) => {
  if (!b) return <div style={{ fontSize: 13, color: C.dust, marginBottom: 14 }}>{label}: not enough history (needs profits for several quarters).</div>;
  const lo = Math.min(b.low, b.now), hi = Math.max(b.high, b.now); const pos = (v: number) => `${((v - lo) / (hi - lo || 1)) * 100}%`;
  return (<div style={{ marginBottom: 18 }}>
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap' }}>
      <span style={{ fontFamily: mono, fontSize: 12, color: C.latte, width: 40 }}>{label}</span>
      <span style={{ fontFamily: mono, fontSize: 18, color: C.cream }}>{x(b.now)}</span>
      <span style={{ fontSize: 13.5, color: C.latte }}>higher than <b style={{ color: b.percentile_now >= 80 ? C.amber : b.percentile_now <= 20 ? C.up : C.cream }}>{Math.round(b.percentile_now)}%</b> of trading days since {since?.slice(0, 4)} · its usual median is {x(b.median)}</span></div>
    <div style={{ position: 'relative', height: 26, marginTop: 8 }}>
      <div style={{ position: 'absolute', top: 11, left: pos(b.low), width: `calc(${pos(b.high)} - ${pos(b.low)})`, height: 4, background: '#3a2920', borderRadius: 2 }} />
      <div title="median" style={{ position: 'absolute', top: 6, left: pos(b.median), width: 2, height: 14, background: C.dust }} />
      <div title="today" style={{ position: 'absolute', top: 2, left: `calc(${pos(b.now)} - 6px)`, width: 12, height: 22, borderRadius: 3, background: C.gold }} />
    </div>
    <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: mono, fontSize: 10, color: C.cocoa }}><span>usual low {x(b.low)}</span><span>usual high {x(b.high)}</span></div>
  </div>);
};

const ValuationTab: React.FC<{ ticker: string; peersChart?: React.ReactNode; fullModel?: React.ReactNode }> = ({ ticker, peersChart, fullModel }) => {
  const [v, setV] = useState<any>(null); const [openFull, setOpenFull] = useState(false);
  useEffect(() => { setV(null); api.get(`/api/v6/valuation-view/${ticker}`).then(r => setV(r.data)).catch(e => setV({ error: e?.response?.data?.detail || 'valuation unavailable' })); }, [ticker]);
  if (!v) return <div style={{ ...card, fontFamily: mono, fontSize: 11, color: C.dust }}>building the valuation view…</div>;
  if (v.error) return <div style={card}>{v.error}</div>;
  const rows = v.peers?.rows || []; const md = v.peers?.median || {};
  return (<div>
    <div style={card}>
      <H t="WHAT THE PRICE ASSUMES" />
      {v.loss_making ? <p style={{ fontSize: 15, color: C.cream, lineHeight: 1.7, margin: 0 }}>{ticker} is <b>loss-making</b> over the last 12 months, so earnings-based valuation doesn't apply. Compare it on <b>price-to-sales</b> against its own history and its peers below{v.actual_sales_growth != null ? <> — its sales grew <b>{pc(v.actual_sales_growth)}</b> in the latest quarter vs a year ago</> : null}.</p>
        : v.implied_growth != null ? <p style={{ fontSize: 15, color: C.cream, lineHeight: 1.7, margin: 0 }}>At ${v.price?.toFixed(2)}, the market is pricing in about <b style={{ color: C.gold }}>{pc(v.implied_growth)} a year</b> of cash-flow growth for the next ten years. For comparison, its sales grew <b>{pc(v.actual_sales_growth)}</b> in the latest quarter vs a year ago. {v.implied_growth > (v.actual_sales_growth ?? 0) ? 'The price assumes growth faster than today’s — that has to happen for the stock to be fairly priced.' : 'The price assumes growth slower than today’s — if current growth holds, that leaves room.'}</p>
        : <p style={{ fontSize: 14, color: C.dust, margin: 0 }}>Not enough cash-flow history to estimate what the price assumes.</p>}
    </div>
    <div style={card}>
      <H t="VALUATION VS ITS OWN HISTORY" sub="EARNINGS AS KNOWN ON EACH DATE · SPLIT-ADJUSTED" />
      <Band label="P/E" b={v.own_history?.pe} since={v.own_history?.since} />
      <Band label="P/S" b={v.own_history?.ps} since={v.own_history?.since} />
    </div>
    <div style={card}>
      <H t="VALUATION VS PEERS" sub={(v.peers?.group || '').toUpperCase() + ' · CLOSEST IN SIZE · SEC DATA'} />
      <div style={{ overflowX: 'auto' }}><table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: mono, fontSize: 12 }}>
        <thead><tr style={{ color: C.cocoa, textAlign: 'right' }}>{['Company', 'Market value', 'P/E', 'P/S', 'Sales growth', 'Op. margin', '1-yr return'].map((h, i) => <th key={h} style={{ padding: '6px 8px', fontWeight: 400, textAlign: i === 0 ? 'left' : 'right', borderBottom: `1px solid ${C.b1}` }}>{h}</th>)}</tr></thead>
        <tbody>{rows.map((r: any) => (<tr key={r.ticker} style={{ background: r.self ? 'rgba(218,165,32,0.08)' : 'none', color: r.self ? C.cream : C.latte }}>
          <td style={{ padding: '7px 8px', textAlign: 'left' }}><b style={{ color: C.gold }}>{r.ticker}</b> <span style={{ color: C.dust }}>{(r.name || '').replace(/ (Inc|Corp|Corporation|Holdings|Company)\.?.*$/, '').slice(0, 22)}</span></td>
          <td style={{ textAlign: 'right', padding: '7px 8px' }}>{r.market_cap ? `$${(r.market_cap / 1e9).toFixed(0)}B` : '—'}</td>
          <td style={{ textAlign: 'right', padding: '7px 8px' }}>{r.pe != null ? x(r.pe) : <span style={{ color: C.cocoa }}>{r.pe_note || '—'}</span>}</td>
          <td style={{ textAlign: 'right', padding: '7px 8px' }}>{x(r.ps)}</td>
          <td style={{ textAlign: 'right', padding: '7px 8px', color: (r.sales_growth ?? 0) >= 0 ? C.up : C.dn }}>{pc(r.sales_growth)}</td>
          <td style={{ textAlign: 'right', padding: '7px 8px' }}>{r.op_margin != null ? `${(r.op_margin * 100).toFixed(0)}%` : '—'}</td>
          <td style={{ textAlign: 'right', padding: '7px 8px', color: (r.ret_1y ?? 0) >= 0 ? C.up : C.dn }}>{pc(r.ret_1y)}</td></tr>))}
          <tr style={{ color: C.dust, borderTop: `1px solid ${C.b1}` }}><td style={{ padding: '7px 8px' }}>Peer median</td><td /><td style={{ textAlign: 'right', padding: '7px 8px' }}>{x(md.pe)}</td><td style={{ textAlign: 'right', padding: '7px 8px' }}>{x(md.ps)}</td>
            <td style={{ textAlign: 'right', padding: '7px 8px' }}>{pc(md.sales_growth)}</td><td style={{ textAlign: 'right', padding: '7px 8px' }}>{md.op_margin != null ? `${(md.op_margin * 100).toFixed(0)}%` : '—'}</td><td style={{ textAlign: 'right', padding: '7px 8px' }}>{pc(md.ret_1y)}</td></tr>
        </tbody></table></div>
    </div>
    <div style={card}>
      <H t="VALUATION METHODS" sub="EACH SHOWS WHAT YOU'D HAVE TO BELIEVE — NOT A PRICE TARGET" />
      {(v.methods || []).map((m: any, i: number) => (<div key={i} style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1.1fr) minmax(0,2fr) 90px 80px', gap: 14, alignItems: 'baseline', padding: '8px 0', borderTop: i ? `1px solid ${C.b1}` : 'none', opacity: m.value == null || m.outlier ? 0.6 : 1 }}>
        <span style={{ fontSize: 13.5, color: C.cream }}>{m.method}</span>
        <span style={{ fontSize: 12.5, color: C.dust }}>{m.assumes}{m.note ? <span style={{ color: C.amber }}> — {m.note}</span> : null}</span>
        <span style={{ fontFamily: mono, fontSize: 12.5, color: C.latte, textAlign: 'right' }}>{m.value != null ? `$${m.value.toFixed(0)}` : 'n/a'}</span>
        <span style={{ fontFamily: mono, fontSize: 12.5, textAlign: 'right', color: m.vs_price == null ? C.cocoa : m.vs_price >= 0 ? C.up : C.dn }}>{m.vs_price != null ? pc(m.vs_price) : ''}</span>
      </div>))}
      <div style={{ fontFamily: mono, fontSize: 9.5, color: C.cocoa, marginTop: 10 }}>Discounted-cash-flow models almost always value great, fast-growing companies below their price; their inputs (discount rate, growth, how long it lasts) move the answer enormously. Read them as scenarios.</div>
    </div>
    {peersChart && <div style={{ marginBottom: 16 }}>{peersChart}</div>}
    {fullModel && <div style={card}>
      <div onClick={() => setOpenFull(o => !o)} style={{ display: 'flex', alignItems: 'baseline', gap: 10, cursor: 'pointer' }}>
        <span style={{ fontFamily: mono, fontSize: 10, letterSpacing: 2, color: C.gold }}>FULL VALUATION MODEL</span>
        <span style={{ fontSize: 12.5, color: C.dust }}>every signal behind the Valuation part of the QuantEdge score</span>
        <span style={{ marginLeft: 'auto', fontFamily: mono, color: C.dust }}>{openFull ? '▾' : '▸'}</span></div>
      {openFull && <div style={{ marginTop: 12 }}>{fullModel}</div>}
    </div>}
    <div style={{ fontFamily: mono, fontSize: 9.5, color: C.cocoa, margin: '0 0 30px' }}>{v.note}</div>
  </div>);
};
export default ValuationTab;
