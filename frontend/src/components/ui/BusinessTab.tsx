// Business: "is this a good business?" — at a glance (verified SEC figures + quality scores in
// plain words), then the former Financial, Business, Management, Competitive and Industry tabs
// as sections that load only when opened.
import React, { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
const C = { s1: '#1c130e', s2: '#241610', b1: '#3a2920', gold: '#daa520', cocoa: '#8a7560', dust: '#9d8b7a', latte: '#d4c4b0', cream: '#f4e8d8', up: '#22c55e', dn: '#ef4444', amber: '#e0ad3a' };
const mono = "'Fira Code',monospace";
const pc = (v: any, d = 0) => v == null ? '—' : `${(v * 100).toFixed(d)}%`;
const big = (v: any) => v == null ? '—' : v >= 1e12 ? `$${(v / 1e12).toFixed(2)}T` : v >= 1e9 ? `$${(v / 1e9).toFixed(1)}B` : `$${(v / 1e6).toFixed(0)}M`;
export type Section = { id: string; title: string; hint: string; node: React.ReactNode; open?: boolean };
const Tile: React.FC<{ k: string; v: React.ReactNode; s: string; col?: string }> = ({ k, v, s, col }) => (
  <div style={{ background: C.s2, borderRadius: 8, padding: '12px 14px' }}>
    <div style={{ fontFamily: mono, fontSize: 9.5, letterSpacing: 1.4, color: C.cocoa }}>{k}</div>
    <div style={{ fontFamily: mono, fontSize: 17, color: col || C.cream, marginTop: 5 }}>{v}</div>
    <div style={{ fontSize: 12, color: C.dust, marginTop: 4, lineHeight: 1.45 }}>{s}</div></div>);

const BusinessTab: React.FC<{ ticker: string; sections: Section[] }> = ({ ticker, sections }) => {
  const [sm, setSm] = useState<any>(null); const [fin, setFin] = useState<any>(null); const [mg, setMg] = useState<any>(null);
  const [open, setOpen] = useState<Record<string, boolean>>(() => Object.fromEntries(sections.map(x => [x.id, !!x.open])));
  useEffect(() => { setSm(null); setFin(null); setMg(null);
    api.get(`/api/v6/summary/${ticker}`).then(r => setSm(r.data)).catch(() => setSm({}));
    api.get(`/api/v6/financial/${ticker}`).then(r => setFin(r.data?.data || r.data)).catch(() => setFin({}));
    api.get(`/api/v6/management/${ticker}`).then(r => setMg(r.data?.data || r.data)).catch(() => setMg({})); }, [ticker]);
  const f = sm?.facts || {}; const k = fin?.key_metrics || {}; const m = mg?.key_metrics || {};
  const az = k.altman_z, pf = k.piotroski_f, bm = k.beneish_m, roic = k.roic, cc = m.cash_conversion, py = m.total_payout_yield, sc = m.share_count_change;
  const card = { background: C.s1, border: `1px solid ${C.b1}`, borderRadius: 10, marginBottom: 12 };
  const go = (id: string) => { setOpen(o => ({ ...o, [id]: true })); setTimeout(() => document.getElementById(`sec-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50); };
  return (<div>
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 14 }}>
      {[{ id: 'glance', title: 'At a glance' }, ...sections].map(x => <button key={x.id} onClick={() => x.id === 'glance' ? document.getElementById('sec-glance')?.scrollIntoView({ behavior: 'smooth' }) : go(x.id)}
        style={{ fontFamily: mono, fontSize: 10.5, padding: '6px 11px', background: 'none', border: `1px solid ${C.b1}`, borderRadius: 999, color: C.dust, cursor: 'pointer' }}>{x.title}</button>)}
    </div>
    <div id="sec-glance" style={{ ...card, padding: 18 }}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'baseline', marginBottom: 12 }}>
        <span style={{ fontFamily: mono, fontSize: 10, letterSpacing: 2, color: C.gold }}>AT A GLANCE</span>
        <span style={{ fontFamily: mono, fontSize: 10, color: C.cocoa }}>{f.last_quarter ? `SEC FILINGS · QUARTER TO ${f.last_quarter}` : ''}</span></div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(210px,1fr))', gap: 10 }}>
        <Tile k="REVENUE · LAST 12 MONTHS" v={big(f.revenue_ttm)} s={f.sales_yoy != null ? `latest quarter ${f.sales_yoy >= 0 ? '+' : ''}${(f.sales_yoy * 100).toFixed(0)}% vs a year ago` : ''} col={C.cream} />
        <Tile k="MARGINS · LAST 12 MONTHS · GROSS / OP / NET" v={`${pc(f.gross_margin_ttm)} / ${pc(f.op_margin_ttm)} / ${pc(f.net_margin_ttm)}`} s="of each $1 of sales: kept after costs, after running the business, after everything" />
        <Tile k="RETURN ON CAPITAL (ROIC)" v={pc(roic)} col={roic == null ? undefined : roic > 0.15 ? C.up : roic < 0.06 ? C.dn : C.cream} s={roic == null ? '' : roic > 0.15 ? 'excellent — earns far more than its cost of capital' : roic < 0.06 ? 'weak — below a typical cost of capital' : 'decent'} />
        <Tile k="PROFITS BACKED BY CASH" v={cc != null ? `${cc.toFixed(2)}×` : '—'} col={cc == null ? undefined : cc >= 0.9 ? C.up : C.amber} s={cc == null ? '' : cc >= 0.9 ? 'operating cash covers reported profit — healthy' : 'cash lags reported profit — worth checking'} />
        <Tile k="BANKRUPTCY RISK (ALTMAN Z)" v={az != null ? az.toFixed(1) : '—'} col={az == null ? undefined : az > 3 ? C.up : az < 1.8 ? C.dn : C.amber} s={az == null ? 'not meaningful for banks and insurers' : az > 3 ? 'very low (above 3 is the safe zone)' : az < 1.8 ? 'elevated (below 1.8 is the distress zone)' : 'grey zone (1.8–3)'} />
        <Tile k="FINANCIAL HEALTH (PIOTROSKI)" v={pf != null ? `${pf} / 9` : '—'} col={pf == null ? undefined : pf >= 7 ? C.up : pf <= 3 ? C.dn : C.cream} s={pf == null ? '' : pf >= 7 ? 'strong — most health checks pass' : pf <= 3 ? 'weak — most health checks fail' : 'mixed'} />
        <Tile k="ACCOUNTING RED FLAGS (BENEISH)" v={bm != null ? bm.toFixed(2) : '—'} col={bm == null ? undefined : bm > -1.78 ? C.amber : C.up} s={bm == null ? '' : bm > -1.78 ? 'above −1.78: pattern seen in earnings manipulation — check the filings' : 'below −1.78: no manipulation pattern'} />
        <Tile k="RETURNED TO SHAREHOLDERS" v={py != null ? pc(py, 1) : '—'} s={sc != null ? `share count ${sc <= 0 ? 'down' : 'up'} ${Math.abs(sc * 100).toFixed(1)}% in a year (${sc <= 0 ? 'buybacks' : 'dilution'})` : 'dividends + buybacks as a share of market value'} />
      </div>
    </div>
    {sections.map(x => (<div id={`sec-${x.id}`} key={x.id} style={card}>
      <div onClick={() => setOpen(o => ({ ...o, [x.id]: !o[x.id] }))} style={{ display: 'flex', alignItems: 'baseline', gap: 12, padding: '14px 18px', cursor: 'pointer' }}>
        <span style={{ fontFamily: mono, fontSize: 11, letterSpacing: 2, color: C.gold }}>{x.title.toUpperCase()}</span>
        <span style={{ fontSize: 12.5, color: C.dust }}>{x.hint}</span>
        <span style={{ marginLeft: 'auto', fontFamily: mono, fontSize: 12, color: C.dust }}>{open[x.id] ? '▾' : '▸'}</span></div>
      {open[x.id] && <div style={{ padding: '0 14px 14px' }}>{x.node}</div>}
    </div>))}
  </div>);
};
export default BusinessTab;
