// Filings, owners & analysts: insiders, institutions, share count and filings at a glance;
// analyst recommendation trend (Finnhub); filings timeline; institutional holders; money flow.
import React, { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
const C = { s1: '#1c130e', s2: '#241610', b1: '#3a2920', gold: '#daa520', cocoa: '#8a7560', dust: '#9d8b7a', latte: '#d4c4b0', cream: '#f4e8d8', up: '#22c55e', dn: '#ef4444', amber: '#e0ad3a' };
const mono = "'Fira Code',monospace";
const card = { background: C.s1, border: `1px solid ${C.b1}`, borderRadius: 10, padding: 18, marginBottom: 16 };
const M = (v: number) => `$${Math.abs(v) >= 1e9 ? (Math.abs(v) / 1e9).toFixed(1) + 'B' : (Math.abs(v) / 1e6).toFixed(1) + 'M'}`;
const H: React.FC<{ t: string; sub?: string }> = ({ t, sub }) => <div style={{ display: 'flex', gap: 10, alignItems: 'baseline', marginBottom: 12, flexWrap: 'wrap' }}>
  <span style={{ fontFamily: mono, fontSize: 10, letterSpacing: 2, color: C.gold }}>{t}</span>{sub && <span style={{ fontFamily: mono, fontSize: 10, color: C.cocoa }}>{sub}</span>}</div>;
const Tile: React.FC<{ k: string; v: React.ReactNode; s: string; col?: string }> = ({ k, v, s, col }) => (
  <div style={{ background: C.s2, borderRadius: 8, padding: '12px 14px' }}><div style={{ fontFamily: mono, fontSize: 9.5, letterSpacing: 1.4, color: C.cocoa }}>{k}</div>
    <div style={{ fontFamily: mono, fontSize: 17, color: col || C.cream, marginTop: 5 }}>{v}</div><div style={{ fontSize: 12, color: C.dust, marginTop: 4, lineHeight: 1.45 }}>{s}</div></div>);
const SEG = [['strong_buy', 'Strong buy', '#16a34a'], ['buy', 'Buy', '#4ade80'], ['hold', 'Hold', '#b09c86'], ['sell', 'Sell', '#f59e0b'], ['strong_sell', 'Strong sell', '#ef4444']];

const FilingsTab: React.FC<{ ticker: string; filings: React.ReactNode; holders: React.ReactNode; flow: React.ReactNode }> = ({ ticker, filings, holders, flow }) => {
  const [sm, setSm] = useState<any>(null); const [mg, setMg] = useState<any>(null); const [ow, setOw] = useState<any>(null); const [an, setAn] = useState<any>(null);
  const [openFlow, setOpenFlow] = useState(false);
  useEffect(() => { setSm(null); setMg(null); setOw(null); setAn(null);
    api.get(`/api/v6/summary/${ticker}`).then(r => setSm(r.data)).catch(() => setSm({}));
    api.get(`/api/v6/management/${ticker}`).then(r => setMg(r.data?.data || r.data)).catch(() => setMg({}));
    api.get(`/api/v6/ownership/${ticker}`).then(r => setOw(r.data?.data || r.data)).catch(() => setOw({}));
    api.get(`/api/v6/analysts/${ticker}`).then(r => setAn(r.data)).catch(e => setAn({ error: e?.response?.data?.detail || 'analyst data unavailable' })); }, [ticker]);
  const ins = sm?.insiders_90d; const net = ins ? ins.bought - ins.sold : null;
  const m = mg?.key_metrics || {}; const ok = ow?.key_metrics || {};
  const months = an?.months || [];
  const maxN = Math.max(1, ...months.map((x: any) => SEG.reduce((a, [k]) => a + (x[k] || 0), 0)));
  return (<div>
    <div style={card}>
      <H t="AT A GLANCE" sub="SEC FORM 4 · 13F · QUARTERLY FILINGS" />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(210px,1fr))', gap: 10 }}>
        <Tile k="INSIDERS · LAST 90 DAYS" v={net == null ? '—' : net === 0 ? 'no trades' : `${net > 0 ? 'net bought ' : 'net sold '}${M(net)}`} col={net == null || net === 0 ? undefined : net > 0 ? C.up : C.amber}
          s={ins ? `bought ${M(ins.bought)} · sold ${M(ins.sold)} on the open market (some sales are pre-planned)` : ''} />
        <Tile k="INSTITUTIONAL HOLDERS" v={ok.major_holder_count != null ? Number(ok.major_holder_count).toLocaleString() : '—'} s={ok.top_holder_pct != null ? `largest single holder owns ${Number(ok.top_holder_pct).toFixed(1)}% · from 13F filings` : 'from 13F filings'} />
        <Tile k="SHARE COUNT · 1 YEAR" v={m.share_count_change != null ? `${m.share_count_change >= 0 ? '+' : ''}${(m.share_count_change * 100).toFixed(1)}%` : '—'} col={m.share_count_change == null ? undefined : m.share_count_change <= 0 ? C.up : C.amber}
          s={m.share_count_change == null ? '' : m.share_count_change <= 0 ? 'shrinking — buybacks raise each share’s slice' : 'growing — new shares dilute existing holders'} />
        <Tile k="RETURNED TO SHAREHOLDERS" v={m.total_payout_yield != null ? `${(m.total_payout_yield * 100).toFixed(1)}%` : '—'} s="dividends + buybacks over 12 months, as a share of market value" />
        <Tile k="MATERIAL SEC FILINGS · 90 DAYS" v={sm?.material_filings_90d ?? '—'} s="8-K events classed as material — see the timeline below" />
      </div>
    </div>
    <div style={card}>
      <H t="WHAT ANALYSTS SAY" sub={an?.source ? `${an.source.toUpperCase()} · MONTHLY` : ''} />
      {an?.error ? <div style={{ fontSize: 13.5, color: C.dust }}>{an.error}</div> : !an ? <div style={{ fontFamily: mono, fontSize: 11, color: C.dust }}>loading…</div> : !months.length ? <div style={{ fontSize: 13.5, color: C.dust }}>No analyst coverage recorded for {ticker}.</div> : (<>
        <p style={{ fontSize: 15, color: C.cream, lineHeight: 1.7, margin: '0 0 14px' }}>
          <b>{Math.round((an.buy_share_now || 0) * an.analysts_now)}</b> of <b>{an.analysts_now}</b> analysts rate {ticker} a buy ({Math.round((an.buy_share_now || 0) * 100)}%)
          {an.buy_share_then != null ? <>, {an.buy_share_now > an.buy_share_then + 0.02 ? 'up' : an.buy_share_now < an.buy_share_then - 0.02 ? 'down' : 'about the same'} from {Math.round(an.buy_share_then * 100)}% in {an.then_period?.slice(0, 7)}.</> : '.'}</p>
        <div style={{ display: 'flex', gap: 6, alignItems: 'flex-end', height: 130 }}>
          {months.map((x: any) => { const n = SEG.reduce((a, [k]) => a + (x[k] || 0), 0);
            return (<div key={x.period} title={`${x.period}: ${SEG.map(([k, l]) => `${l} ${x[k] || 0}`).join(' · ')}`} style={{ flex: 1, display: 'flex', flexDirection: 'column-reverse', height: `${(n / maxN) * 100}%`, minWidth: 14 }}>
              {SEG.map(([k, , col]) => (x[k] ? <div key={k} style={{ height: `${(x[k] / (n || 1)) * 100}%`, background: col }} /> : null))}</div>); })}
        </div>
        <div style={{ display: 'flex', gap: 6, fontFamily: mono, fontSize: 9.5, color: C.cocoa, marginTop: 4 }}>{months.map((x: any) => <span key={x.period} style={{ flex: 1, textAlign: 'center', minWidth: 14 }}>{x.period?.slice(5, 7)}/{x.period?.slice(2, 4)}</span>)}</div>
        <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', fontFamily: mono, fontSize: 10.5, color: C.dust, marginTop: 10 }}>{SEG.map(([k, l, col]) => <span key={k}><i style={{ display: 'inline-block', width: 9, height: 9, background: col, marginRight: 5, borderRadius: 2 }} />{l}</span>)}</div>
        <div style={{ fontSize: 12, color: C.cocoa, marginTop: 10, lineHeight: 1.5 }}>Firm-by-firm ratings and price targets (e.g. JPMorgan, Goldman) need a paid analyst-data feed and aren’t shown yet.</div>
      </>)}
    </div>
    <div style={{ marginBottom: 16 }}>{filings}</div>
    <div style={{ ...card, padding: 14 }}><H t="INSTITUTIONAL OWNERSHIP" sub="13F FILINGS" />{holders}</div>
    <div style={card}>
      <div onClick={() => setOpenFlow(o => !o)} style={{ display: 'flex', alignItems: 'baseline', gap: 10, cursor: 'pointer' }}>
        <span style={{ fontFamily: mono, fontSize: 10, letterSpacing: 2, color: C.gold }}>MONEY FLOW</span>
        <span style={{ fontSize: 12.5, color: C.dust }}>accumulation and distribution read from price and volume</span>
        <span style={{ marginLeft: 'auto', fontFamily: mono, color: C.dust }}>{openFlow ? '▾' : '▸'}</span></div>
      {openFlow && <div style={{ marginTop: 12 }}>{flow}</div>}
    </div>
  </div>);
};
export default FilingsTab;
