// News: what happened (the company's own 8-K filings), news vs price, attention and tone,
// most-covered stories, commentary (collapsed), and the full news model that feeds the score.
import React, { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
const C = { s1: '#1c130e', s2: '#241610', b1: '#3a2920', gold: '#daa520', cocoa: '#8a7560', dust: '#9d8b7a', latte: '#d4c4b0', cream: '#f4e8d8', up: '#22c55e', dn: '#ef4444', amber: '#e0ad3a', blue: '#60a5fa' };
const mono = "'Fira Code',monospace";
const card = { background: C.s1, border: `1px solid ${C.b1}`, borderRadius: 10, padding: 18, marginBottom: 16 };
const pc = (v: any) => v == null ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`;
const col = (v: any) => v == null ? C.cocoa : v >= 0 ? C.up : C.dn;
const TYPE: Record<string, string> = { results: 'Results', leadership: 'Leadership', agreement: 'Agreement', deal: 'Deal', capital: 'Capital', restructuring: 'Restructuring',
  impairment: 'Impairment', auditor: 'Auditor', restatement: 'Restatement', announcement: 'Announcement', disclosure: 'Disclosure', vote: 'Vote', governance: 'Governance',
  product: 'Product', legal: 'Legal / regulatory', analyst: 'Analyst', market: 'Market', commentary: 'Opinion', other: 'Other' };
const H: React.FC<{ t: string; sub?: string }> = ({ t, sub }) => <div style={{ display: 'flex', gap: 10, alignItems: 'baseline', marginBottom: 12, flexWrap: 'wrap' }}>
  <span style={{ fontFamily: mono, fontSize: 10, letterSpacing: 2, color: C.gold }}>{t}</span>{sub && <span style={{ fontFamily: mono, fontSize: 10, color: C.cocoa }}>{sub}</span>}</div>;
const Chip: React.FC<{ t: string; c?: string }> = ({ t, c }) => <span style={{ fontFamily: mono, fontSize: 9.5, letterSpacing: 1, textTransform: 'uppercase', color: c || C.gold, border: `1px solid ${(c || C.gold)}55`, borderRadius: 4, padding: '2px 7px' }}>{t}</span>;

const NewsChart: React.FC<{ series: any[] }> = ({ series }) => {
  if (!series?.length) return null;
  const W = 1000, P = 150, B = 60; const cs = series.map(s => s.close); const lo = Math.min(...cs), hi = Math.max(...cs); const mxN = Math.max(1, ...series.map(s => s.n));
  const x = (i: number) => (i / Math.max(1, series.length - 1)) * W; const y = (c: number) => 8 + (1 - (c - lo) / (hi - lo || 1)) * (P - 16);
  const bw = Math.max(2, W / series.length - 2);
  return (<svg viewBox={`0 0 ${W} ${P + B + 18}`} style={{ width: '100%', height: 230 }} preserveAspectRatio="none">
    <polyline fill="none" stroke={C.gold} strokeWidth={2} points={series.map((s, i) => `${x(i)},${y(s.close)}`).join(' ')} />
    {series.map((s, i) => { const h = (s.n / mxN) * (B - 4); const hn = s.n ? (s.neg / s.n) * h : 0; const hp = s.n ? (s.pos / s.n) * h : 0; const base = P + B;
      return (<g key={i}><rect x={x(i) - bw / 2} y={base - h} width={bw} height={h - hn - hp} fill="#6b5a48" />
        <rect x={x(i) - bw / 2} y={base - hp - hn} width={bw} height={hp} fill={C.up} opacity={0.75} />
        <rect x={x(i) - bw / 2} y={base - hn} width={bw} height={hn} fill={C.dn} opacity={0.8} /></g>); })}
    <text x={4} y={P + B + 15} fill={C.cocoa} fontSize={11} fontFamily={mono}>{series[0].d}</text>
    <text x={W - 4} y={P + B + 15} fill={C.cocoa} fontSize={11} fontFamily={mono} textAnchor="end">{series[series.length - 1].d}</text>
  </svg>);
};

const NewsTab: React.FC<{ ticker: string; fullModel?: React.ReactNode }> = ({ ticker, fullModel }) => {
  const [d, setD] = useState<any>(null); const [nx, setNx] = useState<any>(null); const [wk, setWk] = useState<any>(null);
  const [open, setOpen] = useState<Record<string, boolean>>({}); const [showOp, setShowOp] = useState(false); const [showFull, setShowFull] = useState(false);
  useEffect(() => { setD(null); setOpen({});
    api.get(`/api/v6/news-view/${ticker}`).then(r => setD(r.data)).catch(e => setD({ error: e?.response?.data?.detail || 'news unavailable' }));
    api.get(`/api/v6/summary/${ticker}`).then(r => setNx(r.data?.next_results_est)).catch(() => {});
    setWk(null); api.get(`/api/v6/wiki-attention/${ticker}`).then(r => setWk(r.data)).catch(() => setWk({ available: false })); }, [ticker]);
  if (!d) return <div style={{ ...card, fontFamily: mono, fontSize: 11, color: C.dust }}>reading 90 days of filings and news…</div>;
  if (d.error) return <div style={card}>{d.error}</div>;
  const at = d.attention || {}; const tn = d.tones || {}; const tt = (tn.positive || 0) + (tn.neutral || 0) + (tn.negative || 0);
  const media = (d.top || []).filter((o: any) => !o.commentary); const opinion = (d.timeline || []).filter((o: any) => o.commentary);
  return (<div>
    <div style={card}>
      <H t="WHAT HAPPENED" sub={`FROM ${ticker}'S OWN SEC FILINGS · LAST 90 DAYS`} />
      {!(d.what_happened || []).length && <div style={{ fontSize: 13.5, color: C.dust }}>No material 8-K filings in the last 90 days.</div>}
      {(d.what_happened || []).map((o: any, i: number) => (<div key={i} style={{ padding: '11px 0', borderTop: i ? `1px solid ${C.b1}` : 'none' }}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'baseline', flexWrap: 'wrap' }}>
          <span style={{ fontFamily: mono, fontSize: 11.5, color: C.cocoa, width: 86 }}>{o.event_day || o.filed?.slice(0, 10)}</span>
          <Chip t={TYPE[o.type] || o.type} c={['restatement', 'auditor', 'impairment', 'restructuring'].includes(o.type) ? C.dn : undefined} />
          <span style={{ fontSize: 14.5, color: C.cream, flex: 1, minWidth: 220 }}>{o.what}</span>
          <span style={{ fontFamily: mono, fontSize: 12 }}>that day <b style={{ color: col(o.move_event_day) }}>{pc(o.move_event_day)}</b> · next <b style={{ color: col(o.move_next_day) }}>{pc(o.move_next_day)}</b></span>
        </div>
        {o.key_sentence && <div style={{ fontSize: 13, color: C.latte, margin: '6px 0 0 96px', lineHeight: 1.55 }}>“{o.key_sentence}”</div>}
        <div style={{ margin: '6px 0 0 96px', display: 'flex', gap: 14, flexWrap: 'wrap', fontFamily: mono, fontSize: 11 }}>
          {o.filing_url && <a href={o.filing_url} target="_blank" rel="noopener noreferrer" style={{ color: C.gold }}>filing →</a>}
          {o.coverage_articles > 0 && <span onClick={() => setOpen(p => ({ ...p, [i]: !p[i] }))} style={{ color: C.dust, cursor: 'pointer' }}>{open[i] ? '▾' : '▸'} {o.coverage_articles} articles in the next 3 days</span>}
        </div>
        {open[i] && <div style={{ margin: '6px 0 0 96px' }}>{(o.coverage_headlines || []).map((h: any, j: number) => <div key={j} style={{ fontSize: 12.5, color: C.dust, padding: '3px 0' }}>
          <a href={h.url} target="_blank" rel="noopener noreferrer" style={{ color: C.latte }}>{h.title}</a> <span style={{ fontFamily: mono, fontSize: 10.5 }}>· {h.source}{h.commentary ? ' · opinion' : ''}</span></div>)}</div>}
      </div>))}
      {nx && <div style={{ fontFamily: mono, fontSize: 11.5, color: C.dust, marginTop: 10, paddingTop: 10, borderTop: `1px solid ${C.b1}` }}>Coming up: next results expected around <b style={{ color: C.cream }}>{nx}</b> (estimate).</div>}
    </div>
    <div style={card}>
      <H t="NEWS VS PRICE" sub="90 DAYS · GOLD = PRICE · BARS = ARTICLES PER DAY (GREEN POSITIVE · RED NEGATIVE · BROWN NEUTRAL)" />
      <NewsChart series={d.series || []} />
    </div>
    <div style={card}>
      <H t="ATTENTION & TONE" />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(220px,1fr))', gap: 10 }}>
        <div style={{ background: C.s2, borderRadius: 8, padding: '12px 14px' }}><div style={{ fontFamily: mono, fontSize: 9.5, color: C.cocoa, letterSpacing: 1.4 }}>COVERAGE · LAST 30 DAYS</div>
          <div style={{ fontFamily: mono, fontSize: 17, color: C.cream, marginTop: 5 }}>{at.news_30d ?? '—'} articles</div>
          <div style={{ fontSize: 12, color: C.dust, marginTop: 4 }}>{at.peer_median_30d != null ? `peer median ${at.peer_median_30d} · ${at.news_30d > at.peer_median_30d * 1.5 ? 'far more attention than peers' : at.news_30d < at.peer_median_30d * 0.67 ? 'less attention than peers' : 'about the same as peers'}` : ''}</div></div>
        <div style={{ background: C.s2, borderRadius: 8, padding: '12px 14px' }}><div style={{ fontFamily: mono, fontSize: 9.5, color: C.cocoa, letterSpacing: 1.4 }}>TONE OF COVERAGE · 90 DAYS</div>
          <div style={{ display: 'flex', height: 10, borderRadius: 5, overflow: 'hidden', margin: '10px 0 6px' }}>{tt > 0 && <><i style={{ width: `${tn.positive / tt * 100}%`, background: C.up }} /><i style={{ width: `${tn.neutral / tt * 100}%`, background: '#6b5a48' }} /><i style={{ width: `${tn.negative / tt * 100}%`, background: C.dn }} /></>}</div>
          <div style={{ fontSize: 12, color: C.dust }}>{tn.positive} positive · {tn.neutral} neutral · {tn.negative} negative</div></div>
        <div style={{ background: C.s2, borderRadius: 8, padding: '12px 14px' }}><div style={{ fontFamily: mono, fontSize: 9.5, color: C.cocoa, letterSpacing: 1.4 }}>ABOUT {ticker} VS MENTIONS</div>
          <div style={{ fontFamily: mono, fontSize: 17, color: C.cream, marginTop: 5 }}>{d.n_articles} about it</div>
          <div style={{ fontSize: 12, color: C.dust, marginTop: 4 }}>{d.n_mentions_only} more only mentioned it (headline about another company) — not counted as its news</div></div>
      </div>
      {wk?.available && (() => { const v = wk.series.map((x: any) => x.views); const mx = Math.max(...v), mn = Math.min(...v);
        return (<div style={{ background: C.s2, borderRadius: 8, padding: '12px 14px', marginTop: 10, display: 'grid', gridTemplateColumns: 'minmax(0,1fr) minmax(0,2fr)', gap: 16, alignItems: 'center' }}>
          <div><div style={{ fontFamily: mono, fontSize: 9.5, color: C.cocoa, letterSpacing: 1.4 }}>PUBLIC ATTENTION · WIKIPEDIA</div>
            <div style={{ fontFamily: mono, fontSize: 17, color: wk.ratio >= 1.5 ? C.amber : C.cream, marginTop: 5 }}>{wk.ratio != null ? `${wk.ratio.toFixed(1)}× usual` : '—'}</div>
            <div style={{ fontSize: 12, color: C.dust, marginTop: 4 }}>{Math.round(wk.avg_7d).toLocaleString()} views a day this week vs {Math.round(wk.avg_prior_90d).toLocaleString()} usual · peak {wk.peak?.d} · <a href={wk.url} target="_blank" rel="noopener noreferrer" style={{ color: C.gold }}>{wk.article}</a></div></div>
          <svg viewBox="0 0 400 60" style={{ width: '100%', height: 60 }} preserveAspectRatio="none">
            <polyline fill="none" stroke={C.blue} strokeWidth={1.5} points={v.map((y: number, i: number) => `${(i / (v.length - 1)) * 400},${56 - ((y - mn) / (mx - mn || 1)) * 52}`).join(' ')} /></svg>
        </div>); })()}
      <div style={{ fontFamily: mono, fontSize: 9.5, color: C.cocoa, marginTop: 10 }}>{d.tone_note}{wk?.available ? ' ' + wk.note : ''}</div>
    </div>
    <div style={card}>
      <H t="MOST-COVERED STORIES" sub="SAME STORY FROM SEVERAL OUTLETS MERGED" />
      {media.length === 0 && <div style={{ fontSize: 13.5, color: C.dust }}>No widely covered stories beyond the filings above.</div>}
      {media.map((o: any, i: number) => (<div key={i} style={{ display: 'flex', gap: 10, alignItems: 'baseline', padding: '8px 0', borderTop: i ? `1px solid ${C.b1}` : 'none', flexWrap: 'wrap' }}>
        <span style={{ fontFamily: mono, fontSize: 11.5, color: C.cocoa, width: 86 }}>{o.event_day}</span><Chip t={TYPE[o.type] || o.type} c={C.dust} />
        <a href={o.url} target="_blank" rel="noopener noreferrer" style={{ fontSize: 13.5, color: C.cream, flex: 1, minWidth: 220 }}>{o.headline}</a>
        <span style={{ fontFamily: mono, fontSize: 11, color: C.dust }}>{o.n_articles} articles · {o.source}</span></div>))}
    </div>
    <div style={card}>
      <div onClick={() => setShowOp(v => !v)} style={{ display: 'flex', gap: 10, alignItems: 'baseline', cursor: 'pointer' }}>
        <span style={{ fontFamily: mono, fontSize: 10, letterSpacing: 2, color: C.gold }}>COMMENTARY & OPINION</span><span style={{ fontSize: 12.5, color: C.dust }}>{opinion.length} pieces — views, not events</span>
        <span style={{ marginLeft: 'auto', fontFamily: mono, color: C.dust }}>{showOp ? '▾' : '▸'}</span></div>
      {showOp && <div style={{ marginTop: 10 }}>{opinion.slice(0, 40).map((o: any, i: number) => <div key={i} style={{ fontSize: 12.5, padding: '4px 0', color: C.dust }}>
        <span style={{ fontFamily: mono, fontSize: 10.5, color: C.cocoa, marginRight: 8 }}>{o.published?.slice(0, 10)}</span><a href={o.url} target="_blank" rel="noopener noreferrer" style={{ color: C.latte }}>{o.headline}</a> · {o.source}</div>)}</div>}
    </div>
    {fullModel && <div style={card}>
      <div onClick={() => setShowFull(v => !v)} style={{ display: 'flex', gap: 10, alignItems: 'baseline', cursor: 'pointer' }}>
        <span style={{ fontFamily: mono, fontSize: 10, letterSpacing: 2, color: C.gold }}>FULL NEWS MODEL</span><span style={{ fontSize: 12.5, color: C.dust }}>the signals behind the News part of the QuantEdge score</span>
        <span style={{ marginLeft: 'auto', fontFamily: mono, color: C.dust }}>{showFull ? '▾' : '▸'}</span></div>
      {showFull && <div style={{ marginTop: 12 }}>{fullModel}</div>}
    </div>}
    <div style={{ fontFamily: mono, fontSize: 9.5, color: C.cocoa, margin: '0 0 30px' }}>{d.note}</div>
  </div>);
};
export default NewsTab;
