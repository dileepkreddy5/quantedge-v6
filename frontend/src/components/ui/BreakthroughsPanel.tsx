// Breakthroughs from the company's own press releases, at the top of Company Intel.
import React, { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
const C = { s1: '#1c130e', s2: '#241610', b1: '#3a2920', gold: '#daa520', cocoa: '#8a7560', dust: '#9d8b7a', latte: '#d4c4b0', cream: '#f4e8d8', up: '#3ec27a', dn: '#ef7d5a' };
const mono = "'Fira Code',monospace";
const pct = (v: any) => v == null ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`;
const BreakthroughsPanel: React.FC<{ ticker: string }> = ({ ticker }) => {
  const [d, setD] = useState<any>(null);
  useEffect(() => { setD(null); api.get(`/api/v6/intel/${ticker}/breakthroughs`).then(r => setD(r.data)).catch(() => setD({ findings: [] })); }, [ticker]);
  if (!d || !(d.findings || []).length) return null;
  return (<div style={{ background: C.s1, border: `1px solid ${C.b1}`, borderRadius: 10, padding: 18, marginBottom: 16 }}>
    <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: 2, color: C.gold, marginBottom: 12 }}>BREAKTHROUGHS · FROM THE COMPANY'S OWN SEC FILINGS · LAST 6 MONTHS</div>
    {d.findings.slice(0, 8).map((f: any, i: number) => (<div key={i} style={{ display: 'grid', gridTemplateColumns: '96px 180px minmax(0,1fr) 150px', gap: 14, padding: '9px 0', borderTop: i ? `1px solid ${C.b1}` : 'none', alignItems: 'baseline' }}>
      <span style={{ fontFamily: mono, fontSize: 11.5, color: C.cocoa }}>{f.date}</span>
      <span style={{ fontFamily: mono, fontSize: 11.5, color: f.direction === 'negative' ? C.dn : f.direction === 'positive' ? C.up : C.dust }}>{f.label}</span>
      <span style={{ fontSize: 13.5, color: C.latte, lineHeight: 1.55 }}>“{f.sentence}” {f.url && <a href={f.url} target="_blank" rel="noopener noreferrer" style={{ color: C.gold, fontFamily: mono, fontSize: 11 }}>filing →</a>}</span>
      <span style={{ fontFamily: mono, fontSize: 11.5, color: C.dust }}>on news <span style={{ color: (f.move_on_news ?? 0) >= 0 ? C.up : C.dn }}>{pct(f.move_on_news)}</span><br />since <span style={{ color: (f.move_since ?? 0) >= 0 ? C.up : C.dn }}>{pct(f.move_since)}</span></span>
    </div>))}
    <div style={{ fontFamily: mono, fontSize: 10.5, color: C.cocoa, marginTop: 10 }}>{d.note} A breakthrough raises the odds of growth; it doesn't guarantee it.</div>
  </div>);
};
export default BreakthroughsPanel;
