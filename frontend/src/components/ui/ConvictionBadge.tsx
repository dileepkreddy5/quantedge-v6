import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

interface Mod { id: string; label: string; weight: number; status: string; score: number | null; }
interface Conv { conviction_score: number | null; verdict: string;
  coverage: { live_weight: number; total_weight: number; pct: number; modules_live: number; modules_total: number };
  modules: Mod[]; }

function vColor(v: string): string {
  if (v.includes('STRONG_BUY')) return '#0f9d6e';
  if (v.includes('BUY')) return '#1d9e75';
  if (v.includes('NEUTRAL')) return '#c9a227';
  if (v.includes('STRONG_SELL')) return '#c0392b';
  if (v.includes('SELL')) return '#d35400';
  return '#9d8b7a';
}

export default function ConvictionBadge({ ticker }: { ticker: string }) {
  const [c, setC] = useState<Conv | null>(null);
  const [show, setShow] = useState(false);
  useEffect(() => {
    if (!ticker) return;
    api.get(`/api/v7/conviction/${ticker}`)
      .then(r => setC(r.data?.data || null))
      .catch(() => setC(null));
  }, [ticker]);
  if (!c || c.conviction_score == null) return null;
  const col = vColor(c.verdict);
  return (
    <div style={{ position: 'relative', display: 'inline-block' }}
      onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
      <div style={{ background: `${col}12`, border: `1px solid ${col}55`, borderRadius: 6,
        padding: '6px 14px', cursor: 'help', textAlign: 'center' }}>
        <div style={{ fontSize: 9, color: '#9d8b7a', letterSpacing: 2 }}>QUANTEDGE CONVICTION</div>
        <div style={{ fontSize: 18, fontWeight: 700, color: col, letterSpacing: 1 }}>
          {c.verdict.replace('_', ' ')} · {c.conviction_score}</div>
        <div style={{ fontSize: 8, color: '#7a7266', letterSpacing: 1 }}>
          {(c.coverage.pct * 100).toFixed(0)}% coverage · {c.coverage.modules_live}/{c.coverage.modules_total} modules live</div>
      </div>
      {show && (
        <div style={{ position: 'absolute', top: '100%', right: 0, marginTop: 6, zIndex: 50,
          background: '#111', border: '1px solid #2a2a2a', borderRadius: 8, padding: 12, minWidth: 260,
          boxShadow: '0 8px 24px rgba(0,0,0,.5)' }}>
          <div style={{ fontSize: 10, color: '#9d8b7a', marginBottom: 8, letterSpacing: 1 }}>
            {c.modules.length}-MODULE CONVICTION TREE — {c.coverage.live_weight}/{c.coverage.total_weight} weight live</div>
          {c.modules.map(m => (
            <div key={m.id} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 3,
              opacity: m.score == null ? 0.4 : 1 }}>
              <span style={{ color: '#cdbfae' }}>{m.label.replace(' Intelligence', '')}
                <span style={{ color: '#6f665b', fontSize: 9 }}> · {m.weight}%</span></span>
              <span style={{ color: m.score == null ? '#7a7266' : vColor(c.verdict), fontWeight: 600 }}>
                {m.score == null ? 'building' : m.score.toFixed(1)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
