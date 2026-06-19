// ============================================================
// QuantEdge v6.0 — Ascent Radar Teaser (homepage widget)
// Shows the top 5 climbers; links to the full board.
// Renders nothing if the board isn't ready yet.
// ============================================================

import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../auth/authStore';

interface Row {
  rank: number; ticker: string; name: string; sector: string;
  ascent_score: number; tier: string; is_new: boolean;
}

const C = {
  panel: '#1a0f0a', border: '#3a2920', gold: '#daa520',
  text: '#d4c4b0', textDim: '#9d8b7a', green: '#22c55e', bg: '#0a0505',
};

const AscentTeaser: React.FC = () => {
  const navigate = useNavigate();
  const [rows, setRows] = useState<Row[]>([]);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const res = await api.get('/api/v6/ascent/top/5');
        if (res.data?.rows?.length) {
          setRows(res.data.rows);
          setReady(true);
        }
      } catch { /* board not ready — render nothing */ }
    })();
  }, []);

  if (!ready || rows.length === 0) return null;

  return (
    <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 10, padding: '20px 22px', maxWidth: 1400, margin: '0 auto 48px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10, marginBottom: 14 }}>
        <div>
          <div style={{ color: C.gold, fontWeight: 700, letterSpacing: 1, fontSize: 15 }}>★ ASCENT RADAR — TOP CLIMBERS</div>
          <div style={{ color: C.textDim, fontSize: 11, marginTop: 3 }}>Companies climbing toward larger-cap tiers on sustained strength &amp; volume</div>
        </div>
        <button onClick={() => navigate('/ascent')} style={{ background: 'none', border: `1px solid ${C.gold}`, color: C.gold, borderRadius: 6, padding: '7px 14px', fontSize: 12, fontWeight: 700, cursor: 'pointer' }}>View full board →</button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 10 }}>
        {rows.map((r) => (
          <div key={r.ticker} onClick={() => navigate('/ascent')} style={{ background: C.bg, border: `1px solid ${C.border}`, borderRadius: 8, padding: '12px 14px', cursor: 'pointer' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <span style={{ color: C.gold, fontWeight: 700, fontSize: 16 }}>{r.ticker}{r.is_new && <span style={{ marginLeft: 6, fontSize: 8, color: C.bg, background: C.gold, padding: '1px 4px', borderRadius: 3 }}>NEW</span>}</span>
              <span style={{ color: C.green, fontWeight: 700, fontSize: 15 }}>{r.ascent_score.toFixed(0)}</span>
            </div>
            <div style={{ color: C.text, fontSize: 11, marginTop: 4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.name || r.ticker}</div>
            <div style={{ color: C.textDim, fontSize: 10, marginTop: 2, textTransform: 'capitalize' }}>{r.tier}-cap · #{r.rank}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AscentTeaser;
