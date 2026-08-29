// ============================================================
// QuantEdge v6.0 — Live Tape
// A slow terminal tape under the nav cycling REAL system output:
// board leaders, retrain age, scan ages. Nothing invented — every
// segment comes from an endpoint, and segments that fail to load
// simply don't appear.
// ============================================================

import React, { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

const C = { gold: '#daa520', caramel: '#d4956c', cocoa: '#8a7560',
            dust: '#9d8b7a', latte: '#d4c4b0', bull: '#22c55e', bear: '#ef4444' };
const mono = "'Fira Code',monospace";

interface Seg { text: string; tone?: string }

const ago = (h: number) => h < 1 ? `${Math.max(1, Math.round(h * 60))}M AGO`
  : h < 48 ? `${Math.round(h)}H AGO` : `${Math.round(h / 24)}D AGO`;

const LiveTape: React.FC = () => {
  const [segs, setSegs] = useState<Seg[]>([]);

  useEffect(() => {
    (async () => {
      const out: Seg[] = [];
      try {
        const st = (await api.get('/api/v6/system/stats')).data;
        if (st?.panel?.trained_at) {
          const h = (Date.now() - new Date(st.panel.trained_at).getTime()) / 36e5;
          out.push({ text: `PANEL RETRAINED ${ago(h)} · ${st.panel.n_tickers} TICKERS · ${st.panel.n_features} FEATURES`, tone: C.caramel });
        }
        if (st?.signals) out.push({ text: `${st.signals.signals_live} SIGNALS COMPUTING LIVE ACROSS ${st.signals.catalogs} CATALOGS` });
        (st?.boards || []).forEach((b: any) => {
          if (b.available && b.age_hours != null)
            out.push({ text: `${String(b.board).toUpperCase()} SCAN ${ago(b.age_hours)}`, tone: b.stale ? C.bear : C.bull });
        });
      } catch {}
      try {
        const sc = (await api.get('/api/v6/scan/tiers')).data;
        const rows = (sc?.tiers?.small || []).slice(0, 3);
        rows.forEach((r: any, i: number) => out.push({
          text: `MULTIBAGGER #${i + 1} ${r.ticker} · SCORE ${r.score?.toFixed(1)} · YoY ${r.qtr_yoy_growth != null ? (r.qtr_yoy_growth >= 0 ? '+' : '') + (r.qtr_yoy_growth * 100).toFixed(0) + '%' : '—'}`,
          tone: C.gold,
        }));
      } catch {}
      try {
        const a = (await api.get('/api/v6/ascent/top/5')).data;
        (a?.rows || []).slice(0, 3).forEach((r: any) => out.push({
          text: `ASCENT ${r.ticker} ${r.ascent_score} · ${String(r.tier || '').toUpperCase()}`,
        }));
      } catch {}
      setSegs(out);
    })();
  }, []);

  if (segs.length === 0) return null;
  const loop = [...segs, ...segs]; // seamless wrap

  return (
    <div style={{ position: 'relative', zIndex: 1, overflow: 'hidden',
                  borderBottom: '1px solid #3a2920', background: 'rgba(16,10,7,0.92)' }}>
      <style>{`@keyframes qe-tape { from { transform: translateX(0); } to { transform: translateX(-50%); } }`}</style>
      <div style={{ display: 'inline-flex', gap: 0, whiteSpace: 'nowrap',
                    animation: `qe-tape ${Math.max(30, segs.length * 7)}s linear infinite`,
                    padding: '9px 0' }}>
        {loop.map((s, i) => (
          <span key={i} style={{ fontFamily: mono, fontSize: 10, letterSpacing: 1.6,
                                 color: s.tone || C.dust, padding: '0 26px',
                                 borderRight: '1px solid #2a1c14' }}>
            {s.text}
          </span>
        ))}
      </div>
    </div>
  );
};

export default LiveTape;
