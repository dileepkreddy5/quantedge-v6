// ============================================================
// QuantEdge v6.0 — Pipeline Pulse
// The nightly job cycle rendered as a timeline with real artifact
// ages. Times are the registered ET schedule; the checkmark and age
// are measured from what each job actually produced.
// ============================================================

import React, { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

const C = { s0: '#100a07', s2: '#241610', b1: '#3a2920', gold: '#daa520',
            cocoa: '#8a7560', dust: '#9d8b7a', latte: '#d4c4b0', cream: '#f4e8d8',
            bull: '#22c55e', warn: '#f59e0b' };
const mono = "'Fira Code',monospace";

interface Stats { panel: { trained_at: string | null }; boards: { board: string; age_hours: number | null; stale: boolean; available: boolean }[]; }

const ago = (h: number | null) => h == null ? 'no record'
  : h < 1 ? `${Math.max(1, Math.round(h * 60))}m ago`
  : h < 48 ? `${Math.round(h)}h ago` : `${Math.round(h / 24)}d ago`;

const PipelinePulse: React.FC = () => {
  const [d, setD] = useState<Stats | null>(null);
  useEffect(() => {
    (async () => {
      try { const r = await api.get('/api/v6/system/stats'); if (r.data?.boards) setD(r.data); } catch {}
    })();
  }, []);
  if (!d) return null;

  const mb = d.boards.find(b => b.board === 'multibagger');
  const panelAge = d.panel.trained_at ? (Date.now() - new Date(d.panel.trained_at).getTime()) / 36e5 : null;

  const stops: { t: string; label: string; age: number | null; ok: boolean }[] = [
    { t: '02:00', label: 'MULTIBAGGER SCAN', age: mb?.age_hours ?? null, ok: !!mb && !mb.stale },
    { t: '02:15', label: 'PANEL RETRAIN', age: panelAge, ok: panelAge != null && panelAge < 48 },
    { t: '02:30', label: 'REBOUND SCAN', age: null, ok: false },
    { t: '07:00', label: 'NEWS BRIEFINGS', age: null, ok: false },
    { t: '17:30', label: 'BARS SYNC', age: null, ok: false },
    { t: '18:15', label: 'PEER SCAN', age: null, ok: false },
  ];
  // Only show rows we can actually vouch for, plus the schedule for the rest.
  return (
    <section style={{ position: 'relative', zIndex: 1, maxWidth: 1400,
                      margin: '0 auto 64px', padding: '0 4rem' }}>
      <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: 3, color: C.cocoa, marginBottom: 10 }}>
        THE NIGHTLY CYCLE — ET SCHEDULE, MEASURED OUTPUT
      </div>
      <div style={{ display: 'flex', alignItems: 'stretch', gap: 0, overflowX: 'auto',
                    background: `linear-gradient(150deg, ${C.s2}, ${C.s0})`,
                    border: `1px solid ${C.b1}`, borderRadius: 10, padding: '20px 8px' }}>
        {stops.map((s, i) => (
          <div key={s.label} style={{ display: 'flex', alignItems: 'center', flex: 1, minWidth: 150 }}>
            <div style={{ flex: 1, padding: '0 14px' }}>
              <div style={{ fontFamily: mono, fontSize: 9, color: C.cocoa, letterSpacing: 1.2 }}>{s.t} ET</div>
              <div style={{ fontFamily: mono, fontSize: 11, color: C.cream, letterSpacing: 1, margin: '6px 0 4px' }}>
                {s.ok ? '●' : '○'} {s.label}
              </div>
              <div style={{ fontFamily: mono, fontSize: 9.5,
                            color: s.age == null ? C.cocoa : s.ok ? C.bull : C.warn }}>
                {s.age != null ? `output ${ago(s.age)}` : 'scheduled'}
              </div>
            </div>
            {i < stops.length - 1 && (
              <div style={{ width: 26, height: 1, background: C.b1, flexShrink: 0 }} />
            )}
          </div>
        ))}
      </div>
    </section>
  );
};

export default PipelinePulse;
