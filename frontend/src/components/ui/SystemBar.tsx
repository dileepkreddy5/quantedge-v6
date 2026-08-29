// ============================================================
// QuantEdge v6.0 — System Bar
// Live inventory from /api/v6/system/stats. Nothing here is a
// literal: every count is computed from the catalogs, the model
// directory and the scan artifacts at request time.
// ============================================================

import React, { useEffect, useState } from 'react';

interface Tab { tab: string; categories: number; signals: number; live: number; needs_source: number; reference: number; }
interface Board { board: string; available: boolean; generated: string | null; age_hours: number | null; stale: boolean; }
interface Stats {
  signals: { signals_total: number; signals_live: number; signals_needs_source: number;
             signals_reference: number; categories: number; catalogs: number; per_tab: Tab[] };
  panel: { panel_models: number; trained_at: string | null; n_tickers: number | null;
           n_features: number | null; horizons: number; any_reliable: boolean };
  boards: Board[]; tabs: number; price_history_years: number; universe_note: string;
}

const C = {
  s0: '#100a07', s2: '#241610', b1: '#3a2920', b2: '#4a3428',
  gold: '#daa520', caramel: '#d4956c', burnt: '#c9762f',
  cocoa: '#8a7560', dust: '#9d8b7a', latte: '#d4c4b0', cream: '#f4e8d8',
  bull: '#22c55e', warn: '#f59e0b',
};
const mono = "'Fira Code',monospace";

const age = (h: number | null) =>
  h == null ? '—' : h < 48 ? `${Math.round(h)}h ago` : `${Math.round(h / 24)}d ago`;

const Metric: React.FC<{ v: string; k: string; tone?: string; sub?: string }> =
  ({ v, k, tone, sub }) => (
  <div style={{ minWidth: 128 }}>
    <div style={{ fontFamily: mono, fontSize: 30, fontWeight: 700, color: tone || C.cream,
                  lineHeight: 1.1, letterSpacing: -0.5 }}>{v}</div>
    <div style={{ fontFamily: mono, fontSize: 8.5, letterSpacing: 1.6, color: C.cocoa, marginTop: 6 }}>{k}</div>
    {sub && <div style={{ fontFamily: mono, fontSize: 10, color: C.dust, marginTop: 3 }}>{sub}</div>}
  </div>
);

const SystemBar: React.FC = () => {
  const [d, setD] = useState<Stats | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch('/api/v6/system/stats');
        if (r.ok) setD(await r.json());
      } catch { /* render nothing */ }
    })();
  }, []);

  if (!d) return null;

  const s = d.signals, p = d.panel;
  const livePct = s.signals_total ? (s.signals_live / s.signals_total) * 100 : 0;
  const stale = d.boards.filter(b => b.stale);

  return (
    <section style={{ position: 'relative', zIndex: 1, maxWidth: 1400,
                      margin: '0 auto 56px', padding: '0 4rem' }}>
      <div style={{
        background: `linear-gradient(150deg, ${C.s2} 0%, ${C.s0} 100%)`,
        border: `1px solid ${C.b1}`, borderRadius: 10, overflow: 'hidden',
      }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 44, padding: '28px 30px 24px' }}>
          <Metric v={String(s.signals_live)} k="LIVE SIGNALS" tone={C.gold}
                  sub={`of ${s.signals_total} defined`} />
          <Metric v={String(s.categories)} k="SCORED CATEGORIES"
                  sub={`across ${s.catalogs} catalogs`} />
          <Metric v={String(d.tabs)} k="ANALYSIS TABS" />
          <Metric v={String(p.panel_models)} k="TRAINED MODELS" tone={C.caramel}
                  sub={p.n_features ? `${p.n_features} features · ${p.horizons} horizons` : undefined} />
          <Metric v={p.n_tickers ? String(p.n_tickers) : '—'} k="PANEL TICKERS"
                  sub={p.trained_at ? `retrained ${age((Date.now() - new Date(p.trained_at).getTime()) / 36e5)}` : undefined} />
          <Metric v={`${d.price_history_years}Y`} k="PRICE HISTORY" sub="plan ceiling" />
        </div>

        <div style={{ padding: '0 30px 20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between',
                        fontFamily: mono, fontSize: 9, letterSpacing: 1.3,
                        color: C.cocoa, marginBottom: 7 }}>
            <span>SIGNAL COVERAGE — {livePct.toFixed(1)}% COMPUTING LIVE</span>
            <span>{s.signals_needs_source} AWAITING SOURCE · {s.signals_reference} REFERENCE</span>
          </div>
          <div style={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden',
                        background: 'rgba(0,0,0,0.35)' }}>
            <div style={{ width: `${livePct}%`, background: `linear-gradient(90deg,${C.gold},${C.caramel})` }} />
            <div style={{ width: `${(s.signals_needs_source / s.signals_total) * 100}%`, background: C.b2 }} />
          </div>

          <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', marginTop: 16 }}>
            {s.per_tab.slice(0, 12).map(t => (
              <div key={t.tab} title={`${t.live}/${t.signals} live · ${t.categories} categories`}
                   style={{
                     fontFamily: mono, fontSize: 9, letterSpacing: 0.8, padding: '5px 9px',
                     border: `1px solid ${t.needs_source > 0 ? C.b2 : 'rgba(218,165,32,0.28)'}`,
                     borderRadius: 3, color: t.needs_source > 0 ? C.dust : C.latte,
                   }}>
                {t.tab.toUpperCase()} <span style={{ color: C.cocoa }}>{t.signals}</span>
              </div>
            ))}
          </div>
        </div>

        {stale.length > 0 && (
          <div style={{
            borderTop: `1px solid ${C.b1}`, background: 'rgba(245,158,11,0.05)',
            padding: '12px 30px', fontFamily: mono, fontSize: 10.5, color: C.warn,
          }}>
            ⚠ {stale.map(b => `${b.board} scan ${age(b.age_hours)}`).join(' · ')} — rows below
            are from the last completed scan, not today.
          </div>
        )}
      </div>
    </section>
  );
};

export default SystemBar;
