// ============================================================
// QuantEdge v6.0 — Live Intelligence Board (homepage)
// Real rows from the nightly scans: multibagger tiers + ascent radar.
// Every value is computed server-side; nothing here is illustrative.
// ============================================================

import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../../auth/authStore';

interface MBRow {
  ticker: string; score: number;
  qtr_yoy_growth: number | null; piotroski: number | null;
  margin_trend: number | null; accruals: number | null; debt_trend: number | null;
  market_cap: number | null; price_ladder: Record<string, number> | null;
  vol_change_pct: number | null; up_vol_ratio: number | null;
  quiet_price: boolean | null; price_move_6mo: number | null;
}
interface RbRow {
  ticker: string; name?: string; score?: number | null; prior_high?: number | null;
  discount_pct?: number | null; drawdown_pct?: number | null;
  recovery?: { progress_pct?: number | null } | null; [k: string]: any;
}
interface AscRow {
  rank: number; ticker: string; name: string; sector: string;
  ascent_score: number; tier: string; is_new: boolean;
}

const C = {
  s0: '#100a07', s2: '#241610', s3: '#2f1e16', b1: '#3a2920', b2: '#4a3428',
  gold: '#daa520', caramel: '#d4956c', burnt: '#c9762f',
  cocoa: '#8a7560', dust: '#9d8b7a', latte: '#d4c4b0', cream: '#f4e8d8',
  bull: '#22c55e', bear: '#ef4444', warn: '#f59e0b',
};

const heat = (s: number | null) =>
  s == null ? C.b2 : s >= 70 ? C.gold : s >= 50 ? C.caramel : s >= 30 ? C.burnt : C.bear;

const LADDER = ['1d', '3d', '1w', '2w', '1m', '2m', '3m'];
const mono = "'Fira Code',monospace";

const pct = (v: number | null, d = 1) =>
  v == null ? '—' : (v >= 0 ? '+' : '') + (v * 100).toFixed(d) + '%';
const cap = (v: number | null) =>
  v == null ? '—' : v >= 1000 ? `$${(v / 1000).toFixed(1)}T`
    : v >= 1 ? `$${v.toFixed(1)}B` : `$${(v * 1000).toFixed(0)}M`;

/** 9-segment Piotroski quality meter. */
const Piotroski: React.FC<{ v: number | null }> = ({ v }) => (
  <div style={{ display: 'flex', gap: 2 }}>
    {Array.from({ length: 9 }).map((_, i) => (
      <div key={i} style={{
        width: 6, height: 12, borderRadius: 1,
        background: v != null && i < v
          ? (v >= 7 ? C.gold : v >= 5 ? C.caramel : C.burnt)
          : 'rgba(255,255,255,0.06)',
      }} />
    ))}
  </div>
);

/** Diverging bars across the price ladder — each window's move, zero-centred. */
const Ladder: React.FC<{ l: Record<string, number> | null }> = ({ l }) => {
  if (!l) return <span style={{ color: C.cocoa, fontSize: 10 }}>no ladder</span>;
  const vals = LADDER.map(k => l[k]).filter(v => typeof v === 'number');
  const max = Math.max(0.02, ...vals.map(v => Math.abs(v)));
  return (
    <div style={{ display: 'flex', gap: 3, alignItems: 'center', height: 34 }}>
      {LADDER.map(k => {
        const v = l[k];
        if (typeof v !== 'number') return <div key={k} style={{ width: 9 }} />;
        const h = Math.max(2, (Math.abs(v) / max) * 15);
        return (
          <div key={k} title={`${k}: ${pct(v)}`}
               style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 9 }}>
            <div style={{ height: 15, display: 'flex', alignItems: 'flex-end' }}>
              {v >= 0 && <div style={{ width: 7, height: h, background: C.bull, borderRadius: 1, opacity: 0.85 }} />}
            </div>
            <div style={{ width: 9, height: 1, background: C.b2 }} />
            <div style={{ height: 15 }}>
              {v < 0 && <div style={{ width: 7, height: h, background: C.bear, borderRadius: 1, opacity: 0.85 }} />}
            </div>
          </div>
        );
      })}
    </div>
  );
};

const Chip: React.FC<{ label: string; value: string; tone?: string }> = ({ label, value, tone }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
    <span style={{ fontFamily: mono, fontSize: 8, letterSpacing: 1.1, color: C.cocoa }}>{label}</span>
    <span style={{ fontFamily: mono, fontSize: 11.5, color: tone || C.latte }}>{value}</span>
  </div>
);

const LiveTrackers: React.FC = () => {
  const navigate = useNavigate();
  const [tab, setTab] = useState<'mb' | 'asc' | 'rb'>('mb');
  const [tier, setTier] = useState<'small' | 'mid' | 'large'>('small');
  const [mb, setMb] = useState<Record<string, MBRow[]> | null>(null);
  const [gen, setGen] = useState<string>('');
  const [asc, setAsc] = useState<AscRow[]>([]);
  const [rb, setRb] = useState<RbRow[]>([]);

  useEffect(() => {
    (async () => {
      try {
        const r = await api.get('/api/v6/scan/tiers');
        if (r.data?.tiers) { setMb(r.data.tiers); setGen(r.data.generated || ''); }
      } catch { /* scan not ready */ }
      try {
        const a = await api.get('/api/v6/ascent/top/5');
        if (a.data?.rows?.length) setAsc(a.data.rows);
      } catch { /* board not ready */ }
      try {
        const b = await api.get('/api/v6/rebound/list');
        const tiers = b.data?.tiers;
        if (tiers) {
          const flat: RbRow[] = Object.values(tiers).flat() as RbRow[];
          flat.sort((x, y) => (y.score ?? 0) - (x.score ?? 0));
          setRb(flat.slice(0, 6));
        }
      } catch { /* scan not ready */ }
    })();
  }, []);

  if (!mb && asc.length === 0) return null;

  const rows = (mb?.[tier] || []).slice(0, 6);
  const tabBtn = (active: boolean): React.CSSProperties => ({
    fontFamily: mono, fontSize: 10, letterSpacing: 1.6, padding: '9px 16px',
    background: active ? 'rgba(218,165,32,0.10)' : 'transparent',
    border: `1px solid ${active ? C.gold : C.b1}`, borderRadius: 4,
    color: active ? C.gold : C.dust, cursor: 'pointer', transition: 'all .15s',
  });

  return (
    <section style={{ position: 'relative', zIndex: 1, maxWidth: 1400, margin: '0 auto 72px', padding: '0 4rem' }}>
      <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: 3, color: C.cocoa, marginBottom: 10 }}>
        LIVE INTELLIGENCE · REGENERATED NIGHTLY
      </div>
      <h2 style={{ fontSize: 40, fontWeight: 800, color: C.cream, margin: '0 0 12px', letterSpacing: -0.5 }}>
        THE BOARD
      </h2>
      <p style={{ color: C.dust, fontSize: 15, lineHeight: 1.7, maxWidth: 720, marginBottom: 24 }}>
        Roughly 5,150 investable US names scored every night against SEC bulk fundamentals
        and live Polygon prices. These are the current leaders — real rows, real scores.
      </p>

      <div style={{ display: 'flex', gap: 10, marginBottom: 18, flexWrap: 'wrap', alignItems: 'center' }}>
        <button style={tabBtn(tab === 'mb')} onClick={() => setTab('mb')}>◆ MULTIBAGGER</button>
        <button style={tabBtn(tab === 'asc')} onClick={() => setTab('asc')}>★ ASCENT RADAR</button>
        {rb.length > 0 && (
          <button style={tabBtn(tab === 'rb')} onClick={() => setTab('rb')}>↻ REBOUND</button>
        )}
        {tab === 'mb' && (
          <div style={{ display: 'flex', gap: 6, marginLeft: 8 }}>
            {(['small', 'mid', 'large'] as const).map(t => (
              <button key={t} onClick={() => setTier(t)} style={{
                fontFamily: mono, fontSize: 9, letterSpacing: 1.2, padding: '7px 12px',
                background: 'none', border: `1px solid ${tier === t ? C.caramel : C.b1}`,
                borderRadius: 3, color: tier === t ? C.caramel : C.cocoa, cursor: 'pointer',
              }}>{t.toUpperCase()}-CAP</button>
            ))}
          </div>
        )}
        <span style={{ fontFamily: mono, fontSize: 9.5, color: C.cocoa, marginLeft: 'auto' }}>
          {tab === 'mb'
            ? `${(mb?.[tier] || []).length} ${tier.toUpperCase()}-CAP RANKED` +
              (gen ? ` · SCAN ${new Date(gen).toLocaleDateString()}` : '')
            : tab === 'asc' ? `${asc.length} CLIMBERS` : `${rb.length} DISCOUNTED-QUALITY NAMES`}
        </span>
      </div>

      <div style={{ display: 'grid', gap: 10 }}>
        {tab === 'mb' && rows.length === 0 && (
          <div style={{ padding: '30px 20px', border: `1px solid ${C.b1}`, borderRadius: 8,
                        color: C.cocoa, fontFamily: mono, fontSize: 12 }}>
            No {tier}-cap rows in the current scan artifact.
          </div>
        )}
        {tab === 'mb' && rows.map((r, i) => (
          <div key={r.ticker} onClick={() => navigate(`/dashboard?ticker=${r.ticker}`)}
            style={{
              display: 'grid', gridTemplateColumns: '54px 128px 1fr auto', gap: 20,
              alignItems: 'center', padding: '16px 20px', cursor: 'pointer', minHeight: 104,
              background: `linear-gradient(90deg, ${C.s2}, ${C.s0})`,
              border: `1px solid ${C.b1}`, borderLeft: `3px solid ${heat(r.score)}`,
              borderRadius: 8, transition: 'all .15s',
            }}>
            <div style={{ fontFamily: mono, fontSize: 20, color: C.b2, fontWeight: 700 }}>
              {String(i + 1).padStart(2, '0')}
            </div>
            <div>
              <div style={{ fontFamily: mono, fontSize: 17, color: C.cream, fontWeight: 700, letterSpacing: 0.5 }}>
                {r.ticker}
              </div>
              <div style={{ fontFamily: mono, fontSize: 10, color: C.cocoa, marginTop: 3 }}>{cap(r.market_cap)}</div>
              <div style={{ marginTop: 7, height: 4, background: 'rgba(0,0,0,0.35)', borderRadius: 2 }}>
                <div style={{
                  height: 4, width: `${Math.min(100, (r.score / 130) * 100)}%`,
                  background: heat(r.score), borderRadius: 2,
                }} />
              </div>
              <div style={{ fontFamily: mono, fontSize: 12, color: heat(r.score), marginTop: 5 }}>
                {r.score?.toFixed(1)}
              </div>
            </div>
            <div style={{ display: 'flex', gap: 26, flexWrap: 'wrap', alignItems: 'center' }}>
              <Chip label="QTR YoY GROWTH" value={pct(r.qtr_yoy_growth, 0)}
                    tone={(r.qtr_yoy_growth ?? 0) > 0 ? C.bull : C.dust} />
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <span style={{ fontFamily: mono, fontSize: 8, letterSpacing: 1.1, color: C.cocoa }}>
                  PIOTROSKI {r.piotroski ?? '—'}/9
                </span>
                <Piotroski v={r.piotroski} />
              </div>
              <Chip label="ACCRUALS" value={r.accruals == null ? '—' : r.accruals.toFixed(3)}
                    tone={(r.accruals ?? 0) < 0 ? C.bull : C.warn} />
              <Chip label="DEBT TREND" value={r.debt_trend == null ? '—' : r.debt_trend.toFixed(3)}
                    tone={(r.debt_trend ?? 0) < 0 ? C.bull : C.warn} />
              <Chip label="UP-DAY VOL" value={r.up_vol_ratio == null ? '—' : (r.up_vol_ratio * 100).toFixed(0) + '%'} />
              {r.quiet_price && (
                <span style={{
                  fontFamily: mono, fontSize: 8.5, letterSpacing: 1, padding: '4px 8px',
                  border: `1px solid ${C.caramel}`, borderRadius: 3, color: C.caramel,
                }}>QUIET PRICE</span>
              )}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 5 }}>
              <span style={{ fontFamily: mono, fontSize: 8, letterSpacing: 1.1, color: C.cocoa }}>
                PRICE LADDER 1d → 3m
              </span>
              <Ladder l={r.price_ladder} />
            </div>
          </div>
        ))}

        {tab === 'asc' && asc.map(r => (
          <div key={r.ticker} onClick={() => navigate(`/dashboard?ticker=${r.ticker}`)}
            style={{
              display: 'grid', gridTemplateColumns: '54px 1fr auto', gap: 20, alignItems: 'center',
              padding: '16px 20px', cursor: 'pointer',
              background: `linear-gradient(90deg, ${C.s2}, ${C.s0})`,
              border: `1px solid ${C.b1}`, borderLeft: `3px solid ${heat(r.ascent_score)}`,
              borderRadius: 8,
            }}>
            <div style={{ fontFamily: mono, fontSize: 20, color: C.b2, fontWeight: 700 }}>
              {String(r.rank).padStart(2, '0')}
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{ fontFamily: mono, fontSize: 17, color: C.cream, fontWeight: 700 }}>{r.ticker}</span>
                {r.is_new && (
                  <span style={{
                    fontFamily: mono, fontSize: 8, letterSpacing: 1, padding: '3px 7px',
                    background: 'rgba(34,197,94,0.12)', border: `1px solid ${C.bull}`,
                    borderRadius: 3, color: C.bull,
                  }}>NEW</span>
                )}
              </div>
              <div style={{ color: C.dust, fontSize: 12.5, marginTop: 4 }}>{r.name}</div>
              <div style={{ fontFamily: mono, fontSize: 10, color: C.cocoa, marginTop: 3 }}>
                {r.sector} · {r.tier}
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontFamily: mono, fontSize: 24, color: heat(r.ascent_score), fontWeight: 700 }}>
                {r.ascent_score}
              </div>
              <div style={{ fontFamily: mono, fontSize: 8.5, letterSpacing: 1.2, color: C.cocoa }}>
                ASCENT SCORE
              </div>
            </div>
          </div>
        ))}
        {tab === 'rb' && rb.map((r, i) => {
          const prog = r.recovery?.progress_pct;
          const disc = r.discount_pct ?? r.drawdown_pct ?? (r as any).drawdown_from_high_pct;
          return (
            <div key={r.ticker} onClick={() => navigate(`/dashboard?ticker=${r.ticker}`)}
              style={{
                display: 'grid', gridTemplateColumns: '54px 1fr auto', gap: 20, alignItems: 'center',
                padding: '16px 20px', cursor: 'pointer', minHeight: 84,
                background: `linear-gradient(90deg, ${C.s2}, ${C.s0})`,
                border: `1px solid ${C.b1}`, borderLeft: `3px solid ${heat(r.score ?? null)}`,
                borderRadius: 8,
              }}>
              <div style={{ fontFamily: mono, fontSize: 20, color: C.b2, fontWeight: 700 }}>
                {String(i + 1).padStart(2, '0')}
              </div>
              <div>
                <div style={{ fontFamily: mono, fontSize: 17, color: C.cream, fontWeight: 700 }}>{r.ticker}</div>
                {r.name && <div style={{ color: C.dust, fontSize: 12.5, marginTop: 4 }}>{r.name}</div>}
                {prog != null && (
                  <div style={{ marginTop: 9, maxWidth: 320 }}>
                    <div style={{ fontFamily: mono, fontSize: 8, letterSpacing: 1.1, color: C.cocoa, marginBottom: 4 }}>
                      RECOVERY TOWARD PRIOR HIGH — {Number(prog).toFixed(0)}%
                    </div>
                    <div style={{ height: 5, background: 'rgba(0,0,0,0.35)', borderRadius: 3 }}>
                      <div style={{ height: 5, width: `${Math.max(0, Math.min(100, Number(prog)))}%`,
                                    background: `linear-gradient(90deg,${C.caramel},${C.bull})`, borderRadius: 3 }} />
                    </div>
                  </div>
                )}
              </div>
              <div style={{ textAlign: 'right' }}>
                {disc != null && (
                  <>
                    <div style={{ fontFamily: mono, fontSize: 22, color: C.bear, fontWeight: 700 }}>
                      {Number(disc) > 0 ? '-' : ''}{Math.abs(Number(disc)).toFixed(0)}%
                    </div>
                    <div style={{ fontFamily: mono, fontSize: 8.5, letterSpacing: 1.2, color: C.cocoa }}>
                      OFF PRIOR HIGH
                    </div>
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <div style={{
        marginTop: 14, fontFamily: mono, fontSize: 10.5, color: C.cocoa, lineHeight: 1.7,
      }}>
        A disciplined shortlist, not a predictor. Volume figures are honest measurements —
        share of recent volume on up-days and change against a 60-day baseline. True
        buy/sell split is not derivable from daily bars, so it is not shown.
      </div>
    </section>
  );
};

export default LiveTrackers;
