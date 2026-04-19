// ============================================================
// QuantEdge v6.0 — Screener Page
// Ranked universe scan across 3 horizons with regime overlay
// ============================================================

import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../auth/authStore';
import toast from 'react-hot-toast';

type HorizonKey = 'short_term' | 'medium_term' | 'long_term';

interface RankedRow {
  rank: number;
  ticker: string;
  composite_score: number;
  composite_score_raw?: number;
  quality: number;
  momentum: number;
  accumulation: number;
  trend: number;
  regime_multiplier?: number;
  metrics: Record<string, any>;
}

interface RegimeState {
  regime: string;
  multiplier: number;
  vix_level?: number | null;
  spy_vol_20d?: number | null;
  breadth_pct_above_200ma?: number | null;
  reasoning: string;
  timestamp: string;
}

interface ScanResponse {
  scan_timestamp: number;
  duration_seconds: number;
  universe_size: number;
  tickers_scored: number;
  tickers_failed: number;
  regime: RegimeState;
  rankings: Record<HorizonKey, RankedRow[]>;
}

const HORIZON_TABS: Array<{ id: HorizonKey; label: string; description: string }> = [
  {
    id: 'long_term',
    label: '⏳ LONG TERM',
    description: '2-5 year horizon · Quality 55% · Trend 20% · Momentum 15% · Accumulation 10%',
  },
  {
    id: 'medium_term',
    label: '⏱ MEDIUM TERM',
    description: '6-12 month horizon · Quality 35% · Momentum 25% · Trend 20% · Accumulation 20%',
  },
  {
    id: 'short_term',
    label: '⚡ SHORT TERM',
    description: '1-3 month horizon · Momentum 40% · Accumulation 30% · Trend 20% · Quality 10%',
  },
];

function scoreColor(v: number): string {
  if (v >= 75) return '#22c55e';  // green
  if (v >= 55) return '#b8860b';  // blue
  if (v >= 45) return '#9d8b7a';  // gray
  if (v >= 30) return '#f59e0b';  // orange
  return '#ef4444';                // red
}

function regimeColor(regime: string): string {
  switch (regime) {
    case 'calm':     return '#22c55e';
    case 'normal':   return '#b8860b';
    case 'elevated': return '#f59e0b';
    case 'panic':    return '#ef4444';
    default:         return '#9d8b7a';
  }
}

function ExplainerPanel() {
  const [open, setOpen] = React.useState(false);
  return (
    <div style={{
      background: '#1a0f0a',
      border: '1px solid #3a2920',
      borderRadius: 6,
      marginBottom: 16,
      overflow: 'hidden',
    }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          width: '100%',
          padding: '12px 18px',
          background: 'transparent',
          border: 'none',
          color: '#daa520',
          fontSize: 11,
          fontFamily: "'Fira Code', monospace",
          letterSpacing: 2,
          textAlign: 'left',
          cursor: 'pointer',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <span>NEW HERE? WHAT IS THIS?</span>
        <span style={{ fontSize: 10, color: '#9d8b7a' }}>{open ? '▼ HIDE' : '▶ EXPAND'}</span>
      </button>
      {open && (
        <div style={{ padding: '0 24px 20px', color: '#d4c4b0', fontSize: 13, lineHeight: 1.7 }}>
          <p style={{ marginTop: 0 }}>
            This screener scans 500+ liquid US stocks daily and ranks them across three time horizons.
            Each stock gets a <strong style={{ color: '#daa520' }}>composite score (0–100)</strong> built
            from four independent factors. Higher is better.
          </p>

          <div style={{
            fontFamily: "'Fira Code', monospace", fontSize: 10, letterSpacing: 2,
            color: '#9d8b7a', margin: '20px 0 10px',
          }}>
            THE FOUR FACTORS
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10 }}>
            <div>
              <div style={{ color: '#daa520', fontWeight: 600, fontSize: 12 }}>QUALITY</div>
              <div style={{ fontSize: 12 }}>Business fundamentals — ROIC, margins, debt levels, Piotroski score. How good is the underlying company.</div>
            </div>
            <div>
              <div style={{ color: '#daa520', fontWeight: 600, fontSize: 12 }}>MOMENTUM</div>
              <div style={{ fontSize: 12 }}>Price strength over 3/6/12 months. Is the stock trending up vs peers.</div>
            </div>
            <div>
              <div style={{ color: '#daa520', fontWeight: 600, fontSize: 12 }}>ACCUMULATION</div>
              <div style={{ fontSize: 12 }}>Volume signals suggesting institutional buying. Are big players adding positions.</div>
            </div>
            <div>
              <div style={{ color: '#daa520', fontWeight: 600, fontSize: 12 }}>TREND</div>
              <div style={{ fontSize: 12 }}>Moving average alignment, long-term direction. Is the stock above its 50/200-day lines.</div>
            </div>
          </div>

          <div style={{
            fontFamily: "'Fira Code', monospace", fontSize: 10, letterSpacing: 2,
            color: '#9d8b7a', margin: '20px 0 10px',
          }}>
            THE THREE HORIZONS
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div><strong style={{ color: '#daa520' }}>LONG TERM</strong> (2–5 years) — weights quality heavily. For patient investors prioritizing durable businesses.</div>
            <div><strong style={{ color: '#daa520' }}>MEDIUM TERM</strong> (6–12 months) — balanced factor mix. Suitable for core allocation.</div>
            <div><strong style={{ color: '#daa520' }}>SHORT TERM</strong> (1–3 months) — momentum-heavy. Captures trends; requires frequent rebalancing.</div>
          </div>

          <div style={{
            fontFamily: "'Fira Code', monospace", fontSize: 10, letterSpacing: 2,
            color: '#9d8b7a', margin: '20px 0 10px',
          }}>
            READING A ROW
          </div>
          <div style={{ fontSize: 12 }}>
            Each row shows the composite score and the four factor sub-scores. Colors:{' '}
            <span style={{ color: '#22c55e' }}>green ≥75 strong</span>,{' '}
            <span style={{ color: '#daa520' }}>amber 55–75 constructive</span>,{' '}
            <span style={{ color: '#9d8b7a' }}>cream 45–55 neutral</span>,{' '}
            <span style={{ color: '#f59e0b' }}>yellow 30–45 cautious</span>,{' '}
            <span style={{ color: '#ef4444' }}>red &lt;30 weak</span>.
            Click any row to see the underlying metrics.
          </div>

          <div style={{
            marginTop: 18, paddingTop: 14, borderTop: '1px solid #3a2920',
            fontSize: 11, color: '#9d8b7a', lineHeight: 1.6,
          }}>
            <strong style={{ color: '#b8860b' }}>Honest caveat:</strong> Scores are universe-relative signals,
            not buy/sell recommendations. Backtest validation shows only short-term
            configuration retained edge out-of-sample (~55% Sharpe retention,
            ~5–8% annualized alpha realistic forward expectation).
            Consider paper trading 3–6 months before deploying capital.
          </div>
        </div>
      )}
    </div>
  );
}

function FactorBar({ label, value }: { label: string; value: number }) {
  const width = `${Math.max(2, value)}%`;
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        fontSize: 11, color: '#9d8b7a', marginBottom: 3,
      }}>
        <span>{label}</span>
        <span style={{ fontFamily: 'JetBrains Mono, monospace', color: scoreColor(value) }}>
          {value.toFixed(1)}
        </span>
      </div>
      <div style={{ height: 4, background: '#3a2920', borderRadius: 2 }}>
        <div style={{
          width, height: '100%', borderRadius: 2,
          background: scoreColor(value), transition: 'width 0.3s ease',
        }} />
      </div>
    </div>
  );
}

function RegimeBanner({ regime }: { regime: RegimeState }) {
  const color = regimeColor(regime.regime);
  return (
    <div style={{
      background: '#1a0f0a',
      border: `1px solid ${color}`,
      borderRadius: 8,
      padding: '14px 20px',
      marginBottom: 20,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: 20,
    }}>
      <div>
        <div style={{ fontSize: 11, color: '#9d8b7a', letterSpacing: 1.5, marginBottom: 4 }}>
          MARKET REGIME
        </div>
        <div style={{
          fontSize: 20, fontWeight: 700, color, letterSpacing: 2, textTransform: 'uppercase',
        }}>
          {regime.regime}
          <span style={{
            fontSize: 14, color: '#9d8b7a', marginLeft: 12, letterSpacing: 0, fontWeight: 400,
          }}>
            × {regime.multiplier}
          </span>
        </div>
      </div>
      <div style={{ flex: 1, fontSize: 12, color: '#9d8b7a', lineHeight: 1.5 }}>
        {regime.reasoning || 'No stress signals detected'}
      </div>
    </div>
  );
}

function FactorDetails({ row }: { row: RankedRow }) {
  const m = row.metrics || {};
  const format = (v: any, suffix = '') => {
    if (v === null || v === undefined) return '—';
    if (typeof v === 'number') return `${v.toFixed(2)}${suffix}`;
    return String(v);
  };
  return (
    <div style={{
      background: '#0a0505', padding: 16, borderTop: '1px solid #3a2920',
      fontSize: 12, color: '#d4c4b0',
    }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 20 }}>
        <div>
          <div style={{ fontWeight: 600, color: '#daa520', marginBottom: 8 }}>MOMENTUM</div>
          <div>12-1mo: {format(m.mom_12_1, '%')}</div>
          <div>6mo: {format(m.mom_6m, '%')}</div>
          <div>3mo: {format(m.mom_3m, '%')}</div>
          <div>Sharpe 3m: {format(m.sharpe_3m)}</div>
        </div>
        <div>
          <div style={{ fontWeight: 600, color: '#daa520', marginBottom: 8 }}>ACCUMULATION</div>
          <div>Vol surge: {format(m.volume_surge, 'x')}</div>
          <div>OBV slope: {format(m.obv_slope_norm)}</div>
          <div>Amihud: {format(m.amihud)}</div>
        </div>
        <div>
          <div style={{ fontWeight: 600, color: '#daa520', marginBottom: 8 }}>TREND</div>
          <div>% above MA50: {format(m.pct_above_ma50, '%')}</div>
          <div>% above MA200: {format(m.pct_above_ma200, '%')}</div>
          <div>MA alignment: {m.ma_alignment > 0 ? '↑ bullish' : m.ma_alignment < 0 ? '↓ bearish' : '— mixed'}</div>
          <div>Hurst: {format(m.hurst)}</div>
        </div>
        <div>
          <div style={{ fontWeight: 600, color: '#daa520', marginBottom: 8 }}>QUALITY</div>
          <div>Piotroski: {m.quality_piotroski !== undefined ? `${m.quality_piotroski} / 9` : '—'}</div>
          <div>Altman Z: {format(m.quality_altman_z)}</div>
          <div>Data: {m.quality_data_q || '—'}</div>
        </div>
      </div>
    </div>
  );
}

export default function Screener({ embedded = false }: { embedded?: boolean } = {}) {
  const navigate = useNavigate();
  const [scan, setScan] = useState<ScanResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<HorizonKey>('long_term');
  const [expanded, setExpanded] = useState<string | null>(null);
  const [topN, setTopN] = useState(25);
  const [maxTickers, setMaxTickers] = useState(200);

  const fetchScan = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get(`/api/v6/screener/all?top_n=${topN}&max_tickers=${maxTickers}`);
      setScan(res.data.data);
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err.message || 'Scan failed';
      setError(msg);
      toast.error(`Scan failed: ${msg}`);
    } finally {
      setLoading(false);
    }
  }, [topN, maxTickers]);

  useEffect(() => {
    fetchScan();
    // eslint-disable-next-line
  }, []);

  const handleForceRescan = async () => {
    setLoading(true);
    setError(null);
    try {
      await api.post(`/api/v6/screener/rescan?max_tickers=${maxTickers}`);
      toast.success('Rescan triggered', { icon: '🔄' });
      await fetchScan();
    } catch (err: any) {
      toast.error('Rescan failed');
    } finally {
      setLoading(false);
    }
  };

  const currentRankings = scan?.rankings?.[activeTab] || [];
  const activeTabDesc = HORIZON_TABS.find((t) => t.id === activeTab)?.description || '';

  return (
    <div style={{
      minHeight: embedded ? 'auto' : '100vh',
      background: embedded ? 'transparent' : '#0a0505',
      color: '#d4c4b0',
      fontFamily: 'Inter, sans-serif',
      padding: embedded ? '0' : '20px 30px',
    }}>
      {/* Header */}
      {!embedded && (
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          marginBottom: 24,
        }}>
          <div>
            <div style={{
              fontSize: 26, fontWeight: 700, letterSpacing: 3, color: '#daa520',
            }}>
              QUANTEDGE SCREENER
            </div>
            <div style={{ fontSize: 12, color: '#9d8b7a', marginTop: 4 }}>
              Ranked universe across three horizons · Regime-adjusted · 4 orthogonal factors
            </div>
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <button
              onClick={() => navigate('/dashboard')}
              style={btnStyle('#2d1e18')}
            >
              ← Dashboard
            </button>
            <button
              onClick={handleForceRescan}
              disabled={loading}
              style={btnStyle('#daa520')}
            >
              {loading ? '...' : '🔄 Rescan'}
            </button>
          </div>
        </div>
      )}
      {embedded && (
        <div style={{
          display: 'flex', justifyContent: 'flex-end', marginBottom: 12,
        }}>
          <button
            onClick={handleForceRescan}
            disabled={loading}
            style={{ ...btnStyle('#daa520'), fontSize: 11 }}
          >
            {loading ? '...' : '🔄 Rescan'}
          </button>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          padding: 12, marginBottom: 16,
          background: '#2d1410', border: '1px solid #ef4444', borderRadius: 6,
          color: '#f4e8d8',
        }}>
          ⚠ {error}
        </div>
      )}

      {/* Beginner explainer */}
      <ExplainerPanel />

      {/* Regime banner */}
      {scan?.regime && <RegimeBanner regime={scan.regime} />}

      {/* Meta info */}
      {scan && (
        <div style={{
          display: 'flex', gap: 24, fontSize: 12, color: '#9d8b7a',
          marginBottom: 20, flexWrap: 'wrap',
        }}>
          <span>📊 Universe: {scan.universe_size}</span>
          <span>✓ Scored: {scan.tickers_scored}</span>
          {scan.tickers_failed > 0 && <span>✗ Failed: {scan.tickers_failed}</span>}
          <span>⏱ Duration: {scan.duration_seconds}s</span>
          <span>🕐 Scanned: {new Date(scan.scan_timestamp * 1000).toLocaleString()}</span>
        </div>
      )}

      {/* Horizon tabs */}
      <div style={{
        display: 'flex', borderBottom: '1px solid #3a2920', marginBottom: 0,
      }}>
        {HORIZON_TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => { setActiveTab(tab.id); setExpanded(null); }}
            style={{
              padding: '12px 24px',
              background: 'transparent',
              border: 'none',
              borderBottom: activeTab === tab.id ? '2px solid #daa520' : '2px solid transparent',
              color: activeTab === tab.id ? '#daa520' : '#9d8b7a',
              fontSize: 13,
              fontWeight: 600,
              letterSpacing: 1.5,
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div style={{
        fontSize: 11, color: '#4a3428', padding: '10px 0 18px 0', letterSpacing: 0.5,
      }}>
        {activeTabDesc}
      </div>

      {/* Rankings table */}
      {loading && !scan && (
        <div style={{ textAlign: 'center', padding: 60, color: '#9d8b7a' }}>
          Running scan... this takes 10-30 seconds on first load
        </div>
      )}

      {scan && (
        <div style={{
          background: '#1a0f0a', border: '1px solid #3a2920', borderRadius: 8,
          overflow: 'hidden',
        }}>
          {/* Header row */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '50px 100px 110px 80px 80px 80px 80px 40px',
            padding: '10px 16px', fontSize: 11, color: '#9d8b7a',
            letterSpacing: 1.5, borderBottom: '1px solid #3a2920',
            fontWeight: 600,
          }}>
            <div>RANK</div>
            <div>TICKER</div>
            <div style={{ textAlign: 'right' }}>COMPOSITE</div>
            <div style={{ textAlign: 'right' }}>QUAL</div>
            <div style={{ textAlign: 'right' }}>MOM</div>
            <div style={{ textAlign: 'right' }}>ACC</div>
            <div style={{ textAlign: 'right' }}>TRND</div>
            <div></div>
          </div>

          {currentRankings.map((r) => {
            const isExpanded = expanded === r.ticker;
            return (
              <div key={r.ticker}>
                <div
                  onClick={() => setExpanded(isExpanded ? null : r.ticker)}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '50px 100px 110px 80px 80px 80px 80px 40px',
                    padding: '10px 16px',
                    fontSize: 13,
                    fontFamily: 'JetBrains Mono, monospace',
                    borderBottom: '1px solid #3a2920',
                    cursor: 'pointer',
                    background: isExpanded ? '#241510' : 'transparent',
                    transition: 'background 0.15s',
                  }}
                  onMouseEnter={(e) => {
                    if (!isExpanded) (e.currentTarget as HTMLDivElement).style.background = '#1f130d';
                  }}
                  onMouseLeave={(e) => {
                    if (!isExpanded) (e.currentTarget as HTMLDivElement).style.background = 'transparent';
                  }}
                >
                  <div style={{ color: '#4a3428' }}>{r.rank}</div>
                  <div style={{ color: '#d4c4b0', fontWeight: 600 }}>{r.ticker}</div>
                  <div style={{ textAlign: 'right', color: scoreColor(r.composite_score), fontWeight: 600 }}>
                    {r.composite_score.toFixed(1)}
                  </div>
                  <div style={{ textAlign: 'right', color: scoreColor(r.quality) }}>
                    {r.quality.toFixed(1)}
                  </div>
                  <div style={{ textAlign: 'right', color: scoreColor(r.momentum) }}>
                    {r.momentum.toFixed(1)}
                  </div>
                  <div style={{ textAlign: 'right', color: scoreColor(r.accumulation) }}>
                    {r.accumulation.toFixed(1)}
                  </div>
                  <div style={{ textAlign: 'right', color: scoreColor(r.trend) }}>
                    {r.trend.toFixed(1)}
                  </div>
                  <div style={{ textAlign: 'right', color: '#4a3428' }}>
                    {isExpanded ? '▼' : '▶'}
                  </div>
                </div>
                {isExpanded && (
                  <>
                    <div style={{ padding: '14px 16px', background: '#0a0505', borderBottom: '1px solid #3a2920' }}>
                      <FactorBar label="Quality"      value={r.quality} />
                      <FactorBar label="Momentum"     value={r.momentum} />
                      <FactorBar label="Accumulation" value={r.accumulation} />
                      <FactorBar label="Trend"        value={r.trend} />
                      <div style={{ marginTop: 10, display: 'flex', gap: 8 }}>
                        <button
                          onClick={(e) => { e.stopPropagation(); navigate(`/dashboard?ticker=${r.ticker}`); }}
                          style={{ ...btnStyle('#daa520'), fontSize: 11, padding: '6px 12px' }}
                        >
                          Full analysis →
                        </button>
                      </div>
                    </div>
                    <FactorDetails row={r} />
                  </>
                )}
              </div>
            );
          })}
          {currentRankings.length === 0 && !loading && (
            <div style={{ padding: 40, textAlign: 'center', color: '#9d8b7a' }}>
              No rankings available. Try forcing a rescan.
            </div>
          )}
        </div>
      )}

      {/* Footer */}
      <div style={{
        marginTop: 30, padding: '16px 0', fontSize: 11, color: '#4a3428',
        textAlign: 'center', borderTop: '1px solid #3a2920',
      }}>
        Scores are universe-relative · All signals Polygon-sourced · Not investment advice
      </div>
    </div>
  );
}

function btnStyle(bg: string): React.CSSProperties {
  return {
    padding: '8px 16px',
    background: bg,
    color: '#fff',
    border: 'none',
    borderRadius: 6,
    fontSize: 12,
    fontWeight: 600,
    letterSpacing: 1,
    cursor: 'pointer',
  };
}
