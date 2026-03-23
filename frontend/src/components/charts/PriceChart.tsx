// ============================================================
// QuantEdge v6.0 — Live Price Chart
// Calls /api/v6/chart/{ticker}?timeframe=1Y (real Polygon data)
// Timeframes: 1D, 5D, 1M, 3M, 1Y, 5Y
// Overlays: SMA20, SMA50, EMA9, VWAP, Bollinger Bands
// Sub-panels: RSI(14), MACD(12,26,9), Volume with spike detection
// ============================================================

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  ComposedChart, LineChart, BarChart,
  Line, Bar, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { api } from '../../auth/authStore';

// ── Re-export all panels ─────────────────────────────────────
export { SignalPanel }     from '../ui/Panels';
export { MLModelsPanel }   from '../ui/Panels';
export { VolatilityPanel } from '../ui/Panels';
export { RegimePanel }     from '../ui/Panels';
export { OptionsPanel }    from '../ui/Panels';
export { SentimentPanel }  from '../ui/Panels';
export { MonteCarloPanel } from '../ui/Panels';
export { RiskPanel }       from '../ui/Panels';
export { FundamentalsPanel } from '../ui/Panels';
export { ScenarioPanel }   from '../ui/Panels';
export { Watchlist }       from '../ui/Panels';

// ── Indicator math ───────────────────────────────────────────
function sma(arr: number[], n: number): (number | null)[] {
  return arr.map((_, i) =>
    i < n - 1 ? null : arr.slice(i - n + 1, i + 1).reduce((s, v) => s + v, 0) / n
  );
}
function ema(arr: number[], n: number): (number | null)[] {
  const k = 2 / (n + 1);
  const out: (number | null)[] = new Array(arr.length).fill(null);
  let seed = arr.slice(0, n).reduce((s, v) => s + v, 0) / n;
  out[n - 1] = seed;
  for (let i = n; i < arr.length; i++) out[i] = arr[i] * k + (out[i - 1] as number) * (1 - k);
  return out;
}
function bollinger(closes: number[], n = 20, mult = 2) {
  const mid = sma(closes, n);
  const upper: (number | null)[] = [], lower: (number | null)[] = [];
  closes.forEach((_, i) => {
    if (i < n - 1) { upper.push(null); lower.push(null); return; }
    const sl = closes.slice(i - n + 1, i + 1);
    const m  = mid[i] as number;
    const sd = Math.sqrt(sl.reduce((s, v) => s + (v - m) ** 2, 0) / n);
    upper.push(+(m + mult * sd).toFixed(2));
    lower.push(+(m - mult * sd).toFixed(2));
  });
  return { mid, upper, lower };
}
function rsi(closes: number[], n = 14): (number | null)[] {
  if (closes.length < n + 1) return closes.map(() => null);
  const out: (number | null)[] = new Array(closes.length).fill(null);
  let ag = 0, al = 0;
  for (let i = 1; i <= n; i++) {
    const d = closes[i] - closes[i - 1];
    if (d > 0) ag += d; else al -= d;
  }
  ag /= n; al /= n;
  out[n] = al === 0 ? 100 : 100 - 100 / (1 + ag / al);
  for (let i = n + 1; i < closes.length; i++) {
    const d = closes[i] - closes[i - 1];
    ag = (ag * (n - 1) + (d > 0 ? d : 0)) / n;
    al = (al * (n - 1) + (d < 0 ? -d : 0)) / n;
    out[i] = al === 0 ? 100 : +(100 - 100 / (1 + ag / al)).toFixed(2);
  }
  return out;
}
function macdCalc(closes: number[]) {
  const e12 = ema(closes, 12), e26 = ema(closes, 26);
  const line = closes.map((_, i) =>
    e12[i] != null && e26[i] != null ? +((e12[i] as number) - (e26[i] as number)).toFixed(3) : null
  );
  const validFrom = line.findIndex(v => v != null);
  const sigInput  = line.map(v => v ?? 0);
  const sigRaw    = ema(sigInput, 9);
  const signal    = sigRaw.map((v, i) => i < validFrom + 8 ? null : v != null ? +v.toFixed(3) : null);
  const hist      = line.map((v, i) =>
    v != null && signal[i] != null ? +(v - (signal[i] as number)).toFixed(3) : null
  );
  return { line, signal, hist };
}

// ── OHLCV bar type ───────────────────────────────────────────
interface Bar {
  label: string;
  o: number; h: number; l: number; c: number;
  v: number; vw: number;
  sma20?: number | null; sma50?: number | null;
  ema9?: number | null;
  bbUpper?: number | null; bbLower?: number | null;
  rsi?: number | null;
  macdLine?: number | null; macdSignal?: number | null; macdHist?: number | null;
  volColor?: string;
}

const MONO = "'IBM Plex Mono', 'Fira Code', monospace";
const TF_LABELS: Record<string, string> = {
  '1D':'5-min','5D':'15-min','1M':'Daily','3M':'Daily','1Y':'Weekly','5Y':'Monthly'
};

// ── Custom tooltip ───────────────────────────────────────────
const ChartTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload as Bar;
  if (!d) return null;
  const isUp = d.c >= d.o;
  const chg  = d.c - d.o;
  const pct  = ((chg / d.o) * 100).toFixed(2);
  return (
    <div style={{ background:'#0d1016', border:'1px solid rgba(255,255,255,0.12)', borderRadius:4, padding:'8px 12px', fontFamily:MONO, fontSize:10 }}>
      <div style={{ color:'#8a8880', marginBottom:4 }}>{d.label}</div>
      <div style={{ color:'#e8e6e0', fontWeight:600, fontSize:12 }}>${d.c.toFixed(2)}</div>
      <div style={{ color: isUp ? '#4caf82' : '#e05252' }}>{isUp?'+':''}{chg.toFixed(2)} ({isUp?'+':''}{pct}%)</div>
      <div style={{ color:'#555350', marginTop:4 }}>
        O:{d.o.toFixed(2)} H:{d.h.toFixed(2)} L:{d.l.toFixed(2)}
      </div>
      <div style={{ color:'#555350' }}>Vol: {(d.v / 1e6).toFixed(2)}M</div>
      {d.vw > 0 && <div style={{ color:'#a88a52' }}>VWAP: ${d.vw.toFixed(2)}</div>}
    </div>
  );
};

// ── Main component ───────────────────────────────────────────
interface PriceChartProps { ticker: string; data: any; }

export function PriceChart({ ticker, data: analysisData }: PriceChartProps) {
  const [tf, setTf]           = useState('1Y');
  const [bars, setBars]       = useState<Bar[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState('');
  const [chartType, setChartType] = useState<'line'|'area'>('area');
  const [overlays, setOverlays]   = useState({ sma20:true, sma50:true, ema9:false, bb:false, vwap:true });
  const [subs, setSubs]           = useState({ rsi:true, macd:false });
  const [meta, setMeta]           = useState<any>({});
  const abortRef = useRef<AbortController | null>(null);

  const fetchChart = useCallback(async (timeframe: string) => {
    if (!ticker) return;
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();

    setLoading(true); setError('');
    try {
      const res = await api.get(`/api/v6/chart/${ticker}`, { params: { timeframe } });
      const d   = res.data;
      setMeta(d);

      const closes  = d.closes  as number[];
      const opens   = d.opens   as number[];
      const highs   = d.highs   as number[];
      const lows    = d.lows    as number[];
      const volumes = d.volumes as number[];
      const vwaps   = d.vwaps   as number[];
      const labels  = d.labels  as string[];

      // Compute indicators
      const sma20v  = sma(closes, 20);
      const sma50v  = sma(closes, 50);
      const ema9v   = ema(closes, 9);
      const bbv     = bollinger(closes, 20, 2);
      const rsiV    = rsi(closes, 14);
      const macdV   = macdCalc(closes);
      const avgVol  = volumes.reduce((s, v) => s + v, 0) / volumes.length;

      const built: Bar[] = closes.map((c, i) => ({
        label:      labels[i],
        o: opens[i], h: highs[i], l: lows[i], c,
        v: volumes[i], vw: vwaps[i],
        sma20:      sma20v[i],
        sma50:      sma50v[i],
        ema9:       ema9v[i],
        bbUpper:    bbv.upper[i],
        bbLower:    bbv.lower[i],
        rsi:        rsiV[i],
        macdLine:   macdV.line[i],
        macdSignal: macdV.signal[i],
        macdHist:   macdV.hist[i],
        volColor:   volumes[i] > avgVol * 1.8
          ? 'rgba(224,82,82,0.80)'
          : c >= opens[i] ? 'rgba(76,175,130,0.45)' : 'rgba(224,82,82,0.40)',
      }));

      setBars(built);
    } catch (e: any) {
      if (e.name !== 'CanceledError') setError('Chart data unavailable');
    } finally {
      setLoading(false);
    }
  }, [ticker]);

  useEffect(() => { fetchChart(tf); }, [ticker, tf, fetchChart]);

  // Derived display values
  const isUp        = meta.changePct >= 0;
  const priceColor  = isUp ? '#4caf82' : '#e05252';
  const lastClose   = meta.closes ? meta.closes[meta.closes.length - 1] : (analysisData?.price || 0);
  const regime      = analysisData?.current_regime || 'UNKNOWN';
  const regColor    = regime.includes('BULL') ? '#4caf82' : regime.includes('BEAR') ? '#e05252' : '#d4943a';

  const rsiNow = bars.length ? bars[bars.length - 1]?.rsi : null;
  const rsiColor = rsiNow == null ? '#555350' : rsiNow > 70 ? '#e05252' : rsiNow < 30 ? '#4caf82' : '#d4943a';

  const macdNow   = bars.length ? bars[bars.length - 1]?.macdLine   : null;
  const signalNow = bars.length ? bars[bars.length - 1]?.macdSignal : null;

  const toggleOverlay = (k: keyof typeof overlays) =>
    setOverlays(p => ({ ...p, [k]: !p[k] }));
  const toggleSub = (k: keyof typeof subs) =>
    setSubs(p => ({ ...p, [k]: !p[k] }));

  const btnStyle = (active: boolean): React.CSSProperties => ({
    background: active ? 'rgba(91,143,207,0.18)' : 'transparent',
    border: `1px solid ${active ? 'rgba(91,143,207,0.45)' : 'rgba(255,255,255,0.08)'}`,
    color: active ? '#5b8fcf' : '#555350',
    fontFamily: MONO, fontSize: 9, fontWeight: 600,
    letterSpacing: '0.08em', padding: '2px 7px',
    cursor: 'pointer', transition: 'all 0.12s',
  });

  const tfBtn = (t: string): React.CSSProperties => ({
    background: tf === t ? 'rgba(200,169,110,0.15)' : 'transparent',
    border: `1px solid ${tf === t ? '#a88a52' : 'rgba(255,255,255,0.08)'}`,
    color: tf === t ? '#c8a96e' : '#555350',
    fontFamily: MONO, fontSize: 10, fontWeight: 600,
    letterSpacing: '0.1em', padding: '3px 9px',
    cursor: 'pointer', transition: 'all 0.12s',
  });

  // Y-axis domain with padding
  const allPrices = bars.flatMap(b => [b.h, b.l]).filter(Boolean);
  const yMin = allPrices.length ? Math.min(...allPrices) * 0.98 : 0;
  const yMax = allPrices.length ? Math.max(...allPrices) * 1.02 : 1;

  return (
    <div style={{ background:'#0f1217', border:'1px solid rgba(255,255,255,0.06)', borderRadius:0 }}>

      {/* ── ROW 1: TICKER + TF + OVERLAYS ── */}
      <div style={{ display:'flex', alignItems:'center', gap:6, padding:'8px 14px', borderBottom:'1px solid rgba(255,255,255,0.05)', flexWrap:'wrap' }}>
        {/* Ticker + price */}
        <div style={{ display:'flex', alignItems:'center', gap:8, marginRight:8 }}>
          <span style={{ fontFamily:MONO, fontSize:13, fontWeight:600, color:'#c8a96e', letterSpacing:'0.06em' }}>{ticker}</span>
          <span style={{ fontFamily:MONO, fontSize:13, color:'#e8e6e0' }}>${lastClose.toFixed(2)}</span>
          {meta.changePct != null && (
            <span style={{ fontFamily:MONO, fontSize:11, color:priceColor }}>
              {isUp?'+':''}{meta.changePct?.toFixed(2)}%
            </span>
          )}
        </div>

        {/* Timeframe buttons */}
        <div style={{ display:'flex', gap:2 }}>
          {['1D','5D','1M','3M','1Y','5Y'].map(t => (
            <button key={t} style={tfBtn(t)} onClick={() => setTf(t)}>{t}</button>
          ))}
        </div>

        {/* Chart type */}
        <div style={{ display:'flex', gap:2, marginLeft:6 }}>
          {(['line','area'] as const).map(ct => (
            <button key={ct} style={btnStyle(chartType === ct)} onClick={() => setChartType(ct)}>
              {ct === 'line' ? '—' : '◺'}
            </button>
          ))}
        </div>

        {/* Overlay toggles */}
        <div style={{ display:'flex', gap:2, marginLeft:6 }}>
          {(Object.keys(overlays) as (keyof typeof overlays)[]).map(k => (
            <button key={k} style={btnStyle(overlays[k])} onClick={() => toggleOverlay(k)}>
              {k.toUpperCase()}
            </button>
          ))}
        </div>

        {/* Sub-indicator toggles */}
        <div style={{ display:'flex', gap:2, marginLeft:6 }}>
          {(Object.keys(subs) as (keyof typeof subs)[]).map(k => (
            <button key={k} style={btnStyle(subs[k])} onClick={() => toggleSub(k)}>
              {k.toUpperCase()}
            </button>
          ))}
        </div>

        {/* Status */}
        <div style={{ marginLeft:'auto', display:'flex', alignItems:'center', gap:10 }}>
          {loading && (
            <span style={{ fontFamily:MONO, fontSize:9, color:'#c8a96e', animation:'blink 1s infinite', letterSpacing:'0.1em' }}>
              LOADING...
            </span>
          )}
          {!loading && (
            <span style={{ fontFamily:MONO, fontSize:8, color:'#444340', letterSpacing:'0.1em' }}>
              ● POLYGON LIVE
            </span>
          )}
        </div>
      </div>

      {/* ── ROW 2: OHLCV BAR ── */}
      {meta.n_bars && (
        <div style={{ display:'flex', gap:0, background:'#0a0c0f', borderBottom:'1px solid rgba(255,255,255,0.04)', padding:'5px 14px', overflowX:'auto' }}>
          {[
            ['O', `$${meta.opens?.[meta.opens.length-1]?.toFixed(2) ?? '—'}`, '#e8e6e0'],
            ['H', `$${meta.periodHigh?.toFixed(2) ?? '—'}`, '#4caf82'],
            ['L', `$${meta.periodLow?.toFixed(2) ?? '—'}`, '#e05252'],
            ['C', `$${lastClose?.toFixed(2) ?? '—'}`, priceColor],
            ['CHG', `${isUp?'+':''}${meta.change?.toFixed(2)} (${isUp?'+':''}${meta.changePct?.toFixed(2)}%)`, priceColor],
            ['VOL', meta.avgVolume ? `${(meta.avgVolume/1e6).toFixed(1)}M avg` : '—', '#8a8880'],
            ['BARS', `${meta.n_bars} · ${meta.interval ?? TF_LABELS[tf]}`, '#555350'],
            ['REGIME', regime.replace(/_/g,' '), regColor],
          ].map(([l,v,c]) => (
            <div key={l as string} style={{ padding:'0 12px', borderRight:'1px solid rgba(255,255,255,0.04)', flexShrink:0 }}>
              <div style={{ fontFamily:MONO, fontSize:8, color:'#333230', letterSpacing:'0.15em', marginBottom:1 }}>{l}</div>
              <div style={{ fontFamily:MONO, fontSize:11, color:c as string, fontWeight:500 }}>{v}</div>
            </div>
          ))}
        </div>
      )}

      {/* ── PRICE CHART ── */}
      {error ? (
        <div style={{ textAlign:'center', padding:40, fontFamily:MONO, fontSize:11, color:'#555350' }}>
          {error} — check API key or try another ticker
        </div>
      ) : (
        <div style={{ padding:'8px 14px 0' }}>
          <ResponsiveContainer width="100%" height={240}>
            <ComposedChart data={bars} margin={{ top:4, right:4, bottom:0, left:0 }}>
              <defs>
                <linearGradient id="qeAreaGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={isUp ? '#4caf82' : '#c8a96e'} stopOpacity={0.18} />
                  <stop offset="95%" stopColor={isUp ? '#4caf82' : '#c8a96e'} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="2 4" stroke="rgba(255,255,255,0.03)" vertical={false} />
              <XAxis dataKey="label" tick={{ fontFamily:MONO, fontSize:8, fill:'#444340' }}
                tickLine={false} axisLine={false}
                interval={Math.max(1, Math.floor(bars.length / 8))} />
              <YAxis domain={[yMin, yMax]} orientation="right"
                tick={{ fontFamily:MONO, fontSize:8, fill:'#444340' }}
                tickLine={false} axisLine={false}
                tickFormatter={v => `$${v.toFixed(0)}`} width={52} />
              <Tooltip content={<ChartTooltip />} />

              {/* Bollinger Bands */}
              {overlays.bb && <Line dataKey="bbUpper" stroke="rgba(91,143,207,0.25)" strokeWidth={1} dot={false} strokeDasharray="3 3" />}
              {overlays.bb && <Line dataKey="bbLower" stroke="rgba(91,143,207,0.25)" strokeWidth={1} dot={false} strokeDasharray="3 3" />}

              {/* Main price */}
              {chartType === 'area'
                ? <Area type="monotone" dataKey="c" stroke={isUp ? '#4caf82' : '#c8a96e'}
                    strokeWidth={1.5} fill="url(#qeAreaGrad)" dot={false} />
                : <Line type="monotone" dataKey="c" stroke={isUp ? '#4caf82' : '#c8a96e'}
                    strokeWidth={1.5} dot={false} />
              }

              {/* Overlays */}
              {overlays.sma20 && <Line dataKey="sma20" stroke="rgba(212,148,58,0.70)" strokeWidth={1} dot={false} />}
              {overlays.sma50 && <Line dataKey="sma50" stroke="rgba(91,143,207,0.70)" strokeWidth={1} dot={false} />}
              {overlays.ema9  && <Line dataKey="ema9"  stroke="rgba(200,169,110,0.65)" strokeWidth={1} strokeDasharray="4 2" dot={false} />}
              {overlays.vwap  && <Line dataKey="vw"    stroke="rgba(168,138,82,0.50)"  strokeWidth={1} strokeDasharray="2 4" dot={false} />}
            </ComposedChart>
          </ResponsiveContainer>

          {/* ── VOLUME ── */}
          <div style={{ display:'flex', justifyContent:'space-between', padding:'3px 0 2px' }}>
            <span style={{ fontFamily:MONO, fontSize:8, color:'#444340', letterSpacing:'0.12em' }}>VOLUME</span>
            {meta.avgVolume && (
              <span style={{ fontFamily:MONO, fontSize:8, color:'#444340' }}>
                avg {(meta.avgVolume/1e6).toFixed(1)}M
              </span>
            )}
          </div>
          <ResponsiveContainer width="100%" height={50}>
            <BarChart data={bars} margin={{ top:0, right:4, bottom:0, left:0 }}>
              <XAxis dataKey="label" hide />
              <YAxis orientation="right" tick={{ fontFamily:MONO, fontSize:7, fill:'#333230' }}
                tickLine={false} axisLine={false} width={52}
                tickFormatter={v => `${(v/1e6).toFixed(0)}M`} tickCount={2} />
              <Bar dataKey="v" fill="#c8a96e" opacity={0.35} radius={0}
                label={false}
                // per-bar color via Cell would need Cell import — use shape fill workaround
                shape={(props: any) => {
                  const { x, y, width, height, payload } = props;
                  return <rect x={x} y={y} width={width} height={height} fill={payload?.volColor || 'rgba(200,169,110,0.35)'} />;
                }}
              />
            </BarChart>
          </ResponsiveContainer>

          {/* ── RSI ── */}
          {subs.rsi && bars.some(b => b.rsi != null) && (
            <>
              <div style={{ display:'flex', justifyContent:'space-between', padding:'3px 0 2px' }}>
                <span style={{ fontFamily:MONO, fontSize:8, color:'#444340', letterSpacing:'0.12em' }}>RSI(14)</span>
                {rsiNow != null && (
                  <span style={{ fontFamily:MONO, fontSize:9, color:rsiColor, fontWeight:600 }}>
                    {rsiNow.toFixed(1)}
                    {rsiNow > 70 ? ' OVERBOUGHT' : rsiNow < 30 ? ' OVERSOLD' : ''}
                  </span>
                )}
              </div>
              <ResponsiveContainer width="100%" height={46}>
                <ComposedChart data={bars} margin={{ top:0, right:4, bottom:0, left:0 }}>
                  <XAxis dataKey="label" hide />
                  <YAxis domain={[0, 100]} orientation="right"
                    tick={{ fontFamily:MONO, fontSize:7, fill:'#333230' }}
                    tickLine={false} axisLine={false} width={52}
                    ticks={[30, 70]} />
                  <ReferenceLine y={70} stroke="rgba(224,82,82,0.20)" strokeDasharray="3 3" />
                  <ReferenceLine y={30} stroke="rgba(76,175,130,0.20)" strokeDasharray="3 3" />
                  <ReferenceLine y={50} stroke="rgba(255,255,255,0.04)" />
                  <Line dataKey="rsi" stroke="rgba(212,148,58,0.75)" strokeWidth={1.2} dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </>
          )}

          {/* ── MACD ── */}
          {subs.macd && bars.some(b => b.macdLine != null) && (
            <>
              <div style={{ display:'flex', justifyContent:'space-between', padding:'3px 0 2px' }}>
                <span style={{ fontFamily:MONO, fontSize:8, color:'#444340', letterSpacing:'0.12em' }}>MACD(12,26,9)</span>
                {macdNow != null && signalNow != null && (
                  <span style={{ fontFamily:MONO, fontSize:9,
                    color: macdNow > signalNow ? '#4caf82' : '#e05252', fontWeight:600 }}>
                    {macdNow > signalNow ? '▲ BULL CROSS' : '▼ BEAR CROSS'} · {macdNow.toFixed(3)}
                  </span>
                )}
              </div>
              <ResponsiveContainer width="100%" height={50}>
                <ComposedChart data={bars} margin={{ top:0, right:4, bottom:0, left:0 }}>
                  <XAxis dataKey="label" hide />
                  <YAxis orientation="right" tick={{ fontFamily:MONO, fontSize:7, fill:'#333230' }}
                    tickLine={false} axisLine={false} width={52}
                    tickFormatter={v => v.toFixed(2)} tickCount={3} />
                  <ReferenceLine y={0} stroke="rgba(255,255,255,0.06)" />
                  <Bar dataKey="macdHist"
                    shape={(props: any) => {
                      const { x, y, width, height, payload } = props;
                      const col = (payload?.macdHist ?? 0) >= 0
                        ? 'rgba(76,175,130,0.55)' : 'rgba(224,82,82,0.55)';
                      return <rect x={x} y={y} width={width} height={height} fill={col} />;
                    }}
                  />
                  <Line dataKey="macdLine"   stroke="rgba(91,143,207,0.85)"  strokeWidth={1} dot={false} />
                  <Line dataKey="macdSignal" stroke="rgba(224,82,82,0.75)"   strokeWidth={1} dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </>
          )}

          {/* ── REGIME + TECH SUMMARY FOOTER ── */}
          <div style={{ display:'flex', alignItems:'center', gap:12, borderTop:'1px solid rgba(255,255,255,0.05)', padding:'6px 0', marginTop:4, flexWrap:'wrap' }}>
            <div style={{ fontFamily:MONO, fontSize:8, color:regColor,
              background:`${regColor}12`, padding:'2px 8px', letterSpacing:'0.1em' }}>
              ● {regime.replace(/_/g,' ')}
            </div>
            {overlays.sma20 && bars.length && bars[bars.length-1]?.sma20 != null && (
              <span style={{ fontFamily:MONO, fontSize:8, color:'rgba(212,148,58,0.8)' }}>
                SMA20: ${bars[bars.length-1].sma20?.toFixed(2)}
              </span>
            )}
            {overlays.sma50 && bars.length && bars[bars.length-1]?.sma50 != null && (
              <span style={{ fontFamily:MONO, fontSize:8, color:'rgba(91,143,207,0.8)' }}>
                SMA50: ${bars[bars.length-1].sma50?.toFixed(2)}
              </span>
            )}
            {rsiNow != null && subs.rsi && (
              <span style={{ fontFamily:MONO, fontSize:8, color:rsiColor }}>
                RSI: {rsiNow.toFixed(1)}
              </span>
            )}
            <span style={{ marginLeft:'auto', fontFamily:MONO, fontSize:8, color:'#333230', letterSpacing:'0.1em' }}>
              SOURCE: POLYGON.IO · REAL-TIME
            </span>
          </div>
        </div>
      )}

      <style>{`@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}`}</style>
    </div>
  );
}

export default PriceChart;
