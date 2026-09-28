// Summary tab: plain-English summary from verified facts, a modern price chart, where the
// company shows up on QuantEdge, key numbers in plain words, and macro. Classic overview kept
// one click away while the new layout is reviewed.
import React, { useEffect, useRef, useState } from 'react';
import { createChart, LineStyle } from 'lightweight-charts';
import { api } from '../../auth/authStore';
const C = { s1: '#1c130e', s2: '#241610', b1: '#3a2920', gold: '#daa520', cocoa: '#8a7560', dust: '#9d8b7a', latte: '#d4c4b0', cream: '#f4e8d8', up: '#22c55e', dn: '#ef4444', blue: '#60a5fa' };
const mono = "'Fira Code',monospace";
const RANGES: [string, number][] = [['1M', 21], ['3M', 63], ['6M', 126], ['1Y', 252], ['5Y', 99999]];

const PriceChart2: React.FC<{ ticker: string }> = ({ ticker }) => {
  const ref = useRef<HTMLDivElement>(null); const [p, setP] = useState<any>(null);
  const [range, setRange] = useState('1Y'); const [mode, setMode] = useState<'candle' | 'line'>('candle'); const [leg, setLeg] = useState<any>(null);
  useEffect(() => { setP(null); api.get(`/api/v6/prices/${ticker}`).then(r => setP(r.data)).catch(() => setP({ bars: [] })); }, [ticker]);
  useEffect(() => {
    if (!ref.current || !p?.bars?.length) return;
    const all = p.bars; const n = (RANGES.find(r => r[0] === range) || RANGES[3])[1]; const bars = all.slice(-Math.min(n, all.length));
    const chart = createChart(ref.current, { height: 420, layout: { background: { color: 'rgba(0,0,0,0)' }, textColor: C.dust, fontFamily: mono, fontSize: 10 },
      grid: { vertLines: { color: 'rgba(58,41,32,0.35)' }, horzLines: { color: 'rgba(58,41,32,0.35)' } },
      rightPriceScale: { borderColor: C.b1, scaleMargins: { top: 0.08, bottom: 0.24 } }, timeScale: { borderColor: C.b1, fixLeftEdge: true, fixRightEdge: true },
      crosshair: { mode: 0, vertLine: { color: C.gold, labelBackgroundColor: '#3a2920' }, horzLine: { color: C.gold, labelBackgroundColor: '#3a2920' } } });
    const main: any = mode === 'candle'
      ? chart.addCandlestickSeries({ upColor: C.up, downColor: C.dn, borderUpColor: C.up, borderDownColor: C.dn, wickUpColor: C.up, wickDownColor: C.dn })
      : chart.addAreaSeries({ lineColor: C.gold, topColor: 'rgba(218,165,32,0.25)', bottomColor: 'rgba(218,165,32,0.0)', lineWidth: 2 });
    main.setData(bars.map((b: any) => mode === 'candle' ? { time: b.d, open: b.o, high: b.h, low: b.l, close: b.c } : { time: b.d, value: b.c }));
    const vol = chart.addHistogramSeries({ priceScaleId: 'vol', priceFormat: { type: 'volume' }, lastValueVisible: false, priceLineVisible: false });
    chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    vol.setData(bars.map((b: any) => ({ time: b.d, value: b.v, color: b.c >= b.o ? 'rgba(34,197,94,0.35)' : 'rgba(239,68,68,0.35)' })));
    const idx0 = all.length - bars.length;
    for (const [k, col, ls] of [[50, C.gold, LineStyle.Solid], [200, C.blue, LineStyle.Dashed]] as any) {
      const s = chart.addLineSeries({ color: col, lineWidth: 1, lineStyle: ls, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false });
      const pts: any[] = []; let sum = 0;
      for (let i = 0; i < all.length; i++) { sum += all[i].c; if (i >= k) sum -= all[i - k].c; if (i >= k - 1 && i >= idx0) pts.push({ time: all[i].d, value: sum / k }); }
      s.setData(pts);
    }
    const last252 = all.slice(-252); const hi = Math.max(...last252.map((b: any) => b.h)); const lo = Math.min(...last252.map((b: any) => b.l));
    main.createPriceLine({ price: hi, color: '#d4956c', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: '52W HIGH' });
    main.createPriceLine({ price: lo, color: C.blue, lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: '52W LOW' });
    const first = bars[0].d; const inWin = new Set(bars.map((b: any) => b.d));
    const marks = (p.earnings || []).filter((d: string) => d >= first).map((d: string) => { const b = bars.find((x: any) => x.d >= d); return b && inWin.has(b.d) ? { time: b.d, position: 'aboveBar', color: C.blue, shape: 'circle', text: 'E', size: 0.8 } : null; }).filter(Boolean);
    try { main.setMarkers(marks); } catch { }
    chart.subscribeCrosshairMove((q: any) => { if (!q.time) { setLeg(null); return; } const b = bars.find((x: any) => x.d === q.time); setLeg(b || null); });
    chart.timeScale().fitContent();
    const ro = new ResizeObserver(() => chart.applyOptions({ width: ref.current?.clientWidth || 800 })); ro.observe(ref.current);
    return () => { ro.disconnect(); chart.remove(); };
  }, [p, range, mode]);
  const btn = (on: boolean) => ({ fontFamily: mono, fontSize: 10.5, padding: '6px 12px', background: on ? 'rgba(218,165,32,0.12)' : 'none', border: `1px solid ${on ? C.gold : C.b1}`, borderRadius: 5, color: on ? C.gold : C.dust, cursor: 'pointer' });
  const bars = p?.bars || []; const rn = (RANGES.find(r => r[0] === range) || RANGES[3])[1]; const win = bars.slice(-Math.min(rn, bars.length));
  const chg = win.length > 1 ? win[win.length - 1].c / win[0].c - 1 : null;
  return (<div style={{ background: C.s1, border: `1px solid ${C.b1}`, borderRadius: 10, padding: 14, marginBottom: 16, position: 'relative' }}>
    <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', marginBottom: 8 }}>
      {RANGES.map(([k]) => <button key={k} style={btn(range === k)} onClick={() => setRange(k)}>{k}</button>)}
      {chg != null && <span style={{ fontFamily: mono, fontSize: 12, color: chg >= 0 ? C.up : C.dn, marginLeft: 8 }}>{chg >= 0 ? '+' : ''}{(chg * 100).toFixed(1)}% over {range}</span>}
      <span style={{ flex: 1 }} />
      <button style={btn(mode === 'candle')} onClick={() => setMode('candle')}>Candles</button><button style={btn(mode === 'line')} onClick={() => setMode('line')}>Line</button>
    </div>
    {!p && <div style={{ fontFamily: mono, fontSize: 11, color: C.dust, padding: 20 }}>loading prices…</div>}
    <div ref={ref} style={{ width: '100%' }} />
    <div style={{ position: 'absolute', left: 24, top: 56, fontFamily: mono, fontSize: 10.5, color: C.latte, pointerEvents: 'none', background: 'rgba(16,10,7,0.75)', padding: '3px 8px', borderRadius: 4 }}>
      {leg ? <>{leg.d} · O {leg.o.toFixed(2)} H {leg.h.toFixed(2)} L {leg.l.toFixed(2)} C <b style={{ color: leg.c >= leg.o ? C.up : C.dn }}>{leg.c.toFixed(2)}</b></> : 'gold = 50-day avg · blue dashed = 200-day avg · E = earnings'}</div>
    <div style={{ fontFamily: mono, fontSize: 9.5, color: C.cocoa, marginTop: 6 }}>{p?.note}</div>
  </div>);
};

const SummaryTab: React.FC<{ ticker: string; data: any; macro?: React.ReactNode; classic?: React.ReactNode }> = ({ ticker, data, macro, classic }) => {
  const [s, setS] = useState<any>(null); const [showClassic, setShowClassic] = useState(false);
  useEffect(() => { setS(null); api.get(`/api/v6/summary/${ticker}`).then(r => setS(r.data)).catch(e => setS({ error: e?.response?.data?.detail || 'summary unavailable' })); }, [ticker]);
  const card = { background: C.s1, border: `1px solid ${C.b1}`, borderRadius: 10, padding: 18, marginBottom: 16 };
  const beta = data?.capm_beta ?? data?.beta; const dd = data?.max_drawdown;
  return (<div>
    <div style={card}>
      <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: 2, color: C.gold, marginBottom: 12 }}>IN PLAIN ENGLISH{s?.as_of ? ` · AS OF ${s.as_of}` : ''}</div>
      {!s && <div style={{ fontFamily: mono, fontSize: 11, color: C.dust }}>writing the summary…</div>}
      {s?.error && <div style={{ fontSize: 14, color: C.dust }}>{s.error}</div>}
      {(s?.sentences || []).map((x: any, i: number) => <p key={i} style={{ fontSize: 15.5, lineHeight: 1.7, color: C.cream, margin: '0 0 10px' }}>{x.text}</p>)}
      {s?.next_results_est && <div style={{ fontFamily: mono, fontSize: 11, color: C.dust, marginTop: 6 }}>Next results expected around {s.next_results_est} (estimate).</div>}
      {s?.history_note && <div style={{ fontFamily: mono, fontSize: 10.5, color: C.cocoa, marginTop: 6 }}>Price {s.history_note}.</div>}
    </div>
    <PriceChart2 ticker={ticker} />
    {s && !s.error && (s.trackers?.length || s.has_warnings || s.breakthroughs?.length) ? (<div style={card}>
      <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: 2, color: C.gold, marginBottom: 10 }}>WHERE IT SHOWS UP ON QUANTEDGE</div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
        {(s.trackers || []).map((t: string) => <span key={t} style={{ fontFamily: mono, fontSize: 11, color: C.up, border: '1px solid #1f5c35', borderRadius: 5, padding: '4px 10px' }}>{t}</span>)}
        {s.has_warnings && <span style={{ fontFamily: mono, fontSize: 11, color: '#e0ad3a', border: '1px solid #6b5520', borderRadius: 5, padding: '4px 10px' }}>⚠ Warning signs</span>}
        {(s.breakthroughs || []).map((b: any, i: number) => <span key={i} style={{ fontFamily: mono, fontSize: 11, color: '#f3cf7a', border: '1px solid #6b5520', borderRadius: 5, padding: '4px 10px' }}>★ {b.label} · {b.date}</span>)}
      </div></div>) : null}
    <div style={card}>
      <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: 2, color: C.gold, marginBottom: 10 }}>KEY NUMBERS, IN PLAIN WORDS</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(260px,1fr))', gap: 12, fontSize: 14, color: C.latte, lineHeight: 1.6 }}>
        <div><b style={{ color: C.cream }}>{dd != null ? `${(Math.abs(dd) * (Math.abs(dd) <= 1 ? 100 : 1)).toFixed(0)}%` : '—'}</b> — its worst fall from a peak in the price history we hold.</div>
        <div><b style={{ color: C.cream }}>{beta != null ? `${Number(beta).toFixed(2)}×` : '—'}</b> — how much it tends to move when the whole market moves 1% (its “beta”).</div>
        <div><b style={{ color: C.cream }}>{data?.annual_vol != null ? `${(data.annual_vol * (data.annual_vol <= 1 ? 100 : 1)).toFixed(0)}%` : '—'}</b> — a typical year's price swing (annual volatility).</div>
      </div></div>
    {macro && <div style={{ marginBottom: 16 }}>{macro}</div>}
    <div style={{ textAlign: 'center', margin: '20px 0 40px' }}>
      <button onClick={() => setShowClassic(v => !v)} style={{ fontFamily: mono, fontSize: 11, color: C.dust, background: 'none', border: `1px solid ${C.b1}`, borderRadius: 6, padding: '8px 14px', cursor: 'pointer' }}>
        {showClassic ? 'Hide classic overview' : 'Show classic overview (being retired)'}</button>
      {showClassic && <div style={{ marginTop: 16, textAlign: 'left' }}>{classic}</div>}
    </div>
  </div>);
};
export default SummaryTab;
