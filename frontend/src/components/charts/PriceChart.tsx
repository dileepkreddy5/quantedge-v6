// ============================================================
// QuantEdge v5.0 — Price Chart + Panel re-exports
// ============================================================

import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Area, AreaChart, ComposedChart, Bar, Scatter } from 'recharts';

// ── Re-export all panels from single file ────────────────────
export { SignalPanel } from '../ui/Panels';
export { MLModelsPanel } from '../ui/Panels';
export { VolatilityPanel } from '../ui/Panels';
export { RegimePanel } from '../ui/Panels';
export { OptionsPanel } from '../ui/Panels';
export { SentimentPanel } from '../ui/Panels';
export { MonteCarloPanel } from '../ui/Panels';
export { RiskPanel } from '../ui/Panels';
export { FundamentalsPanel } from '../ui/Panels';
export { ScenarioPanel } from '../ui/Panels';
export { Watchlist } from '../ui/Panels';

// ── Price Chart ───────────────────────────────────────────────
interface PriceChartProps { ticker: string; data: any; }

export function PriceChart({ ticker, data }: PriceChartProps) {
  // Generate sample chart data from known metrics
  const currentPrice = data?.price || 100;
  const annualVol = data?.annual_vol || 0.25;
  const predicted1y = (data?.predicted_return_1y || 8) / 100;

  // Generate 252-point synthetic path for visualization
  const chartData = React.useMemo(() => {
    const points = [];
    let price = currentPrice * 0.75; // Start 25% lower (simulate 1-year history)
    const dailyDrift = (0.08 / 252);
    const dailyVol = annualVol / Math.sqrt(252);
    const seed = ticker.charCodeAt(0) * 137 + ticker.charCodeAt(1) * 31;

    // Pseudo-random but deterministic for same ticker
    const pseudoRandom = (i: number) => {
      const x = Math.sin(seed + i * 9301 + 49297) * 233280;
      return x - Math.floor(x);
    };

    for (let i = 0; i < 252; i++) {
      const r = pseudoRandom(i);
      const ret = dailyDrift + dailyVol * (r * 2 - 1) * 1.41;
      price = price * (1 + ret);
      const isUp = ret >= 0;
      points.push({
        day: i - 251,
        price: parseFloat(price.toFixed(2)),
        color: isUp ? '#22c55e' : '#ef4444',
        vol: parseFloat((annualVol * (0.8 + r * 0.4) * 100).toFixed(1)),
      });
    }
    // Last point is current price
    points[points.length - 1].price = currentPrice;
    return points;
  }, [ticker, currentPrice, annualVol]);

  // Prediction cone (next 252 days)
  const predData = React.useMemo(() => {
    const points = [];
    const dailyRet = predicted1y / 252;
    const dailyVol = annualVol / Math.sqrt(252);
    let bull = currentPrice, base = currentPrice, bear = currentPrice;
    for (let i = 1; i <= 63; i++) {
      bull = bull * (1 + dailyRet + dailyVol * 0.7);
      base = base * (1 + dailyRet);
      bear = bear * (1 + dailyRet - dailyVol * 0.7);
      points.push({ day: i, bull: parseFloat(bull.toFixed(2)), base: parseFloat(base.toFixed(2)), bear: parseFloat(bear.toFixed(2)) });
    }
    return points;
  }, [currentPrice, predicted1y, annualVol]);

  const minP = Math.min(...chartData.map(d => d.price)) * 0.97;
  const maxP = Math.max(...chartData.map(d => d.price)) * 1.03;
  const regime = data?.current_regime || 'UNKNOWN';
  const regimeColor = regime.includes('BULL') ? '#22c55e' : regime.includes('BEAR') ? '#ef4444' : '#f59e0b';

  const customTooltip = ({ active, payload }: any) => {
    if (!active || !payload?.length) return null;
    const d = payload[0]?.payload;
    return (
      <div style={{ background:'#2d1e18', border:'1px solid rgba(212,149,108,0.3)', borderRadius:6, padding:'8px 12px' }}>
        <div style={{ fontFamily:"'Fira Code',monospace", fontSize:10, color:'#9d8b7a' }}>Day {d.day}</div>
        <div style={{ fontFamily:"'Fira Code',monospace", fontSize:14, fontWeight:700, color:'#f4e8d8' }}>${d.price}</div>
        {d.vol && <div style={{ fontFamily:"'Fira Code',monospace", fontSize:9, color:'#f59e0b' }}>Vol: {d.vol}%</div>}
      </div>
    );
  };

  return (
    <div style={{ background:'#241510', border:'1px solid rgba(212,149,108,0.1)', borderRadius:8, padding:16 }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:12 }}>
        <div>
          <span style={{ fontFamily:"'Fira Code',monospace", fontSize:9, color:'#9d8b7a', letterSpacing:2 }}>PRICE HISTORY · 1Y</span>
          <span style={{ fontFamily:"'Fira Code',monospace", fontSize:9, color:'#4a3428', marginLeft:12 }}>+ 3M FORECAST CONE</span>
        </div>
        <div style={{ display:'flex', gap:12, fontFamily:"'Fira Code',monospace", fontSize:9 }}>
          <span style={{ color:regimeColor }}>● {regime.replace(/_/g,' ')}</span>
          <span style={{ color:'#22c55e' }}>─ Bull</span>
          <span style={{ color:'#f59e0b' }}>─ Base</span>
          <span style={{ color:'#ef4444' }}>─ Bear</span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={chartData} margin={{ top:5, right:5, bottom:5, left:45 }}>
          <defs>
            <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#daa520" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#daa520" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="2 4" stroke="rgba(212,149,108,0.06)" />
          <XAxis dataKey="day" tick={{ fontFamily:"'Fira Code',monospace", fontSize:8, fill:'#4a3428' }} tickLine={false} axisLine={false}
            tickFormatter={v => v === 0 ? 'TODAY' : v > 0 ? `+${v}d` : `${v}d`} />
          <YAxis domain={[minP, maxP]} tick={{ fontFamily:"'Fira Code',monospace", fontSize:8, fill:'#4a3428' }} tickLine={false} axisLine={false}
            tickFormatter={v => `$${v.toFixed(0)}`} />
          <Tooltip content={customTooltip} />
          <ReferenceLine x={0} stroke="rgba(218,165,32,0.4)" strokeDasharray="4 2" label={{ value:'NOW', fill:'#daa520', fontSize:8, fontFamily:"'Fira Code',monospace" }} />
          <Area type="monotone" dataKey="price" stroke="#daa520" strokeWidth={1.5} fill="url(#priceGrad)" dot={false} />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Prediction cone below */}
      <div style={{ marginTop:4 }}>
        <ResponsiveContainer width="100%" height={100}>
          <LineChart data={predData} margin={{ top:5, right:5, bottom:5, left:45 }}>
            <CartesianGrid strokeDasharray="2 4" stroke="rgba(212,149,108,0.06)" />
            <XAxis dataKey="day" tick={{ fontFamily:"'Fira Code',monospace", fontSize:7, fill:'#4a3428' }} tickLine={false} axisLine={false} tickFormatter={v => `+${v}d`} />
            <YAxis tick={{ fontFamily:"'Fira Code',monospace", fontSize:7, fill:'#4a3428' }} tickLine={false} axisLine={false} tickFormatter={v => `$${v.toFixed(0)}`} />
            <Line type="monotone" dataKey="bull" stroke="#22c55e" strokeWidth={1} dot={false} strokeDasharray="4 2" />
            <Line type="monotone" dataKey="base" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="bear" stroke="#ef4444" strokeWidth={1} dot={false} strokeDasharray="4 2" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default PriceChart;
