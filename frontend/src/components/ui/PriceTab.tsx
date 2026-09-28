// Price & patterns: at a glance, Pattern Lab (whole), volatility, short interest, price states.
// Replaces the separate Market, Volatility and Regime tabs. Every number states its window.
import React, { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
import PatternLab from './PatternLab';
const C = { s1: '#1c130e', s2: '#241610', b1: '#3a2920', gold: '#daa520', cocoa: '#8a7560', dust: '#9d8b7a', latte: '#d4c4b0', cream: '#f4e8d8', up: '#22c55e', dn: '#ef4444' };
const mono = "'Fira Code',monospace";
const pc = (v: any, d = 1) => v == null ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(d)}%`;
const STATE: Record<string, string> = { BULL_LOW_VOL: 'Rising, calm', BULL_HIGH_VOL: 'Rising, choppy', BEAR_LOW_VOL: 'Falling, calm', BEAR_HIGH_VOL: 'Falling, choppy', MEAN_REVERT: 'Sideways', SIDEWAYS: 'Sideways', NEUTRAL: 'Sideways' };
const card = { background: C.s1, border: `1px solid ${C.b1}`, borderRadius: 10, padding: 18, marginBottom: 16 };
const H: React.FC<{ t: string; sub?: string }> = ({ t, sub }) => <div style={{ display: 'flex', gap: 10, alignItems: 'baseline', marginBottom: 12 }}>
  <span style={{ fontFamily: mono, fontSize: 10, letterSpacing: 2, color: C.gold }}>{t}</span>{sub && <span style={{ fontFamily: mono, fontSize: 10, color: C.cocoa }}>{sub}</span>}</div>;
const Tile: React.FC<{ k: string; v: React.ReactNode; s?: string; col?: string }> = ({ k, v, s, col }) => <div style={{ background: C.s2, borderRadius: 8, padding: '12px 14px' }}>
  <div style={{ fontFamily: mono, fontSize: 9.5, letterSpacing: 1.5, color: C.cocoa }}>{k}</div>
  <div style={{ fontFamily: mono, fontSize: 17, color: col || C.cream, marginTop: 5 }}>{v}</div>{s && <div style={{ fontFamily: mono, fontSize: 10, color: C.dust, marginTop: 3 }}>{s}</div>}</div>;

const PriceTab: React.FC<{ ticker: string; data: any; states?: React.ReactNode }> = ({ ticker, data, states }) => {
  const [ps, setPs] = useState<any>(null); const [mk, setMk] = useState<any>(null);
  useEffect(() => { setPs(null); setMk(null);
    api.get(`/api/v6/price-stats/${ticker}`).then(r => setPs(r.data)).catch(() => setPs({}));
    api.get(`/api/v6/market/${ticker}`).then(r => setMk(r.data?.data || r.data)).catch(() => setMk({})); }, [ticker]);
  const R = ps?.returns || {}; const st = mk?.regime?.regime?.current; const si = mk?.short_interest || {}; const vv = mk?.volatility || {};
  const shares = data?.market_cap && data?.price ? data.market_cap / data.price : null;
  const siPct = si.short_interest_shares && shares ? si.short_interest_shares / shares : null;
  const trend = ps ? (ps.above_50d && ps.above_200d ? 'Above both' : !ps.above_50d && !ps.above_200d ? 'Below both' : ps.above_200d ? 'Above 200-day only' : 'Above 50-day only') : '…';
  return (<div>
    <div style={card}>
      <H t="AT A GLANCE" sub={ps?.as_of ? `AS OF THE ${ps.as_of} CLOSE` : ''} />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(170px,1fr))', gap: 10 }}>
        <Tile k="TREND · 50 & 200-DAY AVG" v={trend} col={trend === 'Above both' ? C.up : trend === 'Below both' ? C.dn : C.cream} />
        {(['1m', '3m', '1y'] as const).map(k => <Tile key={k} k={`${k.toUpperCase()} VS S&P 500`} v={pc(R[k]?.vs_sp500)} col={(R[k]?.vs_sp500 ?? 0) >= 0 ? C.up : C.dn} s={`stock ${pc(R[k]?.stock)} · S&P ${pc(R[k]?.sp500)}`} />)}
        <Tile k="VOLATILITY · 12 MONTHS" v={ps?.vol_1y != null ? `${(ps.vol_1y * 100).toFixed(0)}%` : '—'} s={ps?.daily_move_typical ? `a typical day moves ~${(ps.daily_move_typical * 100).toFixed(1)}%` : ''} />
        <Tile k="CURRENT STATE" v={st ? (STATE[st] || st.replace(/_/g, ' ').toLowerCase()) : '—'} s={mk?.regime?.regime?.confidence ? `${Math.round(mk.regime.regime.confidence * 100)}% confidence · describes now, not next` : ''} />
      </div>
    </div>
    <PatternLab ticker={ticker} />
    <div style={card}>
      <H t="VOLATILITY" sub="HOW MUCH THE PRICE SWINGS" />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))', gap: 10 }}>
        <Tile k="LAST MONTH" v={ps?.vol_1m != null ? `${(ps.vol_1m * 100).toFixed(0)}%` : '—'} s="annualised" />
        <Tile k="LAST 12 MONTHS" v={ps?.vol_1y != null ? `${(ps.vol_1y * 100).toFixed(0)}%` : '—'} s="annualised" />
        <Tile k="VS ITS OWN HISTORY" v={vv.vol_percentile != null ? `${Math.round(vv.vol_percentile)}th pct` : '—'} s="higher = more turbulent than usual" />
        <Tile k="BETA · 12 MONTHS" v={ps?.beta_1y != null ? `${ps.beta_1y.toFixed(2)}×` : '—'} s="moves this much per 1% S&P move" />
        <Tile k="WORST FALL · 12 MONTHS" v={pc(ps?.max_drawdown_1y, 0)} col={C.dn} s={`since ${ps?.history_start || '2021'}: ${pc(ps?.max_drawdown_all, 0)}`} />
        <Tile k="DOWN DAYS VS UP DAYS" v={vv.up_down_vol_ratio != null ? `${vv.up_down_vol_ratio.toFixed(2)}×` : '—'} s="above 1 = falls are sharper than rises" />
      </div>
    </div>
    {si.short_interest_shares ? (<div style={card}>
      <H t="SHORT INTEREST" sub={`EXCHANGE REPORT OF ${si.short_interest_date}`} />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))', gap: 10 }}>
        <Tile k="SHARES SOLD SHORT" v={`${(si.short_interest_shares / 1e6).toFixed(1)}M`} s={siPct != null ? `${(siPct * 100).toFixed(2)}% of all shares` : ''} />
        <Tile k="DAYS TO COVER" v={si.days_to_cover != null ? si.days_to_cover.toFixed(1) : '—'} s="days of normal volume to buy back" />
        <Tile k="CHANGE SINCE LAST REPORT" v={pc(si.short_interest_trend)} col={(si.short_interest_trend ?? 0) <= 0 ? C.up : C.dn} s="fewer shorts = green" />
      </div>
      <div style={{ fontFamily: mono, fontSize: 9.5, color: C.cocoa, marginTop: 10 }}>Reported twice a month by exchanges; the figure lags by about two weeks.</div>
    </div>) : null}
    {states && <div style={card}><H t="PRICE STATES" sub="A DESCRIPTION OF THE PAST, NOT A FORECAST" />{states}</div>}
  </div>);
};
export default PriceTab;
