// Models & risk: risk at a glance (verified sources, loss-maker caveats), the QuantEdge score explained
// (weights, scores, audit status), forward indicators, and honest model validation.
import React, { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
const C = { s1: '#1c130e', s2: '#241610', b1: '#3a2920', gold: '#daa520', cocoa: '#8a7560', dust: '#9d8b7a', latte: '#d4c4b0', cream: '#f4e8d8', up: '#22c55e', dn: '#ef4444', amber: '#e0ad3a' };
const mono = "'Fira Code',monospace";
const card = { background: C.s1, border: `1px solid ${C.b1}`, borderRadius: 10, padding: 18, marginBottom: 16 };
const pc = (v: any, d = 0) => v == null ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(d)}%`;
const H: React.FC<{ t: string; sub?: string }> = ({ t, sub }) => <div style={{ display: 'flex', gap: 10, alignItems: 'baseline', marginBottom: 12, flexWrap: 'wrap' }}>
  <span style={{ fontFamily: mono, fontSize: 10, letterSpacing: 2, color: C.gold }}>{t}</span>{sub && <span style={{ fontFamily: mono, fontSize: 10, color: C.cocoa }}>{sub}</span>}</div>;
const Tile: React.FC<{ k: string; v: React.ReactNode; s: React.ReactNode; col?: string }> = ({ k, v, s, col }) => (
  <div style={{ background: C.s2, borderRadius: 8, padding: '12px 14px' }}><div style={{ fontFamily: mono, fontSize: 9.5, letterSpacing: 1.4, color: C.cocoa }}>{k}</div>
    <div style={{ fontFamily: mono, fontSize: 17, color: col || C.cream, marginTop: 5 }}>{v}</div><div style={{ fontSize: 12, color: C.dust, marginTop: 4, lineHeight: 1.45 }}>{s}</div></div>);
const AUDIT: Record<string, [string, string]> = {
  financial: ['checked', 'inputs verified against SEC filings'], business: ['checked', 'margin windows, consistency, ROIC fixed'],
  valuation: ['checked', 'outlier methods excluded'], management: ['checked', '1-year share count, same-window cash flow'],
  industry: ['checked', 'full price range; beta shown with correlation'], risk: ['checked', 'consistent with price statistics'],
  market: ['checked', 'price statistics verified'], ownership: ['checked', '1-year share count, 12-month buybacks'],
  institutional: ['checked', 'labels corrected'], news: ['checked', 'only articles about the company'],
  forecast: ['checked', 'growth now year-over-year'], peers: ['checked', 'percentile among real peers'],
  competitive: ['not yet audited', 'no contradictions found'], macro: ['not yet audited', 'no contradictions found'],
  alt_data: ['weight 0', 'duplicated Money flow and News inputs'], ml_models: ['not scored', 'no model horizon passes validation'] };

const ModelsRiskTab: React.FC<{ ticker: string; data: any; forward?: React.ReactNode; riskModel?: React.ReactNode }> = ({ ticker, data, forward, riskModel }) => {
  const [rk, setRk] = useState<any>(null); const [ps, setPs] = useState<any>(null); const [sm, setSm] = useState<any>(null);
  const [cv, setCv] = useState<any>(null); const [mv, setMv] = useState<any>(null); const [showRaw, setShowRaw] = useState(false); const [showRisk, setShowRisk] = useState(false);
  useEffect(() => { setRk(null); setPs(null); setSm(null); setCv(null);
    api.get(`/api/v6/risk/${ticker}`).then(r => setRk(r.data?.data || r.data)).catch(() => setRk({}));
    api.get(`/api/v6/price-stats/${ticker}`).then(r => setPs(r.data)).catch(() => setPs({}));
    api.get(`/api/v6/summary/${ticker}`).then(r => setSm(r.data)).catch(() => setSm({}));
    api.get(`/api/v7/conviction/${ticker}`).then(r => setCv(r.data?.data || r.data)).catch(() => setCv({}));
    api.get(`/api/v6/model-validation`).then(r => setMv(r.data)).catch(() => setMv({})); }, [ticker]);
  const k = rk?.key_metrics || {}; const loss = sm?.facts?.ni_ttm != null && sm.facts.ni_ttm <= 0;
  const az = k.altman_z, cr = k.current_ratio, nde = k.net_debt_to_ebitda;
  const vol = ps?.vol_1y; const size = vol ? Math.min(1, 0.10 / vol) : null; const r1 = ps?.returns?.['1y'] || {};
  const mods = [...(cv?.modules || [])].sort((a: any, b: any) => (b.weight || 0) - (a.weight || 0));
  const tw = mods.reduce((a: number, m: any) => a + (m.weight || 0), 0) || 1;
  const hz = Object.values(mv?.horizons || {}) as any[];
  return (<div>
    <div style={card}>
      <H t="RISK AT A GLANCE" sub={ps?.as_of ? `PRICES TO ${ps.as_of} · SEC FILINGS` : ''} />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(220px,1fr))', gap: 10 }}>
        <Tile k="BANKRUPTCY RISK (ALTMAN Z)" v={az != null ? az.toFixed(1) : '—'} col={az == null ? undefined : az > 3 ? C.up : az < 1.8 ? C.dn : C.amber}
          s={<>{az == null ? 'not meaningful for banks and insurers' : az > 3 ? 'safe zone (above 3)' : az < 1.8 ? 'distress zone (below 1.8)' : 'grey zone (1.8–3)'}{loss && cr > 1.5 ? <em style={{ color: C.amber, fontStyle: 'normal' }}> · the formula penalises losses and under-weights cash, so it overstates risk for cash-rich growth companies</em> : null}</>} />
        <Tile k="BALANCE SHEET" v={cr != null ? `${cr.toFixed(2)}× current ratio` : '—'} s={<>current assets ÷ current liabilities{nde != null ? <> · net debt / EBITDA {loss ? <b>n/m (loss-making)</b> : nde.toFixed(2)}</> : null}</>} />
        <Tile k="VOLATILITY · 12 MONTHS" v={vol != null ? `${(vol * 100).toFixed(0)}%` : '—'} s={<>beta {ps?.beta_1y?.toFixed(2) ?? '—'} (correlation {ps?.corr_1y?.toFixed(2) ?? '—'}){ps?.corr_1y != null && ps.corr_1y < 0.3 ? ' — moves largely on its own' : ''}</>} />
        <Tile k="WORST FALL FROM A PEAK" v={pc(ps?.max_drawdown_1y)} col={C.dn} s={`last 12 months · since ${ps?.history_start?.slice(0, 4) || '2021'}: ${pc(ps?.max_drawdown_all)}`} />
        <Tile k="SHARE COUNT · 1 YEAR" v={pc(k.share_dilution, 1)} col={k.share_dilution == null ? undefined : k.share_dilution <= 0 ? C.up : C.amber} s={k.share_dilution == null ? '' : k.share_dilution <= 0 ? 'shrinking (buybacks)' : 'growing (dilution)'} />
        <Tile k="RETURN · 12 MONTHS" v={pc(r1.stock)} col={(r1.stock ?? 0) >= 0 ? C.up : C.dn} s={`S&P 500 ${pc(r1.sp500)} · difference ${pc(r1.vs_sp500)}`} />
      </div>
      {size != null && <div style={{ fontSize: 13, color: C.dust, marginTop: 12, lineHeight: 1.6 }}><b style={{ color: C.latte }}>Sizing rule of thumb:</b> at {(vol * 100).toFixed(0)}% yearly volatility, a position sized to 10% volatility would be about <b style={{ color: C.cream }}>{Math.round(size * 100)}%</b> of a normal one (10% ÷ its volatility). A way to compare riskiness — not a recommendation.</div>}
    </div>
    <div style={card}>
      <H t={`THE QUANTEDGE SCORE · ${cv?.conviction_score ?? '—'} / 100`} sub="NOT YET VALIDATED · RECORDED EVERY TIME IT'S CALCULATED, TO MEASURE WHETHER HIGH SCORES BEAT LOW ONES" />
      <div style={{ overflowX: 'auto' }}><table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: mono, fontSize: 12 }}>
        <thead><tr style={{ color: C.cocoa }}>{['Module', 'Share of score', 'Score', '', 'Audit status'].map((h, i) => <th key={i} style={{ textAlign: i === 0 || i === 4 ? 'left' : 'right', fontWeight: 400, padding: '6px 8px', borderBottom: `1px solid ${C.b1}` }}>{h}</th>)}</tr></thead>
        <tbody>{mods.map((m: any) => { const a = AUDIT[m.id] || ['—', '']; const ac = a[0] === 'checked' ? C.up : a[0] === 'not yet audited' ? C.dust : C.amber;
          return (<tr key={m.id} style={{ opacity: m.score == null || !m.weight ? 0.6 : 1 }}>
            <td style={{ padding: '6px 8px', color: C.latte }}>{(m.label || m.id).replace(' Intelligence', '')}</td>
            <td style={{ padding: '6px 8px', textAlign: 'right', color: C.dust }}>{((m.weight || 0) / tw * 100).toFixed(0)}%</td>
            <td style={{ padding: '6px 8px', textAlign: 'right', color: C.cream }}>{m.score == null ? '—' : Math.round(m.score)}</td>
            <td style={{ padding: '6px 8px', width: 110 }}><div style={{ height: 6, background: '#140d0a', borderRadius: 3 }}><div style={{ height: 6, width: `${m.score ?? 0}%`, background: C.dust, borderRadius: 3 }} /></div></td>
            <td style={{ padding: '6px 8px', color: ac }}>{a[0]} <span style={{ color: C.cocoa }}>· {a[1]}</span></td></tr>); })}</tbody></table></div>
    </div>
    {forward && <div style={card}><H t="FORWARD INDICATORS" sub="GROWTH, REINVESTMENT AND MOMENTUM OF THE BUSINESS — NOT A PRICE FORECAST" />{forward}</div>}
    <div style={card}>
      <H t="MODEL VALIDATION" sub={mv?.split_date ? `TRAINED ON ${mv.n_tickers} STOCKS · TESTED ONLY ON DATA AFTER ${String(mv.split_date).slice(0, 10)}` : ''} />
      <p style={{ fontSize: 14, color: C.latte, lineHeight: 1.65, margin: '0 0 12px' }}>The models rank stocks against each other. A <b>rank correlation</b> above 0 means their rankings lined up with what actually happened; in practice 0.03–0.06 is good. To count as <b>reliable</b>, the result must be statistically solid (t-statistic of 2 or more) on data the models never saw. <b style={{ color: hz.some(h => h.reliable) ? C.up : C.amber }}>{hz.some(h => h.reliable) ? 'Reliable horizons are used on the site.' : 'No horizon currently passes, so no model predictions are used anywhere on the site.'}</b></p>
      <div style={{ overflowX: 'auto' }}><table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: mono, fontSize: 12 }}>
        <thead><tr style={{ color: C.cocoa }}>{['Horizon', 'Rank correlation', 't-stat', 'Right direction', 'Independent windows', 'Tested rows', 'Reliable'].map((h, i) => <th key={i} style={{ textAlign: i ? 'right' : 'left', fontWeight: 400, padding: '6px 8px', borderBottom: `1px solid ${C.b1}` }}>{h}</th>)}</tr></thead>
        <tbody>{hz.map((h: any, i: number) => (<tr key={i} title={h.confidence_note}>
          <td style={{ padding: '6px 8px', color: C.latte }}>{h.horizon_label}</td>
          <td style={{ padding: '6px 8px', textAlign: 'right', color: C.cream }}>{h.oos_rank_ic?.ensemble != null ? (h.oos_rank_ic.ensemble >= 0 ? '+' : '') + h.oos_rank_ic.ensemble.toFixed(3) : '—'}</td>
          <td style={{ padding: '6px 8px', textAlign: 'right' }}>{h.ic_t_stat != null ? h.ic_t_stat.toFixed(2) : '—'}</td>
          <td style={{ padding: '6px 8px', textAlign: 'right' }}>{h.ic_hit_rate != null ? `${Math.round(h.ic_hit_rate * 100)}%` : '—'}</td>
          <td style={{ padding: '6px 8px', textAlign: 'right' }}>{h.n_independent_val_dates ?? '—'}</td>
          <td style={{ padding: '6px 8px', textAlign: 'right', color: C.dust }}>{h.n_val?.toLocaleString() ?? '—'}</td>
          <td style={{ padding: '6px 8px', textAlign: 'right', color: h.reliable ? C.up : C.amber }}>{h.reliable ? 'yes' : 'no'}</td></tr>))}</tbody></table></div>
      <div style={{ fontSize: 12, color: C.cocoa, marginTop: 10 }}>Hover a row for the trainer's own note. A rebuild with stronger, verified inputs and stricter validation is planned.</div>
      <div onClick={() => setShowRaw(v => !v)} style={{ fontFamily: mono, fontSize: 11, color: C.dust, marginTop: 12, cursor: 'pointer' }}>{showRaw ? '▾' : '▸'} per-stock experimental outputs (not used)</div>
      {showRaw && <div style={{ fontSize: 12.5, color: C.dust, marginTop: 8, lineHeight: 1.6 }}>These models are trained on {ticker}'s own price history alone, at the moment the page opens — too few data points to learn from, which is why their training fit looks near-perfect while they disagree with each other.
        <pre style={{ fontFamily: mono, fontSize: 11, color: C.cocoa, whiteSpace: 'pre-wrap', marginTop: 6 }}>{JSON.stringify(data?.ml_predictions || {}, null, 1).slice(0, 1500)}</pre></div>}
    </div>
    {riskModel && <div style={card}>
      <div onClick={() => setShowRisk(v => !v)} style={{ display: 'flex', gap: 10, alignItems: 'baseline', cursor: 'pointer' }}>
        <span style={{ fontFamily: mono, fontSize: 10, letterSpacing: 2, color: C.gold }}>FULL RISK MODEL</span><span style={{ fontSize: 12.5, color: C.dust }}>the signals behind the Risk part of the score</span>
        <span style={{ marginLeft: 'auto', fontFamily: mono, color: C.dust }}>{showRisk ? '▾' : '▸'}</span></div>
      {showRisk && <div style={{ marginTop: 12 }}>{riskModel}</div>}
    </div>}
  </div>);
};
export default ModelsRiskTab;
