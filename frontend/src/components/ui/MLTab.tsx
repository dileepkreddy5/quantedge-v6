// ML models: every model and whether it works; chance of beating the sector (only once validated);
// risk at every horizon; the research model tournament; evidence; the research factor library.
import React, { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
const C = { s1: '#1c130e', s2: '#241610', b1: '#3a2920', gold: '#daa520', cocoa: '#8a7560', dust: '#9d8b7a', latte: '#d4c4b0', cream: '#f4e8d8', up: '#22c55e', dn: '#ef4444', amber: '#e0ad3a' };
const mono = "'Fira Code',monospace";
const card = { background: C.s1, border: `1px solid ${C.b1}`, borderRadius: 10, padding: 18, marginBottom: 16 };
const P = (v: any) => v == null || Number.isNaN(v) ? '—' : `${Math.round(v * 100)}%`;
const pcS = (v: number) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(0)}%`;
const f3 = (v: any) => v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(3)}`;
const H: React.FC<{ t: string; sub?: string }> = ({ t, sub }) => <div style={{ display: 'flex', gap: 10, alignItems: 'baseline', marginBottom: 12, flexWrap: 'wrap' }}>
  <span style={{ fontFamily: mono, fontSize: 10, letterSpacing: 2, color: C.gold }}>{t}</span>{sub && <span style={{ fontFamily: mono, fontSize: 10, color: C.cocoa }}>{sub}</span>}</div>;
const th = (i: number, left = 1) => ({ textAlign: (i < left ? 'left' : 'right') as any, fontWeight: 400, padding: '6px 8px', borderBottom: `1px solid ${C.b1}`, color: C.cocoa });
const td = (i: number, left = 1, col?: string) => ({ padding: '6px 8px', textAlign: (i < left ? 'left' : 'right') as any, color: col || C.latte });
const LBL: Record<string, string> = { '5': '1 week', '10': '2 weeks', '21': '1 month', '63': '3 months', '126': '6 months', '252': '1 year' };
const CONTENDERS: Record<string, [string, string]> = {
  composite: ['QuantEdge Factor Model', 'research priors + each factor\'s track record; weights shrink, never flip'],
  composite_fm: ['Factor model, fast-adapting', 'follows recent factor performance freely'],
  equal_weight: ['Textbook composite', 'every family equally, published signs'],
  lgbm: ['Gradient-boosted trees', 'Gu, Kelly & Xiu 2020'], enet: ['Elastic net', 'Gu, Kelly & Xiu 2020'],
  nn3: ['Neural net NN3 (3 seeds)', 'Gu, Kelly & Xiu 2020'], ranker: ['Learning-to-rank', 'listwise ranking objective'],
  ipca: ['IPCA', 'Kelly, Pruitt & Su 2019'], cae: ['Conditional autoencoder', 'Gu, Kelly & Xiu 2021'], ensemble: ['Ensemble', 'average ranking of the ML models'] };
const FAMILIES: [string, string, string][] = [
  ['Momentum', 'past 12 months excluding the last; nearness to the 52-week high; industry momentum', 'Jegadeesh & Titman 1993 · George & Hwang 2004 · Moskowitz & Grinblatt 1999'],
  ['Short-term reversal', 'last month\'s winners tend to give some back', 'Jegadeesh 1990'],
  ['Earnings surprise', 'profit vs the same quarter a year ago, scaled by its usual swings; sales acceleration', 'Bernard & Thomas 1989'],
  ['Profitability', 'gross profit ÷ assets; return on equity', 'Novy-Marx 2013 · Asness, Frazzini & Pedersen 2019'],
  ['Investment', 'fast asset growth tends to precede weaker returns', 'Cooper, Gulen & Schill 2008'],
  ['Issuance', 'issuing new shares tends to precede weaker returns', 'Pontiff & Woodgate 2008'],
  ['Accruals', 'profits running ahead of cash', 'Sloan 1996'],
  ['Value', 'earnings, book, sales and cash-flow yields', 'Fama & French 1992, 2015'],
  ['Size', 'smaller companies', 'Banz 1981'],
  ['Low risk', 'low company-specific volatility; low beta', 'Ang et al. 2006 · Frazzini & Pedersen 2014'],
  ['Liquidity', 'less-traded shares earn a premium', 'Amihud 2002']];

const MLTab: React.FC<{ ticker: string; data: any }> = ({ ticker, data }) => {
  const [ml, setMl] = useState<any>(null); const [hsel, setHsel] = useState('63');
  useEffect(() => { setMl(null); api.get(`/api/v6/ml/risk/${ticker}`).then(r => setMl(r.data)).catch(() => setMl({})); }, [ticker]);
  if (!ml) return <div style={{ ...card, fontFamily: mono, fontSize: 11, color: C.dust }}>loading models…</div>;
  const hz: any[] = ml.horizons || []; const EH: any = ml.evidence_h || {}; const ev: any = ml.evidence || {}; const T: any = ev.tournament; const alpha: any[] = ml.alpha || [];
  const px: number | null = data?.current_price || null; const d15 = ev.drop15_table, d25 = ev.drop25_table;
  const TH: any = T?.horizons || {}; const passes = (r: any) => r && r.t != null && r.t >= 2;
  const anyPass = Object.values(TH).some((v: any) => Object.entries(v).some(([n, r]: any) => CONTENDERS[n] && passes(r)));
  const retStatus = !T ? ['Being tested on 2012–2026', C.amber] : anyPass ? ['Passed at some horizons — being deployed', C.up] : Object.keys(TH).length < 6 ? ['Tournament running', C.amber] : ['Tested — none passes yet', C.amber];
  const lb = (TH[hsel] ? Object.entries(TH[hsel]).filter(([n, r]: any) => CONTENDERS[n] && r && r.ic != null) : []).sort((a: any, b: any) => b[1].ic - a[1].ic) as [string, any][];
  return (<div>
    <div style={card}>
      <H t="MACHINE LEARNING · EVERY MODEL AND WHETHER IT WORKS" />
      <div style={{ overflowX: 'auto' }}><table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: mono, fontSize: 12 }}>
        <thead><tr>{['Model', 'What it predicts', 'Basis', 'Status'].map((h, i) => <th key={i} style={th(i, 3)}>{h}</th>)}</tr></thead>
        <tbody>{([['Volatility model (gradient-boosted trees)', 'how bumpy the ride will be, 1 week – 6 months', 'beat “the last 3 months repeat”, walk-forward', 'Live · validated', C.up],
          ['Fall-risk tables', 'chance of a 15% or 25% fall', 'how often similar stocks fell; calibrated', 'Live · calibrated', C.up],
          ['Likely price range', 'where the price ends up 8 times in 10, 1 week – 1 year', 'fat-tailed spread around the volatility forecast', 'Live · tested', C.up],
          ['QuantEdge Factor Model + 6 research ML models', 'chance of beating its sector', 'Gu-Kelly-Xiu 2020/2021 · Kelly-Pruitt-Su 2019', retStatus[0], retStatus[1]],
          ['Per-stock XGBoost / LightGBM / LSTM', 'price direction', 'trained on one stock\'s short history', 'Retired · overfit', C.dn]] as any[]).map((r, i) => (
          <tr key={i}>{r.slice(0, 4).map((c: string, j: number) => <td key={j} style={td(j, 3, j === 3 ? r[4] : j === 0 ? C.cream : C.latte)}>{c}</td>)}</tr>))}</tbody></table></div>
    </div>
    <div style={card}>
      <H t="CHANCE OF BEATING ITS SECTOR" sub="SHOWN ONLY FOR HORIZONS WHERE A MODEL PASSED ON DATA IT NEVER SAW · NEVER A PRICE TARGET" />
      {alpha.length ? (<div style={{ overflowX: 'auto' }}><table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: mono, fontSize: 12.5 }}>
        <thead><tr>{['Horizon', 'Chance of beating its sector', 'Rank among sector peers', 'Model', 'Biggest drivers'].map((h, i) => <th key={i} style={th(i)}>{h}</th>)}</tr></thead>
        <tbody>{alpha.map((a: any) => (<tr key={a.horizon}><td style={td(0)}>{LBL[String(a.horizon)]}</td>
          <td style={td(1, 1, a.prob_beat > 0.55 ? C.up : a.prob_beat < 0.45 ? C.dn : C.cream)}>{P(a.prob_beat)}</td><td style={td(2)}>top {Math.max(1, Math.round((1 - a.pct) * 100))}%</td>
          <td style={td(3)}>{CONTENDERS[a.model]?.[0] || a.model}</td>
          <td style={td(4)}>{Object.entries(a.drivers || {}).sort((x: any, y: any) => Math.abs(y[1]) - Math.abs(x[1])).slice(0, 3).map(([k, v]: any) => `${k} ${v >= 0 ? '+' : '−'}`).join(' · ')}</td></tr>))}</tbody></table></div>)
        : <p style={{ fontSize: 14, color: C.latte, lineHeight: 1.65, margin: 0 }}>{!T ? <>The models are being tested on <b>14 years they never trained on (2012–2026)</b>. A probability appears here only for horizons where a model clearly beats the simple research composite — until then, nothing is shown rather than an unproven number.</>
          : anyPass ? <>Models passed at some horizons; per-stock probabilities are being prepared and will appear here.</> : <>No model has yet beaten its sector reliably on 2012–2026 data, so no probability is shown. A second round with insider buying, true post-earnings drift and public attention is next.</>}</p>}
    </div>
    <div style={card}>
      <H t="RISK AT EVERY HORIZON" sub={hz.length ? `FOR ${ticker} · MODELS AS OF ${hz[0].as_of} · RANGES AT TODAY'S PRICE` : ''} />
      {!hz.length ? <div style={{ fontSize: 13.5, color: C.dust }}>No forecast for {ticker} yet — it needs a year of trading and enough volume to be in the monthly panel.</div> : (<>
        <div style={{ overflowX: 'auto' }}><table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: mono, fontSize: 12.5 }}>
          <thead><tr>{['Horizon', 'Expected volatility', 'Likely range (8 in 10)', 'Chance of 15%+ fall', 'Chance of 25%+ fall', 'Range held in testing'].map((h, i) => <th key={i} style={th(i)}>{h}</th>)}</tr></thead>
          <tbody>{hz.map((r: any) => { const e = EH[String(r.horizon)] || {}; const cov = r.horizon === 252 ? (e.baseline_range80_coverage ?? e.range80_coverage) : e.range80_coverage;
            return (<tr key={r.horizon}><td style={td(0)}>{LBL[String(r.horizon)]}{r.method === 'baseline' && <span style={{ color: C.amber }} title="Uses past volatility — the model didn't beat it here"> *</span>}</td>
              <td style={td(1, 1, C.cream)}>{P(r.vol)}</td>
              <td style={td(2, 1, C.cream)}>{px ? `$${(px * r.lo_mult).toFixed(2)} – $${(px * r.hi_mult).toFixed(2)}` : `${pcS(r.lo_mult - 1)} to ${pcS(r.hi_mult - 1)}`}{px && <span style={{ color: C.dust, fontSize: 11 }}> ({pcS(r.lo_mult - 1)} / {pcS(r.hi_mult - 1)})</span>}</td>
              <td style={td(3)}>{P(r.drop15)}</td><td style={td(4, 1, r.drop25 > 0.35 ? C.dn : r.drop25 < 0.1 ? C.up : C.cream)}>{P(r.drop25)}</td><td style={td(5, 1, C.dust)}>{P(cov)}</td></tr>); })}</tbody></table></div>
        <div style={{ fontSize: 12.5, color: C.dust, marginTop: 12, lineHeight: 1.6 }}>These forecast <b style={{ color: C.latte }}>how bumpy</b> the ride is likely to be — not which way the price goes. Each range is where the price ended up 8 times in 10 for similar stocks; the slight upward lean is the market's long-run drift. A “15%+ fall” is from any high to a later low within the period. <span style={{ color: C.amber }}>*</span> 1 year uses past volatility (the model didn't beat it there); its fall chances rest on ~3 years of one-year windows that include the April 2025 crash, so treat them as rough.</div>
      </>)}
    </div>
    <div style={card}>
      <H t="THE MODEL TOURNAMENT" sub={T ? `TESTED ${T.test_from} → ${T.last} · ${T.rows?.toLocaleString()} STOCK-MONTHS · RETRAINED EVERY ${T.retrain_every_years} YEARS ON THE PAST ONLY` : 'RUNNING'} />
      {!T ? <div style={{ fontSize: 13.5, color: C.dust }}>Ten contenders are being tested on 25 years of data — results appear here horizon by horizon as they finish.</div> : (<>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>{Object.keys(LBL).map(h => (<button key={h} onClick={() => setHsel(h)} disabled={!TH[h]}
          style={{ fontFamily: mono, fontSize: 11, padding: '5px 10px', borderRadius: 6, cursor: TH[h] ? 'pointer' : 'default', border: `1px solid ${hsel === h ? C.gold : C.b1}`, background: hsel === h ? '#2c1d14' : 'transparent', color: TH[h] ? (hsel === h ? C.gold : C.latte) : C.cocoa }}>{LBL[h]}{!TH[h] ? ' …' : ''}</button>))}</div>
        {lb.length === 0 ? <div style={{ fontSize: 13, color: C.dust }}>This horizon hasn't finished yet.</div> : (<div style={{ overflowX: 'auto' }}><table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: mono, fontSize: 12 }}>
          <thead><tr>{['Contender', 'Ranking skill', 't-stat', 'Right', 'Top − bottom 10% / yr', 'Sharpe', 'Large / mid / small'].map((h, i) => <th key={i} style={th(i)}>{h}</th>)}</tr></thead>
          <tbody>{lb.map(([n, r]) => (<tr key={n} title={CONTENDERS[n][1]}>
            <td style={td(0, 1, passes(r) ? C.up : C.cream)}>{CONTENDERS[n][0]}<div style={{ fontSize: 10.5, color: C.cocoa }}>{CONTENDERS[n][1]}</div></td>
            <td style={td(1, 1, C.cream)}>{f3(r.ic)}</td><td style={td(2, 1, passes(r) ? C.up : C.amber)}>{r.t == null ? `n/a (${r.independent_windows} periods)` : r.t.toFixed(2)}</td>
            <td style={td(3)}>{P(r.hit)}</td><td style={td(4, 1, (r.ls_annual ?? 0) >= 0 ? C.up : C.dn)}>{r.ls_annual == null ? '—' : pcS(r.ls_annual)}</td>
            <td style={td(5)}>{r.ls_sharpe == null ? '—' : r.ls_sharpe.toFixed(2)}</td>
            <td style={td(6, 1, C.dust)}>{['large', 'mid', 'small'].map(s => r.ic_by_size?.[s] == null ? '—' : f3(r.ic_by_size[s])).join(' / ')}</td></tr>))}</tbody></table></div>)}
        <div style={{ fontSize: 12, color: C.cocoa, marginTop: 10, lineHeight: 1.55 }}>Ranking skill: how well each month's ranking lined up with what happened (0 = none; 0.03–0.06 is good). A model passes only with t ≥ 2 across enough independent periods. Top − bottom 10%: the yearly return gap between its best- and worst-ranked stocks, relative to their sectors, before costs. Results by size show whether it only works among small companies, as the research often finds.</div>
      </>)}
    </div>
    {Object.keys(EH).length > 0 && <div style={card}>
      <H t="WHY YOU CAN TRUST THE RISK FORECASTS" sub="WALK-FORWARD: TRAINED ON THE PAST, TESTED ON THE FOLLOWING 6 MONTHS, FIVE TIMES (2024–2026)" />
      <div style={{ overflowX: 'auto' }}><table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: mono, fontSize: 12 }}>
        <thead><tr>{['Horizon', 'Volatility ranking: model / baseline', 'Typical error: model / baseline', 'Range held (aim 80%)', '25%+ fall: separation', 'Used'].map((h, i) => <th key={i} style={th(i)}>{h}</th>)}</tr></thead>
        <tbody>{Object.entries(EH).filter(([, e]: any) => e && e.available).map(([h, e]: any) => (<tr key={h}><td style={td(0)}>{e.label}</td>
          <td style={td(1)}><span style={{ color: e.vol.model_ic > e.vol.baseline_ic ? C.up : C.dust }}>{e.vol.model_ic.toFixed(2)}</span> / {e.vol.baseline_ic.toFixed(2)}</td>
          <td style={td(2)}><span style={{ color: e.vol.model_error < e.vol.baseline_error ? C.up : C.dust }}>{P(e.vol.model_error)}</span> / {P(e.vol.baseline_error)}</td>
          <td style={td(3)}>{P(h === '252' ? (e.baseline_range80_coverage ?? e.range80_coverage) : e.range80_coverage)}</td>
          <td style={td(4)}>{e.drop?.['25']?.auc?.toFixed(2) ?? '—'}</td><td style={td(5, 1, e.passes ? C.up : C.amber)}>{e.passes ? 'model' : 'baseline'}</td></tr>))}</tbody></table></div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(300px,1fr))', gap: 14, marginTop: 14 }}>
        {[[d15, '15%+ fall'], [d25, '25%+ fall']].map(([t, l]: any) => t && (<div key={l} style={{ background: C.s2, borderRadius: 8, padding: '12px 14px' }}>
          <div style={{ fontFamily: mono, fontSize: 9.5, letterSpacing: 1.4, color: C.cocoa }}>3-MONTH CHANCE OF A {l.toUpperCase()} · CALIBRATION</div>
          <div style={{ fontSize: 12, color: C.dust, margin: '6px 0 8px' }}>separates fallers from non-fallers: {t.auc?.toFixed(2)} (0.5 = coin flip)</div>
          {t.calibration.map((c: any, i: number) => (<div key={i} style={{ display: 'grid', gridTemplateColumns: '90px 1fr 60px', gap: 8, alignItems: 'center', fontFamily: mono, fontSize: 11, padding: '2px 0' }}>
            <span style={{ color: C.dust }}>said {P(c.predicted)}</span>
            <span style={{ position: 'relative', height: 8, background: '#140d0a', borderRadius: 4 }}><i style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${c.actual * 100}%`, background: C.gold, borderRadius: 4 }} /><i style={{ position: 'absolute', top: -2, bottom: -2, width: 2, left: `${c.predicted * 100}%`, background: C.cream }} /></span>
            <span style={{ color: C.cream, textAlign: 'right' }}>{P(c.actual)} fell</span></div>))}
          <div style={{ fontSize: 11, color: C.cocoa, marginTop: 6 }}>Gold = what happened · white tick = what was predicted.</div></div>))}
      </div>
    </div>}
    <div style={card}>
      <H t="WHAT DRIVES STOCK RETURNS — ACCORDING TO RESEARCH" sub="THE 11 FACTOR FAMILIES THE MODELS LEARN FROM · EACH RANKED WITHIN ITS SECTOR, POINT-IN-TIME" />
      <div style={{ overflowX: 'auto' }}><table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <tbody>{FAMILIES.map(([n, d, r]) => (<tr key={n}><td style={{ padding: '7px 8px', color: C.cream, fontFamily: mono, fontSize: 12, verticalAlign: 'top', width: 150 }}>{n}</td>
          <td style={{ padding: '7px 8px', color: C.latte }}>{d}<div style={{ fontFamily: mono, fontSize: 10.5, color: C.cocoa, marginTop: 2 }}>{r}</div></td></tr>))}</tbody></table></div>
      <div style={{ fontSize: 12.5, color: C.dust, marginTop: 12, lineHeight: 1.6 }}><b style={{ color: C.latte }}>Coming next:</b> opportunistic insider buying (Cohen, Malloy & Pomorski 2012), true post-earnings drift from the actual results day (Bernard & Thomas 1989), public attention from Wikipedia views (Da, Engelberg & Gao 2011), and the typical results-day move. Published effects weaken after publication (McLean & Pontiff 2016), which is why every weight is re-learned from data the model could have known at the time.</div>
    </div>
  </div>);
};
export default MLTab;
