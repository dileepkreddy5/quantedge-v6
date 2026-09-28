// Trackers — the four nightly lists in one place. Plain names, one layout,
// "why it's here" on every row, and an honest track record from the cohort store.
import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { api } from '../auth/authStore';
import Screener from './Screener';

const TABS = [
  { id: 'fast-growers', name: 'Fast growers', q: 'Which companies are growing fastest, with quality?', board: 'multibagger' },
  { id: 'comebacks', name: 'Comebacks', q: 'Which solid companies fell hard — and stopped falling?', board: 'rebound' },
  { id: 'climbers', name: 'Steady climbers', q: 'Which companies are rising, with volume behind it?', board: 'ascent' },
  { id: 'filter', name: 'Your filter', q: 'Your own rules, over everything we compute.', board: '' },
];
const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;1,9..144,300&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
.qt{--ink:#0d0806;--ink2:#130c09;--panel:#1a110d;--panel2:#21160f;--line:#33241b;--line2:#46321f;--gold:#e0ad3a;--cream:#f6ecdd;--latte:#d9c9b4;--dust:#b09c86;--mute:#8a7762;--up:#3ec27a;--down:#ef7d5a;
 --serif:'Fraunces',Georgia,serif;--sans:'IBM Plex Sans',system-ui,sans-serif;--mono:'IBM Plex Mono',ui-monospace,monospace;background:var(--ink);color:var(--latte);font-family:var(--sans);min-height:100vh;-webkit-font-smoothing:antialiased}
.qt *{box-sizing:border-box}.qt a{cursor:pointer;color:inherit;text-decoration:none}.qt button{font:inherit;cursor:pointer}
.qt .wrap{max-width:1240px;margin:0 auto;padding:0 36px}
.qt nav{border-bottom:1px solid var(--line)}.qt nav .wrap{display:flex;align-items:center;gap:30px;height:66px}
.qt .logo{font-family:var(--mono);font-weight:600;letter-spacing:.4em;color:var(--gold);font-size:15px}
.qt nav .links{margin-left:auto;display:flex;gap:26px;font-size:14.5px;color:var(--dust)}.qt nav .links a:hover,.qt nav .links a.on{color:var(--cream)}
.qt .eyebrow{font-family:var(--mono);font-size:11.5px;letter-spacing:.26em;text-transform:uppercase;color:var(--gold)}
.qt .ihead{padding:48px 0 22px}.qt h1{font-family:var(--serif);font-weight:300;font-size:48px;color:var(--cream);margin:12px 0 10px;letter-spacing:-.015em}
.qt .ihead p{font-size:17px;color:var(--dust);margin:0;max-width:700px;line-height:1.6}
.qt .switch{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:28px}
.qt .sw{text-align:left;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;color:var(--latte)}
.qt .sw .n{font-family:var(--serif);font-size:20px;color:var(--cream)}.qt .sw .d{font-size:13px;color:var(--dust);margin-top:6px;line-height:1.5}
.qt .sw.on{border-color:var(--gold);background:var(--panel2)}.qt .sw:hover:not(.on){border-color:var(--line2)}
.qt .intro{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:18px;margin:26px 0 18px}
.qt .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px 24px}
.qt .card h4{font-family:var(--mono);font-size:10.5px;letter-spacing:.22em;color:var(--mute);margin:0 0 12px;font-weight:500}
.qt .card p{margin:0;font-size:14.5px;line-height:1.7;color:var(--dust)}.qt .card p b{color:var(--latte);font-weight:500}
.qt .big{font-family:var(--mono);font-size:22px;color:var(--cream);margin-bottom:8px}
.qt .pending{display:inline-block;font-family:var(--mono);font-size:10.5px;letter-spacing:.14em;color:var(--gold);border:1px solid var(--line2);border-radius:4px;padding:3px 8px;margin-bottom:10px}
.qt .bar{display:flex;align-items:center;gap:10px;margin:18px 0 12px;flex-wrap:wrap}
.qt .bar .lbl{font-family:var(--mono);font-size:11px;letter-spacing:.18em;color:var(--mute);margin-right:4px}
.qt .pill{background:none;border:1px solid var(--line);border-radius:999px;padding:7px 14px;font-family:var(--mono);font-size:12px;color:var(--dust)}
.qt .pill.on{border-color:var(--gold);color:var(--gold)}.qt .bar .meta{margin-left:auto;font-family:var(--mono);font-size:11.5px;color:var(--mute)}
.qt .tbl{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:14px}
.qt table{width:100%;border-collapse:collapse;min-width:820px}
.qt th{text-align:left;font-family:var(--mono);font-size:10.5px;letter-spacing:.14em;color:var(--mute);font-weight:500;padding:14px 16px;border-bottom:1px solid var(--line);background:var(--ink2)}
.qt td{padding:13px 16px;border-bottom:1px solid rgba(51,36,27,.6);font-size:14px;color:var(--latte);vertical-align:top}
.qt tbody tr:hover td{background:var(--panel2);cursor:pointer}.qt tr:last-child td{border-bottom:none}
.qt .tk{font-family:var(--mono);font-weight:500;color:var(--gold)}.qt .nm{display:block;font-size:12px;color:var(--mute);margin-top:2px}
.qt td.num{font-family:var(--mono);white-space:nowrap}.qt .why{font-size:13px;color:var(--dust);line-height:1.5}
.qt .stage{font-family:var(--mono);font-size:10.5px;letter-spacing:.12em;border:1px solid var(--line2);border-radius:4px;padding:3px 7px}
.qt .up{color:var(--up)}.qt .dn{color:var(--down)}
.qt .foot{font-size:13px;color:var(--mute);line-height:1.7;margin:16px 0 80px;max-width:900px}
@media (max-width:1000px){.qt .switch,.qt .intro{grid-template-columns:1fr 1fr}.qt nav .links{display:none}.qt h1{font-size:36px}}
@media (max-width:640px){.qt .switch,.qt .intro{grid-template-columns:1fr}.qt .wrap{padding:0 18px}}
`;
const P = (v: any) => v == null || isNaN(v) ? null : (Math.abs(v) <= 3 ? v * 100 : v);      // fraction or percent → percent
const fp = (v: any, d = 0) => v == null ? '—' : `${v >= 0 ? '+' : ''}${Number(v).toFixed(d)}%`;
const tierOf = (mc: any) => mc == null ? '' : mc >= 1e10 ? 'large' : mc >= 2e9 ? 'mid' : 'small';
const nice = (n: string) => (n || '').replace(/ Class A Common Stock| Common Stock/g, '');

const Trackers: React.FC = () => {
  const nav = useNavigate(); const { tab } = useParams();
  const cur = TABS.find(t => t.id === tab) || TABS[0];
  useEffect(() => {
    if (!tab) { let last = 'fast-growers'; try { last = localStorage.getItem('qe:tracker') || last; } catch {} nav(`/trackers/${last}`, { replace: true }); return; }
    if (!TABS.find(t => t.id === tab)) { nav('/trackers/fast-growers', { replace: true }); return; }
    try { localStorage.setItem('qe:tracker', tab); } catch {}
  }, [tab, nav]);
  const [mb, setMb] = useState<any>(null); const [rb, setRb] = useState<any>(null); const [as_, setAs] = useState<any>(null); const [tr, setTr] = useState<any>(null);
  const [f, setF] = useState('all');
  useEffect(() => { setF('all'); }, [tab]);
  useEffect(() => {
    const get = async (u: string, s: (v: any) => void) => { try { s((await api.get(u)).data); } catch { s({ error: true }); } };
    get('/api/v6/scan/tiers', setMb); get('/api/v6/rebound/list', setRb); get('/api/v6/ascent/top/100', setAs); get('/api/v6/boards/track_record', setTr);
  }, []);
  const go = (t: string) => nav(`/dashboard?ticker=${t}`);

  const rows = useMemo(() => {
    if (cur.id === 'fast-growers' && mb?.tiers) {
      const all: any[] = []; Object.entries(mb.tiers).forEach(([t, rs]: any) => rs.forEach((r: any) => all.push({ ...r, tier: t })));
      return all.filter(r => f === 'all' || r.tier === f).sort((a, b) => (b.score ?? 0) - (a.score ?? 0)).map(r => {
        const g = P(r.qtr_yoy_growth); const why: string[] = [];
        if (g != null) why.push(`Sales ${fp(g)} vs a year ago`);
        if (r.piotroski != null) why.push(`quality ${r.piotroski}/9`);
        if (r.margin_trend > 0) why.push('margins widening');
        if (r.accruals != null && r.accruals < 0) why.push('earnings backed by cash');
        if (r.debt_trend > 0.02) why.push('debt rising');
        return { ...r, g, m6: P(r.price_move_6mo), why: why.join(' · ') };
      });
    }
    if (cur.id === 'comebacks' && rb?.tiers) {
      const all: any[] = []; Object.entries(rb.tiers).forEach(([t, rs]: any) => rs.forEach((r: any) => all.push({ ...r, tier: r.tier || t })));
      return all.filter(r => f === 'all' || String(r.stage || '').toLowerCase().startsWith(f)).sort((a, b) => (b.score ?? 0) - (a.score ?? 0)).map(r => {
        const dd = P(r.drawdown); const off = P(r.off_trough_pct);
        const why = r.thesis || `${fp(dd)} from its ${r.prior_high_date || ''} high; ${r.days_since_low ?? '—'} days since the low`;
        return { ...r, dd: dd != null ? -Math.abs(dd) : null, off, why };
      });
    }
    if (cur.id === 'climbers' && as_?.rows) {
      return as_.rows.filter((r: any) => f === 'all' || (f === 'new' ? r.is_new : (r.first_seen && (Date.now() - new Date(r.first_seen).getTime()) / 864e5 >= 56)))
        .map((r: any) => {
          const why: string[] = [];
          if (r.is_new) why.push('new to the list');
          else if (r.first_seen) why.push(`on the list since ${String(r.first_seen).slice(0, 10)}`);
          if (r.delta_1m != null) why.push(r.delta_1m >= 0 ? `strength up ${r.delta_1m} this month` : `strength down ${Math.abs(r.delta_1m)} this month`);
          if (Array.isArray(r.flags) && r.flags.length) why.push(r.flags.slice(0, 2).join(', ').replace(/_/g, ' ').toLowerCase());
          return { ...r, tier: tierOf(r.market_cap), why: why.join(' · ') };
        });
    }
    return [];
  }, [cur.id, mb, rb, as_, f]);

  const trk = cur.board ? tr?.[cur.board] : null; const h21 = trk?.horizons?.['21d'];
  const firstMonth = trk?.first_cohort ? new Date(new Date(trk.first_cohort).getTime() + 31 * 864e5).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : null;
  const updated = cur.id === 'fast-growers' ? mb?.generated : cur.id === 'comebacks' ? rb?.generated : as_?.scan_time;
  const loaded = cur.id === 'fast-growers' ? mb : cur.id === 'comebacks' ? rb : as_;

  const Track = (<div className="card">
    <h4>TRACK RECORD</h4>
    {h21 ? <><div className="big">{h21.beat_universe_pct}% beat the market</div>
      <p>Past lists averaged <b className={h21.mean_pct >= h21.universe_pct ? 'up' : 'dn'}>{fp(h21.mean_pct, 1)}</b> a month later, vs {fp(h21.universe_pct, 1)} for the average US stock ({h21.n} picks).</p></>
      : <><span className="pending">MEASURING SINCE {trk?.first_cohort ? new Date(trk.first_cohort).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }).toUpperCase() : 'TONIGHT'}</span>
        <p>Every night's list is saved and checked later against what actually happened. {firstMonth ? <>First 1-month results arrive around <b>{firstMonth}</b>.</> : null} Until then, no performance claim is made.</p></>}
  </div>);

  return (<div className="qt"><style>{CSS}</style>
    <nav><div className="wrap"><a className="logo" onClick={() => nav('/')}>QUANTEDGE</a>
      <div className="links"><a onClick={() => nav('/')}>Markets</a><a className="on">Trackers</a><a onClick={() => nav('/methodology')}>How it works</a></div></div></nav>
    <div className="wrap">
      <div className="ihead"><div className="eyebrow">Trackers</div><h1>Find companies worth a closer look.</h1>
        <p>Rebuilt every night from every US company. Each tracker answers one question, says why each company is on it, and shows how its past picks actually did.</p>
        <div className="switch">{TABS.map(t => (<button key={t.id} className={`sw ${t.id === cur.id ? 'on' : ''}`} onClick={() => nav(`/trackers/${t.id}`)}>
          <div className="n">{t.name}</div><div className="d">{t.q}</div></button>))}</div></div>

      {cur.id === 'filter' ? <div style={{ margin: '26px 0 80px' }}><Screener /></div> : (<>
        <div className="intro">
          {cur.id === 'fast-growers' && <div className="card"><h4>HOW THIS TRACKER IS BUILT</h4><p>Every US company's latest quarterly results, straight from its SEC filings. Ranked mostly by <b>how fast sales are growing</b>, then checked for <b>quality</b>: healthy finances, widening margins, earnings backed by real cash, debt not piling up. Companies too small to trade easily are left out.</p></div>}
          {cur.id === 'comebacks' && <div className="card"><h4>HOW THIS TRACKER IS BUILT</h4><p>Solid companies trading <b>far below</b> their recent high that have <b>stopped making new lows</b>. Each shows its stage: <b>basing</b> (going sideways), <b>turning</b> (first higher lows), or <b>recovering</b>. For context: of past stocks 35–50% below their high, <b>about 14 in 100</b> got all the way back within a year. Most comebacks are partial.</p></div>}
          {cur.id === 'climbers' && <div className="card"><h4>HOW THIS TRACKER IS BUILT</h4><p>Companies whose price has been <b>beating the market steadily</b>, with <b>volume behind it</b> — more trading on up days than down days — and that are growing into larger size tiers.</p></div>}
          {Track}
        </div>
        <div className="bar">
          {cur.id === 'fast-growers' && <><span className="lbl">SIZE</span>{[['all', 'All'], ['large', 'Large · over $10B'], ['mid', 'Mid · $2–10B'], ['small', 'Small · under $2B']].map(([k, l]) => <button key={k} className={`pill ${f === k ? 'on' : ''}`} onClick={() => setF(k)}>{l}</button>)}</>}
          {cur.id === 'comebacks' && <><span className="lbl">STAGE</span>{[['all', 'All'], ['bas', 'Basing'], ['turn', 'Turning'], ['recov', 'Recovering']].map(([k, l]) => <button key={k} className={`pill ${f === k ? 'on' : ''}`} onClick={() => setF(k)}>{l}</button>)}</>}
          {cur.id === 'climbers' && <><span className="lbl">SHOW</span>{[['all', 'All'], ['new', 'New this week'], ['old', 'On the list 8+ weeks']].map(([k, l]) => <button key={k} className={`pill ${f === k ? 'on' : ''}`} onClick={() => setF(k)}>{l}</button>)}</>}
          <span className="meta">{rows.length} companies{updated ? ` · updated ${String(updated).replace('T', ' ').slice(0, 16)}` : ''}</span>
        </div>
        <div className="tbl"><table>
          {cur.id === 'fast-growers' && <thead><tr><th>#</th><th>Company</th><th>Sales growth</th><th>Quality</th><th>Last 6 months</th><th>Why it's here</th></tr></thead>}
          {cur.id === 'comebacks' && <thead><tr><th>#</th><th>Company</th><th>Below high</th><th>Since the low</th><th>Stage</th><th>Why it's here</th></tr></thead>}
          {cur.id === 'climbers' && <thead><tr><th>#</th><th>Company</th><th>Strength</th><th>Change · 1 month</th><th>Size</th><th>Why it's here</th></tr></thead>}
          <tbody>
            {!loaded && <tr><td colSpan={6} style={{ color: 'var(--mute)' }}>loading…</td></tr>}
            {loaded?.error && <tr><td colSpan={6} style={{ color: 'var(--mute)' }}>this tracker couldn't be loaded right now</td></tr>}
            {rows.map((r: any, i: number) => (<tr key={r.ticker + i} onClick={() => go(r.ticker)}>
              <td className="num">{i + 1}</td>
              <td><span className="tk">{r.ticker}</span><span className="nm">{nice(r.name || '')}{r.tier ? ` · ${r.tier}` : ''}</span></td>
              {cur.id === 'fast-growers' && <><td className={`num ${(r.g ?? 0) >= 0 ? 'up' : 'dn'}`}>{fp(r.g)}</td><td className="num">{r.piotroski ?? '—'} / 9</td>
                <td className={`num ${(r.m6 ?? 0) >= 0 ? 'up' : 'dn'}`}>{fp(r.m6)}</td></>}
              {cur.id === 'comebacks' && <><td className="num dn">{fp(r.dd)}</td><td className="num up">{fp(r.off)} · {r.days_since_low ?? '—'} days</td>
                <td><span className="stage">{String(r.stage || '—').toUpperCase()}</span></td></>}
              {cur.id === 'climbers' && <><td className="num">{r.ascent_score != null ? Math.round(r.ascent_score) : '—'}</td>
                <td className={`num ${(r.delta_1m ?? 0) >= 0 ? 'up' : 'dn'}`}>{r.delta_1m != null ? `${r.delta_1m >= 0 ? '+' : ''}${r.delta_1m}` : '—'}</td><td className="num">{r.tier || '—'}</td></>}
              <td><span className="why">{r.why}</span></td></tr>))}
          </tbody></table></div>
        <p className="foot">A shortlist, not a prediction. Many companies on a tracker won't go on to do well; each one narrows thousands of companies down to a few worth reading about. Research, not advice.</p>
      </>)}
    </div></div>);
};
export default Trackers;
