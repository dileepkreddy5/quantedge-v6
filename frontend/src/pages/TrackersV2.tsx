// Trackers v2 — Great companies on sale · Quiet climbers · Getting better.
// Reads /api/v6/trackers/* (nightly facts sheet). Large · Mid · Small are separate lists.
import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { api } from '../auth/authStore';

const TRACKERS = [
  { id: 'worth', name: 'Worth a look', q: 'Our shortlist: great companies on a dip, early growth not yet priced, fresh breakthroughs.', api: 'worth-a-look' },
  { id: 'on-sale', name: 'Great companies on sale', q: 'The best companies trading far below their high — for reasons that will likely pass.', api: 'on-sale' },
  { id: 'quiet', name: 'Quiet climbers', q: 'Rising steadily, week after week, before everyone notices.', api: 'quiet-climbers' },
  { id: 'better', name: 'Getting better', q: 'Results improving quarter after quarter, straight from SEC filings.', api: 'getting-better' },
  { id: 'warn', name: 'Warning signs', q: 'Good companies showing early cracks — before the price fully reflects it.', api: 'warning-signs' },
  { id: 'rising', name: 'Rising stars', q: 'Growing fast enough to move up a size tier — tomorrow\'s bigger companies.', api: 'rising-stars' },
];
const TIERS = [['large', 'Large', 'over $10B'], ['mid', 'Mid', '$2B–$10B'], ['small', 'Small', '$300M–$2B']];
const STAGE: Record<string, [string, string]> = { falling: ['Still falling', '#ef7d5a'], basing: ['Going sideways', '#e0ad3a'], turning: ['Turning up', '#8fd19e'], recovering: ['Recovering', '#3ec27a'], near_high: ['Near its high', '#b09c86'] };
const CAUSE: Record<string, string> = { market: 'Whole market fell', industry: 'Its industry fell', company: 'Company-specific' };

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;1,9..144,300&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
.t2{--ink:#0d0806;--ink2:#130c09;--panel:#1a110d;--panel2:#21160f;--line:#33241b;--line2:#46321f;--gold:#e0ad3a;--gold2:#f3cf7a;--cream:#f6ecdd;--latte:#d9c9b4;--dust:#b09c86;--mute:#8a7762;--up:#3ec27a;--down:#ef7d5a;
 --serif:'Fraunces',Georgia,serif;--sans:'IBM Plex Sans',system-ui,sans-serif;--mono:'IBM Plex Mono',ui-monospace,monospace;background:var(--ink);color:var(--latte);font-family:var(--sans);min-height:100vh;-webkit-font-smoothing:antialiased}
.t2 *{box-sizing:border-box}.t2 button{font:inherit;cursor:pointer}.t2 a{cursor:pointer;color:inherit;text-decoration:none}
.t2 .wrap{max-width:1720px;margin:0 auto;padding:0 clamp(20px,3vw,56px)}
.t2 nav{border-bottom:1px solid var(--line)}.t2 nav .wrap{display:flex;align-items:center;gap:30px;height:64px}
.t2 .logo{font-family:var(--mono);font-weight:600;letter-spacing:.4em;color:var(--gold);font-size:15px}
.t2 nav .links{margin-left:auto;display:flex;gap:26px;font-size:14.5px;color:var(--dust)}.t2 nav .links a.on{color:var(--cream)}
.t2 .eyebrow{font-family:var(--mono);font-size:11.5px;letter-spacing:.26em;text-transform:uppercase;color:var(--gold)}
.t2 .head{padding:40px 0 18px}.t2 h1{font-family:var(--serif);font-weight:300;font-size:44px;color:var(--cream);margin:10px 0 0;letter-spacing:-.015em}
.t2 .trk{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:24px}
.t2 .srch{margin-left:auto;align-self:center;background:var(--panel);border:1px solid var(--line2);border-radius:8px;color:var(--cream);font-family:var(--mono);font-size:13px;padding:9px 14px;width:280px;outline:none}
.t2 .tb{text-align:left;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;color:var(--latte)}
.t2 .tb .n{font-family:var(--serif);font-size:21px;color:var(--cream)}.t2 .tb .d{font-size:13px;color:var(--dust);margin-top:6px;line-height:1.5}
.t2 .tb.on{border-color:var(--gold);background:var(--panel2)}
.t2 .tiers{display:flex;gap:0;margin-top:22px;border-bottom:1px solid var(--line)}
.t2 .tier{background:none;border:none;border-bottom:2px solid transparent;padding:12px 22px;color:var(--dust);font-size:15px}
.t2 .tier small{display:block;font-family:var(--mono);font-size:10.5px;color:var(--mute);margin-top:2px}
.t2 .tier.on{color:var(--cream);border-bottom-color:var(--gold)}
.t2 .callout{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(0,1fr);gap:14px;margin:20px 0 14px}
.t2 .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px}
.t2 .card .k{font-family:var(--mono);font-size:10px;letter-spacing:.2em;color:var(--mute)}
.t2 .card .v{font-size:14.5px;line-height:1.65;color:var(--dust);margin-top:8px}.t2 .card .v b{color:var(--latte);font-weight:500}
.t2 .card .big{font-family:var(--serif);font-size:24px;color:var(--cream);margin-top:6px}
.t2 .bar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:10px 0 12px}
.t2 .bar .lbl{font-family:var(--mono);font-size:10.5px;letter-spacing:.16em;color:var(--mute);margin:0 2px 0 8px}
.t2 .pill{background:none;border:1px solid var(--line);border-radius:999px;padding:6px 12px;font-family:var(--mono);font-size:11.5px;color:var(--dust)}
.t2 .pill.on{border-color:var(--gold);color:var(--gold)}.t2 .bar .meta{margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--mute)}
.t2 .row{background:var(--panel);border:1px solid var(--line);border-radius:12px;margin-bottom:8px}
.t2 .row:hover{border-color:var(--line2)}
.t2 .rh{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1.5fr) 128px 150px minmax(0,1.3fr) 110px;gap:16px;align-items:center;padding:14px 18px;cursor:pointer}
.t2 .tk{font-family:var(--mono);font-weight:500;color:var(--gold);font-size:15px}.t2 .nm{display:block;font-size:12.5px;color:var(--dust);margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.t2 .sub{display:block;font-family:var(--mono);font-size:10.5px;color:var(--mute);margin-top:3px}
.t2 .gauge{position:relative;height:8px;background:var(--ink2);border-radius:4px;margin-top:6px}
.t2 .gauge .fill{position:absolute;top:0;bottom:0;border-radius:4px;background:linear-gradient(90deg,#ef7d5a,#8a3f28)}
.t2 .gauge .now{position:absolute;top:-4px;width:3px;height:16px;background:var(--cream);border-radius:2px}
.t2 .gl{display:flex;justify-content:space-between;font-family:var(--mono);font-size:10.5px;color:var(--mute);margin-top:4px}
.t2 .chip{display:inline-block;font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;border:1px solid var(--line2);border-radius:5px;padding:3px 8px}
.t2 .num{font-family:var(--mono);font-size:13px}.t2 .up{color:var(--up)}.t2 .dn{color:var(--down)}.t2 .mu{color:var(--mute)}
.t2 .det{border-top:1px solid var(--line);padding:16px 18px;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.3fr);gap:24px}
.t2 .det h5{font-family:var(--mono);font-size:10px;letter-spacing:.2em;color:var(--mute);margin:0 0 10px;font-weight:500}
.t2 .qb{display:flex;align-items:flex-end;gap:5px;height:70px}.t2 .qb i{flex:1;background:linear-gradient(180deg,#3ec27a,#1f6b43);border-radius:3px 3px 0 0}
.t2 .ev{display:flex;gap:12px;padding:6px 0;border-bottom:1px solid var(--line);font-size:13px}.t2 .ev:last-child{border-bottom:none}
.t2 .ev span:first-child{font-family:var(--mono);color:var(--mute);width:92px;flex-shrink:0}
.t2 .open{font-family:var(--mono);font-size:12px;color:var(--gold);background:none;border:1px solid var(--line2);border-radius:6px;padding:8px 12px;margin-top:12px}
.t2 .lb{display:grid;grid-template-columns:36px minmax(0,1.4fr) minmax(0,2.4fr) 150px 150px;gap:16px;align-items:center;padding:13px 18px;background:var(--panel);border:1px solid var(--line);border-radius:12px;margin-bottom:8px;cursor:pointer}
.t2 .lb:hover{border-color:var(--line2)}.t2 .lb .rk{font-family:var(--serif);font-size:22px;color:var(--cream)}
.t2 .rets{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:4px}
.t2 .rline{padding:0 18px 14px}
.t2 .rets div{border-radius:6px;padding:6px 4px;text-align:center;font-family:var(--mono);font-size:11.5px}
.t2 .rets small{display:block;font-size:9.5px;color:var(--mute);margin-bottom:2px;letter-spacing:.08em}
.t2 .meter{height:6px;background:var(--ink2);border-radius:3px;margin-top:6px;overflow:hidden}.t2 .meter i{display:block;height:100%;border-radius:3px}
.t2 .gc{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
.t2 .gcard{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px 20px;display:flex;flex-direction:column;gap:12px;cursor:pointer}
.t2 .gcard:hover{border-color:var(--gold)}
.t2 .streak{margin-left:auto;font-family:var(--mono);font-size:10.5px;color:var(--gold);border:1px solid var(--line2);border-radius:5px;padding:3px 8px}
.t2 .kv{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
.t2 .kv div{background:var(--panel2);border-radius:8px;padding:10px 12px}.t2 .kv .k{font-family:var(--mono);font-size:9.5px;letter-spacing:.14em;color:var(--mute)}
.t2 .kv .v{font-family:var(--mono);font-size:15px;color:var(--cream);margin-top:4px}
.t2 .wco{background:var(--panel);border:1px solid var(--line);border-radius:14px;margin-bottom:12px;overflow:hidden}
.t2 .wco.serious{border-left:3px solid #e5484d}.t2 .wco.watch{border-left:3px solid #e0ad3a}
.t2 .wtop{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,2.4fr) minmax(0,.8fr);gap:20px;align-items:center;padding:16px 20px;cursor:pointer}
.t2 .wsev{text-align:right}.t2 .wsev b{display:block;font-family:var(--serif);font-weight:400;font-size:22px}
.t2 .wsg{display:grid;grid-template-columns:14px 104px 160px minmax(0,1fr);gap:14px;align-items:baseline;padding:9px 20px;border-top:1px solid rgba(51,36,27,.55);font-size:14px}
.t2 .wsg .dt{font-family:var(--mono);font-size:12px;color:var(--mute)}.t2 .wsg .sr{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;color:var(--dust)}
.t2 .wsg small{display:block;color:var(--mute);font-size:12.5px;margin-top:2px}
.t2 .wdot{width:9px;height:9px;border-radius:50%;display:inline-block}
.t2 .feat{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px;margin-bottom:22px}
.t2 .fc{background:linear-gradient(180deg,#221710,#1a110d);border:1px solid var(--line2);border-radius:16px;padding:20px;display:flex;flex-direction:column;gap:12px}
.t2 .fc .rank{font-family:var(--serif);font-size:30px;color:var(--gold);line-height:1}
.t2 .kd{display:inline-block;font-family:var(--mono);font-size:9.5px;letter-spacing:.12em;text-transform:uppercase;border-radius:4px;padding:3px 7px;margin-right:5px}
.t2 .kd.discount{color:#8fb8f0;border:1px solid #2c4a70}.t2 .kd.early{color:#3ec27a;border:1px solid #1f5c35}.t2 .kd.breakthrough{color:#f3cf7a;border:1px solid #6b5520}.t2 .kd.growth{color:#b8e0a0;border:1px solid #3a5a2a}.t2 .kd.pattern{color:#d9a7f0;border:1px solid #533a66}
.t2 .fc ul{margin:0;padding-left:18px;display:flex;flex-direction:column;gap:7px;font-size:13.5px;line-height:1.55;color:var(--latte)}
.t2 .cat{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.t2 .cat div{background:var(--panel2);border-radius:8px;padding:10px 12px}.t2 .cat .k{font-family:var(--mono);font-size:9.5px;letter-spacing:.14em;color:var(--mute)}
.t2 .cat .v{font-size:13px;color:var(--cream);margin-top:4px;line-height:1.45}
.t2 .risk{background:rgba(224,173,58,.07);border:1px solid rgba(224,173,58,.25);border-radius:8px;padding:10px 12px;font-size:12.5px;color:var(--dust);line-height:1.5}
.t2 .risk b{color:#e0ad3a;font-family:var(--mono);font-size:9.5px;letter-spacing:.14em;font-weight:500;display:block;margin-bottom:4px}
.t2 .acts{display:flex;gap:8px;margin-top:auto}.t2 .acts button{flex:1}
.t2 .wr{display:grid;grid-template-columns:36px minmax(0,1.3fr) minmax(0,.9fr) minmax(0,2.6fr) 120px 150px;gap:16px;align-items:center;padding:13px 18px;background:var(--panel);border:1px solid var(--line);border-radius:12px;margin-bottom:8px;cursor:pointer}
.t2 .wr:hover{border-color:var(--line2)}.t2 .wr .rk{font-family:var(--serif);font-size:20px;color:var(--cream)}
@media (max-width:1100px){.t2 .wr{grid-template-columns:30px 1fr}.t2 .wr .hide{display:none}}
.t2 .foot{font-size:13px;color:var(--mute);line-height:1.7;margin:18px 0 70px;max-width:980px}
.t2 .empty{padding:30px;text-align:center;color:var(--mute);font-family:var(--mono);font-size:12.5px}
@media (max-width:1100px){.t2 .rh{grid-template-columns:1fr 1fr}.t2 .lb{grid-template-columns:30px 1fr;}.t2 .lb .hide{display:none}.t2 .gc,.t2 .trk,.t2 .callout,.t2 .det{grid-template-columns:1fr}.t2 nav .links{display:none}.t2 h1{font-size:34px}}
`;

const pct = (v: any, d = 0) => v == null ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(d)}%`;
const bil = (v: any) => v == null ? '—' : v >= 1e12 ? `$${(v / 1e12).toFixed(2)}T` : v >= 1e9 ? `$${(v / 1e9).toFixed(1)}B` : `$${(v / 1e6).toFixed(0)}M`;
const nice = (n: string) => (n || '').replace(/ Class [A-Z] (Common|Ordinary) (Stock|Shares)| Common Stock| Ordinary Shares/g, '');
const cellBg = (v: any) => v == null ? '#1f1510' : v > 0 ? `rgba(62,194,122,${Math.min(0.55, 0.12 + Math.abs(v) * 0.9)})` : `rgba(239,125,90,${Math.min(0.55, 0.12 + Math.abs(v) * 0.9)})`;
const fmtDate = (s: any) => s ? new Date(s + 'T00:00:00Z').toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' }) : '—';

const Rets: React.FC<{ r: any; vol?: any }> = ({ r, vol }) => (<div className="rets">
  {[['1D', '1d'], ['1W', '1w'], ['1M', '1m'], ['3M', '3m'], ['6M', '6m'], ['1Y', '1y']].map(([l, k]) => (
    <div key={k} style={{ background: cellBg(r?.[k]) }}><small>{l}</small>{pct(r?.[k])}</div>))}
  <div style={{ background: '#1f1510', border: '1px solid #33241b' }}><small>VOLUME</small>{vol == null ? '—' : `${vol >= 1 ? '+' : ''}${Math.round((vol - 1) * 100)}%`}</div>
</div>);

const KIND: Record<string, string> = { discount: 'Discount', 'early growth': 'Early growth', breakthrough: 'Breakthrough', growth: 'Growth', pattern: 'Pattern' };
const kcls = (k: string) => k === 'early growth' ? 'early' : k;
const Bars: React.FC<{ v: number[] }> = ({ v }) => {
  const xs = (v || []).filter(x => x != null && x > 0); if (xs.length < 2) return <div className="mu" style={{ fontSize: 12 }}>no quarterly sales on file</div>;
  const mx = Math.max(...xs);
  return (<><div className="qb">{xs.map((x, i) => <i key={i} style={{ height: `${Math.max(6, x / mx * 100)}%` }} />)}</div>
    <div className="gl"><span>{xs.length} quarters ago</span><span>latest</span></div></>);
};

const TrackersV2: React.FC = () => {
  const nav = useNavigate(); const [sp, setSp] = useSearchParams();
  const { tab } = useParams();
  const LEGACY: Record<string, string> = { 'fast-growers': 'better', comebacks: 'on-sale', climbers: 'quiet', filter: 'on-sale' };
  const tr = TRACKERS.find(t => t.id === (sp.get('t') || LEGACY[tab || ''])) || TRACKERS[0];
  const tier = (TIERS.find(x => x[0] === sp.get('tier')) || TIERS[0])[0];
  const etier = tr.id === 'rising' && tier === 'large' ? 'mid' : tier;     // rising stars: mid→large, small→mid
  const [q, setQ] = useState('');
  const set = (k: string, v: string) => { const n = new URLSearchParams(sp); n.set(k, v); setSp(n, { replace: false }); };
  const [raw, setData] = useState<any>(null); const [open, setOpen] = useState<string | null>(null);
  const [sort, setSort] = useState('size'); const [weak, setWeak] = useState(false);
  const [depth, setDepth] = useState('all'); const [stage, setStage] = useState('all'); const [cause, setCause] = useState('all');
  // Each response is labelled with the view it belongs to; a tracker only ever draws its own
  // data. Before this, switching tabs briefly drew the new tracker with the previous one's rows.
  const key = `${tr.id}|${etier}|${sort}|${weak}`;
  useEffect(() => { setData(null); setOpen(null); setQ('');
    const qs = tr.id === 'on-sale' ? `?tier=${etier}&sort=${sort}&quality=${weak ? 'all' : 'strong'}` : tr.id === 'warn' ? `?tier=${etier}&healthy_only=${!weak}` : `?tier=${etier}`;
    api.get(`/api/v6/trackers/${tr.api}${qs}`).then(r => setData({ ...r.data, __key: key })).catch(() => setData({ error: true, companies: [], __key: key }));
  }, [tr.id, etier, sort, weak]); // eslint-disable-line
  const data = raw && raw.__key === key ? raw : null;
  const [sev, setSev] = useState('all');
  const go = (t: string) => nav(`/dashboard?ticker=${t}`);

  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const cs = (data?.companies || []).filter((c: any) => !needle || c.ticker.toLowerCase().includes(needle) || (c.name || '').toLowerCase().includes(needle));
    if (tr.id === 'warn') return cs.filter((c: any) => sev === 'all' || (sev === 'serious' && c.serious > 0) || (sev === 'multi' && c.signs.length >= 2));
    if (tr.id !== 'on-sale') return cs;
    return cs.filter((c: any) => (depth === 'all' || (depth === '20' && c.pct_below_high > -0.30) || (depth === '30' && c.pct_below_high <= -0.30 && c.pct_below_high > -0.50) || (depth === '50' && c.pct_below_high <= -0.50))
      && (stage === 'all' || c.stage === stage) && (cause === 'all' || c.drop_cause === cause));
  }, [data, tr.id, depth, stage, cause, sev, q]);

  const hist = data?.history?.buckets || {};
  const hb = depth === '50' ? hist['50'] : depth === '30' ? hist['30'] : hist['20'];
  const hbLabel = depth === '50' ? '50%' : depth === '30' ? '30%' : '20%';

  return (<div className="t2"><style>{CSS}</style>
    <nav><div className="wrap"><a className="logo" onClick={() => nav('/')}>QUANTEDGE</a>
      <div className="links"><a onClick={() => nav('/')}>Markets</a><a className="on">Trackers</a><a onClick={() => nav('/methodology')}>How it works</a></div></div></nav>
    <div className="wrap">
      <div className="head"><div className="eyebrow">Trackers · rebuilt every night</div><h1>Find the companies worth your attention.</h1>
        <div className="trk">{TRACKERS.map(t => (<button key={t.id} className={`tb ${t.id === tr.id ? 'on' : ''}`} onClick={() => set('t', t.id)}>
          <div className="n">{t.name}</div><div className="d">{t.q}</div></button>))}</div>
        <div className="tiers">{(tr.id === 'rising' ? [['mid', 'Mid → Large', 'to $10B+'], ['small', 'Small → Mid', 'to $2B+']] : TIERS).map(([k, l, sub]) => (
          <button key={k} className={`tier ${etier === k ? 'on' : ''}`} onClick={() => set('tier', k)}>{l}<small>{sub}</small></button>))}
          <label htmlFor="tsearch" style={{ position: 'absolute', left: -9999 }}>Search this tracker</label>
          <input id="tsearch" className="srch" value={q} onChange={e => setQ(e.target.value)} placeholder="Search this tracker — ticker or name" autoComplete="off" /></div>
        {q && data && rows.length === 0 && <div className="empty">{q.toUpperCase()} isn't on this tracker in this size group today.
          <br /><button className="open" onClick={() => go(q.trim().toUpperCase())}>Open its full analysis →</button></div>}
      </div>

      {/* ── WORTH A LOOK ── */}
      {tr.id === 'worth' && (<>
        <div className="callout">
          <div className="card"><div className="k">WHAT THIS IS</div>
            <div className="v">Our shortlist for your research, rebuilt every night. It reads every tracker together — <b>great companies on a dip</b>, <b>growth the price hasn't caught up with</b>, and <b>breakthroughs</b> like a big, lasting jump on results — then drops anything with a serious warning sign or that has <b>already run</b>. Each company comes with its case, its next catalyst, what history says, and what could go wrong.</div></div>
          <div className="card"><div className="k">WHAT IT'S BUILT ON — AND WHAT IT ISN'T</div>
            <div className="v">SEC filings, prices, peer comparisons, measured chart-pattern odds and warning signs. <b>Not</b> the ML forecasts — none currently hold up on recent data, so they're left out. These are <b>candidates to research, not recommendations</b>; some will not work out.</div></div>
        </div>
        <div className="bar"><span className="meta" style={{ marginLeft: 0 }}>{data ? `${rows.length} companies · top 5 highlighted · as of ${data.as_of || ''}` : 'loading…'}</span></div>
        {data && !q && rows.length === 0 && <div className="empty">nothing on the shortlist in this size tier today</div>}
        <div className="feat">{rows.filter((p: any) => p.top5).map((p: any, i: number) => (<div className="fc" key={p.ticker}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}><span className="rank">{i + 1}</span>
            <div><span className="tk">{p.ticker}</span><span className="nm">{nice(p.name)} · {bil(p.market_cap)} · {p.sector}</span></div></div>
          <div>{p.kinds.map((k: string) => <span key={k} className={`kd ${kcls(k)}`}>{KIND[k] || k}</span>)}</div>
          <ul>{p.case.map((c: string, j: number) => <li key={j}>{c}</li>)}</ul>
          <div className="cat">
            <div><div className="k">NEXT CATALYST</div><div className="v">Results ~{fmtDate(p.next_results_est)}{p.next_results_basis ? ' (est.)' : ''}</div></div>
            <div><div className="k">WHAT HISTORY SAYS</div><div className="v">{p.history || '—'}</div></div>
          </div>
          <Rets r={p.returns} vol={p.vol_ratio_20_60} />
          {p.sales_quarters && <Bars v={p.sales_quarters} />}
          <div className="risk"><b>WHAT COULD GO WRONG</b>{p.risks.join(' · ')}</div>
          <div className="acts"><button className="open" onClick={() => go(p.ticker)}>Full analysis →</button>
            <button className="open" onClick={() => nav(`/dashboard?ticker=${p.ticker}&tab=pattern`)}>Pattern chart →</button></div>
        </div>))}</div>
        {rows.filter((p: any) => !p.top5).map((p: any, i: number) => (<div className="wr" key={p.ticker} onClick={() => go(p.ticker)}>
          <span className="rk">{i + 6}</span>
          <div><span className="tk">{p.ticker}</span><span className="nm">{nice(p.name)} · {bil(p.market_cap)}</span></div>
          <div className="hide">{p.kinds.map((k: string) => <span key={k} className={`kd ${kcls(k)}`}>{KIND[k] || k}</span>)}</div>
          <div className="hide" style={{ fontSize: 13, color: 'var(--latte)', lineHeight: 1.5 }}>{p.case[0]}</div>
          <div className="hide num mu" style={{ fontSize: 11.5 }}>results ~{fmtDate(p.next_results_est)}</div>
          <div className="hide num"><span className={((p.returns || {})['1m'] ?? 0) >= 0 ? 'up' : 'dn'}>1M {pct((p.returns || {})['1m'])}</span> · <span className={((p.returns || {})['1y'] ?? 0) >= 0 ? 'up' : 'dn'}>1Y {pct((p.returns || {})['1y'])}</span>
            {p.risks[0] && !p.risks[0].startsWith('No warning') && <span className="sub" style={{ color: '#e0ad3a' }}>⚠ {p.risks[0].slice(0, 60)}</span>}</div>
        </div>))}
        <p className="foot">{data?.note} Research, not advice.</p>
      </>)}

      {/* ── GREAT COMPANIES ON SALE ── */}
      {tr.id === 'on-sale' && (<>
        <div className="callout">
          <div className="card"><div className="k">HOW THIS LIST IS BUILT</div>
            <div className="v">Companies at least <b>20% below their highest price of the last 5 years</b>, whose business is still <b>healthy, profitable and backed by cash</b> — the difference between a dip and a broken company. Each shows <b>why it fell</b>: the whole market, its industry, or something specific to the company.</div></div>
          <div className="card"><div className="k">WHAT HISTORY SAYS · LARGE US COMPANIES DOWN {hbLabel}+</div>
            {hb ? <><div className="big">{Math.round(hb.recovered_pct)} in 100 got back to their high</div>
              <div className="v">about {Math.round(hb.recovered_within_1y_pct)} in 100 within a year · typically after <b>{Math.round(hb.median_sessions_to_recover / 21)} months</b> · {hb.episodes.toLocaleString()} past cases since 2021. Companies that went bankrupt or were delisted aren't in this data, so real odds are somewhat lower.</div></>
              : <div className="v">measuring…</div>}</div>
        </div>
        <div className="bar">
          <span className="lbl">DEPTH</span>{[['all', 'All'], ['20', '20–30%'], ['30', '30–50%'], ['50', '50%+']].map(([k, l]) => <button key={k} className={`pill ${depth === k ? 'on' : ''}`} onClick={() => setDepth(k)}>{l}</button>)}
          <span className="lbl">NOW</span>{[['all', 'Any'], ['falling', 'Still falling'], ['basing', 'Sideways'], ['turning', 'Turning up'], ['recovering', 'Recovering']].map(([k, l]) => <button key={k} className={`pill ${stage === k ? 'on' : ''}`} onClick={() => setStage(k)}>{l}</button>)}
        </div>
        <div className="bar">
          <span className="lbl">WHY IT FELL</span>{[['all', 'Any'], ['market', 'Whole market'], ['industry', 'Its industry'], ['company', 'Company-specific']].map(([k, l]) => <button key={k} className={`pill ${cause === k ? 'on' : ''}`} onClick={() => setCause(k)}>{l}</button>)}
          <span className="lbl">SORT</span>{[['size', 'Biggest'], ['discount', 'Deepest discount'], ['business', 'Strongest business']].map(([k, l]) => <button key={k} className={`pill ${sort === k ? 'on' : ''}`} onClick={() => setSort(k)}>{l}</button>)}
          <button className={`pill ${weak ? 'on' : ''}`} onClick={() => setWeak(w => !w)} style={{ marginLeft: 8 }}>{weak ? '✓ ' : ''}Include weaker businesses</button>
          <span className="meta">{data ? `${rows.length} companies${data.counts ? ` · ${data.counts.weakening} weaker set aside` : ''} · as of ${data.as_of || ''}` : 'loading…'}</span>
        </div>
        {data && !q && rows.length === 0 && <div className="empty">no companies match these filters</div>}
        {rows.map((c: any) => { const st = STAGE[c.stage] || [c.stage, '#b09c86']; const lowPct = c.low_date && c.pct_off_low != null ? (1 + c.pct_below_high) / (1 + c.pct_off_low) - 1 : null;
          const isOpen = open === c.ticker;
          return (<div className="row" key={c.ticker}>
            <div className="rh" onClick={() => setOpen(isOpen ? null : c.ticker)}>
              <div><span className="tk">{c.ticker}</span><span className="nm">{nice(c.name)}</span><span className="sub">{c.sector} · {bil(c.market_cap)}</span></div>
              <div><span className="num dn" style={{ fontSize: 17 }}>{pct(c.pct_below_high)}</span> <span className="mu num" style={{ fontSize: 11 }}>from {fmtDate(c.high_date)} high · {c.months_since_high} mo</span>
                <div className="gauge">{lowPct != null && <div className="fill" style={{ left: 0, width: `${Math.min(100, -lowPct * 100)}%` }} />}
                  <div className="now" style={{ left: `calc(${Math.min(100, -c.pct_below_high * 100)}% - 1px)` }} /></div>
                <div className="gl"><span>high</span><span>{lowPct != null ? `low ${pct(lowPct)}` : ''} · now {pct(c.pct_below_high)}</span></div></div>
              <div><span className="chip" style={{ color: st[1], borderColor: st[1] + '66' }}>{st[0]}</span>{c.pct_off_low != null && <span className="sub">{pct(c.pct_off_low)} off the low</span>}</div>
              <div><span className="chip">{CAUSE[c.drop_cause] || '—'}</span>
                <span className="sub">market {pct(c.mkt_move_since_high)} · peers {pct(c.sector_move_since_high)}</span></div>
              <div className="num"><span className={(c.sales_yoy ?? 0) >= 0 ? 'up' : 'dn'}>sales {pct(c.sales_yoy)}</span> · margin {c.op_margin != null ? `${Math.round(c.op_margin * 100)}%` : '—'}
                <span className="sub">{c.cash_backed ? '✓ profits backed by cash' : 'cash backing weak'}{c.quality === 'weakening' ? ' · weaker business' : ''}</span></div>
              <div className="num mu" style={{ fontSize: 11.5 }}>next results<br /><span style={{ color: 'var(--latte)' }}>~{fmtDate(c.next_results_est)}</span><span className="sub">estimate</span></div>
            </div>
            <div className="rline"><Rets r={c.returns} vol={c.vol_ratio_20_60} /></div>
            {isOpen && (<div className="det">
              <div><h5>SALES · LAST 8 QUARTERS (SEC FILINGS)</h5><Bars v={c.sales_quarters} />
                {c.history_note && <div className="sub" style={{ marginTop: 10 }}>Price {c.history_note}.</div>}</div>
              <div><h5>FILED AROUND THE DROP</h5>
                {(c.around_the_drop || []).length === 0 ? <div className="mu" style={{ fontSize: 13 }}>no material SEC filings between the high and a month after the low</div> :
                  c.around_the_drop.map((e: any, i: number) => <div className="ev" key={i}><span>{fmtDate(e.date)}</span><span>{e.item === '2.02' ? 'Results released' : e.title}</span></div>)}
                <button className="open" onClick={() => go(c.ticker)}>Open full analysis →</button></div>
            </div>)}
          </div>); })}
        <p className="foot">“High” = the highest closing price in the last 5 years (our price history starts September 2021). “Why it fell” compares the company's fall since its high with the S&amp;P 500 and with its closest industry peers over the same dates. Next results dates are estimates from the last report. A company on sale can keep falling — this is a list to research, not advice.</p>
      </>)}

      {/* ── QUIET CLIMBERS ── */}
      {tr.id === 'quiet' && (<>
        <div className="callout">
          <div className="card"><div className="k">HOW THIS LIST IS BUILT</div>
            <div className="v">Companies <b>up over the past year, 6 months and 3 months</b> — rising in both halves of the year, not bouncing back from a fall — that <b>beat the market in at least 15 of the last 26 weeks</b>, while getting <b>no more news coverage than similar-sized companies</b>. Ranked by how steady and how far, against how little attention.</div></div>
          <div className="card"><div className="k">HOW TO READ IT</div>
            <div className="v">The six cells show the move over each period — green up, orange down. <b>Coverage vs peers</b> below 1.0× means fewer news articles than a typical company this size. Our news feed is thin for many companies, so treat coverage as a hint, not a measurement.</div></div>
        </div>
        <div className="bar"><span className="meta" style={{ marginLeft: 0 }}>{data ? `${rows.length} companies · as of ${data.as_of || ''}` : 'loading…'}</span></div>
        {data && !q && rows.length === 0 && <div className="empty">no quiet climbers in this size tier today</div>}
        {rows.map((c: any, i: number) => (<div className="lb" key={c.ticker} onClick={() => go(c.ticker)}>
          <span className="rk">{i + 1}</span>
          <div><span className="tk">{c.ticker}</span><span className="nm">{nice(c.name)}</span><span className="sub">{c.sector} · {bil(c.market_cap)}</span></div>
          <Rets r={c.returns} vol={c.vol_ratio_20_60} />
          <div className="hide"><span className="num">beat market {c.weeks_beat_mkt_26}/26 wks</span>
            <div className="meter"><i style={{ width: `${c.weeks_beat_mkt_26 / 26 * 100}%`, background: 'var(--up)' }} /></div></div>
          <div className="hide"><span className="num">coverage {c.attention_vs_peers.toFixed(1)}× peers</span>
            <div className="meter"><i style={{ width: `${Math.min(100, c.attention_vs_peers / 1.25 * 100)}%`, background: c.attention_vs_peers <= 1 ? 'var(--gold)' : 'var(--dust)' }} /></div>
            <span className="sub">{c.news_180d} articles in 6 months</span></div>
        </div>))}
        <p className="foot">Steady climbs can stop. A stock that has risen a lot has more room to fall, too. Research, not advice.</p>
      </>)}

      {/* ── GETTING BETTER ── */}
      {tr.id === 'better' && (<>
        <div className="callout">
          <div className="card"><div className="k">HOW THIS LIST IS BUILT</div>
            <div className="v">From each company's SEC filings: <b>sales growth speeding up for at least two quarters in a row</b>, <b>operating margin wider than a year ago</b>, sales growing at least 5%, and <b>profits backed by real cash</b>. Each quarter is dated to when it was first filed — nothing from the future.</div></div>
          <div className="card"><div className="k">WHY IT MATTERS</div>
            <div className="v">Improvement often shows up in the numbers before the price fully reflects it. The aim is to catch it <b>while it's happening</b> — check each company's price move to see how much is already priced in.</div></div>
        </div>
        <div className="bar"><span className="meta" style={{ marginLeft: 0 }}>{data ? `${rows.length} companies · as of ${data.as_of || ''}` : 'loading…'}</span></div>
        {data && !q && rows.length === 0 && <div className="empty">no companies in this size tier meet all the conditions today</div>}
        <div className="gc">{rows.map((c: any) => (<div className="gcard" key={c.ticker} onClick={() => go(c.ticker)}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}><span className="tk">{c.ticker}</span>
            <span className="streak">{c.acceleration_streak >= 7 ? '7+' : c.acceleration_streak} QTRS SPEEDING UP</span></div>
          <span className="nm" style={{ marginTop: -8 }}>{nice(c.name)} · {bil(c.market_cap)}</span>
          <Bars v={(c.quarters || []).map((q: any) => q.sales)} />
          <Rets r={c.returns} vol={c.vol_ratio_20_60} />
          <div className="kv">
            <div><div className="k">SALES GROWTH</div><div className="v"><span className="mu">{pct(c.sales_yoy_prev)}</span> → <span className="up">{pct(c.sales_yoy)}</span></div></div>
            <div><div className="k">OPERATING MARGIN</div><div className="v"><span className="mu">{c.op_margin_year_ago != null ? `${Math.round(c.op_margin_year_ago * 100)}%` : '—'}</span> → <span className="up">{c.op_margin != null ? `${Math.round(c.op_margin * 100)}%` : '—'}</span></div></div>
            <div><div className="k">SALES GROWTH · 2 QTRS AGO</div><div className="v mu">{pct((c.quarters || []).slice(-3)[0]?.sales_yoy)}</div></div>
            <div><div className="k">BELOW 5Y HIGH</div><div className="v">{pct(c.pct_below_high)}</div></div>
          </div>
          <span className="sub">{c.cash_backed ? '✓ profits backed by cash' : ''}{c.one_off_suspected ? ' · year-ago margin distorted by a one-off loss' : ''} · last quarter {fmtDate(c.last_quarter)}</span>
        </div>))}</div>
        <p className="foot">Improving results can reverse, and the price may already reflect them. Research, not advice.</p>
      </>)}
      {/* ── WARNING SIGNS ── */}
      {tr.id === 'warn' && (<>
        <div className="callout">
          <div className="card"><div className="k">WHAT IT'S FOR</div>
            <div className="v">Catching a <b>good company that's starting to go wrong</b>, before the price fully reflects it. Every sign comes from the company's <b>own SEC filings</b> or its trading, dated to when it became public. By default only companies that were <b>healthy six months ago</b> are shown — the point is to spot cracks early, not to list companies already broken.</div></div>
          <div className="card"><div className="k">HOW SERIOUS</div>
            <div className="v"><span className="wdot" style={{ background: '#e5484d' }} /> <b>Serious</b> — rare events that usually matter: past financials can't be relied on, auditor change, impairment, a sudden CEO/CFO exit, a late filing.<br />
              <span className="wdot" style={{ background: '#e0ad3a' }} /> <b>Watch</b> — early cracks: sales slowing, margins shrinking, profits outrunning cash, several insiders selling, a trend break, lagging peers.<br />One sign is a question; several deserve attention. <span className="mu">Track record: measuring — five years of filings are being loaded.</span></div></div>
        </div>
        <div className="bar">
          <span className="lbl">SHOW</span>{[['all', 'All signs'], ['serious', 'Serious only'], ['multi', '2+ signs']].map(([k, l]) => <button key={k} className={`pill ${sev === k ? 'on' : ''}`} onClick={() => setSev(k)}>{l}</button>)}
          <button className={`pill ${!weak ? 'on' : ''}`} onClick={() => setWeak(w => !w)} style={{ marginLeft: 8 }}>{!weak ? '✓ ' : ''}Only companies healthy 6 months ago</button>
          <span className="meta">{data ? `${rows.length} companies · signs from the last 30 days · as of ${data.as_of || ''}` : 'loading…'}</span>
        </div>
        {data && !q && rows.length === 0 && <div className="empty">no warning signs in this size tier in the last 30 days</div>}
        {rows.map((c: any) => (<div className={`wco ${c.serious ? 'serious' : 'watch'}`} key={c.ticker}>
          <div className="wtop" onClick={() => go(c.ticker)}>
            <div><span className="tk">{c.ticker}</span><span className="nm">{nice(c.name)}</span>
              <span className="sub">{c.sector} · {bil(c.market_cap)} · first sign {c.days_since_first} days ago</span></div>
            <Rets r={c.returns} vol={c.vol_ratio_20_60} />
            <div className="wsev"><b style={{ color: c.serious ? '#e5484d' : '#e0ad3a' }}>{c.signs.length} sign{c.signs.length > 1 ? 's' : ''}</b>
              <span className="sub">{c.serious ? `${c.serious} serious · ` : ''}{c.watch} watch</span></div>
          </div>
          {c.signs.map((s: any, i: number) => (<div className="wsg" key={i}>
            <span className="wdot" style={{ background: s.severity === 'serious' ? '#e5484d' : '#e0ad3a' }} />
            <span className="dt">{fmtDate(s.date)}</span><span className="sr">{s.source}</span>
            <span>{s.text}{s.note && <small>{s.note}</small>}</span></div>))}
        </div>))}
        <p className="foot">{data?.note} Research, not advice.</p>
      </>)}
      {/* ── RISING STARS ── */}
      {tr.id === 'rising' && (<>
        <div className="callout">
          <div className="card"><div className="k">HOW THIS LIST IS BUILT</div>
            <div className="v">{etier === 'mid' ? 'Mid-size' : 'Small'} companies with <b>sales growing 20%+ a year, sustained across most of the last four quarters</b>, on a real base of sales, with <b>margins widening as they grow</b> and the <b>price beating the market</b> in at least half of the last 26 weeks. Where 13F data exists, we show whether <b>more funds are buying in</b>.</div></div>
          <div className="card"><div className="k">HOW TO READ “MONTHS TO THE NEXT TIER”</div>
            <div className="v">If the company's market value grew as fast as its sales did over the last year, this is how long it would take to reach {etier === 'mid' ? '$10B' : '$2B'}. <b>Arithmetic on the recent pace, not a forecast</b> — growth slows, and prices don't follow sales one-for-one.</div></div>
        </div>
        <div className="bar"><span className="meta" style={{ marginLeft: 0 }}>{data ? `${rows.length} companies · as of ${data.as_of || ''}` : 'loading…'}</span></div>
        {data && !q && rows.length === 0 && <div className="empty">no rising stars in this group today</div>}
        <div className="gc">{rows.map((c: any) => (<div className="gcard" key={c.ticker} onClick={() => go(c.ticker)}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}><span className="tk">{c.ticker}</span>
            {c.funds_arriving && <span className="streak">MORE FUNDS BUYING</span>}</div>
          <span className="nm" style={{ marginTop: -8 }}>{nice(c.name)} · {bil(c.market_cap)} · {c.sector}</span>
          <div className="big" style={{ fontFamily: 'var(--serif)', fontSize: 17, color: 'var(--cream)', lineHeight: 1.4 }}>
            {c.months_to_next_tier_at_sales_pace == null ? '' : c.months_to_next_tier_at_sales_pace <= 0 ? `At the ${c.next_tier}-company threshold now` :
              `~${c.months_to_next_tier_at_sales_pace} months to ${c.next_tier === 'large' ? '$10B' : '$2B'} at its sales pace`}</div>
          <Bars v={c.sales_quarters} />
          <Rets r={c.returns} vol={c.vol_ratio_20_60} />
          <div className="kv">
            <div><div className="k">SALES GROWTH · LAST 4 QTRS</div><div className="v" style={{ fontSize: 13 }}>{(c.sales_yoy_4q || []).map((y: number) => pct(y)).join(' · ')}</div></div>
            <div><div className="k">OPERATING MARGIN</div><div className="v"><span className="mu">{c.op_margin_year_ago != null ? `${Math.round(c.op_margin_year_ago * 100)}%` : '—'}</span> → <span className="up">{c.op_margin != null ? `${Math.round(c.op_margin * 100)}%` : '—'}</span></div></div>
            <div><div className="k">BEAT THE MARKET</div><div className="v">{c.weeks_beat_mkt_26} / 26 weeks</div></div>
            <div><div className="k">FUNDS (13F)</div><div className="v" style={{ fontSize: 13 }}>{c.funds ? `+${c.funds.new_managers ?? 0} new · −${c.funds.exited_managers ?? 0} exited` : 'no data yet'}</div></div>
          </div>
        </div>))}</div>
        <p className="foot">{data?.note} Research, not advice.</p>
      </>)}
    </div></div>);
};
export default TrackersV2;
