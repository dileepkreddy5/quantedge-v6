// QuantEdge homepage — approved design, live data.
import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../auth/authStore';

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;1,9..144,300&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
.qh{--ink:#0d0806;--ink2:#130c09;--panel:#1a110d;--panel2:#21160f;--line:#33241b;--line2:#46321f;--gold:#e0ad3a;--gold2:#f3cf7a;--cream:#f6ecdd;--latte:#d9c9b4;--dust:#b09c86;--mute:#8a7762;--up:#3ec27a;--down:#ef7d5a;
  --serif:'Fraunces',Georgia,serif;--sans:'IBM Plex Sans',system-ui,sans-serif;--mono:'IBM Plex Mono',ui-monospace,monospace;background:var(--ink);color:var(--latte);font-family:var(--sans);min-height:100vh;-webkit-font-smoothing:antialiased}
.qh *{box-sizing:border-box}.qh a{color:inherit;text-decoration:none;cursor:pointer}.qh button{font:inherit;cursor:pointer}
.qh .wrap{max-width:1720px;margin:0 auto;padding:0 clamp(20px,3vw,56px)}
.qh nav{border-bottom:1px solid var(--line)}.qh nav .wrap{display:flex;align-items:center;gap:30px;height:66px}
.qh .logo{font-family:var(--mono);font-weight:600;letter-spacing:.4em;color:var(--gold);font-size:15px}
.qh nav .links{margin-left:auto;display:flex;gap:26px;font-size:14.5px;color:var(--dust)}.qh nav .links a:hover{color:var(--cream)}
.qh .btn{display:inline-flex;align-items:center;gap:8px;border-radius:9px;font-family:var(--mono);font-size:13px;letter-spacing:.1em;padding:12px 20px;background:none;color:var(--latte)}
.qh .btn.ghost{border:1px solid var(--line2)}.qh .btn.solid{background:var(--gold);color:#1a1008;font-weight:600;border:none}
.qh .eyebrow{font-family:var(--mono);font-size:11.5px;letter-spacing:.26em;text-transform:uppercase;color:var(--gold)}
.qh .hero{text-align:center;padding:84px 0 56px}
.qh .hero h1{font-family:var(--serif);font-weight:300;font-size:64px;line-height:1.05;letter-spacing:-.02em;color:var(--cream);margin:18px auto 20px;max-width:880px}
.qh .hero h1 em{font-style:italic;color:var(--gold2)}.qh .hero p{font-size:19px;line-height:1.6;color:var(--dust);max-width:640px;margin:0 auto 34px}
.qh .search{position:relative;display:flex;gap:10px;max-width:620px;margin:0 auto;padding:8px;background:var(--panel);border:1px solid var(--line2);border-radius:14px}
.qh .search input{flex:1;background:transparent;border:none;outline:none;color:var(--cream);font-family:var(--mono);font-size:17px;padding:0 16px;min-height:48px}
.qh .search input::placeholder{color:var(--mute)}
.qh .sugg{position:absolute;left:0;right:0;top:calc(100% + 6px);background:var(--panel);border:1px solid var(--line2);border-radius:12px;overflow:hidden;z-index:20;text-align:left;box-shadow:0 20px 50px rgba(0,0,0,.5)}
.qh .sugg button{display:flex;gap:14px;width:100%;padding:12px 18px;background:none;border:none;border-bottom:1px solid var(--line);color:var(--latte);text-align:left}
.qh .sugg button:hover,.qh .sugg button.on{background:var(--panel2)}.qh .sugg b{font-family:var(--mono);color:var(--gold);font-weight:500;width:70px}
.qh .chips{display:flex;justify-content:center;gap:10px;margin-top:16px;font-family:var(--mono);font-size:13px;color:var(--mute)}
.qh .chips a{color:var(--latte);border:1px solid var(--line);border-radius:999px;padding:5px 12px}.qh .chips a:hover{border-color:var(--gold);color:var(--gold)}
.qh .markets{border-top:1px solid var(--line);border-bottom:1px solid var(--line);background:var(--ink2);padding:34px 0 30px}
.qh .mhead{display:flex;align-items:baseline;gap:16px;margin-bottom:18px;flex-wrap:wrap}
.qh .mhead h2{font-family:var(--serif);font-weight:400;font-size:28px;color:var(--cream);margin:0}.qh .mhead .meta{font-family:var(--mono);font-size:12px;color:var(--mute)}
.qh .mhead .tabs{margin-left:auto;display:flex;gap:6px}
.qh .mhead .tabs button{background:none;border:1px solid var(--line);border-radius:6px;padding:6px 12px;font-family:var(--mono);font-size:11.5px;letter-spacing:.1em;color:var(--dust)}
.qh .mhead .tabs button.on{border-color:var(--gold);color:var(--gold)}
.qh .mood{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin-bottom:14px;font-size:15px;color:var(--dust);line-height:1.6}
.qh .mood b{color:var(--cream);font-weight:500}.qh .mood .dot{width:9px;height:9px;border-radius:50%;flex-shrink:0;align-self:center}
.qh .idx{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}
.qh .ix{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 16px 12px;text-align:left;color:inherit}
.qh .ix .n{font-size:14px;color:var(--cream);font-weight:500}.qh .ix .p{font-family:var(--mono);font-size:11px;color:var(--mute);margin-top:2px}
.qh .ix .t{font-family:var(--mono);font-size:24px;margin-top:12px}.qh .ix .w{font-family:var(--mono);font-size:12px;color:var(--dust);margin-top:4px}
.qh .ix svg{display:block;width:100%;height:34px;margin-top:10px}
.qh .up{color:var(--up)}.qh .dn{color:var(--down)}.qh .flat{color:var(--dust)}
.qh .world{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;margin-top:12px}
.qh .wx{border:1px solid var(--line);border-radius:10px;padding:12px 14px;display:flex;flex-direction:column;gap:4px}
.qh .wx .n{font-size:13.5px;color:var(--latte)}.qh .wx .t{font-family:var(--mono);font-size:17px}.qh .wx .w{font-family:var(--mono);font-size:11px;color:var(--mute)}
.qh .fine{font-family:var(--mono);font-size:11.5px;color:var(--mute);margin-top:16px;line-height:1.6}
.qh section{padding:84px 0}
.qh .shead{display:flex;align-items:flex-end;justify-content:space-between;gap:30px;margin-bottom:34px}
.qh .shead h2{font-family:var(--serif);font-weight:300;font-size:44px;line-height:1.1;color:var(--cream);margin:12px 0 0}
.qh .shead p{max-width:420px;margin:0;color:var(--dust);font-size:16px;line-height:1.65}
.qh .sectors{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
.qh .sec,.qh .mv{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden}
.qh .sec .top{padding:18px 18px 14px;border-bottom:1px solid var(--line)}.qh .sec .top .n{font-size:16px;color:var(--cream);font-weight:500}
.qh .sec .top .row{display:flex;justify-content:space-between;align-items:baseline;margin-top:10px;font-family:var(--mono)}
.qh .sec .top .row b{font-size:20px;font-weight:500}.qh .sec .top .row span{font-size:11.5px;color:var(--mute)}
.qh .co{display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:12px;align-items:baseline;padding:11px 18px;border-bottom:1px solid rgba(51,36,27,.6);font-family:var(--mono);font-size:13px;width:100%;background:none;border-left:none;border-right:none;border-top:none;color:inherit;text-align:left}
.qh .co:last-child{border-bottom:none}.qh .co:hover{background:var(--panel2)}
.qh .co .tk{color:var(--gold);font-weight:500}.qh .co .nm{display:block;font-family:var(--sans);font-size:11.5px;color:var(--mute);margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.qh .co .w{color:var(--mute);font-size:11.5px}
.qh .movers{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.qh .mv .h{padding:16px 18px;border-bottom:1px solid var(--line);font-family:var(--mono);font-size:11.5px;letter-spacing:.22em;text-transform:uppercase}
.qh .doors{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}
.qh .door{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:30px;display:flex;flex-direction:column;gap:12px;min-height:250px;transition:.25s}
.qh .door:hover{border-color:var(--gold)}
.qh .door .k{font-family:var(--mono);font-size:11px;letter-spacing:.24em;color:var(--gold)}
.qh .door h3{font-family:var(--serif);font-weight:400;font-size:27px;line-height:1.2;color:var(--cream);margin:0}
.qh .door p{margin:0;font-size:15px;line-height:1.65;color:var(--dust)}
.qh .door ul{list-style:none;margin:6px 0 0;padding:0;display:flex;flex-direction:column;gap:10px;font-size:14.5px;color:var(--latte)}
.qh .door ul a:before{content:"→ ";color:var(--gold)}.qh .door ul a:hover{color:var(--gold2)}
.qh .door .go{margin-top:auto;font-family:var(--mono);font-size:12.5px;color:var(--gold)}
.qh .qs{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.qh .q{padding:30px 22px 30px 0}.qh .q+.q{padding-left:22px;border-left:1px solid var(--line)}
.qh .q .i{font-family:var(--serif);font-style:italic;font-size:30px;color:var(--gold);font-weight:300}
.qh .q h4{font-family:var(--serif);font-weight:400;font-size:21px;line-height:1.25;color:var(--cream);margin:10px 0}
.qh .q p{margin:0;font-size:14px;line-height:1.65;color:var(--dust)}.qh .q .tabsl{font-family:var(--mono);font-size:11px;color:var(--mute);margin-top:12px;line-height:1.7}
.qh .trust{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:36px}
.qh .trust h4{font-family:var(--serif);font-weight:400;font-size:22px;color:var(--cream);margin:0 0 10px}.qh .trust p{margin:0;font-size:15px;line-height:1.7;color:var(--dust)}
.qh footer{border-top:1px solid var(--line);padding:56px 0 30px;color:var(--mute);background:var(--ink2)}
.qh .fcols{display:grid;grid-template-columns:1.6fr repeat(4,minmax(0,1fr));gap:36px}
.qh .fcols .brand p{font-size:14px;line-height:1.7;color:var(--dust);margin:14px 0 0;max-width:320px}
.qh .fcols .brand .by{font-family:var(--mono);font-size:12px;color:var(--mute);margin-top:18px}.qh .fcols .brand .by b{color:var(--latte);font-weight:500}
.qh .fcols h5{font-family:var(--mono);font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:var(--gold);margin:4px 0 16px;font-weight:500}
.qh .fcols ul{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:11px;font-size:14px;color:var(--dust)}.qh .fcols a:hover{color:var(--cream)}
.qh .fbottom{margin-top:46px;padding-top:22px;border-top:1px solid var(--line);display:flex;flex-wrap:wrap;gap:12px 28px;font-family:var(--mono);font-size:11.5px;color:var(--mute)}
.qh .fbottom .copy{color:var(--latte)}.qh .fnote{margin-top:14px;font-size:12px;line-height:1.7;color:var(--mute);max-width:980px}
.qh .skel{color:var(--mute)}
@media (max-width:1100px){.qh .movers{grid-template-columns:1fr}.qh .idx{grid-template-columns:repeat(3,minmax(0,1fr))}.qh .world{grid-template-columns:repeat(3,minmax(0,1fr))}.qh .sectors{grid-template-columns:repeat(2,minmax(0,1fr))}
  .qh .qs{grid-template-columns:1fr}.qh .q+.q{padding-left:0;border-left:none;border-top:1px solid var(--line)}.qh .doors,.qh .trust{grid-template-columns:1fr}.qh .hero h1{font-size:44px}.qh .shead{flex-direction:column;align-items:flex-start}.qh nav .links{display:none}.qh .fcols{grid-template-columns:1fr 1fr}}
@media (max-width:640px){.qh .idx,.qh .world,.qh .sectors{grid-template-columns:1fr 1fr}.qh .wrap{padding:0 18px}.qh .hero h1{font-size:36px}}
`;

const sgn = (v: any, d = 2) => v == null ? '—' : `${v >= 0 ? '+' : ''}${Number(v).toFixed(d)}%`;
const cls = (v: any) => v == null ? 'flat' : v > 0 ? 'up' : v < 0 ? 'dn' : 'flat';
const Spark: React.FC<{ s: number[] }> = ({ s }) => {
  if (!s || s.length < 2) return null;
  const mn = Math.min(...s), mx = Math.max(...s), up = s[s.length - 1] >= s[0];
  const d = s.map((v, i) => `${i ? 'L' : 'M'}${(i * 100 / (s.length - 1)).toFixed(1)},${(31 - (v - mn) / ((mx - mn) || 1) * 28).toFixed(1)}`).join('');
  return <svg viewBox="0 0 100 34" preserveAspectRatio="none" aria-hidden="true"><path d={d} fill="none" stroke={up ? '#3ec27a' : '#ef7d5a'} strokeWidth={1.6} /></svg>;
};

const Home: React.FC = () => {
  const nav = useNavigate();
  const go = (t: string) => nav(`/dashboard?ticker=${encodeURIComponent(t)}`);
  const [q, setQ] = useState(''); const [sug, setSug] = useState<any[]>([]); const [hi, setHi] = useState(0);
  const [mk, setMk] = useState<any>(null); const [mood, setMood] = useState<any>(null); const [mv, setMv] = useState<any>(null);
  const [span, setSpan] = useState<'today' | 'week'>('today');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const get = async (u: string, set: (v: any) => void) => { try { set((await api.get(u)).data); } catch { set({ error: true }); } };
    get('/api/v6/home/markets', setMk); get('/api/v6/home/mood', setMood); get('/api/v6/home/movers', setMv);
    const t = setInterval(() => get('/api/v6/home/markets', setMk), 60000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => {
    if (!q.trim()) { setSug([]); return; }
    const t = setTimeout(async () => { try { setSug((await api.get(`/api/v6/search/suggest?q=${encodeURIComponent(q.trim())}`)).data.results || []); setHi(0); } catch { setSug([]); } }, 150);
    return () => clearTimeout(t);
  }, [q]);
  const submit = (e: React.FormEvent) => { e.preventDefault(); const pick = sug[hi]?.ticker || q.trim().toUpperCase(); if (pick) go(pick); };
  const focusSearch = () => { window.scrollTo({ top: 0, behavior: 'smooth' }); setTimeout(() => inputRef.current?.focus(), 350); };

  const v = (r: any) => span === 'today' ? r?.today_pct : r?.week_pct;
  const asOf = mk?.as_of ? new Date(mk.as_of).toLocaleString('en-US', { timeZone: 'America/New_York', weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) + ' ET' : '';
  const moodCol = mood?.trend === 'rising' ? (mood?.swings === 'calm' ? '#3ec27a' : '#e0ad3a') : '#ef7d5a';

  return (<div className="qh">
    <style>{CSS}</style>
    <nav><div className="wrap">
      <a className="logo" onClick={focusSearch}>QUANTEDGE</a>
      <div className="links"><a href="#markets">Markets</a><a onClick={() => nav('/trackers')}>Trackers</a><a href="#how">How it works</a></div>
      <button className="btn ghost" onClick={() => nav('/login')}>Log in</button>
    </div></nav>

    <header className="hero"><div className="wrap">
      <div className="eyebrow">Stock research · US markets</div>
      <h1>Understand any stock <em>before</em> you buy it.</h1>
      <p>Search a company and see, in plain English, how its price is moving, how the business is doing, who's buying and selling — and how often similar situations worked out in the past.</p>
      <form className="search" onSubmit={submit} role="search">
        <label htmlFor="q" style={{ position: 'absolute', left: -9999 }}>Search a stock</label>
        <input id="q" ref={inputRef} value={q} onChange={e => setQ(e.target.value)} autoComplete="off"
          placeholder="Company name or ticker — Apple, TSLA, Costco…"
          onKeyDown={e => { if (e.key === 'ArrowDown') { e.preventDefault(); setHi(h => Math.min(h + 1, sug.length - 1)); } if (e.key === 'ArrowUp') { e.preventDefault(); setHi(h => Math.max(h - 1, 0)); } }} />
        <button className="btn solid" type="submit">Search</button>
        {sug.length > 0 && <div className="sugg">{sug.map((s, i) => (
          <button type="button" key={s.ticker} className={i === hi ? 'on' : ''} onMouseEnter={() => setHi(i)} onClick={() => go(s.ticker)}>
            <b>{s.ticker}</b><span>{s.name}</span></button>))}</div>}
      </form>
      <div className="chips"><span>Popular</span>{['AAPL', 'TSLA', 'AMZN', 'MSFT', 'COST'].map(t => <a key={t} onClick={() => go(t)}>{t}</a>)}</div>
    </div></header>

    <div className="markets" id="markets"><div className="wrap">
      <div className="mhead"><h2>Markets {span === 'today' ? 'today' : 'this week'}</h2>
        <span className="meta">{asOf ? `as of ${asOf}` : 'loading…'}</span>
        <div className="tabs"><button className={span === 'today' ? 'on' : ''} onClick={() => setSpan('today')}>Today</button>
          <button className={span === 'week' ? 'on' : ''} onClick={() => setSpan('week')}>This week</button></div></div>
      {mood && !mood.error && <div className="mood"><span className="dot" style={{ background: moodCol, boxShadow: `0 0 0 4px ${moodCol}26` }} />
        <b>Market mood: {mood.label}.</b>
        <span>{mood.higher_month_later_pct != null ? (Math.abs(mood.higher_month_later_pct - mood.base_pct) <= 3
          ? `In ${mood.n_similar_days} similar moments since ${String(mood.since).slice(0, 4)}, the S&P 500 was higher a month later ${mood.higher_month_later_pct}% of the time — about the same as on any day (${mood.base_pct}%). On its own, today's mood doesn't point either way.`
          : `In ${mood.n_similar_days} similar moments since ${String(mood.since).slice(0, 4)}, the S&P 500 was higher a month later ${mood.higher_month_later_pct}% of the time, versus ${mood.base_pct}% on any day.`) : ''}</span></div>}
      <div className="idx">{(mk?.us || [0, 1, 2, 3, 4]).map((r: any, k: number) => r?.ticker ? (
        <div className="ix" key={r.ticker}><div className="n">{r.name}</div><div className="p">{r.proxy}</div>
          <div className={`t ${cls(v(r))}`}>{sgn(v(r))}</div>
          <div className="w">{span === 'today' ? <>week <span className={cls(r.week_pct)}>{sgn(r.week_pct, 1)}</span></> : <>today <span className={cls(r.today_pct)}>{sgn(r.today_pct)}</span></>} · avg day {sgn(r.avg_day_pct)}</div>
          <Spark s={r.spark} /></div>) : <div className="ix skel" key={k}>…</div>)}</div>
      <div className="world">{(mk?.world || []).map((r: any) => (
        <div className="wx" key={r.ticker}><div className="n">{r.name} <span style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--mute)' }}>{r.ticker}</span></div>
          <div className={`t ${cls(v(r))}`}>{sgn(v(r))}</div><div className="w">{span === 'today' ? `week ${sgn(r.week_pct, 1)}` : `today ${sgn(r.today_pct)}`}</div></div>))}</div>
      <div className="fine">{mk?.note}</div>
    </div></div>

    <section style={{ paddingTop: 70 }}><div className="wrap">
      <div className="shead"><div><div className="eyebrow">By sector</div><h2>How the biggest companies are doing</h2></div>
        <p>The largest companies in six sectors, with {span === 'today' ? "today's move" : 'the last five trading days'}. Tap any company to see its full picture.</p></div>
      <div className="sectors">{(mk?.sectors || []).map((s: any) => (
        <div className="sec" key={s.name}><div className="top"><div className="n">{s.name}</div>
          <div className="row"><b className={cls(v(s))}>{sgn(v(s))}</b><span>{span === 'today' ? `week ${sgn(s.week_pct, 1)}` : `today ${sgn(s.today_pct)}`} · {s.etf}</span></div></div>
          {s.companies.map((c: any) => (<button className="co" key={c.ticker} onClick={() => go(c.ticker)}>
            <div><span className="tk">{c.ticker}</span><span className="nm">{c.name}</span></div>
            <span className={cls(v(c))}>{sgn(v(c), 1)}</span><span className="w">{span === 'today' ? `${sgn(c.week_pct, 1)} wk` : `${sgn(c.today_pct, 1)} today`}</span></button>))}
        </div>))}</div>
    </div></section>

    <section style={{ paddingTop: 10 }}><div className="wrap">
      <div className="shead"><div><div className="eyebrow">Today</div><h2>Biggest movers</h2></div>
        <p>Large US companies (over $10B) moving the most today. A big move is a reason to look, not a reason to act.</p></div>
      <div className="movers">{[['Rising', mv?.rising], ['Falling', mv?.falling]].map(([h, rows]: any) => (
        <div className="mv" key={h}><div className={`h ${h === 'Rising' ? 'up' : 'dn'}`}>{h}</div>
          {(rows || []).length === 0 && <div className="co skel" style={{ display: 'block' }}>{mv ? 'none right now' : 'loading…'}</div>}
          {(rows || []).map((c: any) => (<button className="co" key={c.ticker} onClick={() => go(c.ticker)}>
            <div><span className="tk">{c.ticker}</span><span className="nm">{c.name}</span></div>
            <span className={cls(c.today_pct)}>{sgn(c.today_pct, 1)}</span><span className="w">{c.why}</span></button>))}</div>))}</div>
    </div></section>

    <section id="ideas" style={{ paddingTop: 20 }}><div className="wrap">
      <div className="shead"><div><div className="eyebrow">What brings you here?</div><h2>Start wherever you are.</h2></div></div>
      <div className="doors">
        <a className="door" onClick={focusSearch}><div className="k">01 · A stock I'm curious about</div><h3>Check a stock you own or are thinking about.</h3>
          <p>Everything about one company, starting with a plain-English summary at the top.</p><div className="go">Search a stock →</div></a>
        <div className="door"><div className="k">02 · I'm looking for ideas</div><h3>Find companies worth a closer look.</h3>
          <ul><li><a onClick={() => nav('/trackers?t=on-sale')}>Great companies on sale</a></li><li><a onClick={() => nav('/trackers?t=quiet')}>Quiet climbers — rising before anyone notices</a></li>
            <li><a onClick={() => nav('/trackers?t=better')}>Getting better — results improving every quarter</a></li></ul></div>
        <a className="door" href="#markets"><div className="k">03 · How's the market?</div><h3>See the market's mood today.</h3>
          <p>Whether the market is calm or nervous, rising or falling — and what usually followed similar moments.</p><div className="go">Today's market →</div></a>
      </div>
    </div></section>

    <section id="how" style={{ paddingTop: 20 }}><div className="wrap">
      <div className="shead"><div><div className="eyebrow">For every stock</div><h2>Five questions, answered.</h2></div>
        <p>Every company page is organised around the questions investors actually ask. Plain answers first; the detail is there if you want it.</p></div>
      <div className="qs">
        {[['How is the price moving?', 'The trend, the chart patterns forming, and how often similar setups went up.', 'Overview · Chart patterns · Market'],
          ['What do the models expect?', "A combined outlook from eight models — and whether those forecasts have been reliable lately.", 'Models · Forecast'],
          ['Is it a good business?', "Growth, profits, debt, how it's valued, and how it compares to rivals.", 'Financials · Valuation · Business · Management · Peers · Competition · Industry'],
          ["Who's buying and selling?", 'What the company filed, what insiders and big funds did, and what the news is saying.', 'Company filings · Ownership · Fund flows · Analysts · News · Alt-data'],
          ["What's going on around it?", 'Interest rates, the economy, and the market conditions that move every stock.', 'Macro']].map(([h, p, t], i) => (
          <div className="q" key={i}><div className="i">{i + 1}</div><h4>{h}</h4><p>{p}</p><div className="tabsl">{t}</div></div>))}
      </div>
    </div></section>

    <section style={{ paddingTop: 20 }}><div className="wrap"><div className="trust">
      <div><h4>Based on real history</h4><p>When we say "went up 58% of the time," that comes from thousands of real past cases — and we show how many.</p></div>
      <div><h4>Straight from the source</h4><p>Company filings come directly from the SEC, dated to when they became public. If we don't have something, we say so.</p></div>
      <div><h4>Honest about our limits</h4><p>Our forecasts are re-checked every night. When they stop being reliable, the site tells you.</p></div>
    </div></div></section>

    <footer><div className="wrap">
      <div className="fcols">
        <div className="brand"><div className="logo">QUANTEDGE</div>
          <p>Stock research for everyone — plain-English answers built on real market history and primary sources.</p>
          <div className="by">Designed &amp; built by <b>Dileep Kumar Reddy Kapu</b></div></div>
        <div><h5>Explore</h5><ul><li><a onClick={focusSearch}>Search a stock</a></li><li><a href="#markets">Markets today</a></li><li><a onClick={() => nav('/trackers')}>Trackers</a></li><li><a onClick={focusSearch}>Pattern Lab</a></li></ul></div>
        <div><h5>Learn</h5><ul><li><a href="#how">How it works</a></li><li><a onClick={() => nav('/methodology')}>How it's measured</a></li><li><a onClick={() => nav('/methodology')}>System status</a></li></ul></div>
        <div><h5>Legal</h5><ul><li><a onClick={() => nav('/terms')}>Terms of use</a></li><li><a onClick={() => nav('/privacy')}>Privacy policy</a></li><li><a onClick={() => nav('/disclaimer')}>Disclaimer</a></li><li><a onClick={() => nav('/data-sources')}>Data sources</a></li></ul></div>
        <div><h5>Contact</h5><ul><li><a href="mailto:dileepkreddy5@gmail.com">dileepkreddy5@gmail.com</a></li>
          <li><a href="https://www.linkedin.com/in/kapu-dileep-kumar-reddy-1084301a9/" target="_blank" rel="noopener noreferrer">LinkedIn</a></li>
          <li><a href="mailto:dileepkreddy5@gmail.com?subject=QuantEdge%20data%20issue">Report a data issue</a></li></ul></div>
      </div>
      <div className="fbottom"><span className="copy">© 2026 Dileep Kumar Reddy Kapu. All rights reserved.</span>
        <span>Market data: Polygon.io · Company filings: SEC EDGAR</span><span>Analysis refreshed nightly</span></div>
      <div className="fnote">QuantEdge provides research and educational information only. Nothing on this site is investment, financial, legal or tax advice, or a recommendation to buy or sell any security. Historical patterns and model outputs do not guarantee future results. Investing involves risk, including loss of principal.</div>
    </div></footer>
  </div>);
};
export default Home;
