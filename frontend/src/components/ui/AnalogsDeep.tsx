// ============================================================
// Pattern Lab — Historical Analogs (deep). Every figure from
// /api/v6/patterns/analogs: state vector, forward fan (real
// forward closes of matched episodes), six-horizon distributions
// with SPY excess, conditional filters recomputed server-side.
// ============================================================
import React, { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

const C = { s0:'#100a07', s2:'#241610', b1:'#3a2920', b2:'#4a3428', gold:'#daa520',
  caramel:'#d4956c', cocoa:'#8a7560', dust:'#9d8b7a', latte:'#d4c4b0', cream:'#f4e8d8',
  bull:'#22c55e', bear:'#ef4444', warn:'#f59e0b' };
const mono = "'Fira Code',monospace";
type Dist = { n:number; positive_pct:number; negative_pct?:number; median_pct:number; mean_pct:number;
  p10_pct:number; p25_pct?:number; p75_pct?:number; p90_pct:number; outcome_vol_pct?:number };
type Analog = { ticker:string; start:string; end?:string; regime?:string; volume_slope?:number;
  vol_pctile?:number|null; dist_52w_high?:number|null; similarity_pct:number; trajectory:number[];
  fwd:Record<string, number|null> };
type Res = { ticker:string; as_of:string; window_days:number; episodes:number;
  pre_filter_episodes?:number; filters_applied?:Record<string,string>; insufficient?:string;
  search_scope?:{windows_searched:number; tickers_in_library:number};
  distributions:Record<string,Dist>; base_rates:Record<string,Dist>;
  excess_vs_spy?:Record<string,Dist|null>; analogs:Analog[]; query_trajectory:number[];
  forward_fan?:{sessions:number; n_paths:number; median:number[]; p25:number[]; p75:number[]};
  state_vector?:any; method?:Record<string,string>; episode_date_range?:[string,string]; caveat:string };

const pf = (v:number|null|undefined, d=2) => v==null ? '—' : `${v>=0?'+':''}${v.toFixed(d)}%`;

const FanChart: React.FC<{q:number[]; fan?:Res['forward_fan']; analogs:Analog[]}> = ({q, fan, analogs}) => {
  const W=760, H=280, PAD=14, SPLIT=0.44;
  const qx=(i:number)=>PAD+(i/(q.length-1))*(W*SPLIT-PAD);
  const qmn=Math.min(...q), qmx=Math.max(...q);
  const qy=(v:number)=>H-PAD-((v-qmn)/(qmx-qmn||1))*(H-2*PAD);
  const qpath=q.map((v,i)=>`${i?'L':'M'}${qx(i).toFixed(1)},${qy(v).toFixed(1)}`).join('');
  let fanEl=null;
  if (fan) {
    const all=[...fan.p25,...fan.p75]; const fmn=Math.min(...all,0), fmx=Math.max(...all,0);
    const fx=(i:number)=>W*SPLIT+((i)/(fan.sessions-1))*(W*(1-SPLIT)-PAD);
    const fy=(v:number)=>H-PAD-((v-fmn)/(fmx-fmn||1))*(H-2*PAD);
    const line=(a:number[])=>a.map((v,i)=>`${i?'L':'M'}${fx(i).toFixed(1)},${fy(v).toFixed(1)}`).join('');
    const band=line(fan.p75)+' '+[...fan.p25].reverse().map((v,i)=>`L${fx(fan.sessions-1-i).toFixed(1)},${fy(v).toFixed(1)}`).join(' ')+' Z';
    fanEl=(<>
      <path d={band} fill="rgba(218,165,32,0.13)" stroke="none"/>
      <path d={line(fan.median)} fill="none" stroke={C.gold} strokeWidth={2}/>
      <line x1={W*SPLIT} x2={W-PAD} y1={fy(0)} y2={fy(0)} stroke={C.b2} strokeDasharray="3,3"/>
      <text x={W-PAD-4} y={fy(fan.median[fan.sessions-1])-6} fill={C.gold} fontSize={10}
            fontFamily={mono} textAnchor="end">{pf(fan.median[fan.sessions-1])} med</text>
      <text x={W-PAD-4} y={fy(fan.p75[fan.sessions-1])-6} fill={C.dust} fontSize={9}
            fontFamily={mono} textAnchor="end">p75 {pf(fan.p75[fan.sessions-1])}</text>
      <text x={W-PAD-4} y={fy(fan.p25[fan.sessions-1])+12} fill={C.dust} fontSize={9}
            fontFamily={mono} textAnchor="end">p25 {pf(fan.p25[fan.sessions-1])}</text>
    </>);
  }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{width:'100%', background:'rgba(0,0,0,0.25)', borderRadius:8}}>
      {analogs.slice(0,12).map((a,k)=>(
        <path key={k} d={a.trajectory.map((v,i)=>`${i?'L':'M'}${qx(i*(q.length-1)/(a.trajectory.length-1)).toFixed(1)},${qy(v).toFixed(1)}`).join('')}
              fill="none" stroke={C.cocoa} strokeWidth={0.8} opacity={0.25}/>
      ))}
      <path d={qpath} fill="none" stroke={C.cream} strokeWidth={2.2}/>
      <line x1={W*SPLIT} x2={W*SPLIT} y1={PAD} y2={H-PAD} stroke={C.gold} strokeWidth={1} opacity={0.5}/>
      <text x={W*SPLIT-6} y={PAD+10} fill={C.gold} fontSize={9} fontFamily={mono} textAnchor="end">TODAY</text>
      <text x={PAD} y={PAD+10} fill={C.cocoa} fontSize={9} fontFamily={mono}>NORMALIZED SHAPE (z)</text>
      <text x={W*SPLIT+6} y={PAD+10} fill={C.cocoa} fontSize={9} fontFamily={mono}>
        WHAT FOLLOWED — {fan ? `${fan.n_paths} REAL FORWARD PATHS (%)` : 'no fan'}</text>
      {fanEl}
    </svg>);
};

const SV: React.FC<{sv:any}> = ({sv}) => {
  if (!sv) return null;
  const cell=(k:string,v:string,sub?:string,tone?:string)=>(
    <div key={k} style={{flex:'1 1 140px', background:'rgba(0,0,0,0.2)', border:`1px solid ${C.b1}`,
                         borderRadius:6, padding:'10px 12px'}}>
      <div style={{fontFamily:mono, fontSize:8, letterSpacing:1.3, color:C.cocoa, marginBottom:5}}>{k}</div>
      <div style={{fontFamily:mono, fontSize:14, color:tone||C.cream, fontWeight:700}}>{v}</div>
      {sub && <div style={{fontFamily:mono, fontSize:9, color:C.dust, marginTop:3}}>{sub}</div>}
    </div>);
  const p=sv.price||{}, m=sv.momentum||{}, vo=sv.volatility||{}, vl=sv.volume||{};
  return (
    <div style={{display:'flex', flexWrap:'wrap', gap:8, marginBottom:14}}>
      {cell('TREND', (p.trend||'—').toUpperCase(), `slope ${pf(p.slope_20d_ann_pct,1)} ann · ${p.acceleration}`,
            p.trend==='up'?C.bull:C.bear)}
      {cell('MOMENTUM 5/20/60D', `${pf(m['5d_pct'],1)} / ${pf(m['20d_pct'],1)} / ${pf(m['60d_pct'],1)}`)}
      {cell('VOLATILITY', `${vo.realized_21d_ann_pct??'—'}% ann`, `${vo.percentile}th pctile · ${vo.direction}`,
            vo.direction==='rising'?C.warn:C.latte)}
      {cell('VOLUME', `${vl.percentile??'—'}th pctile`, `${vl.trend} · p×v corr ${vl.price_volume_corr_20d}`)}
      {cell('52W POSITION', pf(p.vs_52w_high_pct,1)+' vs high', `${pf(p.vs_52w_low_pct,1)} vs low · dd ${pf(p.drawdown_pct,1)}`)}
      {sv.multi_scale && cell('MULTI-SCALE', sv.multi_scale.verdict,
        Object.entries(sv.multi_scale.signals).map(([k,v])=>`${k}:${String(v)[0]}`).join(' '),
        sv.multi_scale.verdict==='ALIGNED'?C.bull:sv.multi_scale.verdict==='CONFLICTED'?C.warn:C.latte)}
      {cell('VS MOVING AVGS', `${pf(p.vs_sma20_pct,1)} / ${pf(p.vs_sma50_pct,1)}`, `sma20 / sma50 · sma200 ${pf(p.vs_sma200_pct,1)}`)}
    </div>);
};

const FILTERS: {group:string; param:string; opts:[string,string][]}[] = [
  {group:'VOLUME', param:'volume', opts:[['rising','RISING'],['falling','FALLING']]},
  {group:'VOLATILITY', param:'vola', opts:[['high','HIGH'],['low','LOW']]},
  {group:'REGIME', param:'regime', opts:[['BULL_LOW_VOL','BULL·LOW'],['BULL_HIGH_VOL','BULL·HIGH'],['BEAR_LOW_VOL','BEAR·LOW'],['BEAR_HIGH_VOL','BEAR·HIGH']]},
  {group:'52W', param:'extreme', opts:[['near_high','NEAR HIGH'],['near_low','NEAR LOW']]},
  {group:'SCOPE', param:'scope', opts:[['same','SAME STOCK'],['cross','CROSS-STOCK']]},
];

const AnalogsDeep: React.FC<{ticker:string}> = ({ticker}) => {
  const [w,setW]=useState<20|60>(20);
  const [fl,setFl]=useState<Record<string,string>>({});
  const [res,setRes]=useState<Res|null>(null);
  const [busy,setBusy]=useState(false); const [err,setErr]=useState('');
  useEffect(()=>{ let dead=false;
    (async()=>{ setBusy(true); setErr('');
      try{ const qs=new URLSearchParams({window:String(w),...fl}).toString();
        const r=await api.get(`/api/v6/patterns/analogs/${ticker}?${qs}`);
        if(!dead) setRes(r.data);
      }catch(e:any){ if(!dead) setErr(e?.response?.data?.detail||'query failed'); }
      finally{ if(!dead) setBusy(false); } })();
    return()=>{dead=true}; },[ticker,w,fl]);
  const toggle=(param:string,val:string)=>setFl(f=>{const n={...f}; if(n[param]===val) delete n[param]; else n[param]=val; return n;});
  return (
    <div>
      <div style={{display:'flex', flexWrap:'wrap', gap:14, alignItems:'center', marginBottom:14}}>
        {([20,60] as const).map(x=>(
          <button key={x} onClick={()=>setW(x)} style={{fontFamily:mono, fontSize:9.5, letterSpacing:1.2,
            padding:'6px 12px', background:w===x?'rgba(218,165,32,0.1)':'none',
            border:`1px solid ${w===x?C.gold:C.b1}`, borderRadius:4, color:w===x?C.gold:C.dust, cursor:'pointer'}}>
            {x}D SHAPE</button>))}
        {FILTERS.map(g=>(
          <div key={g.group} style={{display:'flex', gap:4, alignItems:'center'}}>
            <span style={{fontFamily:mono, fontSize:8, letterSpacing:1, color:C.cocoa}}>{g.group}</span>
            {g.opts.map(([v,l])=>(
              <button key={v} onClick={()=>toggle(g.param,v)} style={{fontFamily:mono, fontSize:8.5,
                padding:'5px 9px', background:fl[g.param]===v?'rgba(218,165,32,0.12)':'none',
                border:`1px solid ${fl[g.param]===v?C.caramel:C.b1}`, borderRadius:3,
                color:fl[g.param]===v?C.caramel:C.dust, cursor:'pointer'}}>{l}</button>))}
          </div>))}
      </div>
      {busy && <div style={{fontFamily:mono, fontSize:11, color:C.dust}}>
        searching {res?.search_scope?.windows_searched?.toLocaleString()||'the library'}…</div>}
      {err && <div style={{fontFamily:mono, fontSize:11, color:C.warn}}>{err}</div>}
      {res && res.insufficient && (
        <div style={{fontFamily:mono, fontSize:12, color:C.warn, padding:20, border:`1px solid ${C.b1}`, borderRadius:8}}>
          INSUFFICIENT EPISODES — {res.insufficient} ({res.pre_filter_episodes} before filters)</div>)}
      {res && !res.insufficient && (<>
        <SV sv={res.state_vector}/>
        <div style={{display:'grid', gridTemplateColumns:'minmax(340px,1.5fr) minmax(320px,1fr)', gap:14}}>
          <div style={{background:C.s2, border:`1px solid ${C.b1}`, borderRadius:10, padding:18}}>
            <div style={{fontFamily:mono, fontSize:9, letterSpacing:1.5, color:C.cocoa, marginBottom:10}}>
              {res.ticker} — LAST {res.window_days} SESSIONS → {res.episodes} NON-OVERLAPPING EPISODES
              {res.pre_filter_episodes && res.episodes<res.pre_filter_episodes ? ` (OF ${res.pre_filter_episodes} UNFILTERED)` : ''}
            </div>
            <FanChart q={res.query_trajectory} fan={res.forward_fan} analogs={res.analogs}/>
          </div>
          <div style={{background:C.s2, border:`1px solid ${C.b1}`, borderRadius:10, padding:'14px 6px 6px', overflowX:'auto'}}>
            <div style={{fontFamily:mono, fontSize:9, letterSpacing:1.5, color:C.cocoa, padding:'0 12px 8px'}}>
              FORWARD DISTRIBUTIONS — n · %POS · MED · P25/P75 · σ · VS SPY</div>
            <table style={{width:'100%', borderCollapse:'collapse', fontFamily:mono, fontSize:10}}>
              <tbody>{Object.entries(res.distributions).map(([h,d])=>{ if(!d) return null;
                const ex=res.excess_vs_spy?.[h];
                return (<tr key={h} style={{borderBottom:`1px solid rgba(58,41,32,0.5)`}}>
                  <td style={{padding:'8px 10px', color:C.cream, fontWeight:700}}>+{h}</td>
                  <td style={{padding:'8px 6px', color:C.cocoa}}>n={d.n}</td>
                  <td style={{padding:'8px 6px', color:d.positive_pct>=50?C.bull:C.bear}}>{d.positive_pct}%</td>
                  <td style={{padding:'8px 6px', color:C.latte}}>{pf(d.median_pct)}</td>
                  <td style={{padding:'8px 6px', color:C.dust, fontSize:9}}>{pf(d.p25_pct)}/{pf(d.p75_pct)}</td>
                  <td style={{padding:'8px 6px', color:C.dust, fontSize:9}}>σ{d.outcome_vol_pct}</td>
                  <td style={{padding:'8px 6px', fontSize:9,
                       color:ex==null?C.cocoa:ex.positive_pct>=50?C.bull:C.bear}}>
                    {ex==null?'—':`${ex.positive_pct}% beat·${pf(ex.median_pct)}`}</td>
                </tr>);})}</tbody>
            </table>
          </div>
        </div>
        <div style={{background:C.s2, border:`1px solid ${C.b1}`, borderRadius:10, padding:'14px 10px', marginTop:14, overflowX:'auto'}}>
          <div style={{fontFamily:mono, fontSize:9, letterSpacing:1.5, color:C.cocoa, padding:'0 8px 8px'}}>CLOSEST EPISODES</div>
          <table style={{width:'100%', borderCollapse:'collapse', fontFamily:mono, fontSize:10}}>
            <thead><tr>{['TICKER','FORMED','SIM','REGIME','VOL SLOPE','VOL PCTL','VS 52W HI','+5D','+20D','+60D','+120D'].map(h=>(
              <th key={h} style={{textAlign:'left', padding:'5px 8px', color:C.cocoa, fontSize:8.5, letterSpacing:1,
                                  fontWeight:500, borderBottom:`1px solid ${C.b1}`}}>{h}</th>))}</tr></thead>
            <tbody>{res.analogs.slice(0,12).map(a=>(
              <tr key={a.ticker+a.start}>
                <td style={{padding:'6px 8px', color:C.cream, fontWeight:700}}>{a.ticker}</td>
                <td style={{padding:'6px 8px', color:C.dust}}>{a.start}{a.end?` → ${a.end}`:''}</td>
                <td style={{padding:'6px 8px', color:C.gold}}>{a.similarity_pct}%</td>
                <td style={{padding:'6px 8px', color:C.latte, fontSize:9}}>{(a.regime||'—').replace(/_/g,' ')}</td>
                <td style={{padding:'6px 8px', color:(a.volume_slope??0)>=0?C.bull:C.bear}}>
                  {a.volume_slope!=null?(a.volume_slope>=0?'+':'')+a.volume_slope.toFixed(2):'—'}</td>
                <td style={{padding:'6px 8px', color:C.dust}}>{a.vol_pctile!=null?Math.round(a.vol_pctile*100)+'%':'—'}</td>
                <td style={{padding:'6px 8px', color:C.dust}}>{a.dist_52w_high!=null?pf(a.dist_52w_high,1):'—'}</td>
                {(['5d','20d','60d','120d'] as const).map(h=>(
                  <td key={h} style={{padding:'6px 8px', color:a.fwd[h]==null?C.cocoa:(a.fwd[h]!>=0?C.bull:C.bear)}}>
                    {a.fwd[h]==null?'—':pf(a.fwd[h])}</td>))}
              </tr>))}</tbody>
          </table>
        </div>
        <div style={{marginTop:12, fontFamily:mono, fontSize:10, color:C.cocoa, lineHeight:1.7}}>
          {res.search_scope && `SCOPE: ${res.search_scope.windows_searched.toLocaleString()} windows · ${res.search_scope.tickers_in_library} tickers · `}
          {res.method && `${res.method.normalization} · ${res.method.stage1} → ${res.method.stage2} · ${res.method.dedup}. `}
          {res.episode_date_range && `Episodes ${res.episode_date_range[0]} → ${res.episode_date_range[1]}. `}
          {res.caveat} DESCRIPTIVE RESULT — overlapping market history; no significance claimed.
        </div>
      </>)}
    </div>
  );
};
export default AnalogsDeep;
