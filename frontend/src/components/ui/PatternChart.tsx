// Pattern Chart — the picture half of Pattern Lab. Everything drawn comes
// from /patterns/chart; every odds figure is a universe-measured scorecard
// at the chosen horizon, or says it isn't measured yet.
import React, { useEffect, useMemo, useState } from 'react';
import { api } from '../../auth/authStore';
const C={s0:'#100a07',s2:'#241610',b1:'#3a2920',b2:'#4a3428',gold:'#daa520',caramel:'#d4956c',cocoa:'#8a7560',
  dust:'#9d8b7a',latte:'#d4c4b0',cream:'#f4e8d8',bull:'#22c55e',bear:'#ef4444',warn:'#f59e0b',blue:'#60a5fa'};
const mono="'Fira Code',monospace";
const LABEL:Record<string,string>={head_shoulders:'Head & Shoulders',inv_head_shoulders:'Inv. Head & Shoulders',double_top:'Double Top',
  double_bottom:'Double Bottom',triple_top:'Triple Top',triple_bottom:'Triple Bottom',ascending_triangle:'Ascending Triangle',
  descending_triangle:'Descending Triangle',symmetrical_triangle:'Symmetrical Triangle',rectangle:'Rectangle',rising_wedge:'Rising Wedge',
  falling_wedge:'Falling Wedge',doji:'Doji',hammer:'Hammer',hanging_man:'Hanging Man',bullish_engulfing:'Bullish Engulfing',
  bearish_engulfing:'Bearish Engulfing',bullish_harami:'Bullish Harami',bearish_harami:'Bearish Harami',morning_star:'Morning Star',
  evening_star:'Evening Star',three_white_soldiers:'Three White Soldiers',three_black_crows:'Three Black Crows'};
const BEAR_F=new Set(['head_shoulders','double_top','triple_top','rising_wedge','descending_triangle']);
type Occ={family:string;name:string;i:number;direction?:string;points?:{i:number;price:number}[];confirm_i?:number;scorecard:any};
const pf=(v:any,d=1)=>v==null?'—':`${v>=0?'+':''}${Number(v).toFixed(d)}%`;
let BASE_FALLBACK:any=null;   // analog-library base rate at the current horizon, set per render
const edgeOf=(o:Occ)=>{const s=o.scorecard; const st=s?.all||s; const b=s?.base||BASE_FALLBACK; if(!st||st.positive_pct==null)return null;
  return b?.positive_pct!=null? st.positive_pct-b.positive_pct : null;};
const ScoreLine:React.FC<{o:Occ;hz:string}>=({o,hz})=>{const s=o.scorecard; const st=s?.all||s;
  if(!st||st.positive_pct==null) return <span style={{color:C.cocoa}}>{o.family==='candlestick'?'not yet measured (nightly scan pending)':'not enough history'}</span>;
  const e=edgeOf(o); return (<span><span style={{color:st.positive_pct>=50?C.bull:C.bear}}>{st.positive_pct}% up</span>
    <span style={{color:C.dust}}> · med {pf(st.median_pct,2)} · n={st.n?.toLocaleString()}</span>
    {e!=null&&<span style={{color:Math.abs(e)<2?C.cocoa:e>0?C.bull:C.bear}}> · {e>=0?'+':''}{e.toFixed(1)} vs base</span>}
    {o.scorecard?.follow_through_pct!=null&&<span style={{color:C.dust}}> · follow-through {o.scorecard.follow_through_pct}%</span>}
    <span style={{color:C.cocoa}}> @ {hz}</span></span>);};

const PatternChart:React.FC<{ticker:string}>=({ticker})=>{
  const [hz,setHz]=useState('3m'); const [d,setD]=useState<any>(null); const [err,setErr]=useState('');
  const [hover,setHover]=useState<Occ|null>(null); const [show,setShow]=useState<{f:boolean;c:boolean}>({f:true,c:true});
  useEffect(()=>{let dead=false;(async()=>{setD(null);setErr('');
    try{const r=await api.get(`/api/v6/patterns/chart/${ticker}?horizon=${hz}`);if(!dead)setD(r.data);}
    catch(e:any){if(!dead)setErr(e?.response?.data?.detail||'chart unavailable');}})();return()=>{dead=true};},[ticker,hz]);
  const geom=useMemo(()=>{ if(!d) return null;
    const W=1180,H=430,VH=70,PL=8,PR=110,PT=14,PB=8; const cs=d.candles; const n=cs.length;
    const fan=d.analog?.forward_fan; const F=fan?Math.min(fan.sessions,d.outcome_sessions):0;
    const last=cs[n-1].c; const fp=(p:number)=>last*(1+p/100);
    const lows=cs.map((x:any)=>x.l), highs=cs.map((x:any)=>x.h);
    let mn=Math.min(...lows), mx=Math.max(...highs);
    if(fan&&F){ mn=Math.min(mn,...fan.p25.slice(0,F).map(fp)); mx=Math.max(mx,...fan.p75.slice(0,F).map(fp)); }
    mn=Math.min(mn,d.low_52w); mx=Math.max(mx,d.high_52w); const pad=(mx-mn)*0.05; mn-=pad; mx+=pad;
    const cols=n+F+2; const cw=(W-PL-PR)/cols; const x=(i:number)=>PL+(i+0.5)*cw;
    const y=(p:number)=>PT+(1-(p-mn)/(mx-mn))*(H-PT-PB-VH);
    const vmax=Math.max(...cs.map((c:any)=>c.v))||1; const vy=(v:number)=>H-PB-(v/vmax)*(VH-6);
    return {W,H,VH,PL,PR,PT,PB,n,F,cw,x,y,vy,fp,last,fan};},[d]);
  if(err)return <div style={{fontFamily:mono,fontSize:11,color:C.warn}}>{err}</div>;
  if(!d||!geom)return <div style={{fontFamily:mono,fontSize:11,color:C.dust}}>drawing {ticker}…</div>;
  const {W,H,VH,PL,PR,n,F,cw,x,y,vy,fp,last,fan}=geom;
  const occs:Occ[]=[...(show.f?d.formations:[]),...(show.c?d.candlesticks:[])];
  const catalog=Array.from(new Map<string,Occ>([...d.formations,...d.candlesticks].map((o:Occ)=>[o.name,o] as [string,Occ])).values())
    .sort((a:any,b:any)=>Math.abs(edgeOf(b)??-99)-Math.abs(edgeOf(a)??-99));
  const cur=d.current_match||[]; const dist=d.analog?.distribution; const base=d.analog?.base; BASE_FALLBACK=base;
  const line=(arr:(number|null)[],col:string,dash?:string)=>{let p='';arr.forEach((v,i)=>{if(v==null)return;p+=`${p?'L':'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`;});
    return <path d={p} fill="none" stroke={col} strokeWidth={1} strokeDasharray={dash} opacity={.7}/>;};
  return (<div>
    {/* header: horizon + you-are-here */}
    <div style={{display:'flex',flexWrap:'wrap',gap:10,alignItems:'center',marginBottom:10}}>
      <span style={{fontFamily:mono,fontSize:10,letterSpacing:2.5,color:C.cocoa}}>PATTERN CHART · {d.ticker}</span>
      {['1w','1m','3m','6m','12m'].map(h=><button key={h} onClick={()=>setHz(h)} style={{fontFamily:mono,fontSize:9.5,letterSpacing:1.2,padding:'6px 12px',
        background:hz===h?'rgba(218,165,32,0.12)':'none',border:`1px solid ${hz===h?C.gold:C.b1}`,borderRadius:4,color:hz===h?C.gold:C.dust,cursor:'pointer'}}>{h.toUpperCase()}</button>)}
      <span style={{fontFamily:mono,fontSize:8.5,color:C.cocoa}}>outcome window {d.outcome_sessions} sessions</span>
      <span style={{flex:1}}/>
      {(['f','c'] as const).map(k=><button key={k} onClick={()=>setShow(s=>({...s,[k]:!s[k]}))} style={{fontFamily:mono,fontSize:8.5,padding:'5px 9px',
        background:show[k]?'rgba(218,165,32,0.08)':'none',border:`1px solid ${show[k]?C.caramel:C.b1}`,borderRadius:3,color:show[k]?C.caramel:C.dust,cursor:'pointer'}}>
        {k==='f'?'FORMATIONS':'CANDLESTICKS'}</button>)}
    </div>
    <div style={{background:C.s2,border:`1px solid ${cur.length?C.gold:C.b1}`,borderRadius:8,padding:'10px 14px',marginBottom:10,fontFamily:mono,fontSize:11}}>
      {cur.length?(<><span style={{color:C.gold}}>NOW · </span><span style={{color:C.cream}}>{d.ticker} completed {cur.map((o:Occ)=>LABEL[o.name]||o.name).join(', ')} within the last 3 sessions. </span>
        <ScoreLine o={cur[0]} hz={hz.toUpperCase()}/></>):(<span style={{color:C.dust}}>No catalog pattern completed in the last 3 sessions.</span>)}
    </div>
    {dist&&(()=>{const cell=(k:string,v:string,col:string,sub?:string)=>(
        <div key={k} style={{flex:'1 1 120px',background:'rgba(0,0,0,0.22)',border:`1px solid ${C.b1}`,borderRadius:6,padding:'9px 12px'}}>
          <div style={{fontFamily:mono,fontSize:8,letterSpacing:1.3,color:C.cocoa,marginBottom:4}}>{k}</div>
          <div style={{fontFamily:mono,fontSize:16,fontWeight:700,color:col}}>{v}</div>
          {sub&&<div style={{fontFamily:mono,fontSize:8.5,color:C.dust,marginTop:2}}>{sub}</div>}</div>);
      const e=base?dist.positive_pct-base.positive_pct:null;
      return (<div style={{background:C.s2,border:`1px solid ${C.b1}`,borderRadius:8,padding:'12px 14px',marginBottom:10}}>
        <div style={{fontFamily:mono,fontSize:9,letterSpacing:1.5,color:C.cocoa,marginBottom:8}}>
          HISTORICAL OUTLOOK · WHAT FOLLOWED {d.analog.episodes} SIMILAR SHAPES OVER {d.outcome_sessions} SESSIONS · ENVELOPE, NOT A FORECAST</div>
        <div style={{display:'flex',flexWrap:'wrap',gap:8}}>
          {cell('PROBABILITY UP',`${dist.positive_pct}%`,dist.positive_pct>=50?C.bull:C.bear,`base rate ${base?.positive_pct??'?'}%${e!=null?` · ${e>=0?'+':''}${e.toFixed(1)} edge`:''}`)}
          {cell('TYPICAL (MEDIAN)',pf(dist.median_pct,2),dist.median_pct>=0?C.bull:C.bear,`expected value ${pf(dist.mean_pct,2)}`)}
          {cell('UPSIDE CASE (P75)',pf(dist.p75_pct),C.bull,`best 10%: ${pf(dist.p90_pct)}`)}
          {cell('DOWNSIDE CASE (P25)',pf(dist.p25_pct),C.bear,`worst 10%: ${pf(dist.p10_pct)}`)}
          {cell('SPREAD (σ)',`${dist.outcome_vol_pct??'—'}%`,C.latte,`n=${dist.n} non-overlapping episodes`)}
        </div></div>);})()}
    {/* chart */}
    <div style={{background:'rgba(0,0,0,0.28)',border:`1px solid ${C.b1}`,borderRadius:10,padding:6,position:'relative'}}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{width:'100%',display:'block'}} onMouseLeave={()=>setHover(null)}>
        {/* 52w lines */}
        {[['52W HIGH',d.high_52w,C.caramel],['52W LOW',d.low_52w,C.blue]].map(([lab,p,col]:any)=>(<g key={lab}>
          <line x1={PL} x2={W-PR} y1={y(p)} y2={y(p)} stroke={col} strokeDasharray="4,4" opacity={.5}/>
          <text x={W-PR+4} y={y(p)+3} fill={col} fontSize={8} fontFamily={mono}>{lab} {p.toFixed(2)}</text></g>))}
        {d.sma20&&line(d.sma20,C.gold)}{d.sma50&&line(d.sma50,C.caramel,'3,3')}{d.sma200&&line(d.sma200,C.dust,'1,3')}
        {/* volume */}
        {d.candles.map((c:any,i:number)=><rect key={'v'+i} x={x(i)-cw*0.35} y={vy(c.v)} width={cw*0.7} height={H-8-vy(c.v)} fill={c.c>=c.o?C.bull:C.bear} opacity={.18}/>)}
        {/* candles */}
        {d.candles.map((c:any,i:number)=>{const up=c.c>=c.o;const col=up?C.bull:C.bear;return (<g key={i}>
          <line x1={x(i)} x2={x(i)} y1={y(c.h)} y2={y(c.l)} stroke={col} strokeWidth={1}/>
          <rect x={x(i)-cw*0.35} y={y(Math.max(c.o,c.c))} width={Math.max(1,cw*0.7)} height={Math.max(1,Math.abs(y(c.o)-y(c.c)))} fill={up?col:C.s0} stroke={col} strokeWidth={1}/></g>);})}
        {/* formations: points + neckline-ish polyline */}
        {show.f&&d.formations.map((f:Occ,k:number)=>{const pts=f.points||[];const col=BEAR_F.has(f.name)?C.bear:C.bull;
          return (<g key={'f'+k} onMouseEnter={()=>setHover(f)} style={{cursor:'pointer'}}>
            <polyline points={pts.map(p=>`${x(p.i)},${y(p.price)}`).join(' ')} fill="none" stroke={col} strokeWidth={1.5} opacity={.9}/>
            {pts.map((p,j)=><circle key={j} cx={x(p.i)} cy={y(p.price)} r={3} fill={col}/>)}
            {f.confirm_i!=null&&<line x1={x(f.confirm_i)} x2={x(f.confirm_i)} y1={y(pts[pts.length-1].price)-16} y2={y(pts[pts.length-1].price)+16} stroke={col} strokeDasharray="2,2"/>}
            <text x={x(pts[0].i)} y={y(Math.max(...pts.map(p=>p.price)))-8} fill={col} fontSize={9} fontFamily={mono}>{LABEL[f.name]}</text></g>);})}
        {/* candlestick markers */}
        {show.c&&d.candlesticks.map((o:Occ,k:number)=>{const c=d.candles[o.i];if(!c)return null;const up=o.direction==='bullish',neu=o.direction==='neutral';
          const col=neu?C.gold:up?C.bull:C.bear; const yy=up?y(c.l)+12:y(c.h)-12;
          return (<g key={'c'+k} onMouseEnter={()=>setHover(o)} style={{cursor:'pointer'}}>
            {neu?<circle cx={x(o.i)} cy={y(c.h)-8} r={3} fill="none" stroke={col}/>:
              <polygon points={up?`${x(o.i)-4},${yy+6} ${x(o.i)+4},${yy+6} ${x(o.i)},${yy-2}`:`${x(o.i)-4},${yy-6} ${x(o.i)+4},${yy-6} ${x(o.i)},${yy+2}`} fill={col} opacity={.9}/>}
          </g>);})}
        {/* forward envelope */}
        {fan&&F>0&&(()=>{const xs=(j:number)=>x(n-1+j);
          const band=fan.p75.slice(0,F).map((v:number,j:number)=>`${j?'L':'M'}${xs(j)},${y(fp(v))}`).join('')+' '+[...fan.p25.slice(0,F)].reverse().map((v:number,j:number)=>`L${xs(F-1-j)},${y(fp(v))}`).join(' ')+' Z';
          const med=fan.median.slice(0,F).map((v:number,j:number)=>`${j?'L':'M'}${xs(j)},${y(fp(v))}`).join('');
          return (<g><line x1={x(n-1)} x2={x(n-1)} y1={14} y2={H-VH-8} stroke={C.gold} opacity={.5}/>
            <text x={x(n-1)+4} y={22} fill={C.gold} fontSize={8} fontFamily={mono}>TODAY → what followed ({fan.n_paths} real paths)</text>
            <path d={band} fill="rgba(218,165,32,0.13)"/><path d={med} fill="none" stroke={C.gold} strokeWidth={2}/>
            <text x={xs(F-1)+4} y={y(fp(fan.median[F-1]))+3} fill={C.gold} fontSize={9} fontFamily={mono}>med {pf(fan.median[F-1])}</text>
            <text x={xs(F-1)+4} y={y(fp(fan.p75[F-1]))-2} fill={C.dust} fontSize={8} fontFamily={mono}>p75 {pf(fan.p75[F-1])}</text>
            <text x={xs(F-1)+4} y={y(fp(fan.p25[F-1]))+10} fill={C.dust} fontSize={8} fontFamily={mono}>p25 {pf(fan.p25[F-1])}</text></g>);})()}
        {d.candles.map((c:any,i:number)=>{if(i<20)return null;const avg=d.candles.slice(i-20,i).reduce((a:number,z:any)=>a+z.v,0)/20;
          return c.v>=2.5*avg?<g key={'cl'+i}><rect x={x(i)-cw*0.35} y={vy(c.v)} width={cw*0.7} height={H-8-vy(c.v)} fill={C.gold} opacity={.55}/>
            <text x={x(i)} y={vy(c.v)-3} fill={C.gold} fontSize={7} fontFamily={mono} textAnchor="middle">{(c.v/avg).toFixed(1)}×</text></g>:null;})}
        {(()=>{const cands=[...d.formations,...d.candlesticks].filter((o:Occ)=>{const st=o.scorecard?.all||o.scorecard;return st&&st.median_pct!=null&&st.p25_pct!=null;}).sort((a:Occ,b:Occ)=>b.i-a.i);
          const o=cands[0]; if(!o) return null; const st=o.scorecard?.all||o.scorecard; const i0=o.confirm_i??o.i; const p0=d.candles[i0]?.c; if(!p0) return null;
          const K=d.outcome_sessions; const col=st.positive_pct>=50?C.bull:C.bear; const pt=(pct:number)=>y(p0*(1+pct/100));
          return (<g opacity={.85}>
            <line x1={x(i0)} x2={x(i0+K)} y1={y(p0)} y2={pt(st.median_pct)} stroke={col} strokeWidth={1.5} strokeDasharray="5,3"/>
            <line x1={x(i0)} x2={x(i0+K)} y1={y(p0)} y2={pt(st.p75_pct)} stroke={col} strokeWidth={1} strokeDasharray="2,3"/>
            <line x1={x(i0)} x2={x(i0+K)} y1={y(p0)} y2={pt(st.p25_pct)} stroke={col} strokeWidth={1} strokeDasharray="2,3"/>
            <text x={x(i0)+4} y={y(p0)-10} fill={col} fontSize={8} fontFamily={mono}>{LABEL[o.name]} cone · {st.positive_pct}% up · med {pf(st.median_pct,1)} @{K}s (n={st.n})</text></g>);})()}
        <text x={PL+4} y={H-VH-14} fill={C.cocoa} fontSize={8} fontFamily={mono}>{d.candles[0].d} → {d.candles[n-1].d} · last {last}</text>
      </svg>
      {hover&&<div style={{position:'absolute',left:14,top:14,background:'rgba(16,10,7,0.95)',border:`1px solid ${C.gold}`,borderRadius:6,padding:'8px 12px',fontFamily:mono,fontSize:10,maxWidth:460}}>
        <div style={{color:C.cream,fontWeight:700}}>{LABEL[hover.name]||hover.name} <span style={{color:C.cocoa,fontWeight:400}}>· {hover.family} · {d.candles[hover.i]?.d}</span></div>
        <div style={{marginTop:4}}><ScoreLine o={hover} hz={hz.toUpperCase()}/></div>
        {hover.scorecard?.by_regime&&<div style={{marginTop:4,color:C.dust}}>by regime: {Object.entries(hover.scorecard.by_regime).map(([k,v]:any)=>`${k.replace(/_/g,' ')} ${v?v.positive_pct+'%':'n/a'}`).join(' · ')}</div>}
        {hover.scorecard?.by_period&&<div style={{color:C.dust}}>2021–24 {hover.scorecard.by_period['2021-2024']?.positive_pct??'n/a'}% · 2025+ {hover.scorecard.by_period['2025+']?.positive_pct??'n/a'}%</div>}
      </div>}
    </div>
    {/* catalog in window, ranked by measured edge */}
    {(()=>{const cs=d.candles; const H20=d.outcome_sessions;
      const avg=(i:number,k:number)=>cs.slice(Math.max(0,i-k),i).reduce((a:number,z:any)=>a+z.v,0)/Math.max(1,Math.min(k,i));
      const recent=[...d.formations,...d.candlesticks].filter((o:Occ)=>o.i>=n-25&&cs[o.i]).sort((a:Occ,b:Occ)=>b.i-a.i).slice(0,8);
      const ago=(i:number)=>n-1-i;
      const real=(i:number,k:number)=>(i+k<n)?(cs[i+k].c/cs[i].c-1)*100:null;
      const upShare=(()=>{let u=0,t=0;for(let i=Math.max(1,n-20);i<n;i++){t+=cs[i].v;if(cs[i].c>=cs[i-1].c)u+=cs[i].v;}return t?u/t*100:null;})();
      const obv=(()=>{let o=0;const a:number[]=[];for(let i=1;i<n;i++){o+=cs[i].c>cs[i-1].c?cs[i].v:cs[i].c<cs[i-1].c?-cs[i].v:0;a.push(o);}return a;})();
      const obvTrend=obv.length>20?(obv[obv.length-1]>obv[obv.length-21]?'rising':'falling'):'—';
      const v20=avg(n,20), v60=avg(n,60), vtoday=cs[n-1].v;
      return (<>
      <div style={{display:'grid',gridTemplateColumns:'minmax(340px,1.6fr) minmax(260px,1fr)',gap:12,marginTop:12}}>
        <div style={{background:C.s2,border:`1px solid ${C.b1}`,borderRadius:10,padding:'12px 14px'}}>
          <div style={{fontFamily:mono,fontSize:9,letterSpacing:1.5,color:C.cocoa,marginBottom:8}}>RECENT PATTERNS · WHAT THE ODDS SAID vs WHAT ACTUALLY HAPPENED</div>
          {recent.length===0&&<div style={{fontFamily:mono,fontSize:10.5,color:C.cocoa}}>no catalog pattern in the last 25 sessions</div>}
          {recent.map((o:Occ,k:number)=>{const st=o.scorecard?.all||o.scorecard; const since=real(o.i,ago(o.i)); const at5=real(o.i,5); const atH=real(o.i,H20);
            const vr=cs[o.i].v/Math.max(1,avg(o.i,20)); const bull=o.direction?o.direction==='bullish':!BEAR_F.has(o.name);
            return (<div key={k} style={{display:'grid',gridTemplateColumns:'86px 170px 1fr',gap:10,padding:'7px 0',borderBottom:'1px solid rgba(58,41,32,0.45)',fontFamily:mono,fontSize:10.5,alignItems:'center'}}>
              <div><div style={{color:C.cream}}>{cs[o.i].d.slice(5)}</div><div style={{color:C.cocoa,fontSize:8.5}}>{ago(o.i)===0?'today':`${ago(o.i)} sessions ago`}</div></div>
              <div><div style={{color:bull?C.bull:o.direction==='neutral'?C.gold:C.bear}}>{LABEL[o.name]||o.name}</div>
                <div style={{color:vr>=1.5?C.gold:C.cocoa,fontSize:8.5}}>volume {vr.toFixed(1)}× 20d avg{vr>=1.5?' · confirmed':''}</div></div>
              <div style={{fontSize:10}}>
                <div style={{color:C.dust}}>odds @{hz.toUpperCase()}: {st?.positive_pct!=null?<span style={{color:st.positive_pct>=50?C.bull:C.bear}}>{st.positive_pct}% up · med {pf(st.median_pct,2)}</span>:<span style={{color:C.cocoa}}>not yet measured</span>}</div>
                <div>actual: {since!=null?<span style={{color:since>=0?C.bull:C.bear}}>{pf(since,2)} since</span>:'—'}
                  {at5!=null&&<span style={{color:at5>=0?C.bull:C.bear}}> · {pf(at5,2)} @5s</span>}
                  {atH!=null?<span style={{color:atH>=0?C.bull:C.bear}}> · {pf(atH,2)} @{H20}s</span>:<span style={{color:C.cocoa}}> · @{H20}s pending</span>}</div>
              </div></div>);})}
        </div>
        <div style={{background:C.s2,border:`1px solid ${C.b1}`,borderRadius:10,padding:'12px 14px'}}>
          <div style={{fontFamily:mono,fontSize:9,letterSpacing:1.5,color:C.cocoa,marginBottom:8}}>VOLUME</div>
          {[['TODAY vs 20D AVG',`${(vtoday/Math.max(1,v20)).toFixed(2)}×`,vtoday>=1.5*v20?C.gold:C.latte],
            ['20D vs 60D AVG',`${(v20/Math.max(1,v60)).toFixed(2)}×`,v20>v60?C.bull:C.dust,],
            ['UP-DAY VOLUME SHARE (20D)',upShare!=null?`${upShare.toFixed(0)}%`:'—',upShare!=null&&upShare>=55?C.bull:upShare!=null&&upShare<=45?C.bear:C.latte],
            ['OBV TREND (20D)',obvTrend.toUpperCase(),obvTrend==='rising'?C.bull:obvTrend==='falling'?C.bear:C.latte],
            ['CLIMAX DAYS IN WINDOW',String(cs.filter((c:any,i:number)=>i>=20&&c.v>=2.5*avg(i,20)).length),C.gold]].map(([k,v,col]:any)=>(
            <div key={k} style={{display:'flex',justifyContent:'space-between',fontFamily:mono,fontSize:11,padding:'5px 0'}}><span style={{color:C.dust}}>{k}</span><span style={{color:col,fontWeight:700}}>{v}</span></div>))}
          <div style={{fontFamily:mono,fontSize:8.5,color:C.cocoa,marginTop:8,lineHeight:1.6}}>Up-day share and OBV are accumulation proxies (true buy/sell split is not derivable from daily bars). Climax = volume ≥ 2.5× its 20d average, marked gold on the chart. Volume-confirmed pattern odds arrive with tomorrow's scan.</div>
        </div>
      </div>
      </>);})()}
    {(()=>{const measured=catalog.filter((o:any)=>edgeOf(o)!=null); const top=measured.slice(0,5); const rest=catalog.filter((o:any)=>!top.includes(o));
      const count=(n:string)=>[...d.formations,...d.candlesticks].filter((z:Occ)=>z.name===n).length;
      return (<>
      <div style={{background:C.s2,border:`1px solid ${C.gold}`,borderRadius:10,padding:'12px 14px',marginTop:12}}>
        <div style={{fontFamily:mono,fontSize:9,letterSpacing:1.5,color:C.gold,marginBottom:8}}>TOP {top.length} PATTERNS IN THIS WINDOW · LARGEST MEASURED EDGE AT {hz.toUpperCase()} · MEASURED ACROSS THE WHOLE MARKET</div>
        {top.length===0&&<div style={{fontFamily:mono,fontSize:10.5,color:C.cocoa}}>No measured pattern in this window yet — candlestick odds arrive with the nightly scan.</div>}
        {top.map((o:any,i:number)=>{const e=edgeOf(o)!; const st=o.scorecard?.all||o.scorecard; return (
          <div key={o.name} style={{display:'grid',gridTemplateColumns:'28px 200px 1fr',gap:12,alignItems:'center',padding:'8px 0',borderBottom:'1px solid rgba(58,41,32,0.5)'}}>
            <span style={{fontFamily:mono,fontSize:18,color:C.b2,fontWeight:700}}>{i+1}</span>
            <div><div style={{fontFamily:mono,fontSize:12,color:C.cream,fontWeight:700}}>{LABEL[o.name]||o.name}</div>
              <div style={{fontFamily:mono,fontSize:8.5,color:C.cocoa}}>{o.family} · ×{count(o.name)} here · {e>=0?'bullish':'bearish'} lean</div></div>
            <div><div style={{display:'flex',gap:14,fontFamily:mono,fontSize:11}}>
                <span style={{color:st.positive_pct>=50?C.bull:C.bear,fontWeight:700}}>{st.positive_pct}% up</span>
                <span style={{color:C.dust}}>median {pf(st.median_pct,2)}</span>
                <span style={{color:Math.abs(e)<2?C.cocoa:e>0?C.bull:C.bear}}>{e>=0?'+':''}{e.toFixed(1)} pts vs base</span>
                <span style={{color:C.cocoa}}>n={st.n?.toLocaleString()}</span>
                {o.scorecard?.follow_through_pct!=null&&<span style={{color:C.dust}}>follow-through {o.scorecard.follow_through_pct}%</span>}</div>
              <div style={{height:5,background:'rgba(0,0,0,0.3)',borderRadius:3,marginTop:6,maxWidth:360}}>
                <div style={{height:5,width:`${Math.min(100,st.positive_pct)}%`,background:`linear-gradient(90deg,${st.positive_pct>=50?C.bull:C.bear},${C.b2})`,borderRadius:3}}/></div>
            </div></div>);})}
      </div>
      <details style={{marginTop:10}}><summary style={{fontFamily:mono,fontSize:9,letterSpacing:1.5,color:C.cocoa,cursor:'pointer'}}>OTHER PATTERNS IN THIS WINDOW ({rest.length})</summary>
      <div style={{background:C.s2,border:`1px solid ${C.b1}`,borderRadius:10,padding:'10px 14px',marginTop:8}}>
      {rest.map((o:any)=>(<div key={o.name} style={{display:'grid',gridTemplateColumns:'190px 1fr',gap:12,padding:'5px 0',borderBottom:'1px solid rgba(58,41,32,0.4)',fontFamily:mono,fontSize:10.5}}>
        <span style={{color:C.latte}}>{LABEL[o.name]||o.name} <span style={{color:C.cocoa,fontSize:8.5}}>×{count(o.name)}</span></span>
        <ScoreLine o={o} hz={hz.toUpperCase()}/></div>))}
      </div></details>
      <div style={{background:C.s2,border:`1px solid ${C.b1}`,borderRadius:10,padding:'10px 14px',marginTop:10}}>
      <div style={{fontFamily:mono,fontSize:8.5,color:C.cocoa,lineHeight:1.6}}>{d.scorecards_note} Formations: Lo–Mamaysky–Wang templates. Candlesticks: textbook geometry, entry at the next session's close. DESCRIPTIVE RESULTS — historical frequencies with n and base rates; nothing here is a prediction.</div>
    </div></>);})()}
  </div>);
};
export default PatternChart;
