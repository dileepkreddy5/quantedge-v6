// Volume Tracker — weekly participation: average volume per week, week-over-week
// change, 4w/13w/52w averages. Weekly, because daily volume is noise.
import React, { useEffect, useMemo, useState } from 'react';
import { api } from '../../auth/authStore';
const C={s2:'#241610',b1:'#3a2920',b2:'#4a3428',gold:'#daa520',caramel:'#d4956c',cocoa:'#8a7560',dust:'#9d8b7a',latte:'#d4c4b0',cream:'#f4e8d8',bull:'#22c55e',bear:'#ef4444'};
const mono="'Fira Code',monospace";
const HZ:Record<string,number>={'1m':5,'3m':13,'6m':26,'12m':52,'24m':104};
const fmt=(v:number)=>v>=1e9?`${(v/1e9).toFixed(2)}B`:v>=1e6?`${(v/1e6).toFixed(1)}M`:`${(v/1e3).toFixed(0)}K`;
const VolumeTracker:React.FC<{ticker:string}>=({ticker})=>{
  const [hz,setHz]=useState('12m'); const [d,setD]=useState<any>(null); const [err,setErr]=useState('');
  useEffect(()=>{let dead=false;(async()=>{setD(null);setErr('');
    try{const r=await api.get(`/api/v6/patterns/chart/${ticker}?horizon=12m`);if(!dead)setD(r.data);}
    catch(e:any){if(!dead)setErr(e?.response?.data?.detail||'unavailable');}})();return()=>{dead=true};},[ticker]);
  const weeks=useMemo(()=>{ if(!d) return []; const m=new Map<string,any>();
    for(const c of d.candles){const dt=new Date(c.d+'T00:00:00Z'); const day=(dt.getUTCDay()+6)%7; const mon=new Date(dt); mon.setUTCDate(dt.getUTCDate()-day);
      const k=mon.toISOString().slice(0,10); const w=m.get(k)||{week:k,vols:[] as number[],closes:[] as number[],up:0,tot:0};
      w.vols.push(c.v); w.closes.push(c.c); w.tot+=c.v; if(c.c>=c.o) w.up+=c.v; m.set(k,w);}
    const arr=Array.from(m.values()).map((w:any)=>({week:w.week,avg:w.tot/w.vols.length,tot:w.tot,close:w.closes[w.closes.length-1],
      ret:(w.closes[w.closes.length-1]/w.closes[0]-1)*100,upShare:w.tot?w.up/w.tot*100:null,days:w.vols.length}));
    return arr.slice(-HZ[hz]);},[d,hz]);
  if(err)return <div style={{fontFamily:mono,fontSize:11,color:C.bear}}>{err}</div>;
  if(!d)return <div style={{fontFamily:mono,fontSize:11,color:C.dust}}>bucketing {ticker} volume by week…</div>;
  const avgOf=(k:number)=>{const s=weeks.slice(-k);return s.length?s.reduce((a,w)=>a+w.avg,0)/s.length:0;};
  const a4=avgOf(4),a13=avgOf(13),a52=avgOf(52); const W=1180,H=300,PL=8,PR=70,PB=30,PT=14;
  const vmax=Math.max(...weeks.map(w=>w.avg))||1; const cw=(W-PL-PR)/Math.max(1,weeks.length);
  const x=(i:number)=>PL+i*cw; const yv=(v:number)=>PT+(1-v/vmax)*(H-PT-PB);
  const pmn=Math.min(...weeks.map(w=>w.close)),pmx=Math.max(...weeks.map(w=>w.close)); const yp=(p:number)=>PT+(1-(p-pmn)/((pmx-pmn)||1))*(H-PT-PB);
  const biggest=[...weeks].sort((a,b)=>b.avg-a.avg).slice(0,3);
  return (<div>
    <div style={{display:'flex',gap:8,alignItems:'center',marginBottom:10}}>
      <span style={{fontFamily:mono,fontSize:10,letterSpacing:2.5,color:C.cocoa}}>VOLUME TRACKER · {ticker} · WEEKLY</span>
      {Object.keys(HZ).map(h=><button key={h} onClick={()=>setHz(h)} style={{fontFamily:mono,fontSize:9.5,letterSpacing:1.2,padding:'6px 12px',
        background:hz===h?'rgba(218,165,32,0.12)':'none',border:`1px solid ${hz===h?C.gold:C.b1}`,borderRadius:4,color:hz===h?C.gold:C.dust,cursor:'pointer'}}>{h.toUpperCase()}</button>)}
    </div>
    <div style={{display:'flex',flexWrap:'wrap',gap:8,marginBottom:10}}>
      {[['4-WEEK AVG / DAY',fmt(a4),a4>a13?C.bull:C.bear,`${a13?((a4/a13-1)*100).toFixed(0):'—'}% vs 13w`],['13-WEEK AVG / DAY',fmt(a13),C.latte,`${a52?((a13/a52-1)*100).toFixed(0):'—'}% vs 52w`],
        ['52-WEEK AVG / DAY',fmt(a52),C.latte,'baseline'],['THIS WEEK vs LAST',weeks.length>1?`${((weeks[weeks.length-1].avg/weeks[weeks.length-2].avg-1)*100).toFixed(0)}%`:'—',weeks.length>1&&weeks[weeks.length-1].avg>weeks[weeks.length-2].avg?C.bull:C.bear,'week-over-week'],
        ['BIGGEST WEEKS',biggest.map(w=>w.week.slice(5)).join(' · '),C.gold,biggest.map(w=>fmt(w.avg)).join(' · ')]].map(([k,v,col,sub]:any)=>(
        <div key={k} style={{flex:'1 1 150px',background:C.s2,border:`1px solid ${C.b1}`,borderRadius:6,padding:'9px 12px'}}>
          <div style={{fontFamily:mono,fontSize:8,letterSpacing:1.3,color:C.cocoa,marginBottom:4}}>{k}</div>
          <div style={{fontFamily:mono,fontSize:14,fontWeight:700,color:col}}>{v}</div><div style={{fontFamily:mono,fontSize:8.5,color:C.dust,marginTop:2}}>{sub}</div></div>))}
    </div>
    <div style={{background:'rgba(0,0,0,0.28)',border:`1px solid ${C.b1}`,borderRadius:10,padding:6}}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{width:'100%',display:'block'}}>
        <line x1={PL} x2={W-PR} y1={yv(a52)} y2={yv(a52)} stroke={C.dust} strokeDasharray="4,4" opacity={.6}/><text x={W-PR+4} y={yv(a52)+3} fill={C.dust} fontSize={8} fontFamily={mono}>52w avg</text>
        {weeks.map((w,i)=>(<g key={w.week}><rect x={x(i)+1} y={yv(w.avg)} width={Math.max(1,cw-2)} height={H-PB-yv(w.avg)} fill={w.ret>=0?C.bull:C.bear} opacity={w.avg>=1.5*a52?.95:.5}/>
          {i%Math.max(1,Math.floor(weeks.length/12))===0&&<text x={x(i)+cw/2} y={H-PB+12} fill={C.cocoa} fontSize={7} fontFamily={mono} textAnchor="middle">{w.week.slice(5)}</text>}</g>))}
        <path d={weeks.map((w,i)=>`${i?'L':'M'}${x(i)+cw/2},${yp(w.close)}`).join('')} fill="none" stroke={C.gold} strokeWidth={1.5}/>
        <text x={PL+4} y={PT+10} fill={C.cocoa} fontSize={8} fontFamily={mono}>bars: average daily volume per week (green = up week) · gold line: weekly close · bright bars ≥1.5× 52w avg</text>
      </svg>
    </div>
    <div style={{background:C.s2,border:`1px solid ${C.b1}`,borderRadius:10,padding:'10px 14px',marginTop:12,overflowX:'auto'}}>
      <div style={{fontFamily:mono,fontSize:9,letterSpacing:1.5,color:C.cocoa,marginBottom:6}}>WEEK BY WEEK · MOST RECENT FIRST</div>
      <table style={{width:'100%',borderCollapse:'collapse',fontFamily:mono,fontSize:10}}>
        <thead><tr>{['WEEK OF','AVG VOL/DAY','vs PRIOR WEEK','vs 52W AVG','UP-DAY SHARE','WEEK RETURN','CLOSE'].map(h=><th key={h} style={{textAlign:'left',padding:'5px 8px',color:C.cocoa,fontSize:8.5,letterSpacing:1,fontWeight:500,borderBottom:`1px solid ${C.b1}`}}>{h}</th>)}</tr></thead>
        <tbody>{[...weeks].reverse().slice(0,16).map((w,i,arr)=>{const prev=[...weeks].reverse()[i+1]; const ch=prev?(w.avg/prev.avg-1)*100:null; return (
          <tr key={w.week}><td style={{padding:'5px 8px',color:C.cream}}>{w.week}</td><td style={{padding:'5px 8px',color:C.latte}}>{fmt(w.avg)}</td>
            <td style={{padding:'5px 8px',color:ch==null?C.cocoa:ch>=0?C.bull:C.bear}}>{ch==null?'—':`${ch>=0?'+':''}${ch.toFixed(0)}%`}</td>
            <td style={{padding:'5px 8px',color:w.avg>=a52?C.bull:C.dust}}>{a52?`${((w.avg/a52-1)*100).toFixed(0)}%`:'—'}</td>
            <td style={{padding:'5px 8px',color:(w.upShare??50)>=55?C.bull:(w.upShare??50)<=45?C.bear:C.latte}}>{w.upShare!=null?`${w.upShare.toFixed(0)}%`:'—'}</td>
            <td style={{padding:'5px 8px',color:w.ret>=0?C.bull:C.bear}}>{w.ret>=0?'+':''}{w.ret.toFixed(1)}%</td><td style={{padding:'5px 8px',color:C.dust}}>{w.close}</td></tr>);})}</tbody>
      </table>
      <div style={{fontFamily:mono,fontSize:8.5,color:C.cocoa,marginTop:8,lineHeight:1.6}}>Up-day share = volume on up days ÷ week volume (an accumulation proxy; a true buy/sell split is not derivable from daily bars). Partial current week is shown as-is.</div>
    </div>
  </div>);
};
export default VolumeTracker;
