// Shown while the full analysis runs: everything known instantly about the company,
// plus an honest account of what is being computed. No fake progress.
import React, { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
const C={s1:'#1c130e',s2:'#241610',b1:'#3a2920',gold:'#daa520',cocoa:'#8a7560',dust:'#9d8b7a',latte:'#d4c4b0',cream:'#f4e8d8',up:'#3ec27a',dn:'#ef7d5a'};
const mono="'Fira Code',monospace";
const pct=(v:any,d=2)=>v==null?'—':`${v>=0?'+':''}${Number(v).toFixed(d)}%`;
const cap=(v:any)=>v==null?'—':v>=1e12?`$${(v/1e12).toFixed(2)}T`:v>=1e9?`$${(v/1e9).toFixed(1)}B`:`$${(v/1e6).toFixed(0)}M`;
const money=(v:number)=>v>=1e6?`$${(v/1e6).toFixed(1)}M`:`$${Math.round(v/1e3)}K`;
const STEPS=['Five years of daily prices','Volatility model (GJR-GARCH)','Market regime (Hidden Markov model)','Trend filter (Kalman)',
  'Forecasts from boosted-tree models, six horizons','News tone (FinBERT)','10,000-path Monte Carlo simulation','Risk engine and position sizing'];

const LoadingSnapshot:React.FC<{ticker:string;elapsed:number}>=({ticker,elapsed})=>{
  const [q,setQ]=useState<any>(null);
  useEffect(()=>{let dead=false;(async()=>{try{const r=await api.get(`/api/v6/quick/${ticker}`);if(!dead)setQ(r.data);}catch{if(!dead)setQ({error:true});}})();return()=>{dead=true};},[ticker]);
  const cl=q?.closes||[]; const W=520,H=150;
  const mn=Math.min(...cl),mx=Math.max(...cl);
  const path=cl.map((v:number,i:number)=>`${i?'L':'M'}${(i*W/Math.max(1,cl.length-1)).toFixed(1)},${(H-8-(v-mn)/((mx-mn)||1)*(H-16)).toFixed(1)}`).join('');
  const up=cl.length>1&&cl[cl.length-1]>=cl[0];
  const rangePos=q?.high_52w&&q?.low_52w&&q?.price?Math.max(0,Math.min(1,(q.price-q.low_52w)/((q.high_52w-q.low_52w)||1))):null;
  return (<div style={{maxWidth:1180,margin:'0 auto',padding:'40px 24px',display:'grid',gridTemplateColumns:'minmax(0,1.5fr) minmax(0,1fr)',gap:22}}>
    <div style={{background:C.s1,border:`1px solid ${C.b1}`,borderRadius:12,padding:24}}>
      <div style={{fontFamily:mono,fontSize:10,letterSpacing:2,color:C.cocoa}}>WHAT WE KNOW INSTANTLY</div>
      {!q&&<div style={{fontFamily:mono,fontSize:12,color:C.dust,marginTop:16}}>looking up {ticker}…</div>}
      {q&&q.is_company&&(<>
        <div style={{display:'flex',alignItems:'baseline',gap:14,marginTop:12,flexWrap:'wrap'}}>
          <span style={{fontFamily:mono,fontSize:26,fontWeight:700,color:C.gold}}>{ticker}</span>
          <span style={{fontSize:16,color:C.latte}}>{q.name}</span>
          <span style={{fontFamily:mono,fontSize:11,color:C.cocoa}}>{q.exchange}</span></div>
        <div style={{display:'flex',alignItems:'baseline',gap:18,marginTop:10,fontFamily:mono}}>
          <span style={{fontSize:30,color:C.cream}}>{q.price!=null?`$${Number(q.price).toFixed(2)}`:'—'}</span>
          <span style={{fontSize:15,color:(q.today_pct??0)>=0?C.up:C.dn}}>{pct(q.today_pct)} today</span>
          <span style={{fontSize:13,color:C.dust}}>{pct(q.week_pct,1)} this week</span></div>
        {cl.length>1&&<svg viewBox={`0 0 ${W} ${H}`} style={{width:'100%',height:H,marginTop:14,display:'block'}} aria-label="Six-month price chart">
          <path d={path} fill="none" stroke={up?C.up:C.dn} strokeWidth={1.8}/>
          <text x={0} y={H-1} fill={C.cocoa} fontSize={10} fontFamily="Fira Code">{q.dates?.[0]}</text>
          <text x={W} y={H-1} fill={C.cocoa} fontSize={10} fontFamily="Fira Code" textAnchor="end">6 months</text></svg>}
        {rangePos!=null&&<div style={{marginTop:14}}>
          <div style={{display:'flex',justifyContent:'space-between',fontFamily:mono,fontSize:11,color:C.dust}}>
            <span>52-week low ${q.low_52w.toFixed(2)}</span><span>high ${q.high_52w.toFixed(2)}</span></div>
          <div style={{position:'relative',height:6,background:C.s2,borderRadius:3,marginTop:6}}>
            <div style={{position:'absolute',left:`calc(${rangePos*100}% - 6px)`,top:-3,width:12,height:12,borderRadius:6,background:C.gold}}/></div></div>}
        <div style={{display:'grid',gridTemplateColumns:'repeat(2,minmax(0,1fr))',gap:14,marginTop:20}}>
          <div style={{background:C.s2,borderRadius:8,padding:14}}>
            <div style={{fontFamily:mono,fontSize:9.5,letterSpacing:1.5,color:C.cocoa}}>MARKET VALUE</div>
            <div style={{fontFamily:mono,fontSize:18,color:C.cream,marginTop:6}}>{cap(q.market_cap)}</div></div>
          <div style={{background:C.s2,borderRadius:8,padding:14}}>
            <div style={{fontFamily:mono,fontSize:9.5,letterSpacing:1.5,color:C.cocoa}}>INSIDERS · 90 DAYS · SEC FORM 4</div>
            <div style={{fontFamily:mono,fontSize:14,color:C.cream,marginTop:6}}>
              {q.insiders_90d.buys+q.insiders_90d.sells===0?'no open-market trades':<>{q.insiders_90d.buys} buys ({money(q.insiders_90d.buy_value)}) · {q.insiders_90d.sells} sells ({money(q.insiders_90d.sell_value)})</>}</div></div></div>
        <div style={{marginTop:20}}>
          <div style={{fontFamily:mono,fontSize:9.5,letterSpacing:1.5,color:C.cocoa,marginBottom:8}}>LATEST SEC FILINGS</div>
          {(q.filings||[]).length===0&&<div style={{fontSize:13,color:C.dust}}>none on record yet</div>}
          {(q.filings||[]).map((f:any,i:number)=>(<div key={i} style={{display:'flex',gap:14,padding:'6px 0',borderBottom:`1px solid ${C.b1}`,fontSize:13}}>
            <span style={{fontFamily:mono,color:C.cocoa,width:88,flexShrink:0}}>{f.date}</span>
            <span style={{color:f.significance==='MATERIAL'?C.cream:C.latte}}>{f.title}</span></div>))}</div>
      </>)}
    </div>
    <div style={{background:C.s1,border:`1px solid ${C.b1}`,borderRadius:12,padding:24,alignSelf:'start'}}>
      <div style={{fontFamily:mono,fontSize:10,letterSpacing:2,color:C.cocoa}}>FULL ANALYSIS</div>
      <div style={{fontFamily:mono,fontSize:36,color:C.gold,marginTop:12}}>{elapsed}s</div>
      <div style={{fontSize:13.5,lineHeight:1.6,color:C.dust,marginTop:6}}>The first analysis of a stock usually takes 15–25 seconds: the models are fitted to this company's own history. The largest companies are pre-computed and open almost instantly.</div>
      <div style={{fontFamily:mono,fontSize:9.5,letterSpacing:1.5,color:C.cocoa,margin:'20px 0 10px'}}>BEING COMPUTED</div>
      {STEPS.map(s=>(<div key={s} style={{display:'flex',gap:10,alignItems:'baseline',padding:'5px 0',fontSize:13.5,color:C.latte}}>
        <span style={{width:6,height:6,borderRadius:3,background:C.gold,opacity:.7,flexShrink:0}}/>{s}</div>))}
    </div>
  </div>);
};
export default LoadingSnapshot;
