// Situation Report — every section names its source; nothing predicts.
import React, { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
const C={s2:'#241610',b1:'#3a2920',gold:'#daa520',caramel:'#d4956c',cocoa:'#8a7560',dust:'#9d8b7a',
  latte:'#d4c4b0',cream:'#f4e8d8',bull:'#22c55e',bear:'#ef4444',warn:'#f59e0b'};
const mono="'Fira Code',monospace";
const pf=(v:any,d=1)=>v==null?'—':`${v>=0?'+':''}${Number(v).toFixed(d)}%`;
const Box:React.FC<{title:string;source?:string;children:any}>=({title,source,children})=>(
  <div style={{background:C.s2,border:`1px solid ${C.b1}`,borderRadius:10,padding:16}}>
    <div style={{fontFamily:mono,fontSize:9,letterSpacing:1.5,color:C.cocoa,marginBottom:4}}>{title}</div>
    {source&&<div style={{fontFamily:mono,fontSize:8,color:C.cocoa,marginBottom:10,opacity:.8}}>SOURCE: {source}</div>}
    {children}</div>);
const Row:React.FC<{k:string;v:any;tone?:string}>=({k,v,tone})=>(
  <div style={{display:'flex',justifyContent:'space-between',fontFamily:mono,fontSize:10.5,padding:'4px 0'}}>
    <span style={{color:C.dust}}>{k}</span><span style={{color:tone||C.latte}}>{v==null?'—':String(v)}</span></div>);
const SituationMode:React.FC<{ticker:string}>=({ticker})=>{
  const [d,setD]=useState<any>(null); const [err,setErr]=useState('');
  useEffect(()=>{let dead=false;(async()=>{setD(null);setErr('');
    try{const r=await api.get(`/api/v6/patterns/situation/${ticker}`);if(!dead)setD(r.data);}
    catch(e:any){if(!dead)setErr(e?.response?.data?.detail||'report unavailable');}})();return()=>{dead=true};},[ticker]);
  if(err)return <div style={{fontFamily:mono,fontSize:11,color:C.warn}}>{err}</div>;
  if(!d)return <div style={{fontFamily:mono,fontSize:11,color:C.dust}}>assembling {ticker} situation report from all engines…</div>;
  const p=d.price_state?.price||{}, m=d.price_state?.momentum||{}, vo=d.price_state?.volatility||{}, ms=d.price_state?.multi_scale;
  const w20=d.what_followed?.shape_20d, w60=d.what_followed?.shape_60d, cond=d.what_followed?.conditions||{}, cb=d.what_followed?.conditions_base||{};
  const oh=d.off_highs||{}, rr=d.reported_results||{}, ci=d.company_intelligence||{}, nt=d.news_tone||{};
  const dist=(w:any,h:string)=>{const o=w?.outcomes?.[h], b=w?.base_rates?.[h.replace('d','')]||w?.base_rates?.[parseInt(h)], e=w?.excess_vs_spy?.[h];
    if(!o)return null; return <Row key={h} k={`+${h} (n=${o.n})`} v={`${o.positive_pct}% pos (base ${b?.positive_pct??'?'}%) · med ${pf(o.median_pct,2)}${e?` · ${e.positive_pct}% beat SPY`:''}`} tone={o.positive_pct>=50?C.bull:C.bear}/>;};
  return (<div>
    <div style={{fontFamily:mono,fontSize:10,letterSpacing:2.5,color:C.cocoa,marginBottom:4}}>SITUATION REPORT — {d.ticker} · {d.name}</div>
    <div style={{fontFamily:mono,fontSize:9,color:C.cocoa,marginBottom:14}}>as of {d.as_of} · {d.note}</div>
    <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(300px,1fr))',gap:14}}>
      <Box title="1 · PRICE STATE (90D)" source={d.price_state?.source}>
        <Row k="TREND" v={`${(p.trend||'—').toUpperCase()} · slope ${pf(p.slope_20d_ann_pct)} ann · ${p.acceleration||''}`} tone={p.trend==='up'?C.bull:C.bear}/>
        <Row k="VS 52W HIGH / LOW" v={`${pf(p.vs_52w_high_pct)} / ${pf(p.vs_52w_low_pct)}`}/>
        <Row k="DRAWDOWN" v={pf(p.drawdown_pct)}/>
        <Row k="MOMENTUM 5/20/60D" v={`${pf(m['5d_pct'])} / ${pf(m['20d_pct'])} / ${pf(m['60d_pct'])}`}/>
        <Row k="VOLATILITY" v={`${vo.realized_21d_ann_pct??'—'}% ann · ${vo.percentile}th pctile · ${vo.direction||''}`}/>
        <Row k="MULTI-SCALE" v={ms?`${ms.verdict} (${ms.alignment_pct}%)`:'—'} tone={ms?.verdict==='ALIGNED'?C.bull:ms?.verdict==='CONFLICTED'?C.warn:C.latte}/>
      </Box>
      <Box title="2 · WHAT FOLLOWED THIS STATE" source={d.what_followed?.source}>
        <div style={{fontFamily:mono,fontSize:8.5,color:C.cocoa,margin:'4px 0'}}>20-DAY SHAPE {w20?`· ${w20.episodes} EPISODES · ${w20.episode_date_range?.join(' → ')}`:'· INSUFFICIENT'}</div>
        {w20&&['5d','20d','60d'].map(h=>dist(w20,h))}
        <div style={{fontFamily:mono,fontSize:8.5,color:C.cocoa,margin:'8px 0 4px'}}>60-DAY SHAPE {w60?`· ${w60.episodes} EPISODES`:'· INSUFFICIENT'}</div>
        {w60&&['20d','60d','120d'].map(h=>dist(w60,h))}
        {Object.entries(cond).map(([k,c]:any)=>{const cell=c.cell?.fwd_20d, base=cb.fwd_20d; return (
          <Row key={k} k={`${k.toUpperCase()} Q${c.quintile} · +20d`} v={cell?`${cell.positive_pct}% pos (base ${base?.positive_pct}%) · n=${cell.n.toLocaleString()}`:'insufficient'} tone={cell?(cell.positive_pct>=50?C.bull:C.bear):C.cocoa}/>);})}
      </Box>
      <Box title="3 · IF OFF ITS HIGHS" source={oh.source}>
        {oh.applies===false?<div style={{fontFamily:mono,fontSize:10.5,color:C.dust}}>Not applicable — {pf(oh.drawdown_from_52w_high_pct)} from 52w high (threshold −30%).</div>:(<>
          <Row k="DRAWDOWN FROM 52W HIGH" v={pf(oh.drawdown_from_52w_high_pct)} tone={C.bear}/>
          <Row k="ON REBOUND BOARD" v={oh.on_rebound_board?`yes · ${oh.rebound_row?.stage}`:'no — gates not passed'}/>
          {oh.recovery_base_rate&&<Row k={`RECOVERED TO HIGH WITHIN 1Y (${oh.recovery_base_rate.drawdown_bucket}% bucket)`} v={`${oh.recovery_base_rate.rate_pct}% · n=${oh.recovery_base_rate.n}`} tone={C.warn}/>}
          <div style={{fontFamily:mono,fontSize:8.5,color:C.cocoa,marginTop:6}}>{oh.recovery_base_rate?.note}</div></>)}
      </Box>
      <Box title="4 · REPORTED RESULTS" source={rr.source}>
        {(rr.last_quarters||[]).length===0&&<div style={{fontFamily:mono,fontSize:10.5,color:C.cocoa}}>no XBRL quarters in Company Intelligence yet</div>}
        {(rr.last_quarters||[]).map((q:any)=>(<div key={q.period_end} style={{marginBottom:6}}>
          <div style={{fontFamily:mono,fontSize:9,color:C.caramel}}>Q ending {q.period_end} · public {q.public}</div>
          {Object.entries(q.values).map(([k,v]:any)=>v&&'value_usd' in v?<Row key={k} k={k.toUpperCase()} v={`$${(v.value_usd/1e6).toLocaleString(undefined,{maximumFractionDigits:0})}M${v.method?' (derived)':''}`}/>:null)}
        </div>))}
        <div style={{fontFamily:mono,fontSize:8.5,color:C.cocoa,margin:'8px 0 4px'}}>RATIOS</div>
        {Object.entries(rr.ratios||{}).map(([k,v]:any)=><Row key={k} k={k.toUpperCase().replace(/_/g,' ')} v={v==null?'—':(Math.abs(v)<3&&k!=='pe_ratio'&&k!=='price_to_sales'&&k!=='debt_to_equity'?pf(v*100,1):Number(v).toFixed(2))}/>)}
      </Box>
      <Box title="5 · COMPANY INTELLIGENCE (90D)" source={ci.source}>
        <Row k="MATERIAL 8-K EVENTS" v={(ci.material_events||[]).length}/>
        {(ci.material_events||[]).slice(0,4).map((e:any)=><div key={e.id} style={{fontFamily:mono,fontSize:9.5,color:C.latte,padding:'2px 0'}}>{e.available_at?.slice(0,10)} · {e.title}</div>)}
        {ci.insider_open_market&&<Row k="INSIDER OPEN-MARKET NET" v={`${ci.insider_open_market.net_value>=0?'+':'-'}$${(Math.abs(ci.insider_open_market.net_value)/1e6).toFixed(1)}M (${ci.insider_open_market.buys} buys / ${ci.insider_open_market.sells} sells)`} tone={ci.insider_open_market.net_value>=0?C.bull:C.bear}/>}
        <Row k="13F INSTITUTIONAL" v={ci.institutional||'not yet ingested for this ticker'}/>
        {ci.attention_vs_fundamentals&&<Row k="NEWS COVERAGE vs OWN 90D" v={`${ci.attention_vs_fundamentals.attention_ratio??'—'}× (${ci.attention_vs_fundamentals.articles_30d} articles/30d)`}/>}
      </Box>
      <Box title="6 · NEWS TONE" source={nt.source}>
        <Row k="FINBERT COMPOSITE" v={`${nt.composite??'—'} · ${nt.label||''}`} tone={nt.composite>0?C.bull:nt.composite<0?C.bear:C.latte}/>
        <div style={{fontFamily:mono,fontSize:8.5,color:C.cocoa,marginTop:6}}>Headline tone only; article content is not analyzed. Thin for small caps.</div>
      </Box>
      <Box title="7 · NO SOURCE — NOT INFERRED">
        {(d.no_source||[]).map((x:string)=><div key={x} style={{fontFamily:mono,fontSize:10.5,color:C.cocoa,padding:'3px 0'}}>— {x}</div>)}
      </Box>
    </div>
  </div>);
};
export default SituationMode;
