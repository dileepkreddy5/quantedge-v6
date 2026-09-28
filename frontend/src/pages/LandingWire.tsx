// QuantEdge homepage — THE WIRE.
// Left: the market (SPY on the chart engine + outlook). Center: what the platform
// detected across every filer, timestamped and tiered. Right: the boards with their
// measured track record. Bottom: system scoreboard. Search in the header.
import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../auth/authStore';
import PatternChartPro from '../components/ui/PatternChartPro';

const C={bg:'#140d0a',s1:'#1c130e',s2:'#241610',b1:'#3a2920',b2:'#4a3428',gold:'#daa520',caramel:'#d4956c',cocoa:'#8a7560',
  dust:'#9d8b7a',latte:'#d4c4b0',cream:'#f4e8d8',bull:'#22c55e',bear:'#ef4444',warn:'#f59e0b',blue:'#60a5fa'};
const mono="'Fira Code',monospace"; const display="'Bebas Neue','Oswald',sans-serif";
const pf=(v:any,d=1)=>v==null?'—':`${v>=0?'+':''}${Number(v).toFixed(d)}%`;
const TIER_COL:Record<string,string>={PRIMARY:C.gold,SECONDARY:C.blue,DERIVED:C.caramel};
const KIND_TAG:Record<string,string>={material_event:'8-K',capital_allocation:'CAPITAL',insider_cluster:'INSIDERS',attention_spike:'ATTENTION',
  institutional_snapshot:'13F',new_listing:'NEW',pattern_candlestick:'PATTERN',pattern_formation:'PATTERN'};
const ago=(iso:string)=>{const m=(Date.now()-new Date(iso).getTime())/60000; if(m<60)return `${Math.max(1,Math.round(m))}m`; if(m<1440)return `${Math.round(m/60)}h`; return `${Math.round(m/1440)}d`;};

const Panel:React.FC<{title:string;right?:any;children:any;style?:any}>=({title,right,children,style})=>(
  <div style={{background:C.s1,border:`1px solid ${C.b1}`,borderRadius:10,padding:'14px 16px',...style}}>
    <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:12}}>
      <span style={{fontFamily:mono,fontSize:9.5,letterSpacing:2.2,color:C.cocoa}}>{title}</span><span style={{flex:1}}/>{right}</div>
    {children}</div>);

const LandingWire:React.FC=()=>{
  const nav=useNavigate(); const go=(t:string)=>nav(`/dashboard?ticker=${t}`);
  const [q,setQ]=useState(''); const [spy,setSpy]=useState<any>(null); const [wire,setWire]=useState<any>(null);
  const [mb,setMb]=useState<any>(null); const [rb,setRb]=useState<any>(null); const [as_,setAs]=useState<any>(null);
  const [tr,setTr]=useState<any>(null); const [sys,setSys]=useState<any>(null); const [filter,setFilter]=useState('ALL'); const [open,setOpen]=useState<string|null>(null);
  useEffect(()=>{
    const get=async(url:string,set:(v:any)=>void)=>{try{set((await api.get(url)).data);}catch{set({error:true});}};
    get('/api/v6/patterns/chart/SPY?horizon=6m',setSpy); get('/api/v6/wire?hours=72&limit=300',setWire);
    get('/api/v6/scan/tiers',setMb); get('/api/v6/rebound/list',setRb); get('/api/v6/ascent/top/10',setAs);
    get('/api/v6/boards/track_record',setTr); get('/api/v6/system/stats',setSys);
  },[]);

  // Wire: company facts one line each; pattern completions grouped by pattern.
  const feed=useMemo(()=>{ if(!wire?.items) return [];
    const out:any[]=[]; const groups:Record<string,any>={};
    for(const it of wire.items){
      if(String(it.kind).startsWith('pattern_')){ const head=it.title.split(' — ')[0]; const odds=it.title.split(' — ')[1]||'';
        const g=groups[head]||(groups[head]={kind:it.kind,tier:it.tier,ts:it.ts,head,odds,tickers:[] as string[],group:true}); g.tickers.push(it.ticker);
      } else out.push(it); }
    Object.values(groups).forEach((g:any)=>out.push(g));
    out.sort((a,b)=>a.ts<b.ts?1:a.ts>b.ts?-1:(a.group?1:0)-(b.group?1:0));
    const want:Record<string,(x:any)=>boolean>={ALL:()=>true,FILINGS:x=>['material_event','capital_allocation','institutional_snapshot'].includes(x.kind)||x.tier==='PRIMARY'&&!x.group&&x.kind!=='insider_cluster'&&x.kind!=='new_listing',
      INSIDERS:x=>x.kind==='insider_cluster',PATTERNS:x=>!!x.group,ATTENTION:x=>x.kind==='attention_spike'};
    return out.filter(want[filter]); },[wire,filter]);

  const top=(j:any,key='score')=>{ if(!j?.tiers) return []; const all:any[]=[]; Object.entries(j.tiers).forEach(([t,rows]:any)=>rows.forEach((r:any)=>all.push({...r,tier:t})));
    return all.sort((a,b)=>(b[key]??0)-(a[key]??0)).slice(0,5); };
  const badge=(board:string)=>{ const b=tr?.[board]; if(!b) return null; const h=b.horizons?.['21d'];
    return (<div style={{fontFamily:mono,fontSize:8.5,color:h?C.latte:C.cocoa,marginTop:8,lineHeight:1.5,borderTop:`1px solid ${C.b1}`,paddingTop:6}}>
      TRACK RECORD · {h?<>past top-25s at 1M: <span style={{color:h.mean_pct>=h.universe_pct?C.bull:C.bear}}>{pf(h.mean_pct)}</span> vs universe {pf(h.universe_pct)} · beat {h.beat_universe_pct}% · n={h.n}</>:b.note}</div>); };
  const spyOut=spy?.analog?.distribution; const spyBase=spy?.analog?.base; const fc=spy?.forecast;

  return (<div style={{minHeight:'100vh',background:C.bg,color:C.latte}}>
    <style>{`.qe-grid{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1.1fr) minmax(0,0.9fr);gap:14px}
      @media (max-width:1180px){.qe-grid{grid-template-columns:1fr}} .qe-row:hover{background:rgba(218,165,32,0.05)}`}</style>
    {/* header */}
    <div style={{display:'flex',alignItems:'center',gap:14,padding:'14px 24px',borderBottom:`1px solid ${C.b1}`,flexWrap:'wrap'}}>
      <span style={{fontFamily:display,fontSize:26,letterSpacing:6,color:C.gold}}>QUANTEDGE</span>
      <span style={{fontFamily:mono,fontSize:9,color:C.bull}}>● LIVE</span>
      <form onSubmit={e=>{e.preventDefault(); if(q.trim()) go(q.trim().toUpperCase());}} style={{flex:'1 1 280px',maxWidth:520,display:'flex',gap:8}}>
        <input value={q} onChange={e=>setQ(e.target.value)} placeholder="Ticker — NVDA, AAPL, any US stock" style={{flex:1,background:C.s2,border:`1px solid ${C.b2}`,borderRadius:6,
          padding:'9px 12px',color:C.cream,fontFamily:mono,fontSize:12,outline:'none'}}/>
        <button type="submit" style={{background:C.gold,border:'none',borderRadius:6,padding:'0 16px',fontFamily:mono,fontSize:10,letterSpacing:2,color:'#1a120d',fontWeight:700,cursor:'pointer'}}>ANALYZE →</button>
      </form>
      <span style={{flex:1}}/>
      {[['SCREENER','/screener'],['ASCENT','/ascent'],['MULTIBAGGER','/multibagger'],['REBOUND','/rebound'],['METHODOLOGY','/methodology']].map(([l,p])=>(
        <button key={p} onClick={()=>nav(p)} style={{background:'none',border:`1px solid ${C.b1}`,borderRadius:4,padding:'6px 10px',fontFamily:mono,fontSize:9,letterSpacing:1.4,color:C.dust,cursor:'pointer'}}>{l}</button>))}
      <button onClick={()=>nav('/login')} style={{background:'none',border:`1px solid ${C.gold}`,borderRadius:4,padding:'6px 12px',fontFamily:mono,fontSize:9,letterSpacing:1.4,color:C.gold,cursor:'pointer'}}>LOGIN</button>
    </div>
    <div style={{padding:'14px 24px 6px',fontFamily:mono,fontSize:11,color:C.dust}}>
      Every night QuantEdge reads every US filer's SEC filings, prices and patterns. <span style={{color:C.cream}}>This is what it found.</span> Every line is timestamped to when the fact became public; nothing here is a prediction.</div>
    <div className="qe-grid" style={{padding:'10px 24px 18px'}}>
      {/* LEFT — the market */}
      <Panel title="THE MARKET · SPY · 6 MONTHS" right={<button onClick={()=>go('SPY')} style={{background:'none',border:'none',color:C.gold,fontFamily:mono,fontSize:9,cursor:'pointer'}}>OPEN →</button>}>
        {spy&&!spy.error?<>
          <PatternChartPro d={spy} show={{f:false,c:false,doji:false}} height={360}/>
          <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:8,marginTop:10}}>
            {[['WHAT FOLLOWED SIMILAR SHAPES',spyOut?`${spyOut.positive_pct}% up`:'—',spyOut&&spyOut.positive_pct>=(spyBase?.positive_pct??50)?C.bull:C.bear,spyOut?`base ${spyBase?.positive_pct??'?'}% · med ${pf(spyOut.median_pct,1)} · 6M · n=${spyOut.n}`:''],
              ['RELATIVE RANGE',spyOut?`${pf(spyOut.p25_pct)} / ${pf(spyOut.p75_pct)}`:'—',C.latte,'p25 / p75 of those outcomes'],
              ['MODEL FORECAST · 6M',fc&&!fc.error&&fc.produced?(fc.pred_pct!=null?pf(fc.pred_pct,1):'—'):'not produced',fc?.validated?C.gold:C.cocoa,
                fc?.degenerate?'not measurable':fc?.validated?'VALIDATED':'NOT VALIDATED — not a signal']].map(([k,v,col,sub]:any)=>(
              <div key={k} style={{background:C.s2,border:`1px solid ${C.b1}`,borderRadius:6,padding:'8px 10px'}}>
                <div style={{fontFamily:mono,fontSize:7.5,letterSpacing:1.2,color:C.cocoa}}>{k}</div>
                <div style={{fontFamily:mono,fontSize:15,fontWeight:700,color:col,marginTop:3}}>{v}</div>
                <div style={{fontFamily:mono,fontSize:8,color:C.dust,marginTop:2}}>{sub}</div></div>))}
          </div></>:<div style={{fontFamily:mono,fontSize:11,color:C.dust}}>{spy?.error?'market chart unavailable':'loading the market…'}</div>}
      </Panel>
      {/* CENTER — the wire */}
      <Panel title={`THE WIRE · LAST 72 HOURS${wire?.n!=null?` · ${wire.n} DETECTIONS`:''}`}
        right={<div style={{display:'flex',gap:4}}>{['ALL','FILINGS','INSIDERS','PATTERNS','ATTENTION'].map(f=>(
          <button key={f} onClick={()=>setFilter(f)} style={{background:filter===f?'rgba(218,165,32,0.12)':'none',border:`1px solid ${filter===f?C.gold:C.b1}`,borderRadius:3,
            padding:'3px 7px',fontFamily:mono,fontSize:7.5,letterSpacing:1,color:filter===f?C.gold:C.cocoa,cursor:'pointer'}}>{f}</button>))}</div>}
        style={{maxHeight:640,display:'flex',flexDirection:'column'}}>
        <div style={{overflowY:'auto',flex:1}}>
          {!wire&&<div style={{fontFamily:mono,fontSize:11,color:C.dust}}>reading the wire…</div>}
          {feed.map((it:any,k:number)=>it.group?(
            <div key={'g'+k} className="qe-row" style={{padding:'8px 4px',borderBottom:`1px solid rgba(58,41,32,0.5)`,cursor:'pointer'}} onClick={()=>setOpen(open===it.head?null:it.head)}>
              <div style={{display:'flex',gap:8,alignItems:'baseline',fontFamily:mono,fontSize:10.5}}>
                <span style={{color:C.cocoa,width:30,flexShrink:0}}>{ago(it.ts)}</span>
                <span style={{color:TIER_COL.DERIVED,fontSize:8,border:`1px solid ${C.b2}`,borderRadius:2,padding:'0 4px'}}>PATTERN</span>
                <span style={{color:C.cream}}>{it.head.replace(/^./,(c:string)=>c.toUpperCase())} on <b style={{color:C.gold}}>{it.tickers.length}</b> ticker{it.tickers.length>1?'s':''}</span></div>
              <div style={{fontFamily:mono,fontSize:9,color:C.dust,marginLeft:38,marginTop:2}}>{it.odds||'odds not yet measured'}{it.kind==='pattern_formation'?' · provisional (shape at the smoother edge)':''}</div>
              {open===it.head&&<div style={{display:'flex',flexWrap:'wrap',gap:4,marginLeft:38,marginTop:6}}>{it.tickers.map((t:string)=>(
                <span key={t} onClick={e=>{e.stopPropagation();go(t);}} style={{fontFamily:mono,fontSize:9,color:C.gold,border:`1px solid ${C.b2}`,borderRadius:3,padding:'1px 6px',cursor:'pointer'}}>{t}</span>))}</div>}
            </div>):(
            <div key={k} className="qe-row" onClick={()=>go(it.ticker)} style={{display:'flex',gap:8,alignItems:'baseline',padding:'7px 4px',borderBottom:`1px solid rgba(58,41,32,0.5)`,cursor:'pointer',fontFamily:mono,fontSize:10.5}}>
              <span style={{color:C.cocoa,width:30,flexShrink:0}}>{ago(it.ts)}</span>
              <span style={{color:TIER_COL[it.tier]||C.dust,fontSize:8,border:`1px solid ${C.b2}`,borderRadius:2,padding:'0 4px',flexShrink:0}}>{KIND_TAG[it.kind]||it.kind}</span>
              <span style={{color:C.gold,fontWeight:700,width:52,flexShrink:0}}>{it.ticker}</span>
              <span style={{color:it.significance==='MATERIAL'?C.cream:C.latte}}>{it.title}</span></div>))}
          {wire&&feed.length===0&&<div style={{fontFamily:mono,fontSize:10.5,color:C.cocoa,padding:8}}>nothing in this filter in the last 72 hours</div>}
        </div>
        <div style={{fontFamily:mono,fontSize:8,color:C.cocoa,marginTop:8,lineHeight:1.5}}>
          <span style={{color:C.gold}}>■ PRIMARY</span> SEC filings · <span style={{color:C.blue}}>■ SECONDARY</span> news coverage · <span style={{color:C.caramel}}>■ DERIVED</span> computed by QuantEdge. {wire?.note}</div>
      </Panel>
      {/* RIGHT — the boards */}
      <div style={{display:'flex',flexDirection:'column',gap:14}}>
        {[['MULTIBAGGER','/multibagger','multibagger',top(mb),(r:any)=>`${pf(r.qtr_yoy_growth!=null?r.qtr_yoy_growth*(Math.abs(r.qtr_yoy_growth)<5?100:1):null,0)} growth · ${r.tier}`],
          ['REBOUND','/rebound','rebound',top(rb),(r:any)=>`${r.stage||''} · ${pf(r.drawdown!=null?r.drawdown*(Math.abs(r.drawdown)<1.5?100:1):null,0)} from high`],
          ['ASCENT','/ascent','ascent',(as_?.rows||[]).slice(0,5),(r:any)=>`${r.sector||''} · 1M Δ ${r.delta_1m??'—'}`]].map(([title,path,key,rows,sub]:any)=>(
          <Panel key={key} title={title} right={<button onClick={()=>nav(path)} style={{background:'none',border:'none',color:C.gold,fontFamily:mono,fontSize:9,cursor:'pointer'}}>ALL →</button>}>
            {rows.length===0&&<div style={{fontFamily:mono,fontSize:10,color:C.cocoa}}>loading…</div>}
            {rows.map((r:any,i:number)=>(
              <div key={r.ticker} className="qe-row" onClick={()=>go(r.ticker)} style={{display:'grid',gridTemplateColumns:'18px 56px 1fr',gap:8,padding:'5px 2px',cursor:'pointer',fontFamily:mono,fontSize:10.5,alignItems:'baseline'}}>
                <span style={{color:C.b2}}>{i+1}</span><span style={{color:C.gold,fontWeight:700}}>{r.ticker}</span>
                <span style={{color:C.dust,fontSize:9,whiteSpace:'nowrap',overflow:'hidden',textOverflow:'ellipsis'}}>{sub(r)}</span></div>))}
            {badge(key)}
          </Panel>))}
      </div>
    </div>
    {/* scoreboard */}
    <div style={{borderTop:`1px solid ${C.b1}`,padding:'10px 24px',display:'flex',flexWrap:'wrap',gap:18,fontFamily:mono,fontSize:9,color:C.cocoa}}>
      <span>SYSTEM</span>
      {(sys?.boards||[]).map((b:any)=><span key={b.board} style={{color:b.stale?C.warn:C.dust}}>{b.board} {b.age_hours!=null?`${Math.round(b.age_hours)}h`:'—'}{b.stale?' STALE':''}</span>)}
      {sys?.disk&&<span style={{color:sys.disk.warning?C.warn:C.dust}}>disk {sys.disk.used_pct}%</span>}
      <span style={{flex:1}}/>
      <span onClick={()=>nav('/methodology')} style={{color:C.gold,cursor:'pointer'}}>HOW IT'S MEASURED →</span>
      <span onClick={()=>nav('/classic')} style={{cursor:'pointer'}}>classic homepage</span>
    </div>
  </div>);
};
export default LandingWire;
