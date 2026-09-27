// Company Intelligence — evidence timeline. Three timestamps per event,
// tier from source type, NO SOURCE rendered as its own state.
import React, { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
const C={s2:'#241610',b1:'#3a2920',gold:'#daa520',caramel:'#d4956c',cocoa:'#8a7560',dust:'#9d8b7a',
  latte:'#d4c4b0',cream:'#f4e8d8',bull:'#22c55e',bear:'#ef4444',warn:'#f59e0b'};
const mono="'Fira Code',monospace";
const SIG:Record<string,string>={MATERIAL:C.gold,RELEVANT:C.latte,ROUTINE:C.cocoa,UNKNOWN:C.warn};
type Ev={id:number;event_date:string|null;available_at:string;retrieved_at:string;lag_days:number|null;
  item_code:string;event_type:string;significance:string;title:string;evidence_tier:string;
  source:{type:string;accession:string;form:string;url:string};derived:{field:string;value:any}[]};
type Res={ticker:string;events:Ev[];counts?:Record<string,number>;note?:string;
  insider_open_market?:{buys:number;buy_value:number;sells:number;sell_value:number;net_value:number;note:string};
  families:Record<string,string>;pit_note?:string};
const money=(v:number)=>Math.abs(v)>=1e6?`$${(v/1e6).toFixed(1)}M`:`$${(v/1e3).toFixed(0)}K`;
const FAM_LABEL:Record<string,string>={sec_filings:'SEC FILINGS',insider_transactions:'INSIDER TRANSACTIONS',capital_allocation:'CAPITAL ALLOCATION',
  institutional_13f:'13F OWNERSHIP',market_attention:'MARKET ATTENTION',patents:'PATENTS',research_papers:'RESEARCH',
  customers:'CUSTOMERS',government_contracts:'GOV CONTRACTS',capacity_utilization:'CAPACITY',
  job_postings:'JOB POSTINGS',earnings_transcripts:'TRANSCRIPTS'};

const CompanyIntel:React.FC<{ticker:string}>=({ticker})=>{
  const [d,setD]=useState<Res|null>(null); const [err,setErr]=useState('');
  const [sig,setSig]=useState<string>(''); const [days,setDays]=useState(365);
  useEffect(()=>{let dead=false;(async()=>{setD(null);setErr('');
    try{const qs=new URLSearchParams({days:String(days),...(sig?{significance:sig}:{})});
      const r=await api.get(`/api/v6/intel/${ticker}/timeline?${qs}`);if(!dead)setD(r.data);}
    catch(e:any){if(!dead)setErr(e?.response?.data?.detail||'intel unavailable');}})();
    return()=>{dead=true};},[ticker,sig,days]);
  if(err)return <div style={{fontFamily:mono,fontSize:11,color:C.warn}}>{err}</div>;
  if(!d)return <div style={{fontFamily:mono,fontSize:11,color:C.dust}}>loading evidence…</div>;
  const ins=d.insider_open_market;
  return (<div>
    <div style={{display:'flex',flexWrap:'wrap',gap:8,alignItems:'center',marginBottom:14}}>
      <span style={{fontFamily:mono,fontSize:10,letterSpacing:2.5,color:C.cocoa}}>COMPANY INTELLIGENCE — EVIDENCE TIMELINE</span>
      {[90,365,730].map(x=><button key={x} onClick={()=>setDays(x)} style={{fontFamily:mono,fontSize:8.5,padding:'5px 9px',
        background:days===x?'rgba(218,165,32,0.1)':'none',border:`1px solid ${days===x?C.gold:C.b1}`,borderRadius:3,
        color:days===x?C.gold:C.dust,cursor:'pointer'}}>{x}D</button>)}
      {['','MATERIAL','RELEVANT','UNKNOWN','ROUTINE'].map(x=><button key={x||'all'} onClick={()=>setSig(x)} style={{fontFamily:mono,fontSize:8.5,padding:'5px 9px',
        background:sig===x?'rgba(218,165,32,0.1)':'none',border:`1px solid ${sig===x?C.gold:C.b1}`,borderRadius:3,
        color:sig===x?(SIG[x]||C.gold):C.dust,cursor:'pointer'}}>{x||'ALL'}</button>)}
    </div>
    {d.note&&<div style={{fontFamily:mono,fontSize:11,color:C.warn,marginBottom:12}}>{d.note}</div>}
    <div style={{display:'grid',gridTemplateColumns:'minmax(300px,2fr) minmax(220px,1fr)',gap:14}}>
      <div style={{background:C.s2,border:`1px solid ${C.b1}`,borderRadius:10,padding:'6px 4px'}}>
        {d.events.length===0&&<div style={{fontFamily:mono,fontSize:11,color:C.cocoa,padding:14}}>NO EVENTS IN WINDOW</div>}
        {d.events.map(e=>{
          const tx=e.derived.filter(x=>x.field==='transaction').map(x=>x.value);
          const om=tx.filter((t:any)=>t.open_market);
          return (<div key={e.id} style={{display:'grid',gridTemplateColumns:'92px 1fr',gap:12,padding:'10px 12px',
            borderBottom:'1px solid rgba(58,41,32,0.5)',borderLeft:`3px solid ${SIG[e.significance]||C.b1}`}}>
            <div style={{fontFamily:mono,fontSize:9,color:C.dust,lineHeight:1.7}}>
              <div style={{color:C.cream}}>{e.available_at.slice(0,10)}</div>
              <div style={{color:C.cocoa,fontSize:8}}>PUBLIC {e.available_at.slice(11,16)}Z</div>
              {e.event_date&&e.lag_days!=null&&e.lag_days>0&&<div style={{color:C.caramel,fontSize:8}}>event {e.event_date} · +{e.lag_days}d</div>}
            </div>
            <div>
              <div style={{display:'flex',gap:8,alignItems:'center',flexWrap:'wrap'}}>
                <span style={{fontFamily:mono,fontSize:8,letterSpacing:1,color:SIG[e.significance],border:`1px solid ${SIG[e.significance]}`,padding:'2px 6px',borderRadius:3}}>{e.significance}</span>
                <span style={{fontFamily:mono,fontSize:8,letterSpacing:1,color:C.bull,border:`1px solid ${C.bull}`,padding:'2px 6px',borderRadius:3}}>{e.evidence_tier}</span>
                <span style={{fontFamily:mono,fontSize:9,color:C.cocoa}}>{e.source.form}{e.item_code&&e.item_code!=='FORM4'?` · ITEM ${e.item_code}`:''}</span>
                <a href={e.source.url} target="_blank" rel="noopener noreferrer" style={{fontFamily:mono,fontSize:8.5,color:C.dust}}>{e.source.accession} ↗</a>
              </div>
              <div style={{fontFamily:mono,fontSize:11.5,color:C.latte,marginTop:5}}>{e.title}</div>
              {om.length>0&&<div style={{fontFamily:mono,fontSize:9.5,color:C.dust,marginTop:4}}>
                {om.slice(0,3).map((t:any,i:number)=>(<span key={i} style={{marginRight:12,color:t.code==='P'?C.bull:C.bear}}>
                  {t.code==='P'?'BUY':'SELL'} {Math.round(t.shares).toLocaleString()} @ ${t.price} = {money(t.value)}
                  {t.owners?.[0]?.officer_title?` · ${t.owners[0].officer_title}`:''}</span>))}
                {om.length>3&&<span style={{color:C.cocoa}}>+{om.length-3} more</span>}
              </div>}
            </div>
          </div>);})}
      </div>
      <div>
        <div style={{background:C.s2,border:`1px solid ${C.b1}`,borderRadius:10,padding:16,marginBottom:14}}>
          <div style={{fontFamily:mono,fontSize:9,letterSpacing:1.5,color:C.cocoa,marginBottom:10}}>WINDOW SUMMARY</div>
          {d.counts&&Object.entries(d.counts).map(([k,v])=>(<div key={k} style={{display:'flex',justifyContent:'space-between',fontFamily:mono,fontSize:11,padding:'4px 0'}}>
            <span style={{color:SIG[k]||C.dust}}>{k}</span><span style={{color:C.latte}}>{v}</span></div>))}
          {ins&&<div style={{marginTop:12,paddingTop:12,borderTop:`1px solid ${C.b1}`,fontFamily:mono,fontSize:11}}>
            <div style={{fontSize:9,letterSpacing:1.5,color:C.cocoa,marginBottom:6}}>INSIDER OPEN-MARKET</div>
            <div style={{color:C.bull}}>{ins.buys} buys · {money(ins.buy_value)}</div>
            <div style={{color:C.bear}}>{ins.sells} sells · {money(ins.sell_value)}</div>
            <div style={{color:ins.net_value>=0?C.bull:C.bear,marginTop:4,fontWeight:700}}>net {ins.net_value>=0?'+':'-'}{money(Math.abs(ins.net_value))}</div>
            <div style={{fontSize:8.5,color:C.cocoa,marginTop:4}}>{ins.note}</div>
          </div>}
        </div>
        <div style={{background:C.s2,border:`1px solid ${C.b1}`,borderRadius:10,padding:16}}>
          <div style={{fontFamily:mono,fontSize:9,letterSpacing:1.5,color:C.cocoa,marginBottom:10}}>EVIDENCE FAMILIES</div>
          {Object.entries(d.families).map(([k,v])=>(<div key={k} style={{display:'flex',justifyContent:'space-between',fontFamily:mono,fontSize:9.5,padding:'3px 0'}}>
            <span style={{color:C.latte}}>{FAM_LABEL[k]||k}</span>
            <span style={{color:v==='active'?C.bull:v==='no_source'?C.cocoa:C.dust}}>
              {v==='active'?'● ACTIVE':v==='no_source'?'— NO SOURCE':`○ ${v.replace('planned_','').toUpperCase()}`}</span></div>))}
        </div>
      </div>
    </div>
    {d.pit_note&&<div style={{marginTop:12,fontFamily:mono,fontSize:10,color:C.cocoa,lineHeight:1.7}}>{d.pit_note}</div>}
  </div>);
};
export default CompanyIntel;
