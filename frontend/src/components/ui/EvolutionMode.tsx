// Pattern Evolution — measured state transitions, never modeled.
import React, { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
const C={s2:'#241610',b1:'#3a2920',b2:'#4a3428',gold:'#daa520',cocoa:'#8a7560',dust:'#9d8b7a',
  latte:'#d4c4b0',cream:'#f4e8d8',bull:'#22c55e',bear:'#ef4444',warn:'#f59e0b'};
const mono="'Fira Code',monospace";
type St={n:number;positive_pct:number;median_pct:number;p25_pct:number;p75_pct:number};
type Res={ticker:string;current_state:string;inputs:{mom_60d_pct:number;vol_pctile:number};
  state_definition:string;history:{n:number;transitions:Record<string,{pct:number;fwd20:St|null}>}|null;
  all_states:Record<string,number>;note:string;generated:string};
const EvolutionMode:React.FC<{ticker:string}>=({ticker})=>{
  const [d,setD]=useState<Res|null>(null); const [err,setErr]=useState('');
  useEffect(()=>{let dead=false;(async()=>{setD(null);setErr('');
    try{const r=await api.get(`/api/v6/patterns/evolution/${ticker}`);if(!dead)setD(r.data);}
    catch(e:any){if(!dead)setErr(e?.response?.data?.detail||'evolution unavailable');}})();
    return()=>{dead=true};},[ticker]);
  if(err)return <div style={{fontFamily:mono,fontSize:11,color:C.warn}}>{err}</div>;
  if(!d)return <div style={{fontFamily:mono,fontSize:11,color:C.dust}}>classifying state…</div>;
  return (<div>
    <div style={{display:'grid',gridTemplateColumns:'minmax(240px,1fr) minmax(320px,1.8fr)',gap:14}}>
      <div style={{background:C.s2,border:`1px solid ${C.b1}`,borderRadius:10,padding:20}}>
        <div style={{fontFamily:mono,fontSize:9,letterSpacing:1.5,color:C.cocoa,marginBottom:10}}>
          {d.ticker} — CURRENT STATE</div>
        <div style={{fontFamily:mono,fontSize:26,fontWeight:700,color:C.gold,letterSpacing:1}}>
          {d.current_state.replace(/_/g,' · ')}</div>
        <div style={{fontFamily:mono,fontSize:11,color:C.dust,marginTop:10,lineHeight:1.9}}>
          60d momentum {d.inputs.mom_60d_pct>=0?'+':''}{d.inputs.mom_60d_pct}%<br/>
          volatility {d.inputs.vol_pctile}th pctile</div>
        <div style={{marginTop:14,fontFamily:mono,fontSize:9,color:C.cocoa,lineHeight:1.7}}>
          STATES: {d.state_definition}</div>
      </div>
      <div style={{background:C.s2,border:`1px solid ${C.b1}`,borderRadius:10,padding:'16px 18px'}}>
        <div style={{fontFamily:mono,fontSize:9,letterSpacing:1.5,color:C.cocoa,marginBottom:12}}>
          WHERE THIS STATE HISTORICALLY WENT (20 SESSIONS LATER)
          {d.history&&` — ${d.history.n.toLocaleString()} SAMPLES`}</div>
        {d.history?Object.entries(d.history.transitions).map(([s1,t])=>(
          <div key={s1} style={{marginBottom:10}}>
            <div style={{display:'flex',justifyContent:'space-between',fontFamily:mono,fontSize:11,marginBottom:4}}>
              <span style={{color:C.cream}}>→ {s1.replace(/_/g,' · ')}</span>
              <span style={{color:C.latte}}>{t.pct}% of the time
                {t.fwd20&&<span style={{color:t.fwd20.median_pct>=0?C.bull:C.bear}}>
                  {'  '}· med {t.fwd20.median_pct>=0?'+':''}{t.fwd20.median_pct}% over the leg</span>}</span>
            </div>
            <div style={{height:6,background:'rgba(0,0,0,0.3)',borderRadius:3}}>
              <div style={{height:6,width:`${t.pct}%`,borderRadius:3,
                background:`linear-gradient(90deg,${C.gold},${C.b2})`}}/>
            </div>
          </div>
        )):<div style={{fontFamily:mono,fontSize:11,color:C.cocoa}}>INSUFFICIENT SAMPLES FOR THIS STATE</div>}
      </div>
    </div>
    <div style={{marginTop:12,fontFamily:mono,fontSize:10,color:C.cocoa,lineHeight:1.7}}>
      Transition frequencies COUNTED from {Object.values(d.all_states).reduce((a,b)=>a+b,0).toLocaleString()} historical
      state observations across 9 states — measured recurrence, not a forecast model. {d.note} Generated {d.generated}.
    </div>
  </div>);
};
export default EvolutionMode;
