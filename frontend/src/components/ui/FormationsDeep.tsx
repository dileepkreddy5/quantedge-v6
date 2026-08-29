// Formations research report — every figure from formations_scan v2.
import React, { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
const C={s2:'#241610',b1:'#3a2920',gold:'#daa520',caramel:'#d4956c',cocoa:'#8a7560',
  dust:'#9d8b7a',latte:'#d4c4b0',cream:'#f4e8d8',bull:'#22c55e',bear:'#ef4444',warn:'#f59e0b'};
const mono="'Fira Code',monospace";
type St={n:number;positive_pct:number;median_pct:number;mean_pct:number;p25_pct:number;p75_pct:number};
type Form={occurrences:number;raw_detections:number;median_duration:number;duration_p25_p75:[number,number]|null;
  breakout_up_pct:number|null;follow_through_pct:number|null;confirmation:string;
  distributions:Record<string,St|null>;by_regime:Record<string,St|null>;
  by_volume:Record<string,St|null>;by_volatility:Record<string,St|null>;
  examples:{ticker:string;start:string;end:string;duration:number;breakout_up:boolean;
    regime:string;volume_slope:number;vol_pctile:number;fwd_5d:number|null;fwd_20d:number|null;
    fwd_60d:number|null;fwd_120d:number|null}[]};
const LBL:Record<string,string>={head_shoulders:'HEAD & SHOULDERS',inv_head_shoulders:'INV. H&S',
  double_top:'DOUBLE TOP',double_bottom:'DOUBLE BOTTOM',triple_top:'TRIPLE TOP',triple_bottom:'TRIPLE BOTTOM',
  ascending_triangle:'ASC. TRIANGLE',descending_triangle:'DESC. TRIANGLE',symmetrical_triangle:'SYM. TRIANGLE',
  rectangle:'RECTANGLE',rising_wedge:'RISING WEDGE',falling_wedge:'FALLING WEDGE'};
const pf=(v:number|null|undefined,d=2)=>v==null?'—':`${v>=0?'+':''}${v.toFixed(d)}%`;
const Cell:React.FC<{label:string;s:St|null}>=({label,s})=>(
  <div style={{display:'flex',justifyContent:'space-between',padding:'6px 4px',fontFamily:mono,fontSize:10.5}}>
    <span style={{color:C.latte}}>{label}</span>
    <span style={{color:s?(s.positive_pct>=50?C.bull:C.bear):C.cocoa}}>
      {s?`${s.positive_pct}% · med ${pf(s.median_pct)} · n=${s.n}`:'INSUFFICIENT'}</span></div>);
const FormationsDeep:React.FC=()=>{
  const [art,setArt]=useState<{generated:string;method:string;universe:number;
    formations:Record<string,Form>}|null>(null);
  const [sel,setSel]=useState('double_top'); const [err,setErr]=useState('');
  useEffect(()=>{(async()=>{try{const r=await api.get('/api/v6/patterns/formations');setArt(r.data);}
    catch(e:any){setErr(e?.response?.data?.detail||'scan unavailable');}})();},[]);
  if(err)return <div style={{fontFamily:mono,fontSize:11,color:C.warn}}>{err}</div>;
  if(!art)return <div style={{fontFamily:mono,fontSize:11,color:C.dust}}>loading formation library…</div>;
  const f=art.formations[sel];
  return (<div>
    <div style={{display:'flex',flexWrap:'wrap',gap:6,marginBottom:16}}>
      {Object.keys(art.formations).sort().map(k=>(
        <button key={k} onClick={()=>setSel(k)} style={{fontFamily:mono,fontSize:8.5,letterSpacing:1,
          padding:'6px 10px',background:sel===k?'rgba(218,165,32,0.1)':'none',
          border:`1px solid ${sel===k?C.gold:C.b1}`,borderRadius:4,color:sel===k?C.gold:C.dust,
          cursor:'pointer'}}>{LBL[k]||k} · {art.formations[k].occurrences}</button>))}
    </div>
    {f&&(<>
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(280px,1fr))',gap:14}}>
        <div style={{background:C.s2,border:`1px solid ${C.b1}`,borderRadius:10,padding:18}}>
          <div style={{fontFamily:mono,fontSize:9,letterSpacing:1.5,color:C.cocoa,marginBottom:12}}>
            {LBL[sel]} — FORMATION</div>
          {[['OCCURRENCES (NON-OVERLAP)',String(f.occurrences)],['RAW DETECTIONS',String(f.raw_detections)],
            ['MEDIAN LENGTH',`${f.median_duration}d${f.duration_p25_p75?` (p25/75 ${f.duration_p25_p75[0]}–${f.duration_p25_p75[1]}d)`:''}`],
            ['BROKE OUT UPWARD',f.breakout_up_pct!=null?`${f.breakout_up_pct}%`:'—'],
            ['FOLLOW-THROUGH AT +20D',f.follow_through_pct!=null?`${f.follow_through_pct}%`:'INSUFFICIENT'],
          ].map(([k,v])=>(<div key={k} style={{display:'flex',justifyContent:'space-between',
            padding:'7px 0',fontFamily:mono,fontSize:11}}>
            <span style={{color:C.cocoa,fontSize:9,letterSpacing:1}}>{k}</span>
            <span style={{color:C.latte}}>{v}</span></div>))}
          <div style={{marginTop:10,fontFamily:mono,fontSize:9.5,color:C.cocoa,lineHeight:1.6}}>
            CONFIRMATION: {f.confirmation}</div>
        </div>
        <div style={{background:C.s2,border:`1px solid ${C.b1}`,borderRadius:10,padding:18}}>
          <div style={{fontFamily:mono,fontSize:9,letterSpacing:1.5,color:C.cocoa,marginBottom:12}}>
            OUTCOMES AFTER CONFIRMATION</div>
          {Object.entries(f.distributions).map(([h,s])=>(
            <div key={h} style={{display:'flex',justifyContent:'space-between',padding:'6px 0',
              fontFamily:mono,fontSize:11,borderBottom:'1px solid rgba(58,41,32,0.4)'}}>
              <span style={{color:C.cream,fontWeight:700}}>+{h}</span>
              <span style={{color:s?(s.positive_pct>=50?C.bull:C.bear):C.cocoa}}>
                {s?`${s.positive_pct}% pos · med ${pf(s.median_pct)} · p25/75 ${pf(s.p25_pct)}/${pf(s.p75_pct)} · n=${s.n}`:'INSUFFICIENT'}
              </span></div>))}
        </div>
        <div style={{background:C.s2,border:`1px solid ${C.b1}`,borderRadius:10,padding:18}}>
          <div style={{fontFamily:mono,fontSize:9,letterSpacing:1.5,color:C.cocoa,marginBottom:10}}>
            CONDITIONAL (+20D)</div>
          {Object.entries(f.by_regime).sort().map(([k,s])=><Cell key={k} label={k.replace(/_/g,' ')} s={s}/>)}
          <div style={{height:8}}/>
          <Cell label="RISING VOLUME" s={f.by_volume.rising}/>
          <Cell label="FALLING VOLUME" s={f.by_volume.falling}/>
          <Cell label="HIGH VOLATILITY" s={f.by_volatility.high}/>
          <Cell label="LOW VOLATILITY" s={f.by_volatility.low}/>
        </div>
      </div>
      <div style={{background:C.s2,border:`1px solid ${C.b1}`,borderRadius:10,padding:'14px 10px',
                   marginTop:14,overflowX:'auto'}}>
        <div style={{fontFamily:mono,fontSize:9,letterSpacing:1.5,color:C.cocoa,padding:'0 8px 8px'}}>
          MOST RECENT OCCURRENCES</div>
        <table style={{width:'100%',borderCollapse:'collapse',fontFamily:mono,fontSize:10}}>
          <thead><tr>{['TICKER','FORMED','LEN','BREAK','REGIME','VOL SL','VOL %','+5D','+20D','+60D','+120D'].map(h=>(
            <th key={h} style={{textAlign:'left',padding:'5px 8px',color:C.cocoa,fontSize:8.5,
              letterSpacing:1,fontWeight:500,borderBottom:`1px solid ${C.b1}`}}>{h}</th>))}</tr></thead>
          <tbody>{f.examples.map(e=>(<tr key={e.ticker+e.end}>
            <td style={{padding:'6px 8px',color:C.cream,fontWeight:700}}>{e.ticker}</td>
            <td style={{padding:'6px 8px',color:C.dust}}>{e.start} → {e.end}</td>
            <td style={{padding:'6px 8px',color:C.dust}}>{e.duration}d</td>
            <td style={{padding:'6px 8px',color:e.breakout_up?C.bull:C.bear}}>{e.breakout_up?'▲':'▼'}</td>
            <td style={{padding:'6px 8px',color:C.latte,fontSize:9}}>{e.regime.replace(/_/g,' ')}</td>
            <td style={{padding:'6px 8px',color:e.volume_slope>=0?C.bull:C.bear}}>{e.volume_slope>=0?'+':''}{e.volume_slope.toFixed(2)}</td>
            <td style={{padding:'6px 8px',color:C.dust}}>{Math.round(e.vol_pctile*100)}%</td>
            {[e.fwd_5d,e.fwd_20d,e.fwd_60d,e.fwd_120d].map((v,i)=>(
              <td key={i} style={{padding:'6px 8px',color:v==null?C.cocoa:(v>=0?C.bull:C.bear)}}>{pf(v)}</td>))}
          </tr>))}</tbody></table>
      </div>
      <div style={{marginTop:12,fontFamily:mono,fontSize:10,color:C.cocoa,lineHeight:1.7}}>
        METHOD: {art.method} · Universe {art.universe} tickers · Generated {art.generated}. DESCRIPTIVE RESULT.
      </div>
    </>)}
  </div>);
};
export default FormationsDeep;
