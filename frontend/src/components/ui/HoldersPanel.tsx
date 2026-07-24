import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

interface H { manager:string; shares:number; value_usd:number; n_entities:number; pct_of_company:number|null; }
interface Inst {
  available:boolean; quarter?:string; issuer?:string; n_managers?:number;
  institutional_pct?:number; other_pct?:number; shares_outstanding?:number;
  institutional_shares?:number; top_holders?:H[]; note?:string; reason?:string;
}

const ARC = ['#d4a53c','#c98f4a','#bf7a58','#b56666','#a85f7a','#8f6b96','#7a76a8','#6a80b0'];
const qLabel=(q?:string)=>{
  if(!q) return '';
  const m=q.match(/-(\d{2})([a-z]{3})(\d{4})$/i);
  if(!m) return q;
  const mon:Record<string,string>={jan:'January',feb:'February',mar:'March',apr:'April',may:'May',jun:'June',
    jul:'July',aug:'August',sep:'September',oct:'October',nov:'November',dec:'December'};
  return `${m[1]} ${mon[m[2].toLowerCase()]||m[2]} ${m[3]}`;
};

export default function HoldersPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<Inst|null>(null);
  const [hover,setHover]=useState<string|null>(null);

  useEffect(()=>{
    if(!ticker) return;
    setD(null);
    api.get(`/api/v6/ownership/${ticker}`)
      .then(r=>setD(r.data?.data?.institutional||null))
      .catch(()=>setD(null));
  },[ticker]);

  if(!d) return null;
  if(!d.available) return (
    <div style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderRadius:6,padding:'14px 18px',marginBottom:16}}>
      <div style={{fontFamily:'var(--font-mono)',fontSize:9,letterSpacing:2,color:'var(--gold)',marginBottom:4}}>WHO OWNS THIS COMPANY</div>
      <div style={{fontFamily:'var(--font-body)',fontSize:11,color:'var(--cocoa-dust)'}}>
        Institutional holdings unavailable — {d.reason}.
      </div>
    </div>
  );

  const inst=d.institutional_pct??0;
  const top=(d.top_holders||[]).filter(h=>h.pct_of_company!=null) as (H&{pct_of_company:number})[];
  const named=top.slice(0,7);
  const namedSum=named.reduce((s,h)=>s+h.pct_of_company,0);
  const restInst=Math.max(0, inst-namedSum);

  const segs=[
    ...named.map((h,i)=>({label:h.manager, pct:h.pct_of_company, color:ARC[i%ARC.length],
      sub:`${h.n_entities>1?`${h.n_entities} filing entities · `:''}${(h.shares/1e6).toFixed(0)}M shares`})),
    ...(restInst>0.5?[{label:`${((d.n_managers||0)-named.length).toLocaleString()} other institutions`,
      pct:restInst, color:'#5c5142', sub:'each below the largest seven'}]:[]),
    {label:'Retail, insiders and funds below the reporting threshold', pct:d.other_pct??0,
     color:'var(--surface-3)', sub:'not required to file a 13F'},
  ];

  const R=118, r=76, CX=140, CY=140;
  let a0=-Math.PI/2;
  const arcs=segs.map(s=>{
    const a1=a0+(s.pct/100)*Math.PI*2;
    const big=(a1-a0)>Math.PI?1:0;
    const p=[`M ${CX+R*Math.cos(a0)} ${CY+R*Math.sin(a0)}`,
             `A ${R} ${R} 0 ${big} 1 ${CX+R*Math.cos(a1)} ${CY+R*Math.sin(a1)}`,
             `L ${CX+r*Math.cos(a1)} ${CY+r*Math.sin(a1)}`,
             `A ${r} ${r} 0 ${big} 0 ${CX+r*Math.cos(a0)} ${CY+r*Math.sin(a0)}`,'Z'].join(' ');
    const out={...s, path:p};
    a0=a1;
    return out;
  });

  return (
    <div style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderRadius:6,padding:'16px 18px',marginBottom:16}}>
      <div style={{fontFamily:'var(--font-mono)',fontSize:9,letterSpacing:2,color:'var(--gold)',marginBottom:4}}>WHO OWNS THIS COMPANY</div>
      <div style={{fontFamily:'var(--font-body)',fontSize:11,color:'var(--cocoa-dust)',marginBottom:10,lineHeight:1.55}}>
        Positions as at {qLabel(d.quarter)}, from every manager required to file a 13F.
        Filings are due within 45 days of quarter end, so this is the most recent complete picture that exists anywhere.
      </div>

      <div style={{display:'flex',gap:28,flexWrap:'wrap',alignItems:'center'}}>
        <svg viewBox="0 0 280 280" style={{width:280,minWidth:220,display:'block'}}>
          {arcs.map(s=>(
            <path key={s.label} d={s.path} fill={s.color}
              opacity={hover===null||hover===s.label?1:0.32}
              stroke="var(--surface-2)" strokeWidth={1.5}
              onMouseEnter={()=>setHover(s.label)} onMouseLeave={()=>setHover(null)}
              style={{cursor:'default',transition:'opacity .12s'}}/>
          ))}
          <text x={CX} y={CY-8} textAnchor="middle" fontFamily="var(--font-mono)" fontSize="30" fontWeight="700" fill="var(--gold)">
            {inst.toFixed(0)}%
          </text>
          <text x={CX} y={CY+12} textAnchor="middle" fontFamily="var(--font-body)" fontSize="11" fill="var(--cocoa-dust)">institutional</text>
          <text x={CX} y={CY+28} textAnchor="middle" fontFamily="var(--font-mono)" fontSize="9" fill="var(--cocoa)">
            {(d.n_managers||0).toLocaleString()} filers
          </text>
        </svg>

        <div style={{flex:1,minWidth:280}}>
          {arcs.map(s=>(
            <div key={s.label}
              onMouseEnter={()=>setHover(s.label)} onMouseLeave={()=>setHover(null)}
              style={{display:'grid',gridTemplateColumns:'10px 1fr 62px',gap:10,alignItems:'center',
                padding:'5px 0',borderBottom:'1px solid var(--border-1)',
                opacity:hover===null||hover===s.label?1:0.45}}>
              <div style={{width:10,height:10,borderRadius:2,background:s.color}}/>
              <div>
                <div style={{fontFamily:'var(--font-body)',fontSize:12,color:'var(--latte)'}}>{s.label}</div>
                <div style={{fontFamily:'var(--font-body)',fontSize:10,color:'var(--cocoa)'}}>{s.sub}</div>
              </div>
              <div style={{fontFamily:'var(--font-mono)',fontSize:13,color:'var(--latte)',textAlign:'right'}}>
                {s.pct.toFixed(2)}%
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
