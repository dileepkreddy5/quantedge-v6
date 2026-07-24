import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

interface H { manager:string; shares:number; value_usd:number; n_entities:number; pct_of_company:number|null; }
interface N { manager:string; shares:number; value_usd:number;
  pct_of_company:number|null; pct_of_their_book:number|null; }
interface Inst {
  available:boolean; quarter?:string; issuer?:string; n_managers?:number;
  notable?:{available:boolean; n_notable?:number; holders?:N[]};
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

      {(() => {
        const nb = d.notable;
        if (!nb?.available || !(nb.holders||[]).length) return null;
        // Sorted by weight in their own book, not by size. An index fund holds
        // every large company by construction; a manager with 2.5% of what they
        // run in one name has made a decision, and that ordering surfaces it.
        const rows = [...(nb.holders||[])]
          .filter(h => h.pct_of_their_book != null)
          .sort((a,b) => (b.pct_of_their_book!) - (a.pct_of_their_book!))
          .slice(0, 8);
        if (!rows.length) return null;
        const maxBook = Math.max(...rows.map(r => r.pct_of_their_book!));
        return (
          <div style={{marginTop:18,paddingTop:14,borderTop:'1px solid var(--border-1)'}}>
            <div style={{fontFamily:'var(--font-mono)',fontSize:9,letterSpacing:2,color:'var(--gold)',marginBottom:4}}>
              MANAGERS WHO CHOSE THIS POSITION
            </div>
            <div style={{fontFamily:'var(--font-body)',fontSize:11,color:'var(--cocoa-dust)',marginBottom:10,lineHeight:1.55}}>
              Index funds hold every large company by construction, so their presence says nothing.
              These are discretionary managers, ranked by how much of their own book sits in this name —
              the measure of conviction rather than of size.
            </div>
            {rows.map(h => (
              <div key={h.manager} style={{display:'grid',gridTemplateColumns:'minmax(140px,190px) 1fr 74px 66px',
                gap:12,alignItems:'center',padding:'6px 0',borderBottom:'1px solid var(--border-1)'}}>
                <span style={{fontFamily:'var(--font-body)',fontSize:12,color:'var(--latte)'}}>{h.manager}</span>
                <div style={{height:5,background:'var(--surface-3)',borderRadius:2,minWidth:60}}>
                  <div style={{height:'100%',width:`${(h.pct_of_their_book!/maxBook)*100}%`,
                    background:'var(--gold)',borderRadius:2,opacity:0.85}}/></div>
                <span style={{fontFamily:'var(--font-mono)',fontSize:12,color:'var(--gold)',textAlign:'right'}}>
                  {h.pct_of_their_book!.toFixed(2)}%
                </span>
                <span style={{fontFamily:'var(--font-mono)',fontSize:10,color:'var(--cocoa)',textAlign:'right'}}>
                  ${(h.value_usd/1e9).toFixed(1)}B
                </span>
              </div>
            ))}
            <div style={{fontFamily:'var(--font-body)',fontSize:10.5,color:'var(--cocoa)',marginTop:8}}>
              Percentage is this holding as a share of the manager's entire reported book · dollar figure is the position's value
            </div>
          </div>
        );
      })()}
    </div>
  );
}
