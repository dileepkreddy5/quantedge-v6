import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

interface Q {
  fiscal:string; period_end:string|null;
  diluted_shares:number|null; diluted_shares_yoy_pct:number|null;
  buybacks:number|null; sbc:number|null; dividends_paid:number|null;
}
const bn=(v:number|null)=>v==null?'—':`$${(v/1e9).toFixed(1)}B`;

export default function ClaimPanel({ ticker }:{ ticker:string }){
  const [q,setQ]=useState<Q[]|null>(null);
  const [err,setErr]=useState('');
  useEffect(()=>{
    if(!ticker) return;
    setQ(null); setErr('');
    api.get(`/api/v6/quarters/${ticker}`)
      .then(r=>{const x=r.data?.data; if(!x?.available) setErr(x?.reason||'no data'); else setQ(x.quarters);})
      .catch(e=>setErr(e?.message||'request failed'));
  },[ticker]);

  if(err) return null;
  if(!q || q.length<4) return null;

  const shares=q.map(r=>r.diluted_shares).filter(v=>v!=null) as number[];
  if(shares.length<4) return null;
  const first=shares[0], last=shares[shares.length-1];
  const totalChange=((last/first)-1)*100;
  const bbTot=q.reduce((s,r)=>s+(r.buybacks||0),0);
  const sbcTot=q.reduce((s,r)=>s+(r.sbc||0),0);
  const divTot=q.reduce((s,r)=>s+(r.dividends_paid||0),0);
  const ratio = sbcTot>0 ? bbTot/sbcTot : null;

  const W=1000,H=150,PAD={l:8,r:8,t:14,b:20};
  const sLo=Math.min(...shares), sHi=Math.max(...shares);
  const sx=(i:number)=>PAD.l+(i/(q.length-1))*(W-PAD.l-PAD.r);
  const sy=(v:number)=>H-PAD.b-((v-sLo)/((sHi-sLo)||1))*(H-PAD.t-PAD.b);
  const path=q.map((r,i)=>r.diluted_shares==null?null:`${i===0?'M':'L'}${sx(i).toFixed(1)},${sy(r.diluted_shares).toFixed(1)}`).filter(Boolean).join(' ');

  const flowMax=Math.max(...q.map(r=>Math.max(r.buybacks||0,r.sbc||0)),1);

  return (
    <div style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderRadius:6,padding:'16px 18px',marginBottom:16}}>
      <div style={{fontFamily:'var(--font-mono)',fontSize:9,letterSpacing:2,color:'var(--gold)',marginBottom:4}}>YOUR CLAIM ON THE BUSINESS</div>
      <div style={{fontFamily:'var(--font-body)',fontSize:11,color:'var(--cocoa-dust)',marginBottom:14,lineHeight:1.55}}>
        Who owns the company matters less than whether your share of it is growing. Diluted share count over
        twelve quarters, with buybacks and stock compensation as the opposing flows that move it.
      </div>

      <div style={{display:'flex',gap:26,flexWrap:'wrap',marginBottom:14}}>
        {[['SHARE COUNT', totalChange>=0?`+${totalChange.toFixed(1)}%`:`${totalChange.toFixed(1)}%`,
           totalChange<0?'var(--gold)':'var(--bear)', totalChange<0?'fewer shares — your slice grew':'more shares — your slice shrank'],
          ['BOUGHT BACK', bn(bbTot), 'var(--latte)', 'over twelve quarters'],
          ['ISSUED AS PAY', bn(sbcTot), 'var(--latte)', 'stock compensation'],
          ['DIVIDENDS', bn(divTot), 'var(--latte)', 'paid to holders'],
          ['BUYBACK : SBC', ratio==null?'—':`${ratio.toFixed(1)}x`,
            ratio!=null&&ratio>1?'var(--gold)':'var(--bear)',
            ratio!=null&&ratio>1?'retiring more than it issues':'issuing more than it retires'],
        ].map(([l,v,c,n]:any)=>(
          <div key={l}>
            <div style={{fontFamily:'var(--font-mono)',fontSize:9,letterSpacing:1,color:'var(--cocoa)'}}>{l}</div>
            <div style={{fontFamily:'var(--font-mono)',fontSize:19,fontWeight:700,color:c,marginTop:2}}>{v}</div>
            <div style={{fontFamily:'var(--font-body)',fontSize:10,color:'var(--cocoa)'}}>{n}</div>
          </div>
        ))}
      </div>

      <div style={{fontFamily:'var(--font-mono)',fontSize:9,color:'var(--cocoa)',letterSpacing:1,marginBottom:2}}>DILUTED SHARES OUTSTANDING</div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{width:'100%',display:'block'}}>
        <path d={path} fill="none" stroke="var(--gold)" strokeWidth={2}/>
        {q.map((r,i)=>r.diluted_shares==null?null:<circle key={i} cx={sx(i)} cy={sy(r.diluted_shares)} r={2.5} fill="var(--gold)"/>)}
        <text x={PAD.l} y={H-4} fill="var(--cocoa)" fontSize="9" fontFamily="monospace">{(sLo/1e9).toFixed(2)}B low</text>
        <text x={W-PAD.r} y={H-4} fill="var(--cocoa)" fontSize="9" fontFamily="monospace" textAnchor="end">{q[q.length-1].fiscal}</text>
      </svg>

      <div style={{fontFamily:'var(--font-mono)',fontSize:9,color:'var(--cocoa)',letterSpacing:1,margin:'12px 0 4px'}}>
        BUYBACKS (gold, above) VS STOCK COMPENSATION (caramel, below) — PER QUARTER
      </div>
      <svg viewBox={`0 0 ${W} 110`} style={{width:'100%',display:'block'}}>
        <line x1={0} y1={55} x2={W} y2={55} stroke="var(--border-2)"/>
        {q.map((r,i)=>{
          const bw=(W/q.length)*0.5, cx=(i+0.5)*(W/q.length);
          const bh=((r.buybacks||0)/flowMax)*44, sh=((r.sbc||0)/flowMax)*44;
          return (
            <g key={i}>
              <rect x={cx-bw/2} y={55-bh} width={bw} height={bh} fill="var(--gold)" opacity={0.85} rx={1}/>
              <rect x={cx-bw/2} y={55} width={bw} height={sh} fill="var(--caramel)" opacity={0.7} rx={1}/>
              <text x={cx} y={106} fill="var(--cocoa)" fontSize="8" fontFamily="monospace" textAnchor="middle">{r.fiscal.split(' ')[1]||''}</text>
            </g>
          );
        })}
      </svg>
      <div style={{fontFamily:'var(--font-body)',fontSize:11.5,color:'var(--latte)',marginTop:12,lineHeight:1.55}}>
        {ratio!=null && ratio>1
          ? `Retiring ${ratio.toFixed(1)} dollars of stock for every dollar issued as compensation, taking the share count ${Math.abs(totalChange).toFixed(1)}% ${totalChange<0?'lower':'higher'} over three years.`
          : ratio!=null
          ? `Issuing more stock as compensation than it retires, taking the share count ${Math.abs(totalChange).toFixed(1)}% ${totalChange<0?'lower':'higher'} over three years.`
          : `Share count ${Math.abs(totalChange).toFixed(1)}% ${totalChange<0?'lower':'higher'} over three years; no buyback programme reported.`}
      </div>
    </div>
  );
}
