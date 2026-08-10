import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

interface Q {
  period_end:string|null; fiscal:string;
  revenue:number|null; revenue_yoy_pct:number|null; revenue_qoq_pct:number|null;
  gross_margin_pct:number|null; operating_margin_pct:number|null; net_margin_pct:number|null;
  net_income:number|null; net_income_yoy_pct:number|null; net_income_qoq_pct:number|null;
  eps_diluted:number|null; eps_yoy_pct:number|null; eps_qoq_pct:number|null;
  free_cash_flow:number|null;
}

const money=(v:number|null)=>{ if(v==null)return '—'; const a=Math.abs(v);
  if(a>=1e9)return `$${(v/1e9).toFixed(1)}B`; if(a>=1e6)return `$${(v/1e6).toFixed(0)}M`; return `$${v.toFixed(0)}`; };
const pct=(v:number|null)=>v==null?'—':`${v>=0?'+':''}${v.toFixed(1)}%`;
const dir=(v:number|null)=>v==null?'var(--cocoa)':v>=0?'var(--gold)':'var(--bear)';

export default function QuartersPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<Q[]|null>(null);
  const [filing,setFiling]=useState<string|null>(null);
  const [err,setErr]=useState(''); const [loading,setLoading]=useState(false);

  useEffect(()=>{ if(!ticker)return; setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/quarters/${ticker}`)
      .then(r=>{const x=r.data?.data; if(!x?.available)setErr(x?.reason||'no quarterly data'); else {setD(x.quarters);setFiling(x.latest_filing_date||null);}})
      .catch(e=>setErr(e?.message||'request failed')).finally(()=>setLoading(false));
  },[ticker]);

  if(loading)return <div style={{color:'var(--gold)',padding:20}}>Loading reported quarters…</div>;
  if(err)return <div style={{color:'var(--bear)',padding:20}}>Quarters: {err}</div>;
  if(!d||d.length<2)return null;

  const rows=[...d].reverse();  // newest first
  const yoys=d.map(r=>r.revenue_yoy_pct).filter(v=>v!=null) as number[];
  const recent=yoys.slice(-4), prior=yoys.slice(-8,-4);
  const avg=(a:number[])=>a.length?a.reduce((s,v)=>s+v,0)/a.length:null;
  const rA=avg(recent), pA=avg(prior);
  const trend=(rA!=null&&pA!=null)?(rA-pA>2?'accelerating':rA-pA<-2?'decelerating':'steady'):null;

  const Th=({children,align='right'}:any)=>(
    <th style={{fontFamily:'var(--font-mono)',fontSize:9,letterSpacing:1,color:'var(--cocoa)',
      textAlign:align,padding:'0 9px 8px',fontWeight:400,whiteSpace:'nowrap'}}>{children}</th>);
  const Td=({children,align='right',color}:any)=>(
    <td style={{fontFamily:'var(--font-mono)',fontSize:11.5,color:color||'var(--latte)',
      textAlign:align,padding:'6px 9px',borderBottom:'1px solid var(--border-1)',whiteSpace:'nowrap'}}>{children}</td>);
  // small dual YoY/QoQ cell
  const Dual=({yoy,qoq}:{yoy:number|null;qoq:number|null})=>(
    <td style={{padding:'6px 9px',borderBottom:'1px solid var(--border-1)',whiteSpace:'nowrap',textAlign:'right'}}>
      <div style={{fontFamily:'var(--font-mono)',fontSize:11.5,color:dir(yoy)}}>{pct(yoy)}</div>
      <div style={{fontFamily:'var(--font-mono)',fontSize:9,color:qoq==null?'var(--cocoa)':qoq>=0?'var(--cocoa-dust)':'#c9762f'}}>
        {qoq==null?'':`${qoq>=0?'▲':'▼'} ${Math.abs(qoq).toFixed(1)}% q/q`}</div>
    </td>);

  return (
    <div style={{marginBottom:20}}>
      <div style={{fontFamily:'var(--font-mono)',fontSize:9,letterSpacing:2,color:'var(--gold)',marginBottom:4}}>
        REPORTED QUARTERS · {d.length} periods{filing?` · latest filed ${filing}`:''}</div>
      <div style={{fontFamily:'var(--font-body)',fontSize:11,color:'var(--cocoa-dust)',marginBottom:12,lineHeight:1.55}}>
        Five years as filed. Each metric shows year-over-year (large) and sequential quarter-over-quarter (small).
        {trend&&rA!=null&&pA!=null&&(
          <> Revenue growth is <b style={{color:trend==='decelerating'?'var(--bear)':'var(--gold)'}}>{trend}</b> —
          {' '}{pA.toFixed(1)}% avg over the earlier four quarters vs {rA.toFixed(1)}% over the most recent four.</>
        )}
      </div>
      <div style={{overflowX:'auto'}}>
        <table style={{width:'100%',borderCollapse:'collapse',minWidth:880}}>
          <thead><tr>
            <Th align="left">QUARTER</Th><Th>REVENUE</Th><Th>REV Y/Y</Th>
            <Th>GROSS</Th><Th>OPER</Th><Th>NET</Th>
            <Th>NET INC</Th><Th>NI Y/Y</Th><Th>EPS</Th><Th>EPS Y/Y</Th><Th>FCF</Th>
          </tr></thead>
          <tbody>
            {rows.map(r=>(
              <tr key={r.period_end||r.fiscal}>
                <Td align="left" color="var(--cocoa-dust)">{r.fiscal}<div style={{fontSize:9,color:'var(--cocoa)'}}>{r.period_end}</div></Td>
                <Td>{money(r.revenue)}</Td>
                <Dual yoy={r.revenue_yoy_pct} qoq={r.revenue_qoq_pct}/>
                <Td>{r.gross_margin_pct==null?'—':`${r.gross_margin_pct.toFixed(1)}%`}</Td>
                <Td>{r.operating_margin_pct==null?'—':`${r.operating_margin_pct.toFixed(1)}%`}</Td>
                <Td>{r.net_margin_pct==null?'—':`${r.net_margin_pct.toFixed(1)}%`}</Td>
                <Td>{money(r.net_income)}</Td>
                <Dual yoy={r.net_income_yoy_pct} qoq={r.net_income_qoq_pct}/>
                <Td>{r.eps_diluted==null?'—':`$${r.eps_diluted.toFixed(2)}`}</Td>
                <Dual yoy={r.eps_yoy_pct} qoq={r.eps_qoq_pct}/>
                <Td>{money(r.free_cash_flow)}</Td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{fontFamily:'var(--font-mono)',fontSize:9,color:'var(--cocoa)',marginTop:8}}>
        As reported to the SEC · Y/Y and q/q paired by actual period-end date · a gap means that comparison quarter isn't in the source
      </div>
    </div>
  );
}
