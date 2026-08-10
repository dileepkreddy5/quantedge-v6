import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
import BusinessMoat from './BusinessMoat';

interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; method?:string; }
interface Cat { id:string; label:string; weight:number; score:number|null; confidence:number; n_signals:number; n_scored:number; signals:Sig[]; }
interface BizData {
  ticker:string; available:boolean; score:number|null; moat_rating:string; confidence:number;
  coverage:{scored:number;total:number}; tree:{categories:Cat[]}; key_metrics:Record<string,number|null>; reason?:string;
}
const heat=(s:number|null)=>s==null?'var(--border-1)':s>=75?'var(--bull)':s>=58?'var(--bull)':s>=42?'var(--caramel)':s>=25?'#c9762f':'var(--bear)';
const moatColor=(r:string)=>r.includes('Wide')?'#0f9d6e':r.includes('Narrow')?'var(--bull)':r.includes('Emerging')?'var(--caramel)':'var(--bear)';

const fmtVal=(id:string,v:number|null):string=>{
  if(v==null) return '—';
  if(id.includes('economic_profit')) return '$'+(v/1e9).toFixed(1)+'B';
  if(id.includes('persistence')||id.includes('cycle')) return v.toFixed(1);
  if(id.includes('cash_conversion_ratio')||id.includes('turnover')||id.includes('leverage')||id.includes('productivity')||id.includes('resilience')||id.includes('rule_of_40')||id.includes('roic')||id.includes('consistency')||id.includes('stability')||id.includes('predictability')||id.includes('direction')||id.includes('absorption')||id.includes('margin_proxy')||id.includes('incremental_margin')) return v.toFixed(2);
  return (v*100).toFixed(1)+'%';
};

export default function BusinessPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<BizData|null>(null);
  const [loading,setLoading]=useState(false);
  const [err,setErr]=useState('');
  const [expanded,setExpanded]=useState<Record<string,boolean>>({});
  const [allOpen,setAllOpen]=useState(false);
  useEffect(()=>{
    if(!ticker)return;
    setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/business/${ticker}`)
      .then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No business data');else{setD(x);
        const init:Record<string,boolean>={}; (x.tree?.categories||[]).forEach((c:Cat)=>init[c.id]=true); setExpanded(init);}})
      .catch(e=>setErr(e?.message||'Request failed'))
      .finally(()=>setLoading(false));
  },[ticker]);

  if(!ticker)return <div style={{color:'var(--cocoa-dust)',padding:24}}>Enter a ticker for Business Intelligence.</div>;
  if(loading)return <div style={{color:'var(--gold)',padding:24}}>Computing Business Intelligence — 76 moat & durability signals…</div>;
  if(err)return <div style={{color:'var(--bear)',padding:24}}>Business: {err}</div>;
  if(!d)return null;

  const km=d.key_metrics;
  const radarCats=d.tree.categories.filter(c=>c.score!=null).slice(0,10);
  const N=radarCats.length;
  const cx=150,cy=150,R=105;
  const pt=(i:number,r:number)=>{const ang=(-Math.PI/2)+(2*Math.PI*i/N);return [cx+r*Math.cos(ang),cy+r*Math.sin(ang)];};
  const rings=[0.25,0.5,0.75,1.0];
  const dataPoly=radarCats.map((c,i)=>pt(i,R*(c.score!/100))).map(p=>p.join(',')).join(' ');

  const evidence:string[]=[];
  if((km.excess_return_spread??0)>0.1||(km.roic_wacc_spread??0)>0.1) evidence.push(`Persistent excess returns over cost of capital`);
  if((km.gross_margin_level??0)>0.5) evidence.push(`Strong pricing power (${((km.gross_margin_level??0)*100).toFixed(0)}% gross margin)`);
  if((km.recurring_revenue_ratio??0)>0.1) evidence.push(`Recurring revenue base`);
  if((km.reinvestment_quality??0)>0.08) evidence.push('Value-creating reinvestment');
  if(evidence.length===0) evidence.push('Moderate competitive position');

  const toggleAll=()=>{const v=!allOpen;setAllOpen(v);const m:Record<string,boolean>={};d.tree.categories.forEach(c=>m[c.id]=v);setExpanded(m);};

  return (
    <div style={{padding:'8px 4px',color:'var(--latte)'}}>
      <div style={{display:'flex',alignItems:'center',gap:24,marginBottom:20,flexWrap:'wrap'}}>
        <div style={{display:'flex',alignItems:'baseline',gap:10}}>
          <span style={{fontSize:46,fontWeight:700,color:heat(d.score),lineHeight:1}}>{d.score?.toFixed(1)??'—'}</span>
          <span style={{fontSize:16,color:'var(--cocoa-dust)'}}>/100</span>
        </div>
        <div>
          <div style={{fontSize:22,fontWeight:700,color:moatColor(d.moat_rating),letterSpacing:1}}>{d.moat_rating}</div>
          <div style={{fontSize:11,color:'var(--cocoa-dust)',marginTop:2}}>{d.coverage.scored}/{d.coverage.total} durability signals · {(d.confidence*100).toFixed(0)}% confidence</div>
        </div>
        <button onClick={toggleAll} style={{marginLeft:'auto',background:'var(--surface-2)',border:'1px solid #2a2a2a',color:'var(--cocoa-dust)',borderRadius:8,padding:'6px 14px',fontSize:11,cursor:'pointer'}}>
          {allOpen?'Collapse all':'Expand all'}</button>
      </div>

      <BusinessMoat d={d} />

      <div style={{fontSize:12,color:'var(--cocoa-dust)',letterSpacing:1,marginBottom:8}}>10 BUSINESS DIMENSIONS · {d.coverage.total} SIGNALS</div>
      <div style={{display:'flex',flexDirection:'column',gap:8}}>
        {d.tree.categories.map(cat=>{
          const open=expanded[cat.id];
          return (
            <div key={cat.id} style={{background:'var(--surface-2)',border:'1px solid #2a2a2a',borderRadius:10,overflow:'hidden'}}>
              <div onClick={()=>setExpanded(p=>({...p,[cat.id]:!p[cat.id]}))}
                style={{display:'flex',alignItems:'center',gap:12,padding:'10px 14px',cursor:'pointer',borderLeft:`4px solid ${heat(cat.score)}`}}>
                <span style={{fontSize:11,color:'var(--cocoa)',width:12}}>{open?'▾':'▸'}</span>
                <span style={{fontSize:13,fontWeight:600,color:'var(--latte)',flex:1}}>{cat.label}</span>
                <span style={{fontSize:10,color:'var(--cocoa)'}}>wt {cat.weight.toFixed(2)} · {cat.n_scored}/{cat.n_signals}</span>
                <span style={{fontSize:18,fontWeight:700,color:heat(cat.score),width:36,textAlign:'right'}}>{cat.score?.toFixed(0)??'—'}</span>
              </div>
              {open && (
                <div style={{padding:'4px 14px 12px 30px'}}>
                  {cat.signals.map(s=>{
                    const pending=s.status==='needs_source'||s.score==null;
                    return (
                      <div key={s.id} title={s.evidence} style={{display:'flex',alignItems:'center',gap:10,padding:'5px 0',borderBottom:'1px solid #1e1e1e',opacity:pending?0.5:1}}>
                        <span style={{fontSize:12,color:'var(--latte)',flex:1}}>{s.label}</span>
                        <span style={{fontSize:12,color:'var(--cocoa-dust)',width:80,textAlign:'right'}}>{pending?'pending':fmtVal(s.id,s.raw_value)}</span>
                        <div style={{width:90,height:6,background:'var(--surface-3)',borderRadius:3,overflow:'hidden'}}>
                          {!pending && <div style={{height:'100%',width:`${s.score}%`,background:heat(s.score)}}/>}
                        </div>
                        <span style={{fontSize:11,fontWeight:600,color:pending?'var(--cocoa)':heat(s.score),width:26,textAlign:'right'}}>{pending?'—':s.score!.toFixed(0)}</span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
