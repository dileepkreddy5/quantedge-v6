import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; }
interface Cat { id:string; label:string; weight:number; score:number|null; n_signals:number; n_scored:number; signals:Sig[]; }
interface Holder { holder:string; percent:number|null; }
interface OData { ticker:string; available:boolean; score:number|null; ownership_rating:string; coverage:{scored:number;total:number};
  insider_available:boolean; institutional_available:boolean; top_holders:Holder[]; tree:{categories:Cat[]}; key_metrics:Record<string,number|null>; reason?:string; }
const heat=(s:number|null)=>s==null?'var(--border-2)':s>=70?'var(--gold)':s>=50?'var(--caramel)':s>=30?'#c9762f':'var(--bear)';
const rc=(r:string)=>r.includes('Strong')?'var(--gold)':r.includes('Stable')?'var(--gold)':r.includes('Mixed')?'var(--caramel)':r.includes('Weak')?'#c9762f':'var(--bear)';
const fmt=(id:string,v:number|null):string=>{
  if(v==null) return '—';
  if(id.includes('trend')||id.includes('conviction')||id.includes('pressure')||id.includes('intensity')||id.includes('gap')||id.includes('direction')) return (v*100).toFixed(2)+'%';
  if(id.includes('pct')||id.includes('concentration')||id.includes('stake')) return v.toFixed(1)+'%';
  if(id.includes('ratio')||id.includes('cluster')||id.includes('stability')||id.includes('liquidity')||id.includes('buying')||id.includes('conviction')||id.includes('interest')) { if(Math.abs(v)<=1.01) return (v*100).toFixed(0)+'%'; }
  if(id.includes('count')||id.includes('holder')||id.includes('buyers')) return v.toFixed(0);
  if(id.includes('_b')) return v.toFixed(1)+'B';
  return typeof v==='number'?v.toFixed(2):v;
};
export default function OwnershipPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<OData|null>(null);
  const [loading,setLoading]=useState(false); const [err,setErr]=useState('');
  const [expanded,setExpanded]=useState<Record<string,boolean>>({}); const [allOpen,setAllOpen]=useState(false);
  useEffect(()=>{ if(!ticker)return; setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/ownership/${ticker}`).then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No data');else setD(x);})
      .catch(e=>setErr(e?.message||'Request failed')).finally(()=>setLoading(false));
  },[ticker]);
  if(!ticker)return <div style={{color:'#9d8b7a',padding:24}}>Enter a ticker for Ownership Intelligence.</div>;
  if(loading)return <div style={{color:'#daa520',padding:24}}>Analyzing ownership — holders, insider stakes, share structure…</div>;
  if(err)return <div style={{color:'#c0705a',padding:24}}>Ownership: {err}</div>;
  if(!d)return null;
  const km=d.key_metrics||{}; const holders=d.top_holders||[];
  return (
    <div style={{padding:'8px 4px',color:'#e8ddd0'}}>
      <div style={{display:'flex',alignItems:'center',gap:24,marginBottom:14,flexWrap:'wrap'}}>
        <div style={{display:'flex',alignItems:'baseline',gap:10}}>
          <span style={{fontSize:46,fontWeight:700,color:heat(d.score),lineHeight:1}}>{d.score?.toFixed(0)??'—'}</span>
          <span style={{fontSize:16,color:'#9d8b7a'}}>/100</span>
        </div>
        <div>
          <div style={{fontSize:22,fontWeight:700,color:rc(d.ownership_rating),letterSpacing:0.5}}>{d.ownership_rating}</div>
          <div style={{fontSize:11,color:'#9d8b7a',marginTop:2}}>{d.coverage.scored}/{d.coverage.total} signals{d.institutional_available?' · 13G data':''}</div>
        </div>
      </div>
      {/* 13G holders removed — the 13F ring above names every institutional
          holder, not only those past the 5% disclosure threshold. */}

      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(130px,1fr))',gap:8,marginBottom:14}}>
        {[['Share Trend',km.share_count_trend!=null?(km.share_count_trend>0?'+':'')+(km.share_count_trend*100).toFixed(2)+'%':'—'],
          ['Major Holders',km.major_holder_count!=null?km.major_holder_count.toFixed(0):'—'],
          ['Top Holder',km.top_holder_pct!=null?km.top_holder_pct.toFixed(1)+'%':'—'],
          ['Buyback Intensity',km.buyback_intensity!=null?(km.buyback_intensity*100).toFixed(2)+'%':'—'],
          ['Dilution',km.dilution_pressure!=null?(km.dilution_pressure*100).toFixed(1)+'%':'—']].map(([k,v])=>(
          <div key={k as string} style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderRadius:4,padding:'8px 10px'}}>
            <div style={{fontFamily:'var(--font-mono)',fontSize:9,letterSpacing:1,color:'var(--cocoa)'}}>{k}</div><div style={{fontFamily:'var(--font-mono)',fontSize:14,fontWeight:600,color:'var(--gold)'}}>{v}</div>
          </div>
        ))}
      </div>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
        <span style={{fontFamily:'var(--font-mono)',fontSize:9,letterSpacing:2,color:'var(--gold)'}}>{d.tree.categories.length} DIMENSIONS · {d.tree.categories.reduce((a,c)=>a+c.n_signals,0)} SIGNALS</span>
        <button onClick={()=>{const v=!allOpen;setAllOpen(v);const m:Record<string,boolean>={};d.tree.categories.forEach(c=>m[c.id]=v);setExpanded(m);}}
          style={{background:'transparent',border:'1px solid var(--border-1)',color:'var(--cocoa-dust)',borderRadius:3,padding:'4px 10px',fontFamily:'var(--font-mono)',fontSize:10,cursor:'pointer'}}>{allOpen?'Collapse all':'Expand all'}</button>
      </div>
      <div style={{display:'flex',flexDirection:'column',gap:8}}>
        {d.tree.categories.map(cat=>{const open=expanded[cat.id];return (
          <div key={cat.id} style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderRadius:4,overflow:'hidden'}}>
            <div onClick={()=>setExpanded(p=>({...p,[cat.id]:!p[cat.id]}))}
              style={{display:'flex',alignItems:'center',gap:12,padding:'9px 14px',cursor:'pointer',borderLeft:`3px solid ${heat(cat.score)}`}}>
              <span style={{fontFamily:'var(--font-mono)',fontSize:10,color:'var(--cocoa)',width:12}}>{open?'▾':'▸'}</span>
              <span style={{fontFamily:'var(--font-body)',fontSize:12.5,fontWeight:600,color:'var(--latte)',width:210}}>{cat.label}</span>
              <div style={{flex:1,height:5,background:'var(--surface-3)',borderRadius:2,minWidth:100}}>
                <div style={{height:'100%',width:`${cat.score??0}%`,background:heat(cat.score),borderRadius:2,opacity:0.85}}/></div>
              <span style={{fontFamily:'var(--font-mono)',fontSize:9,color:'var(--cocoa)',width:76,textAlign:'right'}}>wt {cat.weight.toFixed(2)} · {cat.n_scored}/{cat.n_signals}</span>
              <span style={{fontFamily:'var(--font-mono)',fontSize:16,fontWeight:700,color:heat(cat.score),width:32,textAlign:'right'}}>{cat.score?.toFixed(0)??'—'}</span>
            </div>
            {open && (<div style={{padding:'4px 14px 12px 30px'}}>
              {cat.signals.map(s=>{const pending=s.status==='needs_source'||s.score==null;return (
                <div key={s.id} title={s.evidence} style={{display:'grid',gridTemplateColumns:'minmax(150px,220px) 90px 1fr 30px',alignItems:'center',gap:14,padding:'5px 0',borderBottom:'1px solid var(--border-1)',opacity:pending?0.45:1}}>
                  <span style={{fontFamily:'var(--font-body)',fontSize:12,color:'var(--latte)'}}>{s.label}</span>
                  <span style={{fontFamily:'var(--font-mono)',fontSize:11.5,color:'var(--cocoa-dust)',textAlign:'right'}}>{pending?'—':fmt(s.id,s.raw_value)}</span>
                  <div style={{height:5,background:'var(--surface-3)',borderRadius:2,overflow:'hidden'}}>{!pending && <div style={{height:'100%',width:`${s.score}%`,background:heat(s.score),borderRadius:2}}/>}</div>
                  <span style={{fontFamily:'var(--font-mono)',fontSize:11,fontWeight:600,color:pending?'var(--cocoa)':heat(s.score),textAlign:'right'}}>{pending?'—':s.score!.toFixed(0)}</span>
                </div>);})}
            </div>)}
          </div>);})}
      </div>
    </div>
  );
}
