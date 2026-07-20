import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; }
interface Cat { id:string; label:string; weight:number; score:number|null; n_signals:number; n_scored:number; signals:Sig[]; }
interface Holder { holder:string; percent:number|null; }
interface OData { ticker:string; available:boolean; score:number|null; ownership_rating:string; coverage:{scored:number;total:number};
  insider_available:boolean; institutional_available:boolean; top_holders:Holder[]; tree:{categories:Cat[]}; key_metrics:Record<string,number|null>; reason?:string; }
const heat=(s:number|null)=>s==null?'#2a2a2a':s>=75?'#0f6e56':s>=58?'#1d9e75':s>=42?'#8a7519':s>=25?'#a35a1d':'#7a2320';
const rc=(r:string)=>r==='Strong Hands'?'#0f9d6e':r==='Stable'?'#1d9e75':r==='Mixed'?'#c9a227':r==='Weak Hands'?'#c0705a':'#7a2320';
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
      {holders.length>0 && (
        <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:12,padding:'12px 16px',marginBottom:12}}>
          <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:8}}>MAJOR HOLDERS (13G, &gt;5%)</div>
          {holders.map((h,i)=>(
            <div key={i} style={{display:'flex',alignItems:'center',gap:10,padding:'4px 0'}}>
              <span style={{fontSize:12,color:'#cdbfae',flex:1}}>{h.holder}</span>
              <div style={{width:120,height:10,background:'#242424',borderRadius:5,overflow:'hidden'}}>
                <div style={{height:'100%',width:`${Math.min(100,(h.percent||0)*5)}%`,background:'#daa520'}}/></div>
              <span style={{fontSize:12,fontWeight:600,color:'#daa520',width:44,textAlign:'right'}}>{h.percent!=null?h.percent.toFixed(1)+'%':'—'}</span>
            </div>
          ))}
        </div>
      )}
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(130px,1fr))',gap:8,marginBottom:14}}>
        {[['Share Trend',km.share_count_trend!=null?(km.share_count_trend>0?'+':'')+(km.share_count_trend*100).toFixed(2)+'%':'—'],
          ['Major Holders',km.major_holder_count!=null?km.major_holder_count.toFixed(0):'—'],
          ['Top Holder',km.top_holder_pct!=null?km.top_holder_pct.toFixed(1)+'%':'—'],
          ['Buyback Intensity',km.buyback_intensity!=null?(km.buyback_intensity*100).toFixed(2)+'%':'—'],
          ['Insider Buy',km.insider_buy_ratio!=null?(km.insider_buy_ratio*100).toFixed(0)+'%':'—'],
          ['Dilution',km.dilution_pressure!=null?(km.dilution_pressure*100).toFixed(1)+'%':'—']].map(([k,v])=>(
          <div key={k as string} style={{background:'#181818',border:'1px solid #2a2a2a',borderRadius:8,padding:'8px 10px'}}>
            <div style={{fontSize:9,color:'#9d8b7a'}}>{k}</div><div style={{fontSize:14,fontWeight:600,color:'#daa520'}}>{v}</div>
          </div>
        ))}
      </div>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
        <span style={{fontSize:12,color:'#9d8b7a',letterSpacing:1}}>{d.tree.categories.length} DIMENSIONS · {d.tree.categories.reduce((a,c)=>a+c.n_signals,0)} SIGNALS</span>
        <button onClick={()=>{const v=!allOpen;setAllOpen(v);const m:Record<string,boolean>={};d.tree.categories.forEach(c=>m[c.id]=v);setExpanded(m);}}
          style={{background:'#181818',border:'1px solid #2a2a2a',color:'#9d8b7a',borderRadius:8,padding:'5px 12px',fontSize:11,cursor:'pointer'}}>{allOpen?'Collapse all':'Expand all'}</button>
      </div>
      <div style={{display:'flex',flexDirection:'column',gap:8}}>
        {d.tree.categories.map(cat=>{const open=expanded[cat.id];return (
          <div key={cat.id} style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:10,overflow:'hidden'}}>
            <div onClick={()=>setExpanded(p=>({...p,[cat.id]:!p[cat.id]}))}
              style={{display:'flex',alignItems:'center',gap:12,padding:'10px 14px',cursor:'pointer',borderLeft:`4px solid ${heat(cat.score)}`}}>
              <span style={{fontSize:11,color:'#7a7266',width:12}}>{open?'▾':'▸'}</span>
              <span style={{fontSize:13,fontWeight:600,color:'#e8ddd0',flex:1}}>{cat.label}</span>
              <span style={{fontSize:10,color:'#7a7266'}}>wt {cat.weight.toFixed(2)} · {cat.n_scored}/{cat.n_signals}</span>
              <span style={{fontSize:18,fontWeight:700,color:heat(cat.score),width:36,textAlign:'right'}}>{cat.score?.toFixed(0)??'—'}</span>
            </div>
            {open && (<div style={{padding:'4px 14px 12px 30px'}}>
              {cat.signals.map(s=>{const pending=s.status==='needs_source'||s.score==null;return (
                <div key={s.id} title={s.evidence} style={{display:'flex',alignItems:'center',gap:10,padding:'5px 0',borderBottom:'1px solid #1e1e1e',opacity:pending?0.5:1}}>
                  <span style={{fontSize:12,color:'#cdbfae',flex:1}}>{s.label}</span>
                  <span style={{fontSize:12,color:'#9d8b7a',width:70,textAlign:'right'}}>{pending?'—':fmt(s.id,s.raw_value)}</span>
                  <div style={{width:80,height:6,background:'#242424',borderRadius:3,overflow:'hidden'}}>{!pending && <div style={{height:'100%',width:`${s.score}%`,background:heat(s.score)}}/>}</div>
                  <span style={{fontSize:11,fontWeight:600,color:pending?'#555':heat(s.score),width:26,textAlign:'right'}}>{pending?'—':s.score!.toFixed(0)}</span>
                </div>);})}
            </div>)}
          </div>);})}
      </div>
    </div>
  );
}
