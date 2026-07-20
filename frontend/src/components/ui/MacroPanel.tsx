import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; }
interface Cat { id:string; label:string; weight:number; score:number|null; n_signals:number; n_scored:number; signals:Sig[]; }
interface MData { ticker:string; available:boolean; score:number|null; macro_rating:string; coverage:{scored:number;total:number};
  tree:{categories:Cat[]}; key_metrics:Record<string,number|null>; reason?:string; }
const heat=(s:number|null)=>s==null?'#2a2a2a':s>=75?'#0f6e56':s>=58?'#1d9e75':s>=42?'#8a7519':s>=25?'#a35a1d':'#7a2320';
const rc=(r:string)=>r==='Insulated'?'#0f9d6e':r==='Resilient'?'#1d9e75':r==='Balanced'?'#c9a227':r==='Exposed'?'#c0705a':'#7a2320';
const fmt=(v:number|null):string=>v==null?'—':(v>=0?'+':'')+v.toFixed(2);
const BetaBar=({label,val,desc}:{label:string;val:number|null;desc:string})=>{
  const v=val??0; const mag=Math.min(1,Math.abs(v)); const pos=v>=0;
  const col=Math.abs(v)<0.15?'#5a6b5f':Math.abs(v)<0.4?'#c9a227':pos?'#0f9d6e':'#c0705a';
  return (
    <div title={desc} style={{display:'flex',alignItems:'center',gap:10,padding:'5px 0'}}>
      <span style={{fontSize:11,color:'#cdbfae',width:120}}>{label}</span>
      <div style={{flex:1,height:18,position:'relative',background:'#181818',borderRadius:4}}>
        <div style={{position:'absolute',left:'50%',top:0,bottom:0,width:1,background:'#3a3a3a'}}/>
        <div style={{position:'absolute',top:3,bottom:3,borderRadius:3,background:col,
          left:pos?'50%':`${50-mag*50}%`,width:`${mag*50}%`}}/>
      </div>
      <span style={{fontSize:11,fontWeight:600,color:col,width:44,textAlign:'right'}}>{fmt(val)}</span>
    </div>
  );
};
export default function MacroPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<MData|null>(null);
  const [loading,setLoading]=useState(false); const [err,setErr]=useState('');
  const [expanded,setExpanded]=useState<Record<string,boolean>>({}); const [allOpen,setAllOpen]=useState(false);
  useEffect(()=>{ if(!ticker)return; setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/macro/${ticker}`).then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No data');else setD(x);})
      .catch(e=>setErr(e?.message||'Request failed')).finally(()=>setLoading(false));
  },[ticker]);
  if(!ticker)return <div style={{color:'#9d8b7a',padding:24}}>Enter a ticker for Macro Sensitivity.</div>;
  if(loading)return <div style={{color:'#daa520',padding:24}}>Analyzing macro exposures — rates, dollar, inflation, cycle, factors…</div>;
  if(err)return <div style={{color:'#c0705a',padding:24}}>Macro: {err}</div>;
  if(!d)return null;
  const km=d.key_metrics||{};
  const betas=[['Rates (TLT)',km.rate_beta,'Sensitivity to long-bond/rate moves'],
    ['Dollar (UUP)',km.dollar_beta,'Strong-dollar impact (- = multinational)'],
    ['Inflation (Gold)',km.inflation_hedge,'Gold correlation (+ = inflation hedge)'],
    ['Oil (USO)',km.oil_beta,'Energy/oil-price sensitivity'],
    ['Market (SPY)',km.market_beta,'Systematic market risk'],
    ['Credit (HYG)',km.credit_beta,'Risk-on/off sensitivity'],
    ['Value tilt',km.value_tilt,'Value-factor loading'],
    ['Momentum tilt',km.momentum_tilt,'Momentum-factor loading']] as [string,number|null,string][];
  return (
    <div style={{padding:'8px 4px',color:'#e8ddd0'}}>
      <div style={{display:'flex',alignItems:'center',gap:24,marginBottom:14,flexWrap:'wrap'}}>
        <div style={{display:'flex',alignItems:'baseline',gap:10}}>
          <span style={{fontSize:46,fontWeight:700,color:heat(d.score),lineHeight:1}}>{d.score?.toFixed(0)??'—'}</span>
          <span style={{fontSize:16,color:'#9d8b7a'}}>/100</span>
        </div>
        <div>
          <div style={{fontSize:22,fontWeight:700,color:rc(d.macro_rating),letterSpacing:0.5}}>{d.macro_rating}</div>
          <div style={{fontSize:11,color:'#9d8b7a',marginTop:2}}>{d.coverage.scored}/{d.coverage.total} exposures measured</div>
        </div>
      </div>
      <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:12,padding:'12px 16px',marginBottom:14}}>
        <div style={{display:'flex',justifyContent:'space-between',fontSize:11,color:'#9d8b7a',letterSpacing:1,marginBottom:8}}>
          <span>MACRO FACTOR EXPOSURE (β)</span><span style={{fontSize:9}}>← negative · positive →</span></div>
        {betas.map(([l,v,desc])=><BetaBar key={l} label={l} val={v} desc={desc}/>)}
        <div style={{marginTop:8,paddingTop:8,borderTop:'1px solid #242424',fontSize:10,color:'#7a7266'}}>
          Resilience {km.macro_resilience!=null?(km.macro_resilience*100).toFixed(0)+'%':'—'} · Defensiveness {km.defensiveness!=null?(km.defensiveness*100).toFixed(0)+'%':'—'}
        </div>
      </div>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
        <span style={{fontSize:12,color:'#9d8b7a',letterSpacing:1}}>{d.tree.categories.length} REGIMES · {d.tree.categories.reduce((a,c)=>a+c.n_signals,0)} SIGNALS</span>
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
                  <span style={{fontSize:12,color:'#9d8b7a',width:60,textAlign:'right'}}>{pending?'—':fmt(s.raw_value)}</span>
                  <div style={{width:80,height:6,background:'#242424',borderRadius:3,overflow:'hidden'}}>{!pending && <div style={{height:'100%',width:`${s.score}%`,background:heat(s.score)}}/>}</div>
                  <span style={{fontSize:11,fontWeight:600,color:pending?'#555':heat(s.score),width:26,textAlign:'right'}}>{pending?'—':s.score!.toFixed(0)}</span>
                </div>);})}
            </div>)}
          </div>);})}
      </div>
    </div>
  );
}
