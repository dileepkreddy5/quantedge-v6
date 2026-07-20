import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; }
interface Cat { id:string; label:string; weight:number; score:number|null; n_signals:number; n_scored:number; signals:Sig[]; }
interface AData { ticker:string; available:boolean; score:number|null; altdata_rating:string; coverage:{scored:number;total:number};
  tree:{categories:Cat[]}; key_metrics:Record<string,number|null>; reason?:string; }
const heat=(s:number|null)=>s==null?'#2a2a2a':s>=75?'#0f6e56':s>=58?'#1d9e75':s>=42?'#8a7519':s>=25?'#a35a1d':'#7a2320';
const rc=(r:string)=>r==='Strong Signals'?'#0f9d6e':r==='Positive Flow'?'#1d9e75':r==='Neutral Flow'?'#c9a227':r==='Weak Flow'?'#c0705a':'#7a2320';
const fmt=(id:string,v:number|null):string=>{
  if(v==null) return '—';
  if(id.includes('accum')||id.includes('ratio')) { if(Math.abs(v)<=1.01) return (v*100).toFixed(0)+'%'; }
  if(id.includes('vol')||id.includes('trend')||id.includes('surge')||id.includes('velocity')||id.includes('exp')||id.includes('gap')||id.includes('attention')||id.includes('size')||id.includes('slope')||id.includes('flow')) return (v>=0?'+':'')+(v*100).toFixed(1)+'%';
  if(id.includes('sent')||id.includes('mean')||id.includes('disp')) return v.toFixed(2);
  if(id.includes('volume_7d')||id.includes('breadth')||(id.includes('vel')&&id.includes('filing'))) return v.toFixed(0);
  if(id.includes('diverge')) return v>0?'Flag':'None';
  if(id.includes('amihud')) return v.toFixed(4);
  return typeof v==='number'?v.toFixed(2):v;
};
export default function AltDataPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<AData|null>(null);
  const [loading,setLoading]=useState(false); const [err,setErr]=useState('');
  const [expanded,setExpanded]=useState<Record<string,boolean>>({}); const [allOpen,setAllOpen]=useState(false);
  useEffect(()=>{ if(!ticker)return; setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/altdata/${ticker}`).then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No data');else setD(x);})
      .catch(e=>setErr(e?.message||'Request failed')).finally(()=>setLoading(false));
  },[ticker]);
  if(!ticker)return <div style={{color:'#9d8b7a',padding:24}}>Enter a ticker for Alt-Data Intelligence.</div>;
  if(loading)return <div style={{color:'#daa520',padding:24}}>Scanning alt-data — volume microstructure, news flow, smart-money footprint…</div>;
  if(err)return <div style={{color:'#c0705a',padding:24}}>Alt-Data: {err}</div>;
  if(!d)return null;
  const km=d.key_metrics||{};
  const flows=[['Accumulation',km.accumulation_ratio,km.accumulation_ratio!=null?km.accumulation_ratio>0.5:null],
    ['OBV Trend',km.obv_slope,km.obv_slope!=null?km.obv_slope>0:null],
    ['Unusual Volume',km.unusual_volume,km.unusual_volume!=null?km.unusual_volume>0:null],
    ['News Velocity',km.news_velocity,km.news_velocity!=null?km.news_velocity>0:null],
    ['News Sentiment',km.news_sentiment_mean,km.news_sentiment_mean!=null?km.news_sentiment_mean>0:null],
    ['Txn Surge',km.txn_count_surge,km.txn_count_surge!=null?km.txn_count_surge>0:null]] as [string,number|null,boolean|null][];
  return (
    <div style={{padding:'8px 4px',color:'#e8ddd0'}}>
      <div style={{display:'flex',alignItems:'center',gap:24,marginBottom:14,flexWrap:'wrap'}}>
        <div style={{display:'flex',alignItems:'baseline',gap:10}}>
          <span style={{fontSize:46,fontWeight:700,color:heat(d.score),lineHeight:1}}>{d.score?.toFixed(0)??'—'}</span>
          <span style={{fontSize:16,color:'#9d8b7a'}}>/100</span>
        </div>
        <div>
          <div style={{fontSize:22,fontWeight:700,color:rc(d.altdata_rating),letterSpacing:0.5}}>{d.altdata_rating}</div>
          <div style={{fontSize:11,color:'#9d8b7a',marginTop:2}}>{d.coverage.scored}/{d.coverage.total} signals · {d.coverage.total-d.coverage.scored} need premium feeds</div>
        </div>
      </div>
      <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:12,padding:'12px 16px',marginBottom:14}}>
        <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:10}}>FLOW SIGNALS</div>
        <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(150px,1fr))',gap:10}}>
          {flows.map(([l,v,good])=>(
            <div key={l} style={{display:'flex',alignItems:'center',gap:8}}>
              <div style={{width:8,height:8,borderRadius:4,background:good==null?'#555':good?'#0f9d6e':'#c0705a'}}/>
              <span style={{fontSize:11,color:'#cdbfae',flex:1}}>{l}</span>
              <span style={{fontSize:12,fontWeight:600,color:good==null?'#9d8b7a':good?'#0f9d6e':'#c0705a'}}>
                {v==null?'—':l.includes('Sentiment')?v.toFixed(2):(v>=0?'+':'')+(v*100).toFixed(0)+'%'}</span>
            </div>
          ))}
        </div>
      </div>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
        <span style={{fontSize:12,color:'#9d8b7a',letterSpacing:1}}>{d.tree.categories.length} DIMENSIONS · {d.tree.categories.reduce((a,c)=>a+c.n_signals,0)} SIGNALS</span>
        <button onClick={()=>{const v=!allOpen;setAllOpen(v);const m:Record<string,boolean>={};d.tree.categories.forEach(c=>m[c.id]=v);setExpanded(m);}}
          style={{background:'#181818',border:'1px solid #2a2a2a',color:'#9d8b7a',borderRadius:8,padding:'5px 12px',fontSize:11,cursor:'pointer'}}>{allOpen?'Collapse all':'Expand all'}</button>
      </div>
      <div style={{display:'flex',flexDirection:'column',gap:8}}>
        {d.tree.categories.map(cat=>{const open=expanded[cat.id];const allNS=cat.n_scored===0;return (
          <div key={cat.id} style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:10,overflow:'hidden',opacity:allNS?0.6:1}}>
            <div onClick={()=>setExpanded(p=>({...p,[cat.id]:!p[cat.id]}))}
              style={{display:'flex',alignItems:'center',gap:12,padding:'10px 14px',cursor:'pointer',borderLeft:`4px solid ${allNS?'#3a3a3a':heat(cat.score)}`}}>
              <span style={{fontSize:11,color:'#7a7266',width:12}}>{open?'▾':'▸'}</span>
              <span style={{fontSize:13,fontWeight:600,color:'#e8ddd0',flex:1}}>{cat.label}{allNS && <span style={{fontSize:9,color:'#7a7266',marginLeft:6}}>premium feed</span>}</span>
              <span style={{fontSize:10,color:'#7a7266'}}>{cat.n_scored}/{cat.n_signals}</span>
              <span style={{fontSize:18,fontWeight:700,color:allNS?'#555':heat(cat.score),width:36,textAlign:'right'}}>{cat.score?.toFixed(0)??'—'}</span>
            </div>
            {open && (<div style={{padding:'4px 14px 12px 30px'}}>
              {cat.signals.map(s=>{const pending=s.status==='needs_source'||s.score==null;return (
                <div key={s.id} title={s.evidence} style={{display:'flex',alignItems:'center',gap:10,padding:'5px 0',borderBottom:'1px solid #1e1e1e',opacity:pending?0.5:1}}>
                  <span style={{fontSize:12,color:'#cdbfae',flex:1}}>{s.label}{s.status==='needs_source'&&<span style={{fontSize:9,color:'#7a7266',marginLeft:6}}>n/a</span>}</span>
                  <span style={{fontSize:12,color:'#9d8b7a',width:64,textAlign:'right'}}>{pending?'—':fmt(s.id,s.raw_value)}</span>
                  <div style={{width:80,height:6,background:'#242424',borderRadius:3,overflow:'hidden'}}>{!pending && <div style={{height:'100%',width:`${s.score}%`,background:heat(s.score)}}/>}</div>
                  <span style={{fontSize:11,fontWeight:600,color:pending?'#555':heat(s.score),width:26,textAlign:'right'}}>{pending?'—':s.score!.toFixed(0)}</span>
                </div>);})}
            </div>)}
          </div>);})}
      </div>
    </div>
  );
}
