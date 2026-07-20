import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; }
interface Cat { id:string; label:string; weight:number; score:number|null; n_signals:number; n_scored:number; signals:Sig[]; }
interface PData { ticker:string; available:boolean; score:number|null; peers_rating:string; coverage:{scored:number;total:number};
  bucket:string; peer_count:number; tree:{categories:Cat[]}; key_metrics:Record<string,number|null>; reason?:string; }
const heat=(s:number|null)=>s==null?'#2a2a2a':s>=75?'#0f6e56':s>=58?'#1d9e75':s>=42?'#8a7519':s>=25?'#a35a1d':'#7a2320';
const rc=(r:string)=>r==='Peer Leader'?'#0f9d6e':r==='Above Peers'?'#1d9e75':r==='In Line'?'#c9a227':r==='Below Peers'?'#c0705a':'#7a2320';
const pctile=(v:number|null):string=>v==null?'—':Math.round(v*100)+'th';
const RankBar=({label,v}:{label:string;v:number|null}) => {
  const p=(v??0)*100; const col=v==null?'#555':v>=0.75?'#0f9d6e':v>=0.5?'#1d9e75':v>=0.25?'#c9a227':'#c0705a';
  return (
    <div style={{display:'flex',alignItems:'center',gap:10,padding:'4px 0'}}>
      <span style={{fontSize:11,color:'#cdbfae',width:120}}>{label}</span>
      <div style={{flex:1,height:14,position:'relative',background:'linear-gradient(90deg,#2a1a1a,#2a2518,#1a2a1f)',borderRadius:4}}>
        {v!=null && <div style={{position:'absolute',left:`${p}%`,top:-2,bottom:-2,width:3,background:col,borderRadius:2,transform:'translateX(-50%)'}}/>}
        <div style={{position:'absolute',left:'25%',top:0,bottom:0,width:1,background:'#3a3a3a44'}}/>
        <div style={{position:'absolute',left:'50%',top:0,bottom:0,width:1,background:'#3a3a3a66'}}/>
        <div style={{position:'absolute',left:'75%',top:0,bottom:0,width:1,background:'#3a3a3a44'}}/>
      </div>
      <span style={{fontSize:11,fontWeight:600,color:col,width:40,textAlign:'right'}}>{pctile(v)}</span>
    </div>
  );
};
export default function PeersScorePanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<PData|null>(null);
  const [loading,setLoading]=useState(false); const [err,setErr]=useState('');
  const [expanded,setExpanded]=useState<Record<string,boolean>>({}); const [allOpen,setAllOpen]=useState(false);
  useEffect(()=>{ if(!ticker)return; setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/peers_score/${ticker}`).then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No data');else setD(x);})
      .catch(e=>setErr(e?.message||'Request failed')).finally(()=>setLoading(false));
  },[ticker]);
  if(!ticker)return <div style={{color:'#9d8b7a',padding:24}}>Enter a ticker for Peers Intelligence.</div>;
  if(loading)return <div style={{color:'#daa520',padding:24}}>Ranking vs peers — quality, valuation, growth, profitability…</div>;
  if(err)return <div style={{color:'#c0705a',padding:24}}>Peers: {err}</div>;
  if(!d)return null;
  const km=d.key_metrics||{};
  const ranks=[['Overall Rank',km.overall_peer_rank],['Quality',km.quality_composite],['Profitability',km.profitability_composite],
    ['ROIC',km.roic_rank],['Net Margin',km.net_margin_rank],['Growth',km.revenue_growth_rank],['Valuation (cheap)',km.pe_rank]] as [string,number|null][];
  return (
    <div style={{padding:'8px 4px',color:'#e8ddd0'}}>
      <div style={{display:'flex',alignItems:'center',gap:24,marginBottom:14,flexWrap:'wrap'}}>
        <div style={{display:'flex',alignItems:'baseline',gap:10}}>
          <span style={{fontSize:46,fontWeight:700,color:heat(d.score),lineHeight:1}}>{d.score?.toFixed(0)??'—'}</span>
          <span style={{fontSize:16,color:'#9d8b7a'}}>/100</span>
        </div>
        <div>
          <div style={{fontSize:22,fontWeight:700,color:rc(d.peers_rating),letterSpacing:0.5}}>{d.peers_rating}</div>
          <div style={{fontSize:11,color:'#9d8b7a',marginTop:2}}>ranked vs {d.peer_count} {d.bucket} peers</div>
        </div>
      </div>
      <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:12,padding:'14px 16px',marginBottom:14}}>
        <div style={{display:'flex',justifyContent:'space-between',fontSize:11,color:'#9d8b7a',letterSpacing:1,marginBottom:10}}>
          <span>PERCENTILE RANK vs PEERS</span><span style={{fontSize:9}}>0 ··· 50 ··· 100</span></div>
        {ranks.map(([l,v])=><RankBar key={l} label={l} v={v}/>)}
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
                  <span style={{fontSize:12,color:'#9d8b7a',width:56,textAlign:'right'}}>{pending?'—':(s.raw_value!=null&&Math.abs(s.raw_value)<=1.01?Math.round(s.raw_value*100)+'th':s.raw_value?.toFixed(2))}</span>
                  <div style={{width:80,height:6,background:'#242424',borderRadius:3,overflow:'hidden'}}>{!pending && <div style={{height:'100%',width:`${s.score}%`,background:heat(s.score)}}/>}</div>
                  <span style={{fontSize:11,fontWeight:600,color:pending?'#555':heat(s.score),width:26,textAlign:'right'}}>{pending?'—':s.score!.toFixed(0)}</span>
                </div>);})}
            </div>)}
          </div>);})}
      </div>
    </div>
  );
}
