import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; }
interface Cat { id:string; label:string; weight:number; score:number|null; n_signals:number; n_scored:number; signals:Sig[]; }
interface CData { ticker:string; available:boolean; score:number|null; competitive_rating:string; confidence:number;
  coverage:{scored:number;total:number}; bucket:string; peer_count:number; tree:{categories:Cat[]}; key_metrics:Record<string,number|null>; reason?:string; }

const heat=(s:number|null)=>s==null?'#2a2a2a':s>=75?'#0f6e56':s>=58?'#1d9e75':s>=42?'#8a7519':s>=25?'#a35a1d':'#7a2320';
const ratingColor=(r:string)=>r==='Dominant'?'#0f9d6e':r==='Strong'?'#1d9e75':r==='Competitive'?'#c9a227':r==='Challenged'?'#c0705a':'#7a2320';
const fmt=(id:string,v:number|null):string=>{
  if(v==null) return '—';
  if(id.includes('pctile')||id.includes('rank')||id.includes('_pct')||id.includes('proxy')||id.includes('premium')||id.includes('discount')||id.includes('cheap')||id.includes('dominance')||id.includes('durability')||id.includes('persistence')||id.includes('excess')||id.includes('efficiency')) { if(Math.abs(v)<=1.01) return (v*100).toFixed(0)+'th'; }
  if(id.includes('advantage')||id.includes('spread')||id.includes('level')||id.includes('growth')||id.includes('decel')) return (v*100).toFixed(1)+'%';
  if(id.includes('scale')&&id.includes('abs')) return '$'+v.toFixed(0)+'B';
  if(id.includes('employee')) return v.toLocaleString();
  if(id.includes('risk')||id.includes('loss')||id.includes('comp')) return v>0?'Yes':'No';
  if(id.includes('liquidity')||id.includes('rel')) return v.toFixed(2);
  return typeof v==='number'?v.toFixed(2):v;
};

const RadialBars=({metrics}:{metrics:Record<string,number|null>})=>{
  const bars=[['Scale',metrics.scale_rank],['Margin',metrics.net_margin_pctile],['ROIC',metrics.roic_pctile],['Growth',metrics.growth_pctile]];
  return (
    <div style={{display:'flex',gap:16,justifyContent:'center',padding:'8px 0',flexWrap:'wrap'}}>
      {bars.map(([label,v])=>{
        const val=(v as number)||0; const pct=Math.round(val*100);
        const col=val>=0.7?'#0f9d6e':val>=0.4?'#c9a227':'#c0705a';
        const circ=2*Math.PI*32;
        return (
          <div key={label as string} style={{textAlign:'center'}}>
            <svg width="80" height="80" viewBox="0 0 80 80">
              <circle cx="40" cy="40" r="32" fill="none" stroke="#242424" strokeWidth="7"/>
              <circle cx="40" cy="40" r="32" fill="none" stroke={col} strokeWidth="7" strokeLinecap="round"
                strokeDasharray={`${circ*val} ${circ}`} transform="rotate(-90 40 40)"/>
              <text x="40" y="45" textAnchor="middle" fontSize="17" fontWeight="700" fill={col}>{pct}</text>
            </svg>
            <div style={{fontSize:10,color:'#9d8b7a',marginTop:2}}>{label as string}</div>
          </div>
        );
      })}
    </div>
  );
};

export default function CompetitivePanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<CData|null>(null);
  const [loading,setLoading]=useState(false); const [err,setErr]=useState('');
  const [expanded,setExpanded]=useState<Record<string,boolean>>({}); const [allOpen,setAllOpen]=useState(false);

  useEffect(()=>{ if(!ticker)return; setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/competitive/${ticker}`).then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No data');else setD(x);})
      .catch(e=>setErr(e?.message||'Request failed')).finally(()=>setLoading(false));
  },[ticker]);

  if(!ticker)return <div style={{color:'#9d8b7a',padding:24}}>Enter a ticker for Competitive Intelligence.</div>;
  if(loading)return <div style={{color:'#daa520',padding:24}}>Analyzing competitive position — market share, profitability edge, moat vs 100+ peers…</div>;
  if(err)return <div style={{color:'#c0705a',padding:24}}>Competitive: {err}</div>;
  if(!d)return null;
  const km=d.key_metrics||{};

  return (
    <div style={{padding:'8px 4px',color:'#e8ddd0'}}>
      <div style={{display:'flex',alignItems:'center',gap:24,marginBottom:14,flexWrap:'wrap'}}>
        <div style={{display:'flex',alignItems:'baseline',gap:10}}>
          <span style={{fontSize:46,fontWeight:700,color:heat(d.score),lineHeight:1}}>{d.score?.toFixed(0)??'—'}</span>
          <span style={{fontSize:16,color:'#9d8b7a'}}>/100</span>
        </div>
        <div>
          <div style={{fontSize:22,fontWeight:700,color:ratingColor(d.competitive_rating),letterSpacing:0.5}}>{d.competitive_rating}</div>
          <div style={{fontSize:11,color:'#9d8b7a',marginTop:2}}>vs {d.peer_count} {d.bucket} peers · {d.coverage.scored}/{d.coverage.total} signals</div>
        </div>
      </div>

      <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:12,padding:'10px',marginBottom:14}}>
        <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:4,textAlign:'center'}}>COMPETITIVE PERCENTILE vs SECTOR</div>
        <RadialBars metrics={km}/>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(140px,1fr))',gap:8,marginBottom:14}}>
        {[['Market Share',km.market_share_proxy!=null?(km.market_share_proxy*100).toFixed(1)+'%':'—'],
          ['Margin Advantage',km.margin_advantage!=null?(km.margin_advantage>0?'+':'')+(km.margin_advantage*100).toFixed(1)+'pp':'—'],
          ['Moat Spread',km.economic_moat_spread!=null?(km.economic_moat_spread>0?'+':'')+(km.economic_moat_spread*100).toFixed(1)+'%':'—'],
          ['Growth Edge',km.growth_advantage!=null?(km.growth_advantage>0?'+':'')+(km.growth_advantage*100).toFixed(1)+'pp':'—'],
          ['Gross Margin',km.gross_margin_level!=null?(km.gross_margin_level*100).toFixed(0)+'%':'—'],
          ['Valuation Rank',km.pe_discount_vs_peers!=null?(km.pe_discount_vs_peers*100).toFixed(0)+'th':'—']].map(([k,v])=>(
          <div key={k as string} style={{background:'#181818',border:'1px solid #2a2a2a',borderRadius:8,padding:'8px 10px'}}>
            <div style={{fontSize:9,color:'#9d8b7a'}}>{k}</div>
            <div style={{fontSize:15,fontWeight:600,color:'#daa520'}}>{v}</div>
          </div>
        ))}
      </div>

      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
        <span style={{fontSize:12,color:'#9d8b7a',letterSpacing:1}}>11 DIMENSIONS · {d.tree.categories.reduce((a,c)=>a+c.n_signals,0)} SIGNALS</span>
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
