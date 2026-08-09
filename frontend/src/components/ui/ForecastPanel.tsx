import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; }
interface Cat { id:string; label:string; weight:number; score:number|null; n_signals:number; n_scored:number; signals:Sig[]; }
interface FData { ticker:string; available:boolean; score:number|null; forecast_rating:string; coverage:{scored:number;total:number};
  tree:{categories:Cat[]}; key_metrics:Record<string,number|null>; reason?:string; }
// Score ramp matches RiskPanel exactly so an identical score is an identical
// colour across every tab: gold (strong) → caramel → burnt-orange → bear.
const heat=(s:number|null)=>s==null?'var(--border-2)':s>=70?'var(--gold)':s>=50?'var(--caramel)':s>=30?'#c9762f':'var(--bear)';
const rc=(r:string)=>r==='Bullish Outlook'?'var(--gold)':r==='Constructive'?'var(--caramel)':r==='Neutral'?'var(--neutral)':r==='Cautious'?'#c9762f':'var(--bear)';
const pct=(v:number|null,d=1)=>v==null?'—':(v>=0?'+':'')+(v*100).toFixed(d)+'%';
const num=(v:number|null,d=2)=>v==null?'—':v.toFixed(d);
const fmt=(id:string,v:number|null):string=>{
  if(v==null) return '—';
  if(id.includes('roiic')||id.includes('margin')||id.includes('growth')||id.includes('accel')||id.includes('slope')||id.includes('mom')||id.includes('traj')||id.includes('pull')||id.includes('dist')||id.includes('trend')||id.includes('yoy')||id.includes('stack')||id.includes('intrinsic')||id.includes('fcf')||id.includes('ocf')||id.includes('cash_traj')||(id.includes('rule_of_40')&&id!=='rule40_pass')||id.includes('vs_ma')||id.includes('golden')||id.includes('ext')||id.includes('wc_')) return (v*100).toFixed(1)+'%';
  if(id.includes('pass')||id.includes('align')||id.includes('agree')||id.includes('pos')||id.includes('consist')||id.includes('stab')||id.includes('persist')||id.includes('conf')||id.includes('comp')||id.includes('up_days')||id.includes('rsi')||id.includes('quality')||id.includes('anchor')||id.includes('conv_lvl')) { if(Math.abs(v)<=1.5) return (v*100).toFixed(0)+'%'; }
  if(id.includes('leverage')||id.includes('inc_margin')) return v.toFixed(2)+'x';
  return typeof v==='number'?v.toFixed(2):v;
};
const OutlookGauge=({score}:{score:number|null})=>{
  const s=score??50; const angle=-90+(s/100)*180; const col=heat(s);
  return (
    <svg width="150" height="90" viewBox="0 0 150 90">
      <path d="M 15 80 A 60 60 0 0 1 135 80" fill="none" stroke="var(--surface-3)" strokeWidth="10" strokeLinecap="round"/>
      <path d="M 15 80 A 60 60 0 0 1 135 80" fill="none" stroke={col} strokeWidth="10" strokeLinecap="round"
        strokeDasharray={`${(s/100)*188.5} 188.5`}/>
      <line x1="75" y1="80" x2={75+50*Math.cos(angle*Math.PI/180)} y2={80+50*Math.sin(angle*Math.PI/180)}
        stroke={col} strokeWidth="3" strokeLinecap="round"/>
      <circle cx="75" cy="80" r="5" fill={col}/>
      <text x="75" y="55" textAnchor="middle" fontSize="22" fontWeight="700" fill={col}>{score?.toFixed(0)??'—'}</text>
    </svg>
  );
};
const Driver=({label,val,good}:{label:string;val:string;good:boolean|null})=>(
  <div style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderRadius:8,padding:'8px 10px'}}>
    <div style={{fontSize:9,color:'var(--cocoa-dust)'}}>{label}</div>
    <div style={{fontSize:14,fontWeight:600,color:good==null?'var(--gold)':good?'var(--bull)':'var(--bear)'}}>{val}</div>
  </div>
);
export default function ForecastPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<FData|null>(null);
  const [loading,setLoading]=useState(false); const [err,setErr]=useState('');
  const [expanded,setExpanded]=useState<Record<string,boolean>>({}); const [allOpen,setAllOpen]=useState(false);
  useEffect(()=>{ if(!ticker)return; setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/forecast/${ticker}`).then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No data');else setD(x);})
      .catch(e=>setErr(e?.message||'Request failed')).finally(()=>setLoading(false));
  },[ticker]);
  if(!ticker)return <div style={{color:'var(--cocoa-dust)',padding:24}}>Enter a ticker for Forecast Intelligence.</div>;
  if(loading)return <div style={{color:'var(--gold)',padding:24}}>Projecting forward — earnings trajectory, compounding, momentum quality…</div>;
  if(err)return <div style={{color:'var(--bear)',padding:24}}>Forecast: {err}</div>;
  if(!d)return null;
  const km=d.key_metrics||{};
  return (
    <div style={{padding:'8px 4px',color:'var(--latte)'}}>
      <div style={{display:'flex',alignItems:'center',gap:20,marginBottom:14,flexWrap:'wrap'}}>
        <OutlookGauge score={d.score}/>
        <div>
          <div style={{fontSize:22,fontWeight:700,color:rc(d.forecast_rating),letterSpacing:0.5}}>{d.forecast_rating}</div>
          <div style={{fontSize:11,color:'var(--cocoa-dust)',marginTop:2}}>{d.coverage.scored}/{d.coverage.total} forward signals</div>
          <div style={{fontSize:10,color:'var(--cocoa)',marginTop:4,maxWidth:280}}>Model-based projection from historical trajectory — not analyst consensus</div>
        </div>
      </div>
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(120px,1fr))',gap:8,marginBottom:14}}>
        <Driver label="ROIIC" val={pct(km.roiic)} good={km.roiic!=null?km.roiic>0.1:null}/>
        <Driver label="Incremental Margin" val={km.incremental_op_margin!=null?km.incremental_op_margin.toFixed(2)+'x':'—'} good={km.incremental_op_margin!=null?km.incremental_op_margin>0.2:null}/>
        <Driver label="Rule of 40" val={km.rule_of_40!=null?(km.rule_of_40*100).toFixed(0)+'%':'—'} good={km.rule_of_40!=null?km.rule_of_40>=0.4:null}/>
        <Driver label="2Y Stacked Growth" val={pct(km.two_year_stacked_growth)} good={km.two_year_stacked_growth!=null?km.two_year_stacked_growth>0.15:null}/>
        <Driver label="Momentum Quality" val={num(km.momentum_quality)} good={km.momentum_quality!=null?km.momentum_quality>0:null}/>
        <Driver label="Intrinsic Growth" val={pct(km.intrinsic_growth_proxy)} good={km.intrinsic_growth_proxy!=null?km.intrinsic_growth_proxy>0.08:null}/>
        <Driver label="FCF Margin" val={pct(km.fcf_margin_proxy)} good={km.fcf_margin_proxy!=null?km.fcf_margin_proxy>0.1:null}/>
        <Driver label="Fwd Composite" val={km.forward_composite!=null?(km.forward_composite*100).toFixed(0)+'%':'—'} good={km.forward_composite!=null?km.forward_composite>0.5:null}/>
      </div>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
        <span style={{fontSize:12,color:'var(--cocoa-dust)',letterSpacing:1}}>{d.tree.categories.length} DIMENSIONS · {d.tree.categories.reduce((a,c)=>a+c.n_signals,0)} SIGNALS</span>
        <button onClick={()=>{const v=!allOpen;setAllOpen(v);const m:Record<string,boolean>={};d.tree.categories.forEach(c=>m[c.id]=v);setExpanded(m);}}
          style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',color:'var(--cocoa-dust)',borderRadius:8,padding:'5px 12px',fontSize:11,cursor:'pointer'}}>{allOpen?'Collapse all':'Expand all'}</button>
      </div>
      <div style={{display:'flex',flexDirection:'column',gap:8}}>
        {d.tree.categories.map(cat=>{const open=expanded[cat.id];return (
          <div key={cat.id} style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderRadius:10,overflow:'hidden'}}>
            <div onClick={()=>setExpanded(p=>({...p,[cat.id]:!p[cat.id]}))}
              style={{display:'flex',alignItems:'center',gap:12,padding:'10px 14px',cursor:'pointer',borderLeft:`4px solid ${heat(cat.score)}`}}>
              <span style={{fontSize:11,color:'var(--cocoa)',width:12}}>{open?'▾':'▸'}</span>
              <span style={{fontSize:13,fontWeight:600,color:'var(--latte)',flex:1}}>{cat.label}</span>
              <span style={{fontSize:10,color:'var(--cocoa)'}}>wt {(cat.weight??0).toFixed(2)} · {cat.n_scored??0}/{cat.n_signals??0}</span>
              <span style={{fontSize:18,fontWeight:700,color:heat(cat.score),width:36,textAlign:'right'}}>{cat.score?.toFixed(0)??'—'}</span>
            </div>
            {open && (<div style={{padding:'4px 14px 12px 30px'}}>
              {cat.signals.map(s=>{const pending=s.status==='needs_source'||s.score==null;return (
                <div key={s.id} title={s.evidence} style={{display:'flex',alignItems:'center',gap:10,padding:'5px 0',borderBottom:'1px solid var(--border-1)',opacity:pending?0.5:1}}>
                  <span style={{fontSize:12,color:'var(--latte)',flex:1}}>{s.label}</span>
                  <span style={{fontSize:12,color:'var(--cocoa-dust)',width:64,textAlign:'right'}}>{pending?'—':fmt(s.id,s.raw_value)}</span>
                  <div style={{width:80,height:6,background:'var(--surface-3)',borderRadius:3,overflow:'hidden'}}>{!pending && <div style={{height:'100%',width:`${s.score}%`,background:heat(s.score)}}/>}</div>
                  <span style={{fontSize:11,fontWeight:600,color:pending?'var(--cocoa)':heat(s.score),width:26,textAlign:'right'}}>{pending?'—':s.score!.toFixed(0)}</span>
                </div>);})}
            </div>)}
          </div>);})}
      </div>
    </div>
  );
}
