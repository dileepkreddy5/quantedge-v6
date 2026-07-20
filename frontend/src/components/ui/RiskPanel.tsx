import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; }
interface Cat { id:string; label:string; weight:number; score:number|null; n_signals:number; n_scored:number; signals:Sig[]; }
interface RiskData {
  ticker:string; available:boolean; score:number|null; risk_rating:string; confidence:number;
  coverage:{scored:number;total:number}; tree:{categories:Cat[]}; key_metrics:Record<string,number|null>; reason?:string;
}
const heat=(s:number|null)=>s==null?'#2a2a2a':s>=75?'#0f6e56':s>=58?'#1d9e75':s>=42?'#8a7519':s>=25?'#a35a1d':'#7a2320';
const ratingColor=(r:string)=>r.includes('Low')?'#0f9d6e':r.includes('Moderate')?'#1d9e75':r.includes('Elevated')?'#c9a227':r.includes('High')?'#c0705a':'#7a2320';

const altmanZone=(z:number|null)=>{
  if(z==null) return {label:'—',color:'#9d8b7a'};
  if(z>=3) return {label:'SAFE',color:'#0f9d6e'};
  if(z>=1.8) return {label:'GREY ZONE',color:'#c9a227'};
  return {label:'DISTRESS',color:'#c0705a'};
};

const fmtVal=(id:string,v:number|null):string=>{
  if(v==null) return '—';
  if(id.includes('altman')||id.includes('ohlson')||id.includes('ratio')||id.includes('beta')||id.includes('conversion')||id.includes('ebitda')||id.includes('turnover')||id.includes('piotroski')||id.includes('dsri')||id.includes('coverage')||id.includes('pe')) return v.toFixed(2);
  if(id.includes('prob')||id.includes('drawdown')||id.includes('vol')||id.includes('cvar')||id.includes('day')||id.includes('month')||id.includes('deviation')||id.includes('accruals')||id.includes('dilution')||id.includes('intensity')||id.includes('yield')||id.includes('sensitivity')||id.includes('margin')||id.includes('cushion')||id.includes('semivar')) return (v*100).toFixed(1)+'%';
  if(id.includes('days')||id.includes('duration')) return v.toFixed(0);
  return v.toFixed(2);
};

export default function RiskPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<RiskData|null>(null);
  const [loading,setLoading]=useState(false);
  const [err,setErr]=useState('');
  const [expanded,setExpanded]=useState<Record<string,boolean>>({});
  const [allOpen,setAllOpen]=useState(false);

  useEffect(()=>{
    if(!ticker)return;
    setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/risk/${ticker}`)
      .then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No risk data');else setD(x);})
      .catch(e=>setErr(e?.message||'Request failed'))
      .finally(()=>setLoading(false));
  },[ticker]);

  if(!ticker)return <div style={{color:'#9d8b7a',padding:24}}>Enter a ticker for Risk Intelligence.</div>;
  if(loading)return <div style={{color:'#daa520',padding:24}}>Assessing risk — credit models, leverage, tail risk, and 43 signals…</div>;
  if(err)return <div style={{color:'#c0705a',padding:24}}>Risk: {err}</div>;
  if(!d)return null;

  const km=d.key_metrics||{};
  const az=altmanZone(km.altman_z);

  return (
    <div style={{padding:'8px 4px',color:'#e8ddd0'}}>
      <div style={{display:'flex',alignItems:'center',gap:24,marginBottom:18,flexWrap:'wrap'}}>
        <div style={{display:'flex',alignItems:'baseline',gap:10}}>
          <span style={{fontSize:46,fontWeight:700,color:heat(d.score),lineHeight:1}}>{d.score?.toFixed(0)??'—'}</span>
          <span style={{fontSize:16,color:'#9d8b7a'}}>/100</span>
        </div>
        <div>
          <div style={{fontSize:22,fontWeight:700,color:ratingColor(d.risk_rating),letterSpacing:0.5}}>{d.risk_rating}</div>
          <div style={{fontSize:11,color:'#9d8b7a',marginTop:2}}>{d.coverage.scored}/{d.coverage.total} risk signals · higher score = lower risk</div>
        </div>
      </div>

      <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:8}}>CREDIT & SOLVENCY MODELS</div>
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(150px,1fr))',gap:10,marginBottom:18}}>
        <div style={{background:'#141414',border:`1px solid ${az.color}`,borderRadius:10,padding:'12px 14px'}}>
          <div style={{fontSize:10,color:'#9d8b7a'}}>ALTMAN Z-SCORE</div>
          <div style={{fontSize:24,fontWeight:700,color:az.color}}>{km.altman_z?.toFixed(2)??'—'}</div>
          <div style={{fontSize:9,color:az.color,fontWeight:600,letterSpacing:1}}>{az.label}</div>
        </div>
        <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:10,padding:'12px 14px'}}>
          <div style={{fontSize:10,color:'#9d8b7a'}}>BANKRUPTCY PROB.</div>
          <div style={{fontSize:24,fontWeight:700,color:km.bankruptcy_prob!=null?(km.bankruptcy_prob<0.05?'#0f9d6e':km.bankruptcy_prob<0.15?'#c9a227':'#c0705a'):'#9d8b7a'}}>
            {km.bankruptcy_prob!=null?(km.bankruptcy_prob*100).toFixed(1)+'%':'—'}</div>
          <div style={{fontSize:9,color:'#7a7266'}}>Ohlson-derived</div>
        </div>
        <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:10,padding:'12px 14px'}}>
          <div style={{fontSize:10,color:'#9d8b7a'}}>NET DEBT / EBITDA</div>
          <div style={{fontSize:24,fontWeight:700,color:'#daa520'}}>{km.net_debt_to_ebitda!=null?km.net_debt_to_ebitda.toFixed(2)+'x':'—'}</div>
          <div style={{fontSize:9,color:'#7a7266'}}>leverage</div>
        </div>
        <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:10,padding:'12px 14px'}}>
          <div style={{fontSize:10,color:'#9d8b7a'}}>CURRENT RATIO</div>
          <div style={{fontSize:24,fontWeight:700,color:'#daa520'}}>{km.current_ratio!=null?km.current_ratio.toFixed(2):'—'}</div>
          <div style={{fontSize:9,color:'#7a7266'}}>liquidity</div>
        </div>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(120px,1fr))',gap:8,marginBottom:18}}>
        {[['Max Drawdown',km.max_drawdown!=null?(km.max_drawdown*100).toFixed(0)+'%':'—'],
          ['Annual Vol',km.annualized_vol!=null?(km.annualized_vol*100).toFixed(0)+'%':'—'],
          ['Beta',km.beta!=null?km.beta.toFixed(2):'—'],
          ['P/E',km.pe_ratio!=null?km.pe_ratio.toFixed(0):'—'],
          ['Sloan Accruals',km.sloan_accruals!=null?(km.sloan_accruals*100).toFixed(1)+'%':'—'],
          ['Dilution',km.share_dilution!=null?(km.share_dilution*100).toFixed(1)+'%':'—']].map(([k,v])=>(
          <div key={k as string} style={{background:'#181818',border:'1px solid #2a2a2a',borderRadius:8,padding:'8px 10px'}}>
            <div style={{fontSize:9,color:'#9d8b7a'}}>{k}</div>
            <div style={{fontSize:14,fontWeight:600,color:'#cdbfae'}}>{v}</div>
          </div>
        ))}
      </div>

      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
        <span style={{fontSize:12,color:'#9d8b7a',letterSpacing:1}}>10 RISK DIMENSIONS · {d.tree.categories.reduce((a,c)=>a+c.n_signals,0)} SIGNALS</span>
        <button onClick={()=>{const v=!allOpen;setAllOpen(v);const m:Record<string,boolean>={};d.tree.categories.forEach(c=>m[c.id]=v);setExpanded(m);}}
          style={{background:'#181818',border:'1px solid #2a2a2a',color:'#9d8b7a',borderRadius:8,padding:'5px 12px',fontSize:11,cursor:'pointer'}}>
          {allOpen?'Collapse all':'Expand all'}</button>
      </div>
      <div style={{display:'flex',flexDirection:'column',gap:8}}>
        {d.tree.categories.map(cat=>{
          const open=expanded[cat.id];
          return (
            <div key={cat.id} style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:10,overflow:'hidden'}}>
              <div onClick={()=>setExpanded(p=>({...p,[cat.id]:!p[cat.id]}))}
                style={{display:'flex',alignItems:'center',gap:12,padding:'10px 14px',cursor:'pointer',borderLeft:`4px solid ${heat(cat.score)}`}}>
                <span style={{fontSize:11,color:'#7a7266',width:12}}>{open?'▾':'▸'}</span>
                <span style={{fontSize:13,fontWeight:600,color:'#e8ddd0',flex:1}}>{cat.label}</span>
                <span style={{fontSize:10,color:'#7a7266'}}>wt {cat.weight.toFixed(2)} · {cat.n_scored}/{cat.n_signals}</span>
                <span style={{fontSize:18,fontWeight:700,color:heat(cat.score),width:36,textAlign:'right'}}>{cat.score?.toFixed(0)??'—'}</span>
              </div>
              {open && (
                <div style={{padding:'4px 14px 12px 30px'}}>
                  {cat.signals.map(s=>{
                    const pending=s.status==='needs_source'||s.score==null;
                    return (
                      <div key={s.id} title={s.evidence} style={{display:'flex',alignItems:'center',gap:10,padding:'5px 0',borderBottom:'1px solid #1e1e1e',opacity:pending?0.5:1}}>
                        <span style={{fontSize:12,color:'#cdbfae',flex:1}}>{s.label}</span>
                        <span style={{fontSize:12,color:'#9d8b7a',width:70,textAlign:'right'}}>{pending?'—':fmtVal(s.id,s.raw_value)}</span>
                        <div style={{width:80,height:6,background:'#242424',borderRadius:3,overflow:'hidden'}}>
                          {!pending && <div style={{height:'100%',width:`${s.score}%`,background:heat(s.score)}}/>}
                        </div>
                        <span style={{fontSize:11,fontWeight:600,color:pending?'#555':heat(s.score),width:26,textAlign:'right'}}>{pending?'—':s.score!.toFixed(0)}</span>
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
