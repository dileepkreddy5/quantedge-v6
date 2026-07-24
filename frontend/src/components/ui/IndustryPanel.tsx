import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; }
interface Cat { id:string; label:string; weight:number; score:number|null; n_signals:number; n_scored:number; signals:Sig[]; }
interface IndData {
  ticker:string; available:boolean; score:number|null; industry_rating:string; confidence:number;
  coverage:{scored:number;total:number}; sector_name:string; sector_etf:string; sic:string; sic_description:string;
  tree:{categories:Cat[]}; key_metrics:Record<string,number|null>; reason?:string;
}
const heat=(s:number|null)=>s==null?'var(--border-2)':s>=70?'var(--gold)':s>=50?'var(--caramel)':s>=30?'#c9762f':'var(--bear)';
const ratingColor=(r:string)=>r.includes('Leader')?'var(--gold)':r.includes('Above')?'var(--gold)':r.includes('In-Line')?'var(--caramel)':r.includes('Below')?'#c9762f':'var(--bear)';

const fmtVal=(id:string,v:number|null):string=>{
  if(v==null) return '—';
  // Counts and tiers first: 'years_public' was reaching a later percent branch
  // and rendering 27.5 years as 2749.9%, while the summary card said 27.
  if(id.includes('years')) return v.toFixed(0)+'y';
  if(id.includes('tier')) return v.toFixed(0);
  if(id.includes('employee')||id.includes('peer_count')) return v.toLocaleString();
  if(id.includes('cap_eff')) return '$'+v.toFixed(0)+'M';
  if(id.includes('pct')||id.includes('rank')||id.includes('quartile')||id.includes('position')||id.includes('capture')||id.includes('in_favor')||id.includes('leader')||id.includes('cyclical')||id.includes('defensive')||id.includes('correlation')) {
    if(Math.abs(v)<=1) return (v*100).toFixed(0)+(id.includes('pct')||id.includes('rank')||id.includes('position')||id.includes('quartile')?' pct':'%');
  }
  if(id.includes('rs_')||id.includes('trend')||id.includes('vs_')||id.includes('magnitude')||id.includes('momentum')||id.includes('accel')||id.includes('drawdown')||id.includes('vol')) return (v*100).toFixed(1)+'%';
  if(id.includes('years')) return v.toFixed(0);
  if(id.includes('mcap')||id.includes('cap_b')) return '$'+v.toFixed(0)+'B';
  if(id.includes('beta')||id.includes('tier')||id.includes('vs_avg')) return v.toFixed(2);
  if(id.includes('employee')||id.includes('peer_count')) return v.toLocaleString();
  return v.toFixed(2);
};

const RSBar=({label,value}:{label:string,value:number|null})=>{
  if(value==null) return null;
  const pct=Math.max(-1,Math.min(1,value*3));
  const isPos=value>=0;
  return (
    <div style={{display:'flex',alignItems:'center',gap:10,padding:'4px 0'}}>
      <span style={{fontSize:11,color:'#9d8b7a',width:60}}>{label}</span>
      <div style={{flex:1,height:16,position:'relative',background:'#181818',borderRadius:4}}>
        <div style={{position:'absolute',left:'50%',top:0,bottom:0,width:1,background:'#3a3a3a'}}/>
        <div style={{position:'absolute',top:2,bottom:2,borderRadius:2,
          background:isPos?'var(--gold)':'var(--caramel)',
          left:isPos?'50%':`${50+pct*50}%`, width:`${Math.abs(pct)*50}%`}}/>
      </div>
      <span style={{fontFamily:'var(--font-mono)',fontSize:11,fontWeight:600,color:isPos?'var(--gold)':'var(--caramel)',width:52,textAlign:'right'}}>
        {isPos?'+':''}{(value*100).toFixed(1)}%</span>
    </div>
  );
};

export default function IndustryPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<IndData|null>(null);
  const [loading,setLoading]=useState(false);
  const [err,setErr]=useState('');
  const [expanded,setExpanded]=useState<Record<string,boolean>>({});
  const [allOpen,setAllOpen]=useState(false);

  useEffect(()=>{
    if(!ticker)return;
    setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/industry/${ticker}`)
      .then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No industry data');else setD(x);})
      .catch(e=>setErr(e?.message||'Request failed'))
      .finally(()=>setLoading(false));
  },[ticker]);

  if(!ticker)return <div style={{color:'#9d8b7a',padding:24}}>Enter a ticker for Industry Intelligence.</div>;
  if(loading)return <div style={{color:'#daa520',padding:24}}>Analyzing sector positioning — relative strength, peer rank, and 48 signals…</div>;
  if(err)return <div style={{fontFamily:'var(--font-body)',color:'var(--bear)',padding:24}}>Industry: {err}</div>;
  if(!d)return null;

  const km=d.key_metrics||{};
  const posCategory=d.tree.categories.find(c=>c.id==='industry_position');
  const pctSignals=posCategory?posCategory.signals.filter(s=>s.raw_value!=null):[];

  return (
    <div style={{padding:'8px 4px',color:'#e8ddd0'}}>
      <div style={{display:'flex',alignItems:'center',gap:24,marginBottom:18,flexWrap:'wrap'}}>
        <div style={{display:'flex',alignItems:'baseline',gap:10}}>
          <span style={{fontSize:46,fontWeight:700,color:heat(d.score),lineHeight:1}}>{d.score?.toFixed(0)??'—'}</span>
          <span style={{fontSize:16,color:'#9d8b7a'}}>/100</span>
        </div>
        <div>
          <div style={{fontSize:22,fontWeight:700,color:ratingColor(d.industry_rating),letterSpacing:0.5}}>{d.industry_rating}</div>
          <div style={{fontSize:11,color:'#9d8b7a',marginTop:2}}>{d.coverage.scored}/{d.coverage.total} signals · vs sector peers</div>
        </div>
        <div style={{marginLeft:'auto',textAlign:'right'}}>
          <div style={{fontSize:16,fontWeight:700,color:'#daa520'}}>{d.sector_name||'—'}</div>
          <div style={{fontSize:10,color:'#7a7266'}}>{d.sector_etf} · {d.sic_description||''}</div>
        </div>
      </div>

      <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:12,padding:'14px 16px',marginBottom:14}}>
        <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:8}}>RELATIVE STRENGTH vs {d.sector_etf} SECTOR</div>
        <RSBar label="1 month" value={d.tree.categories.find(c=>c.id==='sector_rel_perf')?.signals.find(s=>s.id==='rs_1m')?.raw_value??null}/>
        <RSBar label="3 month" value={km.rs_sector_3m??null}/>
        <RSBar label="1 year" value={km.rs_sector_1y??null}/>
        <div style={{fontSize:10,color:'#7a7266',marginTop:6}}>Center line = sector performance. Right/green = outperforming, left/red = lagging.</div>
      </div>

      {pctSignals.length>0 && (
        <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:12,padding:'14px 16px',marginBottom:14}}>
          <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:8}}>RANK WITHIN SECTOR (percentile)</div>
          {pctSignals.map(s=>(
            <div key={s.id} style={{display:'flex',alignItems:'center',gap:10,padding:'4px 0'}}>
              <span style={{fontSize:11,color:'#cdbfae',width:150}}>{s.label.replace(' percentile','')}</span>
              <div style={{flex:1,height:14,background:'#181818',borderRadius:4,overflow:'hidden'}}>
                <div style={{height:'100%',width:`${(s.raw_value||0)*100}%`,
                  background:(s.raw_value||0)>=0.6?'var(--gold)':(s.raw_value||0)>=0.4?'var(--caramel)':'#c9762f'}}/>
              </div>
              <span style={{fontSize:11,fontWeight:600,color:'#cdbfae',width:44,textAlign:'right'}}>{((s.raw_value||0)*100).toFixed(0)}th</span>
            </div>
          ))}
        </div>
      )}

      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(130px,1fr))',gap:8,marginBottom:16}}>
        {[['Composite Rank',km.composite_sector_rank!=null?(km.composite_sector_rank*100).toFixed(0)+'th':'—'],
          ['Sector Trend 3m',km.sector_trend_3m!=null?(km.sector_trend_3m*100).toFixed(1)+'%':'—'],
          ['Sector vs Market',km.sector_vs_spy_3m!=null?(km.sector_vs_spy_3m>0?'+':'')+(km.sector_vs_spy_3m*100).toFixed(1)+'%':'—'],
          ['Sector in Favor',km.sector_in_favor!=null?(km.sector_in_favor>0?'Yes':'No'):'—'],
          ['Peers in Sector',km.sector_peer_count!=null?km.sector_peer_count.toString():'—'],
          ['Years Public',km.years_public!=null?km.years_public.toFixed(0):'—']].map(([k,v])=>(
          <div key={k as string} style={{background:'#181818',border:'1px solid #2a2a2a',borderRadius:8,padding:'8px 10px'}}>
            <div style={{fontSize:9,color:'#9d8b7a'}}>{k}</div>
            <div style={{fontSize:14,fontWeight:600,color:'#daa520'}}>{v}</div>
          </div>
        ))}
      </div>

      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
        <span style={{fontSize:12,color:'#9d8b7a',letterSpacing:1}}>11 SECTOR DIMENSIONS · {d.tree.categories.reduce((a,c)=>a+c.n_signals,0)} SIGNALS</span>
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
                      <div key={s.id} title={s.evidence} style={{display:'grid',gridTemplateColumns:'minmax(150px,220px) 90px 1fr 30px',alignItems:'center',gap:14,padding:'5px 0',borderBottom:'1px solid var(--border-1)',opacity:pending?0.45:1}}>
                        <span style={{fontFamily:'var(--font-body)',fontSize:12,color:'var(--latte)'}}>{s.label}</span>
                        <span style={{fontFamily:'var(--font-mono)',fontSize:11.5,color:'var(--cocoa-dust)',textAlign:'right'}}>{pending?(s.status==='needs_source'?'pending':'—'):fmtVal(s.id,s.raw_value)}</span>
                        <div style={{height:5,background:'var(--surface-3)',borderRadius:2,overflow:'hidden'}}>
                          {!pending && <div style={{height:'100%',width:`${s.score}%`,background:heat(s.score),borderRadius:2}}/>}
                        </div>
                        <span style={{fontFamily:'var(--font-mono)',fontSize:11,fontWeight:600,color:pending?'var(--cocoa)':heat(s.score),textAlign:'right'}}>{pending?'—':s.score!.toFixed(0)}</span>
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
