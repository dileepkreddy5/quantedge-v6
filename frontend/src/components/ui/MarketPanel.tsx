import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
import MarketPositioning from './MarketPositioning';

interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; method?:string; }
interface Cat { id:string; label:string; weight:number; score:number|null; confidence:number; n_signals:number; n_scored:number; signals:Sig[]; }
interface MktData {
  ticker:string; available:boolean; score:number|null; market_rating:string; confidence:number;
  sector_bucket:string|null; peer_count:number; coverage:{scored:number;total:number};
  tree:{categories:Cat[]};
  regime:{garch?:{current_vol:number|null;vol_regime:string|null};regime?:{current:string|null;confidence:number|null};kalman?:{trend:string|null;state:string|null}}|null;
  momentum_ladder:Record<string,number|null>|null;
  key_metrics:Record<string,number|null>;
  volatility:Record<string,number|null>|null;
  trading_risk:Record<string,number|null>|null;
  volume:Record<string,number|null>|null;
  short_interest:Record<string,any>|null;
  relative_strength:Record<string,number|null>|null;
  price_position:Record<string,number|null>|null;
  reasons:string[]|null;
  sector_breadth:Record<string,any>|null;
  reason?:string;
}
const heat=(s:number|null)=>s==null?'var(--border-1)':s>=75?'var(--bull)':s>=58?'var(--bull)':s>=42?'var(--caramel)':s>=25?'#c9762f':'var(--bear)';
const ratingColor=(r:string)=>r.includes('Strong')||r.includes('Positive')?'var(--bull)':r.includes('Weak')||r.includes('Downtrend')?'var(--bear)':'var(--caramel)';
const LADDER_LABELS:Record<string,string>={mom_1m:'1 Month',mom_3m:'3 Month',mom_6m:'6 Month',mom_12_1:'12-1 Month'};

export default function MarketPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<MktData|null>(null);
  const [loading,setLoading]=useState(false);
  const [err,setErr]=useState('');
  const [expanded,setExpanded]=useState<Record<string,boolean>>({});
  const [allOpen,setAllOpen]=useState(false);
  useEffect(()=>{
    if(!ticker)return;
    setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/market/${ticker}`)
      .then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No market data');else{setD(x);const init:Record<string,boolean>={};(x.tree?.categories||[]).forEach((c:any)=>init[c.id]=false);setExpanded(init);}})
      .catch(e=>setErr(e?.message||'Request failed'))
      .finally(()=>setLoading(false));
  },[ticker]);

  if(!ticker)return <div style={{color:'var(--cocoa-dust)',padding:24}}>Enter a ticker for Market Intelligence.</div>;
  if(loading)return <div style={{color:'var(--gold)',padding:24}}>Computing Market Intelligence — peer-relative momentum, trend, regime…</div>;
  if(err)return <div style={{color:'var(--bear)',padding:24}}>Market: {err}</div>;
  if(!d)return null;

  const ladder=d.momentum_ladder||{};
  const catByCat=(id:string)=>d.tree.categories.find(c=>c.id===id);
  // find the peer-percentile score for each ladder timeframe from the signals
  const ladderScore=(key:string):number|null=>{
    for(const c of d.tree.categories){for(const s of c.signals){if(s.id===key||(s as any).id===key.replace('mom_',''))return s.score;}}
    // fallback: match by raw
    for(const c of d.tree.categories){for(const s of c.signals){if((s as any).label?.toLowerCase().includes(LADDER_LABELS[key]?.toLowerCase()||''))return s.score;}}
    return null;
  };
  const rg=d.regime||{};

  return (
    <div style={{padding:'8px 4px',color:'var(--latte)'}}>
      <div style={{display:'flex',alignItems:'center',gap:24,marginBottom:20,flexWrap:'wrap'}}>
        <div style={{display:'flex',alignItems:'baseline',gap:10}}>
          <span style={{fontSize:46,fontWeight:700,color:heat(d.score),lineHeight:1}}>{d.score?.toFixed(1)??'—'}</span>
          <span style={{fontSize:16,color:'var(--cocoa-dust)'}}>/100</span>
        </div>
        <div>
          <div style={{fontSize:20,fontWeight:700,color:ratingColor(d.market_rating),letterSpacing:1}}>{d.market_rating}</div>
          <div style={{fontSize:11,color:'var(--cocoa-dust)',marginTop:2}}>
            Ranked vs {d.peer_count} {d.sector_bucket} peers · coverage {d.coverage.scored}/{d.coverage.total}</div>
        </div>
        <MarketPositioning d={d} />

        {rg.regime?.current && (
          <div style={{marginLeft:'auto',display:'flex',gap:10}}>
            <div style={{background:'var(--surface-2)',border:'1px solid #2a2a2a',borderRadius:8,padding:'8px 14px',textAlign:'center'}}>
              <div style={{fontSize:10,color:'var(--cocoa-dust)'}}>MARKET REGIME</div>
              <div style={{fontSize:14,fontWeight:600,color:rg.regime.current.includes('BULL')?'var(--bull)':'var(--bear)'}}>{rg.regime.current.replace(/_/g,' ')}</div>
              <div style={{fontSize:9,color:'var(--cocoa)'}}>{rg.regime.confidence!=null?(rg.regime.confidence*100).toFixed(0)+'% conf':''}</div>
            </div>
            {rg.garch?.vol_regime && (
              <div style={{background:'var(--surface-2)',border:'1px solid #2a2a2a',borderRadius:8,padding:'8px 14px',textAlign:'center'}}>
                <div style={{fontSize:10,color:'var(--cocoa-dust)'}}>VOLATILITY</div>
                <div style={{fontSize:14,fontWeight:600,color:rg.garch.vol_regime==='HIGH'?'var(--bear)':rg.garch.vol_regime==='LOW'?'var(--bull)':'var(--caramel)'}}>{rg.garch.vol_regime}</div>
              </div>
            )}
          </div>
        )}
      </div>

      {d.reasons && d.reasons.length>0 && (
        <div style={{background:'#1a1512',border:'1px solid #3a2a1a',borderRadius:12,padding:'12px 16px',marginBottom:14}}>
          <div style={{fontSize:12,color:'var(--caramel)',letterSpacing:1,marginBottom:6,fontWeight:600}}>MARKET SUMMARY</div>
          <div style={{display:'flex',flexWrap:'wrap',gap:'4px 16px'}}>
            {d.reasons.map((r,i)=><span key={i} style={{fontSize:12,color:'var(--latte)'}}>▸ {r}</span>)}
          </div>
        </div>
      )}

      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
        <span style={{fontSize:12,color:'var(--cocoa-dust)',letterSpacing:1}}>MARKET INTELLIGENCE COMPONENTS · {d.tree.categories.reduce((a,c)=>a+c.n_signals,0)} SIGNALS</span>
        <button onClick={()=>{const v=!allOpen;setAllOpen(v);const m:Record<string,boolean>={};d.tree.categories.forEach(c=>m[c.id]=v);setExpanded(m);}}
          style={{background:'var(--surface-2)',border:'1px solid #2a2a2a',color:'var(--cocoa-dust)',borderRadius:8,padding:'5px 12px',fontSize:11,cursor:'pointer'}}>
          {allOpen?'Collapse all':'Expand all'}</button>
      </div>
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
                  {cat.signals.map((s:any)=>{
                    const pending=s.status==='needs_source'||s.score==null;
                    const rv=s.raw_value;
                    const fmt=rv==null?'—':Math.abs(rv)>=1000000?(rv/1e6).toFixed(0)+'M':(Number.isInteger(rv)?rv.toString():(Math.abs(rv)<1&&Math.abs(rv)>0?rv.toFixed(3):rv.toFixed(2)));
                    return (
                      <div key={s.id} title={s.evidence} style={{display:'flex',alignItems:'center',gap:10,padding:'5px 0',borderBottom:'1px solid #1e1e1e',opacity:pending?0.5:1}}>
                        <span style={{fontSize:12,color:'var(--latte)',flex:1}}>{s.label}</span>
                        <span style={{fontSize:12,color:'var(--cocoa-dust)',width:72,textAlign:'right'}}>{pending?'pending':fmt}</span>
                        <div style={{width:80,height:6,background:'var(--surface-3)',borderRadius:3,overflow:'hidden'}}>
                          {!pending && <div style={{height:'100%',width:`${s.score}%`,background:heat(s.score)}}/>}
                        </div>
                        <span style={{fontSize:11,fontWeight:600,color:pending?'var(--cocoa)':heat(s.score),width:26,textAlign:'right'}}>{pending?'—':s.score.toFixed(0)}</span>
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
