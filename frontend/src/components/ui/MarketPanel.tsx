import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

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
const heat=(s:number|null)=>s==null?'#2a2a2a':s>=75?'#0f6e56':s>=58?'#1d9e75':s>=42?'#8a7519':s>=25?'#a35a1d':'#7a2320';
const ratingColor=(r:string)=>r.includes('Strong')||r.includes('Positive')?'#1d9e75':r.includes('Weak')||r.includes('Downtrend')?'#c0705a':'#c9a227';
const LADDER_LABELS:Record<string,string>={mom_1m:'1 Month',mom_3m:'3 Month',mom_6m:'6 Month',mom_12_1:'12-1 Month'};

export default function MarketPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<MktData|null>(null);
  const [loading,setLoading]=useState(false);
  const [err,setErr]=useState('');
  useEffect(()=>{
    if(!ticker)return;
    setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/market/${ticker}`)
      .then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No market data');else setD(x);})
      .catch(e=>setErr(e?.message||'Request failed'))
      .finally(()=>setLoading(false));
  },[ticker]);

  if(!ticker)return <div style={{color:'#9d8b7a',padding:24}}>Enter a ticker for Market Intelligence.</div>;
  if(loading)return <div style={{color:'#daa520',padding:24}}>Computing Market Intelligence — peer-relative momentum, trend, regime…</div>;
  if(err)return <div style={{color:'#c0705a',padding:24}}>Market: {err}</div>;
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
    <div style={{padding:'8px 4px',color:'#e8ddd0'}}>
      <div style={{display:'flex',alignItems:'center',gap:24,marginBottom:20,flexWrap:'wrap'}}>
        <div style={{display:'flex',alignItems:'baseline',gap:10}}>
          <span style={{fontSize:46,fontWeight:700,color:heat(d.score),lineHeight:1}}>{d.score?.toFixed(1)??'—'}</span>
          <span style={{fontSize:16,color:'#9d8b7a'}}>/100</span>
        </div>
        <div>
          <div style={{fontSize:20,fontWeight:700,color:ratingColor(d.market_rating),letterSpacing:1}}>{d.market_rating}</div>
          <div style={{fontSize:11,color:'#9d8b7a',marginTop:2}}>
            Ranked vs {d.peer_count} {d.sector_bucket} peers · coverage {d.coverage.scored}/{d.coverage.total}</div>
        </div>
        {rg.regime?.current && (
          <div style={{marginLeft:'auto',display:'flex',gap:10}}>
            <div style={{background:'#181818',border:'1px solid #2a2a2a',borderRadius:8,padding:'8px 14px',textAlign:'center'}}>
              <div style={{fontSize:10,color:'#9d8b7a'}}>MARKET REGIME</div>
              <div style={{fontSize:14,fontWeight:600,color:rg.regime.current.includes('BULL')?'#1d9e75':'#c0705a'}}>{rg.regime.current.replace(/_/g,' ')}</div>
              <div style={{fontSize:9,color:'#7a7266'}}>{rg.regime.confidence!=null?(rg.regime.confidence*100).toFixed(0)+'% conf':''}</div>
            </div>
            {rg.garch?.vol_regime && (
              <div style={{background:'#181818',border:'1px solid #2a2a2a',borderRadius:8,padding:'8px 14px',textAlign:'center'}}>
                <div style={{fontSize:10,color:'#9d8b7a'}}>VOLATILITY</div>
                <div style={{fontSize:14,fontWeight:600,color:rg.garch.vol_regime==='HIGH'?'#c0705a':rg.garch.vol_regime==='LOW'?'#1d9e75':'#c9a227'}}>{rg.garch.vol_regime}</div>
              </div>
            )}
          </div>
        )}
      </div>

      {d.reasons && d.reasons.length>0 && (
        <div style={{background:'#1a1512',border:'1px solid #3a2a1a',borderRadius:12,padding:'12px 16px',marginBottom:14}}>
          <div style={{fontSize:12,color:'#c9a227',letterSpacing:1,marginBottom:6,fontWeight:600}}>MARKET SUMMARY</div>
          <div style={{display:'flex',flexWrap:'wrap',gap:'4px 16px'}}>
            {d.reasons.map((r,i)=><span key={i} style={{fontSize:12,color:'#cdbfae'}}>▸ {r}</span>)}
          </div>
        </div>
      )}

      {/* RELATIVE STRENGTH vs benchmarks + PRICE POSITION */}
      <div style={{display:'grid',gridTemplateColumns:'1.3fr 1fr',gap:14,marginBottom:14}}>
        {d.relative_strength && Object.keys(d.relative_strength).length>0 && (
          <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:12,padding:16}}>
            <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:12}}>RELATIVE STRENGTH — 3-month vs benchmarks</div>
            {Object.entries(d.relative_strength).map(([k,v])=>{
              const val=v as number; const w=Math.min(50,Math.abs(val));
              return (
                <div key={k} style={{display:'flex',alignItems:'center',gap:10,marginBottom:8}}>
                  <div style={{width:44,fontSize:12,color:'#cdbfae'}}>{k}</div>
                  <div style={{flex:1,position:'relative',height:16,background:'#242424',borderRadius:8}}>
                    <div style={{position:'absolute',left:'50%',top:0,bottom:0,width:1,background:'#555'}}/>
                    <div style={{position:'absolute',left:val>=0?'50%':`${50-w}%`,width:`${w}%`,top:0,bottom:0,
                      background:val>=0?'#1d9e75':'#c0705a',borderRadius:4}}/>
                  </div>
                  <div style={{width:56,fontSize:12,fontWeight:600,color:val>=0?'#1d9e75':'#c0705a',textAlign:'right'}}>{val>=0?'+':''}{val.toFixed(1)}%</div>
                </div>
              );
            })}
            <div style={{fontSize:10,color:'#7a7266',marginTop:4}}>vs SPY (market), QQQ (growth), XLK (tech sector)</div>
          </div>
        )}
        {d.price_position && (
          <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:12,padding:16}}>
            <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:12}}>PRICE POSITION</div>
            <div style={{display:'flex',justifyContent:'space-between',marginBottom:10,fontSize:12}}>
              <span style={{color:'#9d8b7a'}}>From 52-week high</span>
              <span style={{color:'#c0705a',fontWeight:600}}>{d.price_position.pct_from_52w_high?.toFixed(1)}%</span></div>
            <div style={{display:'flex',justifyContent:'space-between',marginBottom:10,fontSize:12}}>
              <span style={{color:'#9d8b7a'}}>From 52-week low</span>
              <span style={{color:'#1d9e75',fontWeight:600}}>+{d.price_position.pct_from_52w_low?.toFixed(1)}%</span></div>
            {d.price_position.range_percentile!=null && (
              <div>
                <div style={{fontSize:10,color:'#7a7266',marginBottom:4}}>Position in 52-week range</div>
                <div style={{height:12,background:'#242424',borderRadius:6,overflow:'hidden'}}>
                  <div style={{height:'100%',width:`${d.price_position.range_percentile}%`,background:'#c9a227'}}/></div>
                <div style={{fontSize:11,color:'#cdbfae',marginTop:3}}>{d.price_position.range_percentile.toFixed(0)}th percentile of range</div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* SECTOR BREADTH */}
      {d.sector_breadth && d.sector_breadth.breadth_score!=null && (
        <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:12,padding:16,marginBottom:14}}>
          <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:12}}>
            {d.sector_breadth.sector} SECTOR BREADTH — {d.sector_breadth.universe_size} peers</div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(150px,1fr))',gap:12}}>
            {[['% above 50-day MA',d.sector_breadth.pct_above_ma50],
              ['% above 200-day MA',d.sector_breadth.pct_above_ma200],
              ['% positive momentum',d.sector_breadth.pct_positive_mom],
              ['Breadth score',d.sector_breadth.breadth_score]].map(([k,v])=>(
              <div key={k as string}>
                <div style={{fontSize:10,color:'#9d8b7a',marginBottom:4}}>{k as string}</div>
                <div style={{display:'flex',alignItems:'center',gap:8}}>
                  <div style={{flex:1,height:8,background:'#242424',borderRadius:4,overflow:'hidden'}}>
                    <div style={{height:'100%',width:`${v}%`,background:(v as number)>=50?'#1d9e75':'#a35a1d'}}/></div>
                  <span style={{fontSize:13,fontWeight:600,color:(v as number)>=50?'#1d9e75':'#c0705a',width:44,textAlign:'right'}}>{(v as number).toFixed(0)}%</span>
                </div>
              </div>
            ))}
          </div>
          <div style={{fontSize:10,color:'#7a7266',marginTop:8}}>Is the sector healthy? High breadth = broad participation, not just a few names.</div>
        </div>
      )}

      {/* MOMENTUM LADDER — hero */}
      <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:12,padding:18,marginBottom:16}}>
        <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:16}}>MOMENTUM LADDER — return & peer-percentile by timeframe</div>
        {['mom_1m','mom_3m','mom_6m','mom_12_1'].map(k=>{
          const raw=ladder[k]; const pct=ladderScore(k);
          return (
            <div key={k} style={{display:'flex',alignItems:'center',gap:14,marginBottom:12}}>
              <div style={{width:90,fontSize:12,color:'#cdbfae'}}>{LADDER_LABELS[k]}</div>
              <div style={{width:70,fontSize:13,fontWeight:600,color:raw!=null&&raw>=0?'#1d9e75':'#c0705a',textAlign:'right'}}>
                {raw!=null?(raw>=0?'+':'')+raw.toFixed(1)+'%':'—'}</div>
              <div style={{flex:1,height:16,background:'#242424',borderRadius:8,overflow:'hidden',position:'relative'}}>
                <div style={{height:'100%',width:`${pct??0}%`,background:heat(pct),transition:'width 0.6s'}}/>
              </div>
              <div style={{width:80,fontSize:11,color:heat(pct),fontWeight:600}}>{pct!=null?pct.toFixed(0)+'th pctile':'—'}</div>
            </div>
          );
        })}
        <div style={{fontSize:10,color:'#7a7266',marginTop:6}}>Percentile = rank vs {d.sector_bucket} sector peers. Higher bar = outperforming sector.</div>
      </div>

      {/* KEY METRICS strip */}
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(130px,1fr))',gap:10,marginBottom:16}}>
        {[['Hurst (persistence)',d.key_metrics.hurst,(v:number)=>v.toFixed(3)],
          ['3M Sharpe',d.key_metrics.sharpe_3m,(v:number)=>v.toFixed(2)],
          ['MA Alignment',d.key_metrics.ma_alignment,(v:number)=>v.toFixed(1)],
          ['% above MA50',d.key_metrics.pct_above_ma50,(v:number)=>v.toFixed(1)+'%'],
          ['% above MA200',d.key_metrics.pct_above_ma200,(v:number)=>v.toFixed(1)+'%']].map(([k,v,fmt])=>(
          <div key={k as string} style={{background:'#181818',border:'1px solid #2a2a2a',borderRadius:8,padding:'10px 12px'}}>
            <div style={{fontSize:10,color:'#9d8b7a',marginBottom:3}}>{k as string}</div>
            <div style={{fontSize:16,fontWeight:600,color:'#daa520'}}>{v!=null?(fmt as any)(v):'—'}</div>
          </div>
        ))}
      </div>

      {/* 6 CATEGORY SCORES */}
      <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:8}}>MARKET INTELLIGENCE COMPONENTS · peer-relative + technical</div>
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(170px,1fr))',gap:8}}>
        {d.tree.categories.map(cat=>(
          <div key={cat.id} style={{background:heat(cat.score),borderRadius:8,padding:'10px 12px'}}>
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
              <span style={{fontSize:11,fontWeight:600,color:'#fff'}}>{cat.label}</span>
              <span style={{fontSize:16,fontWeight:700,color:'#fff'}}>{cat.score?.toFixed(0)??'—'}</span>
            </div>
            <div style={{fontSize:9,color:'#fff',opacity:0.6,marginTop:2}}>wt {cat.weight.toFixed(2)} · {cat.n_scored}/{cat.n_signals}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
