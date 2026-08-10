import { useState } from 'react';

const heat=(s:number|null)=>s==null?'var(--border-2)':s>=70?'var(--gold)':s>=50?'var(--caramel)':s>=30?'#c9762f':'var(--bear)';

export default function MarketPositioning({ d }:{ d:any }){
  const [hover,setHover]=useState<number|null>(null);
  const rg=d.regime||{};
  const ladder=d.momentum_ladder||{};
  const rs=d.relative_strength||{};
  const rating=d.market_rating||'';
  const km=d.key_metrics||{};
  const breadth=d.sector_breadth||{};

  const ratingCol=/Strong|Positive/i.test(rating)?'var(--gold)':/Neutral/i.test(rating)?'var(--neutral)':'var(--bear)';

  // momentum ladder → term structure (real fields: mom_1m/3m/6m/12_1)
  const steps=[
    {tf:'12-1m',v:ladder.mom_12_1},
    {tf:'6m',v:ladder.mom_6m},
    {tf:'3m',v:ladder.mom_3m},
    {tf:'1m',v:ladder.mom_1m},
  ].filter(s=>s.v!=null);

  const regimeCur=rg.regime?.current||'';
  const volReg=rg.garch?.vol_regime||'';
  const isBull=/BULL/i.test(regimeCur);
  const beta=km.beta ?? (d.volatility?.beta);

  // SVG geometry for the term-structure curve
  const W=640,H=220,x0=90,x1=600;
  const rets=steps.map(s=>s.v as number);
  const dMax=Math.max(...rets), dMin=Math.min(...rets);
  const pad=Math.max((dMax-dMin)*0.2, 3);
  const top=dMax+pad, bot=dMin-pad; const span=(top-bot)||1;
  const xof=(i:number)=>x0+i*((x1-x0)/Math.max(1,steps.length-1));
  const yof=(r:number)=>28+(1-(r-bot)/span)*160;   // map data range across full plot height
  const zeroY=28+(1-(0-bot)/span)*160;

  return (
    <div style={{marginBottom:16}}>
      {/* positioning banner + regime */}
      <div style={{display:'flex',gap:14,marginBottom:14,flexWrap:'wrap'}}>
        <div style={{flex:2,minWidth:280,background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:14,padding:'18px 20px'}}>
          <div style={{fontSize:11,letterSpacing:2,color:'#c9762f',marginBottom:6}}>MARKET POSITIONING</div>
          <div style={{fontSize:26,fontWeight:500,color:ratingCol,lineHeight:1}}>{rating||'—'}</div>
          <div style={{fontSize:12,color:'var(--cocoa-dust)',marginTop:8,lineHeight:1.5}}>
            Ranked against {d.peer_count} {d.sector_bucket||'sector'} peers.
            {steps.length>=2 && steps[steps.length-1].v!=null && steps[0].v!=null &&
              ((steps[steps.length-1].v as number)>(steps[0].v as number)
                ? ' Recent momentum is stronger than the longer-term trend — accelerating.'
                : ' Longer-term momentum leads recent — steady or cooling.')}
          </div>
        </div>
        <div style={{flex:1,minWidth:150,background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:14,padding:'16px 18px'}}>
          <div style={{fontSize:10,letterSpacing:1,color:'var(--cocoa)',marginBottom:4}}>MARKET REGIME</div>
          <div style={{fontSize:16,fontWeight:500,color:isBull?'var(--bull)':'var(--bear)'}}>
            {regimeCur?regimeCur.replace(/_/g,' '):'—'}{volReg?` · ${volReg} vol`:''}</div>
          {beta!=null && <div style={{fontSize:10,color:'var(--cocoa-dust)',marginTop:8}}>β {Number(beta).toFixed(2)} · {Number(beta)<1?'defensive to market swings':'amplifies market moves'}</div>}
          {rg.regime?.confidence!=null && <div style={{fontSize:10,color:'var(--cocoa)',marginTop:2}}>{(rg.regime.confidence*100).toFixed(0)}% regime confidence</div>}
        </div>
      </div>

      {/* momentum term-structure curve */}
      {steps.length>=2 && (
        <>
          <div style={{fontSize:10,color:'var(--gold)',letterSpacing:2,margin:'6px 0 4px'}}>MOMENTUM TERM-STRUCTURE · return across timeframes</div>
          <div style={{fontSize:11,color:'var(--cocoa-dust)',marginBottom:12}}>Each point is the return over that lookback. Rising left-to-right means momentum is accelerating.</div>
          <div style={{background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:14,padding:'20px 10px 8px'}}>
            <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{display:'block'}}>
              <line x1={x0-30} y1={zeroY} x2={x1+20} y2={zeroY} stroke="var(--surface-4)" strokeWidth="1" strokeDasharray="3 3"/>
              <text x={x0-38} y={zeroY+4} textAnchor="end" fontSize="9" fill="var(--cocoa)">0%</text>
              <polyline points={steps.map((s,i)=>`${xof(i)},${yof(s.v as number)}`).join(' ')} fill="none" stroke="var(--gold)" strokeWidth="2.5" strokeLinejoin="round"/>
              {steps.map((s,i)=>{
                const v=s.v as number; const col=v>=0?'var(--bull)':'var(--bear)'; const on=hover===i;
                return (
                  <g key={s.tf} onMouseEnter={()=>setHover(i)} onMouseLeave={()=>setHover(null)} style={{cursor:'pointer'}}>
                    <rect x={xof(i)-30} y={0} width={60} height={H} fill="transparent"/>
                    <circle cx={xof(i)} cy={yof(v)} r={on?7:5} fill={col} stroke="var(--surface-2)" strokeWidth="2"/>
                    <text x={xof(i)} y={yof(v)+(v>=0?-12:20)} textAnchor="middle" fontSize="12" fontWeight="600" fill={col}>{v>=0?'+':''}{v.toFixed(1)}%</text>
                    <text x={xof(i)} y="205" textAnchor="middle" fontSize="11" fill="var(--latte)">{s.tf}</text>
                  </g>
                );
              })}
            </svg>
          </div>
        </>
      )}

      {/* relative strength diverging */}
      {(rs.SPY!=null||rs.QQQ!=null||rs.XLK!=null) && (
        <>
          <div style={{fontSize:10,color:'var(--gold)',letterSpacing:2,margin:'20px 0 4px'}}>RELATIVE STRENGTH · 3-month vs benchmarks</div>
          <div style={{background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:14,padding:'16px 22px'}}>
            {[['vs S&P 500',rs.SPY],['vs Nasdaq-100',rs.QQQ],['vs sector ETF',rs.XLK]].filter(b=>b[1]!=null).map((b,i,arr)=>{
              const v=b[1] as number; const pct=Math.min(Math.abs(v)/20*50,50);
              return (
                <div key={b[0] as string} style={{display:'grid',gridTemplateColumns:'110px 1fr 52px',alignItems:'center',gap:12,padding:'7px 0',borderBottom:i<arr.length-1?'1px solid var(--border-1)':'none'}}>
                  <span style={{fontSize:12,color:'var(--latte)'}}>{b[0]}</span>
                  <div style={{position:'relative',height:12}}>
                    <div style={{position:'absolute',left:'50%',top:0,bottom:0,width:1,background:'var(--surface-4)'}}/>
                    <div style={{position:'absolute',top:2,bottom:2,background:v>=0?'var(--bull)':'var(--bear)',borderRadius:2,...(v>=0?{left:'50%',width:`${pct}%`}:{right:'50%',width:`${pct}%`})}}/>
                  </div>
                  <span style={{fontSize:12,fontWeight:600,color:v>=0?'var(--bull)':'var(--bear)',textAlign:'right'}}>{v>=0?'+':''}{v.toFixed(1)}%</span>
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* ── MARKET CONTEXT BAND ── */}
      {(d.price_position||d.sector_breadth) && (
        <div style={{display:'grid',gridTemplateColumns:'1.4fr 1fr',gap:14,marginTop:20}}>
          {/* 52-week range track */}
          {d.price_position?.range_percentile!=null && (
            <div style={{background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:14,padding:'16px 18px'}}>
              <div style={{fontSize:10,color:'var(--gold)',letterSpacing:1,marginBottom:14}}>52-WEEK RANGE POSITION</div>
              <div style={{position:'relative',height:44}}>
                <div style={{position:'absolute',left:0,right:0,top:16,height:8,borderRadius:4,opacity:0.5,
                  background:'linear-gradient(90deg,var(--bear) 0%,var(--caramel) 50%,var(--bull) 100%)'}}/>
                <div style={{position:'absolute',left:`${Math.max(2,Math.min(97,d.price_position.range_percentile))}%`,top:8,width:3,height:24,background:'var(--gold)',borderRadius:2,transform:'translateX(-1px)'}}/>
                <div style={{position:'absolute',left:`${Math.max(2,Math.min(97,d.price_position.range_percentile))}%`,top:-4,transform:'translateX(-50%)',fontSize:11,color:'var(--gold)',fontWeight:600,whiteSpace:'nowrap'}}>now · {d.price_position.range_percentile.toFixed(0)}th %ile</div>
                <div style={{position:'absolute',left:0,top:38,fontSize:10,color:'var(--cocoa)'}}>52w low</div>
                <div style={{position:'absolute',right:0,top:38,fontSize:10,color:'var(--cocoa)'}}>52w high</div>
              </div>
              <div style={{display:'flex',justifyContent:'space-between',marginTop:6,fontSize:11}}>
                {d.price_position.pct_from_52w_high!=null && <span style={{color:'var(--cocoa-dust)'}}>{d.price_position.pct_from_52w_high.toFixed(1)}% from high</span>}
                {d.price_position.pct_from_52w_low!=null && <span style={{color:'var(--bull)'}}>+{d.price_position.pct_from_52w_low.toFixed(1)}% from low</span>}
              </div>
            </div>
          )}
          {/* sector health */}
          {d.sector_breadth?.breadth_score!=null && (
            <div style={{background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:14,padding:'16px 18px'}}>
              <div style={{fontSize:10,color:'var(--gold)',letterSpacing:1,marginBottom:10}}>SECTOR HEALTH · {d.sector_breadth.universe_size} {d.sector_breadth.sector||''} peers</div>
              <div style={{display:'flex',alignItems:'baseline',gap:8}}>
                <span style={{fontSize:28,fontWeight:500,color:heat(d.sector_breadth.breadth_score)}}>{d.sector_breadth.breadth_score.toFixed(0)}</span>
                <span style={{fontSize:11,color:'var(--cocoa-dust)'}}>breadth score</span>
              </div>
              <div style={{marginTop:10,fontSize:11,color:'var(--cocoa-dust)',lineHeight:1.9}}>
                {[['above 50-day MA',d.sector_breadth.pct_above_ma50],['above 200-day MA',d.sector_breadth.pct_above_ma200],['positive momentum',d.sector_breadth.pct_positive_mom]].filter(x=>x[1]!=null).map(x=>(
                  <div key={x[0] as string} style={{display:'flex',justifyContent:'space-between'}}>
                    <span>{x[0]}</span><span style={{color:heat(x[1] as number),fontWeight:600}}>{(x[1] as number).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── TREND QUALITY strip ── */}
      {d.key_metrics && (
        <div style={{background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:14,padding:'16px 18px',marginTop:14}}>
          <div style={{fontSize:10,color:'var(--gold)',letterSpacing:1,marginBottom:12}}>TREND QUALITY</div>
          {[
            {n:'Hurst persistence',v:d.key_metrics.hurst,lo:0.3,hi:0.7,mid:0.5,fmt:(x:number)=>x.toFixed(2),note:(x:number)=>x>0.55?'trending':x<0.45?'mean-reverting':'random walk'},
            {n:'3-month Sharpe',v:d.key_metrics.sharpe_3m,lo:-0.5,hi:1.5,mid:0.5,fmt:(x:number)=>x.toFixed(2),note:(x:number)=>x>0.8?'strong risk-adjusted':x>0.3?'moderate':'weak risk-adjusted'},
            {n:'MA alignment',v:d.key_metrics.ma_alignment,lo:0,hi:1,mid:0.5,fmt:(x:number)=>x.toFixed(2),note:(x:number)=>x>=0.99?'fully aligned uptrend':x>=0.5?'partial alignment':'misaligned'},
            {n:'Price vs 50-day',v:d.key_metrics.pct_above_ma50,lo:-10,hi:10,mid:0,fmt:(x:number)=>x.toFixed(1)+'%',note:(x:number)=>x>0?'above':'below'},
          ].filter(m=>m.v!=null).map((m,i,arr)=>{
            const v=m.v as number; const pct=Math.max(0,Math.min(100,((v-m.lo)/(m.hi-m.lo))*100));
            const col=v>=m.mid?'var(--bull)':'var(--caramel)';
            return (
              <div key={m.n} style={{display:'grid',gridTemplateColumns:'130px 1fr 150px',alignItems:'center',gap:12,padding:'6px 0',borderBottom:i<arr.length-1?'1px solid var(--border-1)':'none'}}>
                <span style={{fontSize:12,color:'var(--latte)'}}>{m.n}</span>
                <div style={{position:'relative',height:6,background:'var(--surface-4)',borderRadius:3}}>
                  <div style={{position:'absolute',left:0,top:0,bottom:0,width:`${pct}%`,background:col,borderRadius:3}}/>
                </div>
                <span style={{fontSize:11,color:'var(--cocoa-dust)',textAlign:'right'}}><b style={{color:col}}>{m.fmt(v)}</b> · {m.note(v)}</span>
              </div>
            );
          })}
        </div>
      )}

    </div>
  );
}
