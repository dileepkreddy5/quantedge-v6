import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; method?:string; }
interface Cat { id:string; label:string; weight:number; score:number|null; confidence:number; n_signals:number; n_scored:number; signals:Sig[]; }
interface ValData {
  ticker:string; available:boolean; score:number|null; valuation_rating:string; confidence:number;
  coverage:{scored:number;total:number}; beta_used:number|null; wacc_used:number; current_price:number|null;
  tree:{categories:Cat[]};
  key_metrics:Record<string,number|null>;
  sensitivity:{waccs:number[];terminal_growths:number[];grid:(number|null)[][];pct_scenarios_above_price:number|null}|null;
  value_range:{current_price:number;methods:Record<string,number|null>}|null;
  assumptions:Record<string,number|null>|null;
  cases:{bear:number|null;base:number|null;bull:number|null;probabilities:Record<string,number>|null;confidence_low:number|null;confidence_high:number|null}|null;
  expected_return:Record<string,number|null>|null;
  model_agreement:{dispersion_cv:number|null;consensus_overvalued:number|null;agreement_score:number|null}|null;
  driver_waterfall:Record<string,number>|null;
  reasons:string[]|null;
  model_confidence:{models:{model:string;confidence:number}[];overall:number|null}|null;
  time_horizon_years:number|null;
  reason?:string;
}

const heat=(s:number|null)=>s==null?'var(--border-2)':s>=70?'var(--gold)':s>=50?'var(--caramel)':s>=30?'#c9762f':'var(--bear)';
const ratingColor=(r:string)=>r.includes('Undervalued')?'var(--bull)':r.includes('Overvalued')?'var(--bear)':'var(--caramel)';
const METHOD_LABELS:Record<string,string>={dcf_bear:'DCF Bear',dcf_base:'DCF Base',dcf_bull:'DCF Bull',
  epv:'EPV',graham:'Graham',residual_income:'Residual Income',ddm:'DDM',nav:'NAV'};

export default function ValuationPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<ValData|null>(null);
  const [loading,setLoading]=useState(false);
  const [err,setErr]=useState('');
  const [expanded,setExpanded]=useState<Record<string,boolean>>({});
  const [allOpen,setAllOpen]=useState(true);
  useEffect(()=>{
    if(!ticker)return;
    setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/valuation/${ticker}`)
      .then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No valuation data');else{setD(x);const init:Record<string,boolean>={};(x.tree?.categories||[]).forEach((c:any)=>init[c.id]=true);setExpanded(init);}})
      .catch(e=>setErr(e?.message||'Request failed'))
      .finally(()=>setLoading(false));
  },[ticker]);

  if(!ticker)return <div style={{color:'var(--cocoa-dust)',padding:24}}>Enter a ticker for Valuation Intelligence.</div>;
  if(loading)return <div style={{color:'var(--gold)',padding:24}}>Computing Valuation — DCF, EPV, Graham, residual income, DDM, reverse-DCF, real CAPM beta…</div>;
  if(err)return <div style={{color:'var(--bear)',padding:24}}>Valuation: {err}</div>;
  if(!d)return null;

  const km=d.key_metrics;
  const price=d.current_price||0;
  const methods=d.value_range?.methods||{};
  const methodEntries=Object.entries(methods).filter(([_,v])=>v!=null) as [string,number][];
  const allVals=[...methodEntries.map(([_,v])=>v),price].filter(v=>v>0);
  const lo=Math.min(...allVals)*0.9, hi=Math.max(...allVals)*1.1;
  const pos=(v:number)=>((v-lo)/(hi-lo))*100;
  const fmtD=(v:number|null)=>v==null?'—':'$'+v.toFixed(2);
  const fmtP=(v:number|null)=>v==null?'—':(v*100).toFixed(1)+'%';
  const fmtX=(v:number|null)=>v==null?'—':v.toFixed(2)+'×';

  return (
    <div style={{padding:'8px 4px',color:'var(--latte)'}}>
      <div style={{display:'flex',alignItems:'center',gap:24,marginBottom:20,flexWrap:'wrap'}}>
        <div style={{display:'flex',alignItems:'baseline',gap:10}}>
          <span style={{fontSize:46,fontWeight:700,color:heat(d.score),lineHeight:1}}>{d.score?.toFixed(1)??'—'}</span>
          <span style={{fontSize:16,color:'var(--cocoa-dust)'}}>/100</span>
        </div>
        <div>
          <div style={{fontSize:20,fontWeight:700,color:ratingColor(d.valuation_rating),letterSpacing:1}}>{d.valuation_rating}</div>
          <div style={{fontSize:11,color:'var(--cocoa-dust)',marginTop:2}}>
            Coverage {d.coverage.scored}/{d.coverage.total} · β {d.beta_used?.toFixed(2)??'—'} · WACC {(d.wacc_used*100).toFixed(1)}%</div>
        </div>
        <div style={{marginLeft:'auto',display:'flex',gap:10,flexWrap:'wrap'}}>
          {[['Fair Value',fmtD(km.fair_value)],['Margin of Safety',fmtP(km.margin_of_safety)],
            ['Buy Zone',fmtD(km.buy_zone)],['Sell Zone',fmtD(km.sell_zone)]].map(([k,v])=>(
            <div key={k} style={{background:'var(--surface-2)',border:'1px solid #2a2a2a',borderRadius:8,padding:'8px 12px',minWidth:90}}>
              <div style={{fontSize:10,color:'var(--cocoa-dust)'}}>{k}</div>
              <div style={{fontSize:16,fontWeight:600,color:'var(--gold)'}}>{v}</div>
            </div>
          ))}
        </div>
      </div>

      {d.reasons && d.reasons.length>0 && (
        <div style={{background:'var(--surface-2)',border:'1px solid #3a2a1a',borderRadius:12,padding:'14px 18px',marginBottom:16}}>
          <div style={{fontSize:12,color:'var(--caramel)',letterSpacing:1,marginBottom:8,fontWeight:600}}>WHY {d.valuation_rating.toUpperCase()}?</div>
          <div style={{display:'flex',flexWrap:'wrap',gap:'6px 18px'}}>
            {d.reasons.map((r,i)=>(
              <span key={i} style={{fontSize:12,color:'var(--latte)'}}>▸ {r}</span>
            ))}
          </div>
        </div>
      )}

      <div style={{background:'var(--surface-2)',border:'1px solid #2a2a2a',borderRadius:12,padding:'18px 20px 18px 130px',marginBottom:18}}>
        <div style={{fontSize:12,color:'var(--cocoa-dust)',letterSpacing:1,marginBottom:22,marginLeft:-110}}>VALUATION RANGE — method fair values vs current price ${price.toFixed(2)}</div>
        <div style={{position:'relative'}}>
          <div style={{position:'absolute',left:`${Math.max(0,Math.min(100,pos(price)))}%`,top:-6,bottom:16,width:2,background:'var(--bear)',zIndex:2}}>
            <div style={{position:'absolute',top:-16,left:'50%',transform:'translateX(-50%)',whiteSpace:'nowrap',fontSize:10,color:'var(--bear)',fontWeight:600}}>${price.toFixed(0)}</div>
          </div>
          {methodEntries.map(([k,v])=>{
            const cheap=v>=price;
            return (
              <div key={k} style={{position:'relative',marginBottom:11,height:20}}>
                <div style={{position:'absolute',left:-118,width:110,color:'var(--latte)',textAlign:'right',fontSize:12,top:2}}>{METHOD_LABELS[k]||k}</div>
                <div style={{position:'absolute',left:0,right:0,top:9,height:2,background:'var(--surface-3)'}}/>
                <div style={{position:'absolute',left:`${Math.max(0,Math.min(100,pos(v)))}%`,transform:'translateX(-50%)',top:0,
                  background:cheap?'var(--bull)':'#c9762f',color:'var(--cream)',borderRadius:4,padding:'1px 7px',fontSize:11,fontWeight:600,whiteSpace:'nowrap',zIndex:1}}>${v.toFixed(0)}</div>
              </div>
            );
          })}
        </div>
        <div style={{marginTop:14,fontSize:11,color:'var(--cocoa)',marginLeft:-110}}>
          <span style={{color:'var(--bull)'}}>green</span> = above price (upside) · <span style={{color:'#c9762f'}}>amber</span> = below price · <span style={{color:'var(--bear)'}}>red line</span> = current price</div>
      </div>

      {/* ASSUMPTIONS PANEL */}
      {d.assumptions && (
        <div style={{background:'var(--surface-2)',border:'1px solid #2a2a2a',borderRadius:12,padding:16,marginBottom:14}}>
          <div style={{fontSize:12,color:'var(--cocoa-dust)',letterSpacing:1,marginBottom:12}}>DCF ASSUMPTIONS — what drives fair value</div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(120px,1fr))',gap:10}}>
            {[['Revenue CAGR',d.assumptions.assumption_revenue_cagr,'%'],
              ['WACC',d.assumptions.assumption_wacc,'%'],
              ['Terminal Growth',d.assumptions.assumption_terminal_growth,'%'],
              ['Operating Margin',d.assumptions.assumption_operating_margin,'%'],
              ['Tax Rate',d.assumptions.assumption_tax_rate,'%'],
              ['FCF Margin',d.assumptions.assumption_fcf_margin,'%'],
              ['Beta',d.assumptions.assumption_beta,'x'],
              ['Forecast Horizon',d.assumptions.assumption_forecast_years,'yr']].map(([k,v,u])=>(
              <div key={k as string}>
                <div style={{fontSize:10,color:'var(--cocoa)'}}>{k}</div>
                <div style={{fontSize:16,fontWeight:600,color:'var(--latte)'}}>
                  {v==null?'—':u==='%'?((v as number)*100).toFixed(1)+'%':u==='x'?(v as number).toFixed(2):(v as number)+' yr'}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* CASES + EXPECTED RETURN + MODEL AGREEMENT */}
      <div style={{display:'grid',gridTemplateColumns:'1.2fr 1fr 0.8fr',gap:14,marginBottom:14}}>
        {d.cases && (
          <div style={{background:'var(--surface-2)',border:'1px solid #2a2a2a',borderRadius:12,padding:16}}>
            <div style={{fontSize:12,color:'var(--cocoa-dust)',letterSpacing:1,marginBottom:12}}>SCENARIO FAIR VALUES</div>
            {[['Bear',d.cases.bear,d.cases.probabilities?.bear,'#c9762f'],
              ['Base',d.cases.base,d.cases.probabilities?.base,'var(--bull)'],
              ['Bull',d.cases.bull,d.cases.probabilities?.bull,'var(--bull)']].map(([k,v,p,c])=>(
              <div key={k as string} style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:8}}>
                <span style={{fontSize:12,color:'var(--latte)',width:44}}>{k}</span>
                <span style={{fontSize:10,color:'var(--cocoa)'}}>{p!=null?((p as number)*100).toFixed(0)+'%':''}</span>
                <span style={{fontSize:16,fontWeight:700,color:c as string}}>{v==null?'—':'$'+(v as number).toFixed(0)}</span>
              </div>
            ))}
            {d.cases.confidence_low!=null && (
              <div style={{marginTop:8,paddingTop:8,borderTop:'1px solid var(--border-1)',fontSize:11,color:'var(--cocoa-dust)'}}>
                Confidence range: <span style={{color:'var(--gold)'}}>${d.cases.confidence_low?.toFixed(0)}–${d.cases.confidence_high?.toFixed(0)}</span></div>
            )}
          </div>
        )}
        {d.expected_return && (
          <div style={{background:'var(--surface-2)',border:'1px solid #2a2a2a',borderRadius:12,padding:16}}>
            <div style={{fontSize:12,color:'var(--cocoa-dust)',letterSpacing:1,marginBottom:6}}>EXPECTED RETURN{d.time_horizon_years?` · ${d.time_horizon_years}YR HORIZON`:''}</div>
            <div style={{fontSize:26,fontWeight:700,color:(d.expected_return.expected_total_return??0)>=0?'var(--bull)':'var(--bear)',marginBottom:10}}>
              {((d.expected_return.expected_total_return??0)*100).toFixed(1)}%<span style={{fontSize:11,color:'var(--cocoa)'}}> /yr</span></div>
            {[['Revenue growth',d.expected_return.exp_return_growth,'var(--bull)'],
              ['Dividend',d.expected_return.exp_return_dividend,'#4bbe8a'],
              ['Buyback',d.expected_return.exp_return_buyback,'#6fae7a'],
              ['Margin',d.expected_return.exp_return_margin,'var(--caramel)'],
              ['Multiple change',d.expected_return.exp_return_multiple,'#c9762f']].map(([k,v,c])=>(
              <div key={k as string} style={{display:'flex',justifyContent:'space-between',fontSize:11,marginBottom:4}}>
                <span style={{color:'var(--cocoa-dust)'}}>{k}</span>
                <span style={{color:c as string,fontWeight:600}}>{v==null?'—':((v as number)>=0?'+':'')+((v as number)*100).toFixed(1)+'%'}</span>
              </div>
            ))}
          </div>
        )}
        {d.model_confidence && (
          <div style={{background:'var(--surface-2)',border:'1px solid #2a2a2a',borderRadius:12,padding:16}}>
            <div style={{fontSize:12,color:'var(--cocoa-dust)',letterSpacing:1,marginBottom:4}}>MODEL CONFIDENCE</div>
            <div style={{fontSize:22,fontWeight:700,color:'var(--gold)',marginBottom:8}}>
              {d.model_confidence.overall!=null?(d.model_confidence.overall*100).toFixed(0)+'%':'—'}
              <span style={{fontSize:10,color:'var(--cocoa)',fontWeight:400}}> overall</span></div>
            {d.model_confidence.models.map(m=>(
              <div key={m.model} style={{marginBottom:5}}>
                <div style={{display:'flex',justifyContent:'space-between',fontSize:10,color:'var(--cocoa-dust)'}}>
                  <span>{m.model}</span><span>{(m.confidence*100).toFixed(0)}%</span></div>
                <div style={{height:4,background:'var(--surface-3)',borderRadius:2,overflow:'hidden'}}>
                  <div style={{height:'100%',width:`${m.confidence*100}%`,background:'var(--bull)'}}/></div>
              </div>
            ))}
            {d.model_agreement?.consensus_overvalued!=null && (
              <div style={{fontSize:10,color:'var(--latte)',marginTop:8}}>
                <span style={{color:d.model_agreement.consensus_overvalued>=0.6?'var(--bear)':'var(--bull)',fontWeight:600}}>
                  {(d.model_agreement.consensus_overvalued*100).toFixed(0)}%</span> of methods say overvalued</div>
            )}
          </div>
        )}
      </div>

      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:14,marginBottom:18}}>
        {d.sensitivity?.grid && (
          <div style={{background:'var(--surface-2)',border:'1px solid #2a2a2a',borderRadius:12,padding:16}}>
            <div style={{fontSize:12,color:'var(--cocoa-dust)',letterSpacing:1,marginBottom:4}}>DCF SENSITIVITY — fair value $/share</div>
            <div style={{fontSize:10,color:'var(--cocoa)',marginBottom:10}}>
              {d.sensitivity.pct_scenarios_above_price!=null?`${(d.sensitivity.pct_scenarios_above_price*100).toFixed(0)}% of scenarios above current price`:''}</div>
            <table style={{width:'100%',borderCollapse:'collapse',fontSize:10}}>
              <thead><tr><th style={{color:'var(--cocoa)',padding:3,textAlign:'left'}}>WACC↓TG→</th>
                {d.sensitivity.terminal_growths.map(t=><th key={t} style={{color:'var(--cocoa)',padding:3}}>{(t*100).toFixed(1)}%</th>)}</tr></thead>
              <tbody>{d.sensitivity.grid.map((row,i)=>(
                <tr key={i}><td style={{color:'var(--cocoa-dust)',padding:3}}>{(d.sensitivity!.waccs[i]*100).toFixed(0)}%</td>
                  {row.map((cell,j)=>(
                    <td key={j} style={{padding:3,textAlign:'center',fontWeight:600,color:'var(--cream)',background:cell==null?'var(--surface-1)':cell>=price?'var(--bull)':'var(--bear)',borderRadius:3}}>
                      {cell==null?'—':cell.toFixed(0)}</td>
                  ))}</tr>
              ))}</tbody>
            </table>
          </div>
        )}
        <div style={{background:'var(--surface-2)',border:'1px solid #2a2a2a',borderRadius:12,padding:16}}>
          <div style={{fontSize:12,color:'var(--cocoa-dust)',letterSpacing:1,marginBottom:10}}>KEY MULTIPLES & SIGNALS</div>
          {[['P/E',fmtX(km.mult_pe)],['EV/EBITDA',fmtX(km.mult_ev_ebitda)],
            ['Reverse-DCF implied growth',fmtP(km.reverse_dcf_implied_growth)],
            ['Intrinsic consensus',fmtD(km.intrinsic_consensus)],
            ['P/E vs 1.5yr history',km.pe_vs_history!=null?km.pe_vs_history.toFixed(2)+'×':'—']].map(([k,v])=>(
            <div key={k} style={{display:'flex',justifyContent:'space-between',padding:'6px 0',borderBottom:'1px solid var(--border-1)',fontSize:12}}>
              <span style={{color:'var(--latte)'}}>{k}</span><span style={{color:'var(--gold)',fontWeight:600}}>{v}</span>
            </div>
          ))}
        </div>
      </div>

      {d.driver_waterfall && (
        <div style={{background:'var(--surface-2)',border:'1px solid #2a2a2a',borderRadius:12,padding:16,marginBottom:18}}>
          <div style={{fontSize:12,color:'var(--cocoa-dust)',letterSpacing:1,marginBottom:14}}>VALUATION DRIVER WATERFALL — what builds intrinsic value ($/share)</div>
          <div style={{display:'flex',alignItems:'flex-end',gap:6,height:140,paddingLeft:4}}>
            {[['Revenue',d.driver_waterfall.revenue_growth,'var(--bull)'],
              ['Margins',d.driver_waterfall.margins,'#4bbe8a'],
              ['Buybacks',d.driver_waterfall.buybacks,'#6fae7a'],
              ['Terminal',d.driver_waterfall.terminal_value,'var(--caramel)'],
              ['WACC drag',d.driver_waterfall.wacc_drag,'#c9762f'],
              ['Intrinsic',d.driver_waterfall.intrinsic_value,'var(--gold)']].map(([k,v,c])=>{
              const val=v as number; const maxv=Math.max(...Object.values(d.driver_waterfall!).map(Math.abs));
              const h=Math.abs(val)/maxv*100;
              return (
                <div key={k as string} style={{flex:1,display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'flex-end',height:'100%'}}>
                  <div style={{fontSize:11,color:c as string,fontWeight:600,marginBottom:3}}>{val>=0?'+':''}{val.toFixed(0)}</div>
                  <div style={{width:'80%',height:`${h}%`,background:c as string,borderRadius:'3px 3px 0 0',minHeight:4}}/>
                  <div style={{fontSize:9,color:'var(--cocoa-dust)',marginTop:5,textAlign:'center'}}>{k}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
        <span style={{fontSize:12,color:'var(--cocoa-dust)',letterSpacing:1}}>10 VALUATION CATEGORIES · {d.tree.categories.reduce((a,c)=>a+c.n_signals,0)} SIGNALS</span>
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
                  {cat.signals.map(s=>{
                    const pending=s.status==='needs_source'||s.score==null;
                    const rv=s.raw_value;
                    const fmt=rv==null?'—':Math.abs(rv)>=1000000?'$'+(rv/1e9).toFixed(1)+'B':Math.abs(rv)<1&&Math.abs(rv)>0?(rv*100).toFixed(1)+'%':rv.toFixed(2);
                    return (
                      <div key={s.id} title={s.evidence} style={{display:'flex',alignItems:'center',gap:10,padding:'5px 0',borderBottom:'1px solid var(--border-1)',opacity:pending?0.5:1}}>
                        <span style={{fontSize:12,color:'var(--latte)',flex:1}}>{s.label}</span>
                        <span style={{fontSize:12,color:'var(--cocoa-dust)',width:80,textAlign:'right'}}>{pending?'pending':fmt}</span>
                        <div style={{width:90,height:6,background:'var(--surface-3)',borderRadius:3,overflow:'hidden'}}>
                          {!pending && <div style={{height:'100%',width:`${s.score}%`,background:heat(s.score)}}/>}
                        </div>
                        <span style={{fontSize:11,fontWeight:600,color:pending?'var(--cocoa)':heat(s.score),width:26,textAlign:'right'}}>{pending?'—':s.score!.toFixed(0)}</span>
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
