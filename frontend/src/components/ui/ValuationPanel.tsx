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
  reason?:string;
}

const heat=(s:number|null)=>s==null?'#2a2a2a':s>=75?'#0f6e56':s>=58?'#1d9e75':s>=42?'#8a7519':s>=25?'#a35a1d':'#7a2320';
const ratingColor=(r:string)=>r.includes('Undervalued')?'#1d9e75':r.includes('Overvalued')?'#c0705a':'#c9a227';
const METHOD_LABELS:Record<string,string>={dcf_bear:'DCF Bear',dcf_base:'DCF Base',dcf_bull:'DCF Bull',
  epv:'EPV',graham:'Graham',residual_income:'Residual Income',ddm:'DDM',nav:'NAV'};

export default function ValuationPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<ValData|null>(null);
  const [loading,setLoading]=useState(false);
  const [err,setErr]=useState('');
  useEffect(()=>{
    if(!ticker)return;
    setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/valuation/${ticker}`)
      .then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No valuation data');else setD(x);})
      .catch(e=>setErr(e?.message||'Request failed'))
      .finally(()=>setLoading(false));
  },[ticker]);

  if(!ticker)return <div style={{color:'#9d8b7a',padding:24}}>Enter a ticker for Valuation Intelligence.</div>;
  if(loading)return <div style={{color:'#daa520',padding:24}}>Computing Valuation — DCF, EPV, Graham, residual income, DDM, reverse-DCF, real CAPM beta…</div>;
  if(err)return <div style={{color:'#c0705a',padding:24}}>Valuation: {err}</div>;
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
    <div style={{padding:'8px 4px',color:'#e8ddd0'}}>
      <div style={{display:'flex',alignItems:'center',gap:24,marginBottom:20,flexWrap:'wrap'}}>
        <div style={{display:'flex',alignItems:'baseline',gap:10}}>
          <span style={{fontSize:46,fontWeight:700,color:heat(d.score),lineHeight:1}}>{d.score?.toFixed(1)??'—'}</span>
          <span style={{fontSize:16,color:'#9d8b7a'}}>/100</span>
        </div>
        <div>
          <div style={{fontSize:20,fontWeight:700,color:ratingColor(d.valuation_rating),letterSpacing:1}}>{d.valuation_rating}</div>
          <div style={{fontSize:11,color:'#9d8b7a',marginTop:2}}>
            Coverage {d.coverage.scored}/{d.coverage.total} · β {d.beta_used?.toFixed(2)??'—'} · WACC {(d.wacc_used*100).toFixed(1)}%</div>
        </div>
        <div style={{marginLeft:'auto',display:'flex',gap:10,flexWrap:'wrap'}}>
          {[['Fair Value',fmtD(km.fair_value)],['Margin of Safety',fmtP(km.margin_of_safety)],
            ['Buy Zone',fmtD(km.buy_zone)],['Sell Zone',fmtD(km.sell_zone)]].map(([k,v])=>(
            <div key={k} style={{background:'#181818',border:'1px solid #2a2a2a',borderRadius:8,padding:'8px 12px',minWidth:90}}>
              <div style={{fontSize:10,color:'#9d8b7a'}}>{k}</div>
              <div style={{fontSize:16,fontWeight:600,color:'#daa520'}}>{v}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:12,padding:'18px 20px 18px 130px',marginBottom:18}}>
        <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:22,marginLeft:-110}}>VALUATION RANGE — method fair values vs current price ${price.toFixed(2)}</div>
        <div style={{position:'relative'}}>
          <div style={{position:'absolute',left:`${Math.max(0,Math.min(100,pos(price)))}%`,top:-6,bottom:16,width:2,background:'#c0392b',zIndex:2}}>
            <div style={{position:'absolute',top:-16,left:'50%',transform:'translateX(-50%)',whiteSpace:'nowrap',fontSize:10,color:'#e07a6a',fontWeight:600}}>${price.toFixed(0)}</div>
          </div>
          {methodEntries.map(([k,v])=>{
            const cheap=v>=price;
            return (
              <div key={k} style={{position:'relative',marginBottom:11,height:20}}>
                <div style={{position:'absolute',left:-118,width:110,color:'#cdbfae',textAlign:'right',fontSize:12,top:2}}>{METHOD_LABELS[k]||k}</div>
                <div style={{position:'absolute',left:0,right:0,top:9,height:2,background:'#242424'}}/>
                <div style={{position:'absolute',left:`${Math.max(0,Math.min(100,pos(v)))}%`,transform:'translateX(-50%)',top:0,
                  background:cheap?'#1d9e75':'#a35a1d',color:'#fff',borderRadius:4,padding:'1px 7px',fontSize:11,fontWeight:600,whiteSpace:'nowrap',zIndex:1}}>${v.toFixed(0)}</div>
              </div>
            );
          })}
        </div>
        <div style={{marginTop:14,fontSize:11,color:'#7a7266',marginLeft:-110}}>
          <span style={{color:'#1d9e75'}}>green</span> = above price (upside) · <span style={{color:'#a35a1d'}}>amber</span> = below price · <span style={{color:'#c0392b'}}>red line</span> = current price</div>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:14,marginBottom:18}}>
        {d.sensitivity?.grid && (
          <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:12,padding:16}}>
            <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:4}}>DCF SENSITIVITY — fair value $/share</div>
            <div style={{fontSize:10,color:'#7a7266',marginBottom:10}}>
              {d.sensitivity.pct_scenarios_above_price!=null?`${(d.sensitivity.pct_scenarios_above_price*100).toFixed(0)}% of scenarios above current price`:''}</div>
            <table style={{width:'100%',borderCollapse:'collapse',fontSize:10}}>
              <thead><tr><th style={{color:'#7a7266',padding:3,textAlign:'left'}}>WACC↓TG→</th>
                {d.sensitivity.terminal_growths.map(t=><th key={t} style={{color:'#7a7266',padding:3}}>{(t*100).toFixed(1)}%</th>)}</tr></thead>
              <tbody>{d.sensitivity.grid.map((row,i)=>(
                <tr key={i}><td style={{color:'#9d8b7a',padding:3}}>{(d.sensitivity!.waccs[i]*100).toFixed(0)}%</td>
                  {row.map((cell,j)=>(
                    <td key={j} style={{padding:3,textAlign:'center',fontWeight:600,color:'#fff',background:cell==null?'#1a1a1a':cell>=price?'#0f6e56':'#7a2320',borderRadius:3}}>
                      {cell==null?'—':cell.toFixed(0)}</td>
                  ))}</tr>
              ))}</tbody>
            </table>
          </div>
        )}
        <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:12,padding:16}}>
          <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:10}}>KEY MULTIPLES & SIGNALS</div>
          {[['P/E',fmtX(km.mult_pe)],['EV/EBITDA',fmtX(km.mult_ev_ebitda)],
            ['Reverse-DCF implied growth',fmtP(km.reverse_dcf_implied_growth)],
            ['Intrinsic consensus',fmtD(km.intrinsic_consensus)],
            ['P/E vs 1.5yr history',km.pe_vs_history!=null?km.pe_vs_history.toFixed(2)+'×':'—']].map(([k,v])=>(
            <div key={k} style={{display:'flex',justifyContent:'space-between',padding:'6px 0',borderBottom:'1px solid #222',fontSize:12}}>
              <span style={{color:'#cdbfae'}}>{k}</span><span style={{color:'#daa520',fontWeight:600}}>{v}</span>
            </div>
          ))}
        </div>
      </div>

      <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:8}}>10 VALUATION CATEGORIES</div>
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(150px,1fr))',gap:8}}>
        {d.tree.categories.map(cat=>(
          <div key={cat.id} style={{background:heat(cat.score),borderRadius:8,padding:'8px 12px'}}>
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
              <span style={{fontSize:11,fontWeight:600,color:'#fff',opacity:0.95}}>{cat.label.replace(' Intelligence','')}</span>
              <span style={{fontSize:15,fontWeight:700,color:'#fff'}}>{cat.score?.toFixed(0)??'—'}</span>
            </div>
            <div style={{fontSize:9,color:'#fff',opacity:0.6,marginTop:2}}>wt {cat.weight.toFixed(2)} · {cat.n_scored}/{cat.n_signals}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
