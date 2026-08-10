import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

/* ── types ─────────────────────────────────────────── */
interface Sig { id:string; label:string; weight:number; status:string; method?:string;
  evidence:string; raw_value:number|null; score:number|null; }
interface Cat { id:string; label:string; weight:number; score:number|null;
  confidence:number; n_signals:number; n_scored:number; signals:Sig[]; }
interface Tree { label:string; weight:number; score:number|null; confidence:number; categories:Cat[]; }
interface FinData { ticker:string; available:boolean; score:number|null; confidence:number;
  weight_in_conviction:number; coverage:{scored:number;total:number};
  market_cap:number|null; wacc_used:number; n_quarters:number; tree:Tree;
  key_metrics:Record<string,number|null>; reason?:string; }
interface Q { period_end:string|null; fiscal:string; revenue:number|null; revenue_yoy_pct:number|null;
  gross_margin_pct:number|null; operating_margin_pct:number|null; net_margin_pct:number|null;
  net_income:number|null; net_income_yoy_pct:number|null; eps_diluted:number|null; eps_yoy_pct:number|null;
  free_cash_flow:number|null; operating_cash_flow:number|null; capex:number|null; }

/* ── palette / format ──────────────────────────────── */
const heat=(s:number|null)=>s==null?'var(--border-2)':s>=70?'var(--gold)':s>=50?'var(--caramel)':s>=30?'#c9762f':'var(--bear)';
const bn=(v:number|null)=>{ if(v==null)return '—'; const a=Math.abs(v);
  if(a>=1e12)return '$'+(v/1e12).toFixed(2)+'T'; if(a>=1e9)return '$'+(v/1e9).toFixed(1)+'B';
  if(a>=1e6)return '$'+(v/1e6).toFixed(0)+'M'; return '$'+v.toFixed(0); };

function fmt(id:string,v:number|null):string{
  if(v==null)return '—';
  const pct=['gross_margin','operating_margin','net_margin','ebitda_margin','fcf_margin','roic','roic_ex_goodwill',
    'roic_wacc_spread','roe','roa','revenue_growth','fcf_growth','revenue_cagr_3y','earnings_cagr_3y','rd_intensity',
    'cogs_ratio','capex_intensity','shareholder_yield','buyback_yield','dividend_yield','owner_earnings_yield',
    'effective_tax_rate','goodwill_ratio','equity_ratio','reinvestment_rate','accruals_ratio','deferred_rev_to_revenue',
    'deferred_rev_growth','revenue_cagr_5y','earnings_cagr_5y','retained_earnings_ratio','net_margin_stability',
    'gross_margin_stability','operating_margin_stability','roic_stability','fcf_stability','revenue_stability','cash_earnings_gap'];
  const days=['dso','dio','dpo','cash_conversion_cycle'];
  const ratio=['current_ratio','quick_ratio','cash_ratio','debt_to_equity','debt_to_ebitda','net_debt_to_ebitda',
    'asset_turnover','asset_turnover_eff','equity_multiplier','fcf_conversion','ocf_to_net_income','earnings_quality',
    'dividend_coverage','sbc_dilution_ratio','ev_ebitda','ev_revenue','price_to_book','adj_debt_to_ebitda'];
  if(pct.includes(id))return (v*100).toFixed(1)+'%';
  if(days.includes(id))return v.toFixed(0)+'d';
  if(ratio.includes(id))return v.toFixed(2)+'×';
  if(id==='piotroski_f')return v.toFixed(0)+'/9';
  if(id==='altman_z'||id==='beneish_m')return v.toFixed(2);
  if(id==='share_count_trend')return (v*100).toFixed(2)+'%/q';
  if(id==='interest_coverage')return v.toFixed(1)+'×';
  if(['net_debt','enterprise_value'].includes(id))return bn(v);
  if(['book_value_per_share','tangible_bvps'].includes(id))return '$'+v.toFixed(2);
  return v.toFixed(2);
}

/* ── mini gauge ────────────────────────────────────── */
const Gauge=({score,size=104}:{score:number|null;size?:number})=>{
  const s=score??0; const r=size*0.37, circ=2*Math.PI*r, dash=(s/100)*circ*0.75, col=heat(score);
  const c=size/2;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={c} cy={c} r={r} fill="none" stroke="var(--surface-3)" strokeWidth="8"
        strokeDasharray={`${circ*0.75} ${circ}`} transform={`rotate(135 ${c} ${c})`} strokeLinecap="round"/>
      <circle cx={c} cy={c} r={r} fill="none" stroke={col} strokeWidth="8"
        strokeDasharray={`${dash} ${circ}`} transform={`rotate(135 ${c} ${c})`} strokeLinecap="round"/>
      <text x={c} y={c-2} textAnchor="middle" fontSize={size*0.30} fontWeight="500" fill="var(--latte)">{score?.toFixed(0)??'—'}</text>
      <text x={c} y={c+size*0.16} textAnchor="middle" fontSize={size*0.09} fill="var(--cocoa)" letterSpacing="1">/ 100</text>
    </svg>
  );
};

export default function FinancialIntelligencePanel({ ticker }:{ ticker:string }){
  const [data,setData]=useState<FinData|null>(null);
  const [quarters,setQuarters]=useState<Q[]|null>(null);
  const [loading,setLoading]=useState(false); const [err,setErr]=useState('');
  const [tab,setTab]=useState<'quality'|'detail'>('quality');
  const [tip,setTip]=useState('');

  useEffect(()=>{ if(!ticker)return; setLoading(true);setErr('');setData(null);setQuarters(null);
    api.get(`/api/v6/financial/${ticker}`).then(r=>{const d=r.data?.data;if(!d?.available)setErr(d?.reason||'No data');else setData(d);})
      .catch(e=>setErr(e?.message||'Request failed')).finally(()=>setLoading(false));
    api.get(`/api/v6/quarters/${ticker}`).then(r=>{const x=r.data?.data;if(x?.available)setQuarters(x.quarters);}).catch(()=>{});
  },[ticker]);

  if(!ticker)return <div style={{color:'var(--cocoa-dust)',padding:24}}>Enter a ticker for Financial Intelligence.</div>;
  if(loading)return <div style={{color:'var(--gold)',padding:24}}>Computing Financial Intelligence — Polygon + SEC EDGAR, six quality models…</div>;
  if(err)return <div style={{color:'var(--bear)',padding:24}}>Financial Intelligence: {err}</div>;
  if(!data)return null;

  const km=data.key_metrics;
  const cats=[...data.tree.categories];
  const scored=cats.filter(c=>c.score!=null).sort((a,b)=>(b.score!)-(a.score!));

  /* master models */
  const piotroski=km.piotroski_f, altman=km.altman_z, beneish=km.beneish_m, spread=km.roic_wacc_spread;
  const roe=(cats.find(c=>c.id==='profitability')?.signals.find(s=>s.id==='roe')?.raw_value)??null;
  const nm=km['net_margin']??(cats.find(c=>c.id==='income_statement')?.signals.find(s=>s.id==='net_margin')?.raw_value)??null;
  const at=(cats.find(c=>c.id==='balance_sheet')?.signals.find(s=>s.id==='asset_turnover')?.raw_value)??null;
  const em=(cats.find(c=>c.id==='efficiency')?.signals.find(s=>s.id==='equity_multiplier')?.raw_value)??null;

  /* reference valuation cross-refs */
  const refs=cats.flatMap(c=>c.signals).filter(s=>s.status==='reference'&&s.raw_value!=null);

  /* quarters chart geometry */
  const qs=quarters?[...quarters]:[];
  const revs=qs.map(q=>q.revenue||0); const maxRev=Math.max(...revs,1);
  const allM=qs.flatMap(q=>[q.gross_margin_pct,q.operating_margin_pct,q.net_margin_pct].filter(v=>v!=null) as number[]);
  const mLo=Math.min(...allM,0), mHi=Math.max(...allM,1);
  const CW=760, CH=200, PADL=8, PADR=8, PADB=28, PADT=10;
  const bx=(i:number)=>PADL+(i+0.5)*((CW-PADL-PADR)/qs.length);
  const my=(v:number)=>PADT+(1-(v-mLo)/((mHi-mLo)||1))*(CH-PADT-PADB);
  const mLine=(key:'gross_margin_pct'|'operating_margin_pct'|'net_margin_pct')=>
    qs.map((q,i)=>q[key]==null?null:`${bx(i)},${my(q[key] as number)}`).filter(Boolean).join(' ');

  /* piotroski components (from the deepest signal if present) */
  const pioSig=cats.find(c=>c.id==='financial_health')?.signals.find(s=>s.id==='piotroski_f');

  return (
    <div style={{padding:'8px 4px',color:'var(--latte)'}}>

      {/* ═══ HEADER: score + context ═══ */}
      <div style={{display:'flex',alignItems:'center',gap:22,marginBottom:16,flexWrap:'wrap'}}>
        <Gauge score={data.score}/>
        <div style={{minWidth:220}}>
          <div style={{fontSize:22,fontWeight:500,color:heat(data.score)}}>Financial Intelligence</div>
          <div style={{fontSize:12,color:'var(--cocoa)',marginTop:4,lineHeight:1.7}}>
            <div>{data.coverage.scored}/{data.coverage.total} signals live · 18% of conviction · conf {(data.confidence*100).toFixed(0)}%</div>
            <div>Market cap {bn(data.market_cap)} · WACC {(data.wacc_used*100).toFixed(1)}% · {data.n_quarters} quarters</div>
          </div>
        </div>
      </div>

      {/* ═══ MASTER MODELS — the quant-desk quality verdict ═══ */}
      <div style={{fontSize:10,color:'var(--gold)',letterSpacing:2,margin:'6px 0 10px'}}>QUALITY MODELS · academic fundamentals</div>
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(215px,1fr))',gap:10,marginBottom:16}}>

        {/* Piotroski */}
        <div style={{background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:12,padding:'14px 16px'}}>
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'baseline'}}>
            <span style={{fontSize:12,color:'var(--cocoa-dust)'}}>Piotroski F-score</span>
            <span style={{fontSize:22,fontWeight:600,color:(piotroski??0)>=7?'var(--bull)':(piotroski??0)>=4?'var(--neutral)':'var(--bear)'}}>{piotroski?.toFixed(0)??'—'}<span style={{fontSize:12,color:'var(--cocoa)'}}>/9</span></span>
          </div>
          <div style={{display:'flex',gap:3,marginTop:10}}>
            {Array.from({length:9}).map((_,i)=>(
              <div key={i} style={{flex:1,height:6,borderRadius:2,background:i<(piotroski??0)?'var(--bull)':'var(--surface-3)'}}/>
            ))}
          </div>
          <div style={{fontSize:10,color:'var(--cocoa)',marginTop:8,lineHeight:1.4}}>{(piotroski??0)>=7?'Strong — profitable, improving, low-debt':(piotroski??0)>=4?'Moderate fundamental health':'Weak fundamentals'}</div>
        </div>

        {/* Altman Z */}
        <div style={{background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:12,padding:'14px 16px'}}>
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'baseline'}}>
            <span style={{fontSize:12,color:'var(--cocoa-dust)'}}>Altman Z-score</span>
            <span style={{fontSize:22,fontWeight:600,color:(altman??0)>=3?'var(--bull)':(altman??0)>=1.8?'var(--neutral)':'var(--bear)'}}>{altman?.toFixed(2)??'—'}</span>
          </div>
          <div style={{position:'relative',height:6,borderRadius:3,marginTop:12,background:'linear-gradient(90deg,var(--bear) 0%,var(--bear) 25%,var(--neutral) 25%,var(--neutral) 42%,var(--bull) 42%,var(--bull) 100%)'}}>
            <div style={{position:'absolute',top:-3,left:`${Math.min(100,(altman??0)/7*100)}%`,width:3,height:12,background:'var(--latte)',borderRadius:1,transform:'translateX(-1px)'}}/>
          </div>
          <div style={{display:'flex',justifyContent:'space-between',fontSize:9,color:'var(--cocoa)',marginTop:4}}>
            <span>distress 1.8</span><span>safe 3.0</span>
          </div>
          <div style={{fontSize:10,color:'var(--cocoa)',marginTop:6}}>{(altman??0)>=3?'Fortress balance sheet':(altman??0)>=1.8?'Grey zone':'Distress risk'}</div>
        </div>

        {/* Beneish M */}
        <div style={{background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:12,padding:'14px 16px'}}>
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'baseline'}}>
            <span style={{fontSize:12,color:'var(--cocoa-dust)'}}>Beneish M-score</span>
            <span style={{fontSize:22,fontWeight:600,color:(beneish??0)<-1.78?'var(--bull)':'var(--bear)'}}>{beneish?.toFixed(2)??'—'}</span>
          </div>
          <div style={{fontSize:11,marginTop:12,padding:'6px 10px',borderRadius:6,display:'inline-block',
            background:(beneish??0)<-1.78?'rgba(34,197,94,0.12)':'rgba(239,68,68,0.12)',
            color:(beneish??0)<-1.78?'var(--bull)':'var(--bear)'}}>
            {(beneish??0)<-1.78?'✓ Clean books':'⚠ Worth scrutiny'}</div>
          <div style={{fontSize:10,color:'var(--cocoa)',marginTop:8,lineHeight:1.4}}>Earnings-manipulation detector · threshold −1.78</div>
        </div>

        {/* ROIC-WACC value creation */}
        <div style={{background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:12,padding:'14px 16px'}}>
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'baseline'}}>
            <span style={{fontSize:12,color:'var(--cocoa-dust)'}}>ROIC − WACC</span>
            <span style={{fontSize:22,fontWeight:600,color:(spread??0)>0?'var(--bull)':'var(--bear)'}}>{spread!=null?(spread>=0?'+':'')+(spread*100).toFixed(1)+'%':'—'}</span>
          </div>
          <div style={{fontSize:11,marginTop:12,color:(spread??0)>0?'var(--bull)':'var(--bear)'}}>
            {(spread??0)>0?'Creating value':'Destroying value'}</div>
          <div style={{fontSize:10,color:'var(--cocoa)',marginTop:8,lineHeight:1.4}}>
            Return on capital {spread!=null&&spread>0?`${(spread*100).toFixed(1)} pts above`:'below'} cost of capital ({(data.wacc_used*100).toFixed(1)}%)</div>
        </div>
      </div>

      {/* ═══ DUPONT — ROE decomposition ═══ */}
      {roe!=null&&nm!=null&&at!=null&&em!=null&&(
        <div style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderRadius:12,padding:'14px 18px',marginBottom:16}}>
          <div style={{fontSize:10,color:'var(--gold)',letterSpacing:2,marginBottom:12}}>DUPONT · what drives return on equity</div>
          <div style={{display:'flex',alignItems:'center',gap:12,flexWrap:'wrap'}}>
            <div style={{textAlign:'center'}}><div style={{fontSize:24,fontWeight:600,color:'var(--gold)'}}>{(roe*100).toFixed(1)}%</div><div style={{fontSize:10,color:'var(--cocoa)'}}>ROE</div></div>
            <span style={{fontSize:18,color:'var(--cocoa)'}}>=</span>
            <div style={{textAlign:'center'}}><div style={{fontSize:18,fontWeight:500,color:'var(--latte)'}}>{(nm*100).toFixed(1)}%</div><div style={{fontSize:10,color:'var(--cocoa)'}}>net margin</div></div>
            <span style={{fontSize:18,color:'var(--cocoa)'}}>×</span>
            <div style={{textAlign:'center'}}><div style={{fontSize:18,fontWeight:500,color:'var(--latte)'}}>{at.toFixed(2)}×</div><div style={{fontSize:10,color:'var(--cocoa)'}}>asset turnover</div></div>
            <span style={{fontSize:18,color:'var(--cocoa)'}}>×</span>
            <div style={{textAlign:'center'}}><div style={{fontSize:18,fontWeight:500,color:'var(--latte)'}}>{em.toFixed(2)}×</div><div style={{fontSize:10,color:'var(--cocoa)'}}>leverage</div></div>
          </div>
          <div style={{fontSize:10,color:'var(--cocoa)',marginTop:10,lineHeight:1.4}}>
            {nm>0.2?'Return driven by exceptional margins':at>1?'Return driven by asset efficiency':'Return leans on leverage'} — the highest-quality ROE comes from margin, not debt.</div>
        </div>
      )}

      {/* ═══ EARNINGS TRAJECTORY chart ═══ */}
      {qs.length>=4&&(
        <div style={{marginBottom:16}}>
          <div style={{fontSize:10,color:'var(--gold)',letterSpacing:2,marginBottom:8}}>EARNINGS TRAJECTORY · {qs.length} quarters · revenue bars + margin lines</div>
          <div style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderRadius:12,padding:'12px 10px 6px',overflowX:'auto'}}>
            <svg width={CW} height={CH} style={{minWidth:CW,display:'block'}}>
              {qs.map((q,i)=>{ const bh=(q.revenue||0)/maxRev*(CH-PADT-PADB); const bw=(CW-PADL-PADR)/qs.length*0.56;
                return <rect key={i} x={bx(i)-bw/2} y={CH-PADB-bh} width={bw} height={bh} rx={2} fill="var(--surface-4)"/>; })}
              <polyline points={mLine('gross_margin_pct')} fill="none" stroke="var(--gold)" strokeWidth="1.8"/>
              <polyline points={mLine('operating_margin_pct')} fill="none" stroke="var(--caramel)" strokeWidth="1.8"/>
              <polyline points={mLine('net_margin_pct')} fill="none" stroke="var(--bull)" strokeWidth="1.8"/>
              {qs.map((q,i)=>(<text key={i} x={bx(i)} y={CH-10} textAnchor="middle" fontSize="8" fill="var(--cocoa)">{q.fiscal?.replace(' ','')}</text>))}
            </svg>
            <div style={{display:'flex',gap:16,padding:'4px 8px 2px',fontSize:11}}>
              <span style={{color:'var(--surface-4)'}}>▮ <span style={{color:'var(--cocoa-dust)'}}>Revenue</span></span>
              <span style={{color:'var(--gold)'}}>— <span style={{color:'var(--cocoa-dust)'}}>Gross margin</span></span>
              <span style={{color:'var(--caramel)'}}>— <span style={{color:'var(--cocoa-dust)'}}>Operating</span></span>
              <span style={{color:'var(--bull)'}}>— <span style={{color:'var(--cocoa-dust)'}}>Net</span></span>
            </div>
          </div>
        </div>
      )}

      {/* ═══ VALUATION cross-refs (reference) ═══ */}
      {refs.length>0&&(
        <div style={{display:'flex',gap:10,flexWrap:'wrap',marginBottom:16}}>
          {refs.map(s=>(
            <div key={s.id} style={{background:'var(--surface-2)',border:'1px dashed var(--border-2)',borderRadius:8,padding:'8px 12px'}}>
              <div style={{fontSize:10,color:'var(--cocoa)'}}>{s.label} <span style={{fontSize:8,color:'var(--cocoa-dust)',border:'1px solid var(--border-2)',borderRadius:3,padding:'0 3px'}}>REF</span></div>
              <div style={{fontSize:15,fontWeight:500,color:'var(--latte)',marginTop:2}}>{fmt(s.id,s.raw_value)}</div>
            </div>
          ))}
          <div style={{alignSelf:'center',fontSize:10,color:'var(--cocoa)',fontStyle:'italic'}}>scored in Valuation — shown here for context</div>
        </div>
      )}

      {/* ═══ QUALITY PROFILE · 12 categories ranked ═══ */}
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',margin:'18px 0 10px'}}>
        <span style={{fontSize:10,color:'var(--gold)',letterSpacing:2}}>QUALITY PROFILE · 12 dimensions ranked</span>
        <button onClick={()=>setTab(t=>t==='quality'?'detail':'quality')}
          style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',color:'var(--cocoa-dust)',borderRadius:8,padding:'4px 12px',fontSize:11,cursor:'pointer'}}>
          {tab==='quality'?'Show all signals':'Show profile'}</button>
      </div>

      {tab==='quality'&&(
        <>
          <div style={{fontSize:10,color:'var(--cocoa)',marginBottom:10,minHeight:14}}>{tip||'\u00A0'}</div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(230px,1fr))',gap:10}}>
            {scored.map(c=>{
              const top=[...c.signals].filter(s=>s.score!=null).sort((a,b)=>(b.score!)-(a.score!))[0];
              const weak=[...c.signals].filter(s=>s.score!=null).sort((a,b)=>(a.score!)-(b.score!))[0];
              return (
                <div key={c.id} onMouseEnter={()=>setTip(`${c.label.replace(' Intelligence','')} — ${c.n_scored}/${c.n_signals} signals · weight ${c.weight.toFixed(2)}`)}
                  onMouseLeave={()=>setTip('')}
                  style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderLeft:`3px solid ${heat(c.score)}`,borderRadius:'0 10px 10px 0',padding:'12px 14px'}}>
                  <div style={{display:'flex',justifyContent:'space-between',alignItems:'baseline'}}>
                    <span style={{fontSize:13,fontWeight:600,color:'var(--latte)'}}>{c.label.replace(' Intelligence','')}</span>
                    <span style={{fontSize:20,fontWeight:500,color:heat(c.score)}}>{c.score?.toFixed(0)}</span>
                  </div>
                  <div style={{height:4,background:'var(--surface-3)',borderRadius:2,margin:'8px 0',overflow:'hidden'}}>
                    <div style={{height:'100%',width:`${c.score}%`,background:heat(c.score)}}/>
                  </div>
                  {top&&<div style={{fontSize:10,color:'var(--cocoa)'}}>▲ {top.label} {fmt(top.id,top.raw_value)}</div>}
                  {weak&&weak.id!==top?.id&&<div style={{fontSize:10,color:'var(--cocoa)'}}>▼ {weak.label} {fmt(weak.id,weak.raw_value)}</div>}
                </div>
              );
            })}
          </div>
        </>
      )}

      {tab==='detail'&&(
        <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(300px,1fr))',gap:10,alignItems:'start'}}>
          {cats.map(cat=>(
            <div key={cat.id} style={{borderRadius:10,overflow:'hidden',border:'1px solid var(--border-1)'}}>
              <div style={{background:'var(--surface-3)',borderLeft:`3px solid ${heat(cat.score)}`,padding:'10px 14px'}}>
                <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
                  <span style={{fontSize:12,fontWeight:600,color:'var(--latte)'}}>{cat.label.replace(' Intelligence','')}</span>
                  <span style={{fontSize:16,fontWeight:600,color:heat(cat.score)}}>{cat.score?.toFixed(0)??'—'}</span>
                </div>
                <div style={{fontSize:10,color:'var(--cocoa)',marginTop:2}}>wt {cat.weight.toFixed(2)} · {cat.n_scored}/{cat.n_signals} live</div>
              </div>
              <div style={{background:'var(--surface-1)',padding:10}}>
                {cat.signals.map(s=>{
                  const isRef=s.status==='reference'; const isPending=s.status==='needs_source';
                  return (
                    <div key={s.id} style={{marginBottom:9,opacity:isPending?0.5:1}}>
                      <div style={{display:'flex',justifyContent:'space-between',fontSize:12}}>
                        <span style={{color:'var(--cocoa-dust)'}}>{s.label}
                          {isRef&&<span style={{fontSize:8,color:'var(--cocoa)',marginLeft:5,border:'1px solid var(--border-2)',borderRadius:3,padding:'0 3px'}}>REF</span>}
                          {isPending&&<span style={{fontSize:8,color:'var(--neutral)',marginLeft:5,border:'1px solid var(--border-2)',borderRadius:3,padding:'0 3px'}}>SOON</span>}
                        </span>
                        <span style={{color:isRef?'var(--cocoa-dust)':heat(s.score),fontWeight:600}}>
                          {isPending?'pending':fmt(s.id,s.raw_value)}{!isRef&&!isPending?' · '+(s.score?.toFixed(0)??'—'):''}</span>
                      </div>
                      {!isRef&&!isPending&&(
                        <div style={{height:4,background:'var(--surface-3)',borderRadius:2,marginTop:3,overflow:'hidden'}}>
                          <div style={{height:'100%',width:`${s.score??0}%`,background:heat(s.score)}}/>
                        </div>)}
                      <div style={{fontSize:9,color:'var(--cocoa)',marginTop:2}}>{s.evidence}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
