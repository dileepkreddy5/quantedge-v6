import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; }
interface Cat { id:string; label:string; weight:number; score:number|null; n_signals:number; n_scored:number; signals:Sig[]; }
interface BriefItem { headline:string; sentiment:string; publisher:string; date:string; url:string; }
interface Headline { title:string; sentiment:string; reason:string; publisher:string; date:string; url:string; materiality?:number; materiality_why?:string[]; about_company?:boolean; kind?:string; event_kind?:string; source_weight?:number; }
interface NewsData {
  ticker:string; available:boolean; score:number|null; news_rating:string; confidence:number;
  coverage:{scored:number;total:number}; article_count:number;
  sentiment_dist:{positive:number;neutral:number;negative:number};
  brief:BriefItem[]; recent_headlines:Headline[]; key_facts?:{fact:string;group:string;source:string;date:string;url:string;headline:string;weight:number}[]; tree:{categories:Cat[]}; key_metrics:Record<string,number|null>; reason?:string;
}
const heat=(s:number|null)=>s==null?'rgba(212,149,108,0.12)':s>=75?'#0f6e56':s>=58?'#1d9e75':s>=42?'#8a7519':s>=25?'#a35a1d':'#7a2320';
const ratingColor=(r:string)=>r.includes('Very Positive')?'#0f9d6e':r.includes('Positive')?'#1d9e75':r.includes('Mixed')?'#c9a227':r.includes('Negative')?'#c0705a':'#7a2320';
const sentDot=(s:string)=>s==='positive'?'#0f9d6e':s==='negative'?'#c0705a':'#9d8b7a';

const fmtVal=(id:string,v:number|null):string=>{
  if(v==null) return '—';
  if(id.includes('count')||id.includes('publishers')||id.includes('topics')||id.includes('diversity')||id.includes('reasoning')||id.includes('flag')||id.includes('spike')) return Number.isInteger(v)?v.toString():v.toFixed(1);
  if(id.includes('hours')) return v.toFixed(0)+'h';
  if(id.includes('ratio')||id.includes('velocity')||id.includes('balance')||id.includes('severity')||id.includes('sentiment')||id.includes('mean')) return v.toFixed(2);
  return (v*100).toFixed(0)+'%';
};

export default function NewsPanel({ ticker, data }:{ ticker:string; data?:any }){
  const [d,setD]=useState<NewsData|null>(null);
  const [loading,setLoading]=useState(false);
  const [err,setErr]=useState('');
  const [expanded,setExpanded]=useState<Record<string,boolean>>({});
  const [showAllSignals,setShowAllSignals]=useState(false);

  useEffect(()=>{
    if(!ticker)return;
    setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/news/${ticker}`)
      .then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No news coverage');else setD(x);})
      .catch(e=>setErr(e?.message||'Request failed'))
      .finally(()=>setLoading(false));
  },[ticker]);

  if(!ticker)return <div style={{color:'#9d8b7a',padding:24}}>Enter a ticker for News Intelligence.</div>;
  if(loading)return <div style={{color:'#daa520',padding:24}}>Analyzing news — sentiment, events, and 47 signals across recent coverage…</div>;
  if(err)return <div style={{color:'#c0705a',padding:24}}>News: {err}</div>;
  if(!d)return null;

  const sd=d.sentiment_dist||{positive:0,neutral:0,negative:0};
  const total=sd.positive+sd.neutral+sd.negative||1;
  const km=d.key_metrics||{};

  // Compose a verdict from the numbers rather than a template.
  const net = km.net_sentiment ?? 0;
  const px30 = km.price_return_30d ?? null;
  const diverge = px30 != null && ((net > 0.1 && px30 < -0.02) || (net < -0.1 && px30 > 0.02));
  const trend = km.sentiment_trend ?? 0;
  const top10 = km.top10_sentiment ?? null;
  const toneOf = (v:number) => v > 0.15 ? 'positive' : v < -0.15 ? 'negative' : 'mixed';
  const verdict = (() => {
    const tone = toneOf(net);
    const dir  = trend > 0.03 ? 'improving' : trend < -0.03 ? 'deteriorating' : 'stable';
    const parts: string[] = [];
    parts.push(`Coverage is broadly ${tone} and ${dir} — ${sd.positive} positive against ${sd.negative} negative across ${d.article_count} articles.`);
    // Does the high-impact subset disagree with the average?
    if (top10 != null && Math.abs(top10 - net) > 0.12) {
      const t10 = toneOf(top10);
      parts.push(t10 === tone
        ? `The highest-impact stories lean the same way but harder (${top10 > 0 ? '+' : ''}${top10.toFixed(2)} against ${net > 0 ? '+' : ''}${net.toFixed(2)} overall).`
        : `The stories most likely to move the price read ${t10}, against ${tone} on average — the aggregate is being carried by lower-impact coverage.`);
    }
    if (px30 != null) {
      parts.push(diverge
        ? `Price has moved the other way, ${px30 > 0 ? 'up' : 'down'} ${Math.abs(px30*100).toFixed(1)}% over 30 days.`
        : `Price is ${px30 >= 0 ? 'up' : 'down'} ${Math.abs(px30*100).toFixed(1)}% over 30 days, in line.`);
    }
    return parts.join(' ');
  })();

  const heads = (d.recent_headlines || []);
  const events = heads.filter(h => h.kind === 'EVENT' || h.kind === 'ANALYST');
  const rest   = heads.filter(h => h.kind !== 'EVENT' && h.kind !== 'ANALYST').slice(0, 12);

  return (
    <div style={{padding:'8px 4px',color:'#e8ddd0'}}>

      <div style={{background:'#241510',border:'1px solid rgba(212,149,108,0.12)',borderRadius:8,padding:'18px 20px',marginBottom:14}}>
        <div style={{display:'flex',gap:26,alignItems:'flex-start',flexWrap:'wrap'}}>
          <div style={{textAlign:'center',minWidth:96}}>
            <div style={{fontFamily:"'Fira Code',monospace",fontSize:44,fontWeight:800,color:heat(d.score),lineHeight:1}}>{d.score?.toFixed(0)??'—'}</div>
            <div style={{fontFamily:"'Fira Code',monospace",fontSize:8,color:'#8a7560',letterSpacing:2,marginTop:4}}>NEWS SCORE</div>
            <div style={{fontFamily:"'Outfit',sans-serif",fontSize:12,color:ratingColor(d.news_rating),marginTop:2}}>{d.news_rating}</div>
          </div>
          <div style={{flex:1,minWidth:300}}>
            <div style={{fontFamily:"'Outfit',sans-serif",fontSize:13.5,color:'#d4c4b0',lineHeight:1.6}}>{verdict}</div>
            <div style={{display:'flex',height:8,borderRadius:4,overflow:'hidden',marginTop:14}}>
              <div style={{width:`${sd.positive/total*100}%`,background:'#22c55e',opacity:0.65}}/>
              <div style={{width:`${sd.neutral/total*100}%`,background:'#8a7560',opacity:0.35}}/>
              <div style={{width:`${sd.negative/total*100}%`,background:'#ef4444',opacity:0.65}}/>
            </div>
            <div style={{display:'flex',justifyContent:'space-between',fontFamily:"'Fira Code',monospace",fontSize:8.5,color:'#6b5d52',marginTop:4}}>
              <span>{sd.positive} positive</span><span>{sd.neutral} neutral</span><span>{sd.negative} negative</span>
            </div>
          </div>
        </div>
        <div style={{display:'grid',gridTemplateColumns:'repeat(5, 1fr)',gap:8,marginTop:16,paddingTop:14,borderTop:'1px solid rgba(212,149,108,0.1)'}}>
          {[
            {l:'NET SENTIMENT', v:net!=null?`${net>0?'+':''}${net.toFixed(2)}`:'—', n:'weighted by relevance'},
            {l:'30-DAY PRICE',  v:px30!=null?`${px30>0?'+':''}${(px30*100).toFixed(1)}%`:'—', n:diverge?'diverging from tone':'in line with tone'},
            {l:'HIGH-IMPACT TONE', v:top10!=null?`${top10>0?'+':''}${top10.toFixed(2)}`:'—', n:'top 10 stories only'},
            {l:'COVERAGE RATE', v:km.news_velocity!=null?`${km.news_velocity.toFixed(1)}/day`:'—', n:`${km.article_count_7d??0} in last 7 days`},
            {l:'SOURCE QUALITY',v:km.tier1_source_share!=null?`${(km.tier1_source_share*100).toFixed(0)}%`:'—', n:'from tier-1 outlets'},
          ].map(x=>(
            <div key={x.l} style={{background:'#1a0f0a',borderRadius:6,padding:'9px 10px'}}>
              <div style={{fontFamily:"'Fira Code',monospace",fontSize:8,color:'#8a7560',letterSpacing:1}}>{x.l}</div>
              <div style={{fontFamily:"'Fira Code',monospace",fontSize:15,fontWeight:700,color:'#d4c4b0',marginTop:3}}>{x.v}</div>
              <div style={{fontFamily:"'Outfit',sans-serif",fontSize:8.5,color:'#6b5d52',marginTop:1}}>{x.n}</div>
            </div>
          ))}
        </div>
        {km.fraud_litigation_flag ? (
          <div style={{marginTop:12,padding:'8px 12px',background:'rgba(239,68,68,0.08)',borderLeft:'2px solid #ef4444',borderRadius:4,
            fontFamily:"'Outfit',sans-serif",fontSize:11,color:'#d4c4b0'}}>
            Litigation or fraud language detected in recent coverage — worth reading the source articles directly.
          </div>
        ) : null}
      </div>

      <div style={{background:'#241510',border:'1px solid rgba(212,149,108,0.12)',borderRadius:8,padding:'16px 20px',marginBottom:14}}>
      {(d.key_facts||[]).length > 0 && (() => {
        const GROUPS: Record<string,{label:string; note:string}> = {
          FINANCIAL:   {label:'FINANCIALS',  note:'margins, earnings, valuation'},
          OPERATIONAL: {label:'OPERATIONS',  note:'volumes, capacity, customers'},
          CAPITAL:     {label:'CAPITAL',     note:'buybacks, dividends, debt'},
          STRATEGIC:   {label:'STRATEGIC',   note:'deals, launches, partnerships'},
        };
        const by: Record<string, any[]> = {};
        (d.key_facts||[]).forEach(k => { (by[k.group] = by[k.group] || []).push(k); });
        const order = ['FINANCIAL','OPERATIONAL','CAPITAL','STRATEGIC'].filter(g => by[g]?.length);
        return (
          <div style={{background:'#241510',border:'1px solid rgba(212,149,108,0.12)',borderRadius:8,padding:'16px 20px',marginBottom:14}}>
            <div style={{fontFamily:"'Fira Code',monospace",fontSize:9,color:'#daa520',letterSpacing:2,marginBottom:4}}>THE NUMBERS THAT MATTER</div>
            <div style={{fontFamily:"'Outfit',sans-serif",fontSize:10.5,color:'#7a6b5d',marginBottom:16}}>
              Specific figures and stated developments pulled from recent coverage, grouped by what they describe.
              Each links to the article it came from.
            </div>
            <div style={{display:'grid',gridTemplateColumns:'repeat(2, 1fr)',gap:'18px 24px'}}>
              {order.map(g => (
                <div key={g}>
                  <div style={{fontFamily:"'Fira Code',monospace",fontSize:8.5,color:'#daa520',letterSpacing:2,marginBottom:2}}>{GROUPS[g].label}</div>
                  <div style={{fontFamily:"'Outfit',sans-serif",fontSize:8.5,color:'#6b5d52',marginBottom:9}}>{GROUPS[g].note}</div>
                  {by[g].slice(0,4).map((k,i) => (
                    <div key={i} style={{marginBottom:11,paddingLeft:10,borderLeft:'2px solid rgba(212,149,108,0.18)'}}>
                      <div style={{fontFamily:"'Outfit',sans-serif",fontSize:11.5,color:'#d4c4b0',lineHeight:1.5}}>{k.fact}</div>
                      <div style={{marginTop:3}}>
                        {k.url
                          ? <a href={k.url} target="_blank" rel="noopener noreferrer"
                              style={{fontFamily:"'Outfit',sans-serif",fontSize:9,color:'#8a7560',textDecoration:'none'}}>
                              {k.source} · {k.date} ↗
                            </a>
                          : <span style={{fontFamily:"'Outfit',sans-serif",fontSize:9,color:'#6b5d52'}}>{k.source} · {k.date}</span>}
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        );
      })()}

      <div style={{background:'#241510',border:'1px solid rgba(212,149,108,0.12)',borderRadius:8,padding:'16px 20px',marginBottom:14}}>
        <div style={{fontFamily:"'Fira Code',monospace",fontSize:9,color:'#daa520',letterSpacing:2,marginBottom:4}}>REPORTED EVENTS</div>
        <div style={{fontFamily:"'Outfit',sans-serif",fontSize:10.5,color:'#7a6b5d',marginBottom:14}}>
          Things that actually happened — results, deals, filings, analyst actions — separated from commentary about them.
          Most financial coverage is opinion; this section is deliberately short, and empty when nothing has been reported.
        </div>
        {events.length === 0 && (
          <div style={{fontFamily:"'Outfit',sans-serif",fontSize:12,color:'#8a7560',padding:'14px 0',lineHeight:1.6}}>
            No reported events in recent coverage. Everything currently written about this company is commentary,
            comparison or speculation — listed below.
          </div>
        )}
        {events.map((h,i)=>(
          <div key={i} style={{paddingBottom:14,marginBottom:14,borderBottom:i<events.length-1?'1px solid rgba(212,149,108,0.08)':'none'}}>
            <div style={{display:'flex',gap:12,alignItems:'flex-start'}}>
              <div style={{minWidth:38,textAlign:'center'}}>
                <div style={{fontFamily:"'Fira Code',monospace",fontSize:16,fontWeight:800,color:heat(h.materiality??50)}}>{h.materiality??'—'}</div>
                <div style={{fontFamily:"'Fira Code',monospace",fontSize:7,color:'#6b5d52',letterSpacing:1}}>IMPACT</div>
              </div>
              <div style={{flex:1}}>
                {h.url
                  ? <a href={h.url} target="_blank" rel="noopener noreferrer" style={{fontFamily:"'Outfit',sans-serif",fontSize:16,fontWeight:600,color:'#e8ddd0',textDecoration:'none',lineHeight:1.35,display:'block'}}>{h.title}</a>
                  : <div style={{fontFamily:"'Outfit',sans-serif",fontSize:16,fontWeight:600,color:'#e8ddd0',lineHeight:1.35}}>{h.title}</div>}
                <div style={{fontFamily:"'Outfit',sans-serif",fontSize:11.5,color:'#9d8b7a',lineHeight:1.5,marginTop:5}}>{h.reason}</div>
                <div style={{display:'flex',gap:10,alignItems:'center',marginTop:7,flexWrap:'wrap'}}>
                  <span style={{fontFamily:"'Fira Code',monospace",fontSize:8,letterSpacing:1,padding:'2px 6px',borderRadius:3,
                    color: h.kind==='EVENT' ? '#22c55e' : '#daa520',
                    border: `1px solid ${h.kind==='EVENT' ? '#22c55e' : '#daa520'}44`,
                    background: `${h.kind==='EVENT' ? '#22c55e' : '#daa520'}12`}}>
                    {h.event_kind || h.kind}
                  </span>
                  <span style={{fontFamily:"'Fira Code',monospace",fontSize:9,color:sentDot(h.sentiment),textTransform:'uppercase',letterSpacing:1}}>{h.sentiment}</span>
                  <span style={{fontFamily:"'Outfit',sans-serif",fontSize:9.5,color:'#6b5d52'}}>{h.publisher} · {h.date}</span>
                  {h.materiality_why?.length ? (
                    <span style={{fontFamily:"'Outfit',sans-serif",fontSize:9,color:'#6b5d52',fontStyle:'italic'}}>ranked for: {h.materiality_why.join('; ')}</span>
                  ) : null}
                </div>
              </div>
            </div>
          </div>
        ))}
        {rest.length > 0 && (
          <div style={{marginTop:16,paddingTop:14,borderTop:'1px solid rgba(212,149,108,0.1)'}}>
            <div style={{fontFamily:"'Fira Code',monospace",fontSize:9,color:'#8a7560',letterSpacing:2,marginBottom:8}}>
              COMMENTARY &amp; ANALYSIS · {rest.length}
            </div>
            {rest.map((h,i)=>(
              <div key={i} style={{display:'grid',gridTemplateColumns:'30px 1fr auto',gap:10,alignItems:'center',padding:'6px 0',
                borderTop:'1px solid rgba(212,149,108,0.06)'}}>
                <span style={{fontFamily:"'Fira Code',monospace",fontSize:10,color:heat(h.materiality??50),textAlign:'right'}}>{h.materiality??'—'}</span>
                {h.url
                  ? <a href={h.url} target="_blank" rel="noopener noreferrer" style={{fontFamily:"'Outfit',sans-serif",fontSize:12,color:'#b8a894',textDecoration:'none'}}>{h.title}</a>
                  : <span style={{fontFamily:"'Outfit',sans-serif",fontSize:12,color:'#b8a894'}}>{h.title}</span>}
                <span style={{fontFamily:"'Fira Code',monospace",fontSize:8.5,color:sentDot(h.sentiment),letterSpacing:1}}>{h.sentiment.slice(0,3).toUpperCase()}</span>
              </div>
            ))}
          </div>
        )}
      </div>


      <div style={{fontFamily:"'Fira Code',monospace",fontSize:9,color:'#daa520',letterSpacing:2,marginBottom:8}}>ALL RECENT COVERAGE · sentiment + reasoning</div>
      <div style={{display:'flex',flexDirection:'column',gap:6,marginBottom:18}}>
        {(d.recent_headlines||[]).slice(0,10).map((h,i)=>(
          <div key={i} style={{background:'#241510',border:'1px solid rgba(212,149,108,0.12)',borderRadius:8,padding:'9px 12px',borderLeft:`3px solid ${sentDot(h.sentiment)}`}}>
            <div style={{display:'flex',justifyContent:'space-between',gap:10}}>
              {h.url?<a href={h.url} target="_blank" rel="noopener noreferrer" style={{fontSize:12.5,color:'#e8ddd0',textDecoration:'none',fontWeight:500,flex:1}}>{h.title}</a>
                :<span style={{fontSize:12.5,color:'#e8ddd0',fontWeight:500,flex:1}}>{h.title}</span>}
              <span style={{fontSize:9,color:sentDot(h.sentiment),textTransform:'uppercase',fontWeight:600,flexShrink:0}}>{h.sentiment}</span>
            </div>
            {h.reason && <div style={{fontSize:11,color:'#9d8b7a',marginTop:4,lineHeight:1.4}}>{h.reason}</div>}
            <div style={{fontSize:9,color:'#7a7266',marginTop:3}}>{h.publisher} · {h.date}</div>
          </div>
        ))}
      </div>

      {data?.sentiment?.news && (() => {
        const nw = data.sentiment.news;
        const p = (nw.positive ?? nw.positive_prob ?? 0) * 100;
        const ng = (nw.negative ?? nw.negative_prob ?? 0) * 100;
        const nu = (nw.neutral ?? nw.neutral_prob ?? 0) * 100;
        const comp = data.sentiment.composite;
        if (!(p || ng || nu)) return null;
        return (
          <div style={{background:'#241510',border:'1px solid rgba(212,149,108,0.12)',borderRadius:8,padding:'16px 20px',marginBottom:14}}>
            <div style={{fontFamily:"'Fira Code',monospace",fontSize:9,color:'#daa520',letterSpacing:2,marginBottom:4}}>LANGUAGE MODEL READ</div>
            <div style={{fontFamily:"'Outfit',sans-serif",fontSize:10.5,color:'#7a6b5d',marginBottom:14}}>
              FinBERT — a transformer trained on financial text — scores the tone of each article independently of the
              headline ranking above. It reads how coverage is written, not what it reports.
            </div>
            <div style={{display:'flex',gap:20,alignItems:'center',flexWrap:'wrap'}}>
              <div style={{textAlign:'center',minWidth:90}}>
                <div style={{fontFamily:"'Fira Code',monospace",fontSize:26,fontWeight:800,
                  color: comp > 0.15 ? '#22c55e' : comp < -0.15 ? '#ef4444' : '#f59e0b', lineHeight:1}}>
                  {comp != null ? (comp > 0 ? '+' : '') + comp.toFixed(2) : '—'}
                </div>
                <div style={{fontFamily:"'Fira Code',monospace",fontSize:8,color:'#8a7560',letterSpacing:1,marginTop:3}}>COMPOSITE</div>
                <div style={{fontFamily:"'Outfit',sans-serif",fontSize:10,color:'#9d8b7a'}}>{data.sentiment.label || ''}</div>
              </div>
              <div style={{flex:1,minWidth:240}}>
                {[{l:'Positive',v:p,c:'#22c55e'},{l:'Neutral',v:nu,c:'#8a7560'},{l:'Negative',v:ng,c:'#ef4444'}].map(x=>(
                  <div key={x.l} style={{display:'grid',gridTemplateColumns:'70px 1fr 46px',gap:10,alignItems:'center',marginBottom:5}}>
                    <span style={{fontFamily:"'Outfit',sans-serif",fontSize:11,color:'#9d8b7a'}}>{x.l}</span>
                    <div style={{height:10,background:'#1a0f0a',borderRadius:2,overflow:'hidden'}}>
                      <div style={{height:'100%',width:`${x.v}%`,background:x.c,opacity:0.55,borderRadius:2}}/>
                    </div>
                    <span style={{fontFamily:"'Fira Code',monospace",fontSize:10.5,color:'#d4c4b0',textAlign:'right'}}>{x.v.toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );
      })()}

      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
        <span style={{fontSize:12,color:'#9d8b7a',letterSpacing:1}}>NEWS SIGNALS · {d.tree.categories.reduce((a,c)=>a+c.n_signals,0)} scored</span>
        <button onClick={()=>{setShowAllSignals(!showAllSignals);const m:Record<string,boolean>={};d.tree.categories.forEach(c=>m[c.id]=!showAllSignals);setExpanded(m);}}
          style={{background:'#1a0f0a',border:'1px solid rgba(212,149,108,0.12)',color:'#9d8b7a',borderRadius:8,padding:'5px 12px',fontSize:11,cursor:'pointer'}}>
          {showAllSignals?'Collapse all':'Expand all'}</button>
      </div>
      <div style={{display:'flex',flexDirection:'column',gap:8}}>
        {d.tree.categories.map(cat=>{
          const open=expanded[cat.id];
          return (
            <div key={cat.id} style={{background:'#241510',border:'1px solid rgba(212,149,108,0.12)',borderRadius:10,overflow:'hidden'}}>
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
                        <span style={{fontSize:12,color:'#9d8b7a',width:64,textAlign:'right'}}>{pending?'—':fmtVal(s.id,s.raw_value)}</span>
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
