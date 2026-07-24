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
const heat=(s:number|null)=>s==null?'var(--border-2)':s>=70?'var(--gold)':s>=50?'var(--caramel)':s>=30?'#c9762f':'var(--bear)';
const ratingColor=(r:string)=>r.includes('Very Positive')?'var(--gold)':r.includes('Positive')?'var(--gold)':r.includes('Mixed')?'var(--caramel)':r.includes('Negative')?'#c9762f':'var(--bear)';
const sentDot=(s:string)=>s==='positive'?'var(--gold)':s==='negative'?'var(--bear)':'var(--cocoa)';

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

  if(!ticker)return <div style={{color:'var(--cocoa-dust)',padding:24}}>Enter a ticker for News Intelligence.</div>;
  if(loading)return <div style={{color:'var(--gold)',padding:24}}>Analyzing news — sentiment, events, and 47 signals across recent coverage…</div>;
  if(err)return <div style={{fontFamily:'var(--font-body)',color:'var(--bear)',padding:24}}>News: {err}</div>;
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
    // State the neutral count too. Reporting only positive and negative against
    // the article total reads as broken arithmetic (43 + 11 != 93).
    parts.push(`Coverage is broadly ${tone} and ${dir} — ${sd.positive} positive, ${sd.neutral} neutral, ${sd.negative} negative across ${d.article_count} articles.`);
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
    <div style={{padding:'8px 4px',color:'var(--latte)'}}>

      <div style={{background:'var(--surface-2)',border:'1px solid rgba(212,149,108,0.12)',borderRadius:8,padding:'18px 20px',marginBottom:14}}>
        <div style={{display:'flex',gap:26,alignItems:'flex-start',flexWrap:'wrap'}}>
          <div style={{textAlign:'center',minWidth:96}}>
            <div style={{fontFamily:"'Fira Code',monospace",fontSize:44,fontWeight:800,color:heat(d.score),lineHeight:1}}>{d.score?.toFixed(0)??'—'}</div>
            <div style={{fontFamily:"'Fira Code',monospace",fontSize:8,color:'var(--cocoa)',letterSpacing:2,marginTop:4}}>NEWS SCORE</div>
            <div style={{fontFamily:"'Outfit',sans-serif",fontSize:12,color:ratingColor(d.news_rating),marginTop:2}}>{d.news_rating}</div>
          </div>
          <div style={{flex:1,minWidth:300}}>
            <div style={{fontFamily:"'Outfit',sans-serif",fontSize:13.5,color:'var(--latte)',lineHeight:1.6}}>{verdict}</div>
            <div style={{display:'flex',height:8,borderRadius:4,overflow:'hidden',marginTop:14}}>
              <div style={{width:`${sd.positive/total*100}%`,background:'var(--gold)',opacity:0.85}}/>
              <div style={{width:`${sd.neutral/total*100}%`,background:'var(--border-2)',opacity:0.7}}/>
              <div style={{width:`${sd.negative/total*100}%`,background:'var(--bear)',opacity:0.75}}/>
            </div>
            <div style={{display:'flex',justifyContent:'space-between',fontFamily:"'Fira Code',monospace",fontSize:8.5,color:'var(--cocoa)',marginTop:4}}>
              <span>{sd.positive} positive</span><span>{sd.neutral} neutral</span><span>{sd.negative} negative</span>
            </div>
          </div>
        </div>
        <div style={{display:'grid',gridTemplateColumns:'repeat(5, 1fr)',gap:8,marginTop:16,paddingTop:14,borderTop:'1px solid rgba(212,149,108,0.1)'}}>
          {[
            {l:'NET SENTIMENT', v:net!=null?`${net>0?'+':''}${net.toFixed(2)}`:'—', n:'materiality-weighted'},
            {l:'30-DAY PRICE',  v:px30!=null?`${px30>0?'+':''}${(px30*100).toFixed(1)}%`:'—', n:diverge?'diverging from tone':'in line with tone'},
            {l:'HIGH-IMPACT TONE', v:top10!=null?`${top10>0?'+':''}${top10.toFixed(2)}`:'—', n:'top 10 stories only'},
            {l:'COVERAGE RATE', v:km.news_velocity!=null?`${km.news_velocity.toFixed(1)}/day`:'—', n:`${km.article_count_7d??0} in last 7 days`},
            {l:'SOURCE QUALITY',v:km.tier1_source_share!=null?`${(km.tier1_source_share*100).toFixed(0)}%`:'—', n:'from tier-1 outlets'},
          ].map(x=>(
            <div key={x.l} style={{background:'var(--surface-3)',borderRadius:6,padding:'9px 10px'}}>
              <div style={{fontFamily:"'Fira Code',monospace",fontSize:8,color:'var(--cocoa)',letterSpacing:1}}>{x.l}</div>
              <div style={{fontFamily:"'Fira Code',monospace",fontSize:15,fontWeight:700,color:'var(--latte)',marginTop:3}}>{x.v}</div>
              <div style={{fontFamily:"'Outfit',sans-serif",fontSize:8.5,color:'var(--cocoa)',marginTop:1}}>{x.n}</div>
            </div>
          ))}
        </div>
        {km.fraud_litigation_flag ? (
          <div style={{marginTop:12,padding:'8px 12px',background:'rgba(192,112,90,0.08)',borderLeft:'2px solid var(--bear)',borderRadius:4,
            fontFamily:"'Outfit',sans-serif",fontSize:11,color:'var(--latte)'}}>
            Litigation or fraud language detected in recent coverage — worth reading the source articles directly.
          </div>
        ) : null}
      </div>

      {(() => {
        // The parent keeps `data` across ticker changes, so this block would
        // render the previous company's earnings until the new analyze lands.
        // META showed GOOG's 2.01-vs-1.9884 for exactly that reason.
        if (data?.analyst_ratings?.ticker &&
            data.analyst_ratings.ticker.toUpperCase() !== ticker.toUpperCase()) return null;
        const er = data?.analyst_ratings?.earnings;
        const an = data?.analyst_ratings?.analytics;
        const hist = (er?.surprise_history || []).filter((s:any)=>s?.surprise_percent!=null);
        if (!er || hist.length === 0) return null;
        // Oldest first, so the reader scans the trend left to right.
        const ordered = [...hist].reverse();
        const beats = ordered.filter((s:any)=>s.surprise_percent > 0).length;
        const first = ordered[0].surprise_percent, last = ordered[ordered.length-1].surprise_percent;
        const narrowing = ordered.length >= 3 && last < first;
        const lastRep = er.last_reported || er.date;
        const days = er.days_since ?? (er.days_to != null ? -er.days_to : null);
        return (
          <div style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderRadius:8,padding:'16px 20px',marginBottom:14}}>
            <div style={{fontFamily:'var(--font-mono)',fontSize:9,color:'var(--gold)',letterSpacing:2,marginBottom:4}}>EARNINGS</div>
            <div style={{fontFamily:'var(--font-body)',fontSize:10.5,color:'var(--cocoa)',marginBottom:14}}>
              Reported results against consensus. The trend across quarters says more than any single beat.
            </div>
            <div style={{display:'flex',gap:28,flexWrap:'wrap',alignItems:'flex-start',marginBottom:16}}>
              <div>
                <div style={{fontFamily:'var(--font-mono)',fontSize:9,color:'var(--cocoa)',letterSpacing:1.5}}>LAST REPORTED</div>
                <div style={{fontFamily:'var(--font-mono)',fontSize:15,color:'var(--latte)',fontWeight:600,marginTop:3}}>
                  {lastRep || '—'}
                </div>
                {days != null && (
                  <div style={{fontFamily:'var(--font-body)',fontSize:10,color:'var(--cocoa)'}}>{days} days ago</div>
                )}
              </div>
              <div>
                <div style={{fontFamily:'var(--font-mono)',fontSize:9,color:'var(--cocoa)',letterSpacing:1.5}}>EPS ACTUAL VS EST</div>
                <div style={{fontFamily:'var(--font-mono)',fontSize:15,color:'var(--latte)',fontWeight:600,marginTop:3}}>
                  {ordered[ordered.length-1].actual ?? '—'} vs {ordered[ordered.length-1].estimate ?? '—'}
                </div>
                <div style={{fontFamily:'var(--font-body)',fontSize:10,color: last>0?'var(--bull)':'var(--bear)'}}>
                  {last>0?'+':''}{last.toFixed(1)}% surprise
                </div>
              </div>
              <div>
                <div style={{fontFamily:'var(--font-mono)',fontSize:9,color:'var(--cocoa)',letterSpacing:1.5}}>BEAT RATE</div>
                <div style={{fontFamily:'var(--font-mono)',fontSize:15,color:'var(--latte)',fontWeight:600,marginTop:3}}>
                  {beats}/{ordered.length}
                </div>
                {an?.avg_surprise_pct != null && (
                  <div style={{fontFamily:'var(--font-body)',fontSize:10,color:'var(--cocoa)'}}>avg {an.avg_surprise_pct.toFixed(1)}%</div>
                )}
              </div>
              <div>
                <div style={{fontFamily:'var(--font-mono)',fontSize:9,color:'var(--cocoa)',letterSpacing:1.5}}>NEXT SCHEDULED</div>
                <div style={{fontFamily:'var(--font-mono)',fontSize:15,color:'var(--cocoa)',fontWeight:600,marginTop:3}}>
                  {er.next_scheduled || 'not available'}
                </div>
                <div style={{fontFamily:'var(--font-body)',fontSize:10,color:'var(--cocoa)'}}>
                  {er.next_scheduled ? '' : 'forward calendar not on this data tier'}
                </div>
              </div>
            </div>
            <div style={{display:'grid',gridTemplateColumns:`repeat(${ordered.length}, 1fr)`,gap:8}}>
              {ordered.map((s:any)=>(
                <div key={s.period} style={{background:'var(--surface-3)',border:'1px solid var(--border-1)',borderRadius:4,padding:'8px 10px'}}>
                  <div style={{fontFamily:'var(--font-mono)',fontSize:9,color:'var(--cocoa)'}}>{s.period}</div>
                  <div style={{fontFamily:'var(--font-mono)',fontSize:14,fontWeight:600,color: s.surprise_percent>0?'var(--bull)':'var(--bear)',marginTop:2}}>
                    {s.surprise_percent>0?'+':''}{s.surprise_percent.toFixed(1)}%
                  </div>
                  <div style={{fontFamily:'var(--font-body)',fontSize:9,color:'var(--cocoa)'}}>
                    {s.actual} vs {s.estimate}
                  </div>
                </div>
              ))}
            </div>
            <div style={{fontFamily:'var(--font-body)',fontSize:11,color:'var(--latte)',marginTop:12,lineHeight:1.5}}>
              {beats === ordered.length
                ? `Beat consensus in all ${ordered.length} quarters shown`
                : `Beat consensus in ${beats} of ${ordered.length} quarters shown`}
              {narrowing
                ? `, but the margin has narrowed from ${first>0?'+':''}${first.toFixed(1)}% to ${last>0?'+':''}${last.toFixed(1)}% — a consistent beat rate can mask a compressing surprise.`
                : '.'}
            </div>
          </div>
        );
      })()}

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
          <div style={{background:'var(--surface-2)',border:'1px solid rgba(212,149,108,0.12)',borderRadius:8,padding:'16px 20px',marginBottom:14}}>
            <div style={{fontFamily:"'Fira Code',monospace",fontSize:9,color:'var(--gold)',letterSpacing:2,marginBottom:4}}>THE NUMBERS THAT MATTER</div>
            <div style={{fontFamily:"'Outfit',sans-serif",fontSize:10.5,color:'var(--cocoa)',marginBottom:16}}>
              Specific figures and stated developments pulled from recent coverage, grouped by what they describe.
              Each links to the article it came from.
            </div>
            <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit, minmax(280px, 1fr))',gap:'18px 24px'}}>
              {order.map(g => (
                <div key={g}>
                  <div style={{fontFamily:"'Fira Code',monospace",fontSize:8.5,color:'var(--gold)',letterSpacing:2,marginBottom:2}}>{GROUPS[g].label}</div>
                  <div style={{fontFamily:"'Outfit',sans-serif",fontSize:8.5,color:'var(--cocoa)',marginBottom:9}}>{GROUPS[g].note}</div>
                  {by[g].slice(0,4).map((k,i) => (
                    <div key={i} style={{marginBottom:11,paddingLeft:10,borderLeft:'2px solid rgba(212,149,108,0.18)'}}>
                      <div style={{fontFamily:"'Outfit',sans-serif",fontSize:11.5,color:'var(--latte)',lineHeight:1.5}}>{k.fact}</div>
                      <div style={{marginTop:3}}>
                        {k.url
                          ? <a href={k.url} target="_blank" rel="noopener noreferrer"
                              style={{fontFamily:"'Outfit',sans-serif",fontSize:9,color:'var(--cocoa)',textDecoration:'none'}}>
                              {k.source} · {k.date} ↗
                            </a>
                          : <span style={{fontFamily:"'Outfit',sans-serif",fontSize:9,color:'var(--cocoa)'}}>{k.source} · {k.date}</span>}
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        );
      })()}

      <div style={{background:'var(--surface-2)',border:'1px solid rgba(212,149,108,0.12)',borderRadius:8,padding:'16px 20px',marginBottom:14}}>
        <div style={{fontFamily:"'Fira Code',monospace",fontSize:9,color:'var(--gold)',letterSpacing:2,marginBottom:4}}>REPORTED EVENTS</div>
        <div style={{fontFamily:"'Outfit',sans-serif",fontSize:10.5,color:'var(--cocoa)',marginBottom:14}}>
          Things that actually happened — results, deals, filings, analyst actions — separated from commentary about them.
          Most financial coverage is opinion; this section is deliberately short, and empty when nothing has been reported.
        </div>
        {events.length === 0 && (
          <div style={{fontFamily:"'Outfit',sans-serif",fontSize:12,color:'var(--cocoa)',padding:'14px 0',lineHeight:1.6}}>
            No reported events in recent coverage. Everything currently written about this company is commentary,
            comparison or speculation — listed below.
          </div>
        )}
        {events.map((h,i)=>(
          <div key={i} style={{paddingBottom:14,marginBottom:14,borderBottom:i<events.length-1?'1px solid rgba(212,149,108,0.08)':'none'}}>
            <div style={{display:'flex',gap:12,alignItems:'flex-start'}}>
              <div style={{minWidth:38,textAlign:'center'}}>
                <div style={{fontFamily:"'Fira Code',monospace",fontSize:16,fontWeight:800,color:heat(h.materiality??50)}}>{h.materiality??'—'}</div>
                <div style={{fontFamily:"'Fira Code',monospace",fontSize:7,color:'var(--cocoa)',letterSpacing:1}}>IMPACT</div>
              </div>
              <div style={{flex:1}}>
                {h.url
                  ? <a href={h.url} target="_blank" rel="noopener noreferrer" style={{fontFamily:"'Outfit',sans-serif",fontSize:16,fontWeight:600,color:'var(--latte)',textDecoration:'none',lineHeight:1.35,display:'block'}}>{h.title}</a>
                  : <div style={{fontFamily:"'Outfit',sans-serif",fontSize:16,fontWeight:600,color:'var(--latte)',lineHeight:1.35}}>{h.title}</div>}
                <div style={{fontFamily:"'Outfit',sans-serif",fontSize:11.5,color:'var(--cocoa-dust)',lineHeight:1.5,marginTop:5}}>{h.reason}</div>
                <div style={{display:'flex',gap:10,alignItems:'center',marginTop:7,flexWrap:'wrap'}}>
                  <span style={{fontFamily:"'Fira Code',monospace",fontSize:8,letterSpacing:1,padding:'2px 6px',borderRadius:3,
                    color: h.kind==='EVENT' ? 'var(--gold)' : 'var(--caramel)',
                    border: `1px solid ${h.kind==='EVENT' ? 'var(--gold)' : 'var(--caramel)'}44`,
                    background: `${h.kind==='EVENT' ? 'var(--gold)' : 'var(--caramel)'}12`}}>
                    {h.event_kind || h.kind}
                  </span>
                  <span style={{fontFamily:"'Fira Code',monospace",fontSize:9,color:sentDot(h.sentiment),textTransform:'uppercase',letterSpacing:1}}>{h.sentiment}</span>
                  <span style={{fontFamily:"'Outfit',sans-serif",fontSize:9.5,color:'var(--cocoa)'}}>{h.publisher} · {h.date}</span>
                  {h.materiality_why?.length ? (
                    <span style={{fontFamily:"'Outfit',sans-serif",fontSize:9,color:'var(--cocoa)',fontStyle:'italic'}}>ranked for: {h.materiality_why.join('; ')}</span>
                  ) : null}
                </div>
              </div>
            </div>
          </div>
        ))}
        {rest.length > 0 && (
          <div style={{marginTop:16,paddingTop:14,borderTop:'1px solid rgba(212,149,108,0.1)'}}>
            <div style={{fontFamily:"'Fira Code',monospace",fontSize:9,color:'var(--cocoa)',letterSpacing:2,marginBottom:8}}>
              COMMENTARY &amp; ANALYSIS · {rest.length}
            </div>
            {rest.map((h,i)=>(
              <div key={i} style={{display:'grid',gridTemplateColumns:'30px 1fr auto',gap:10,alignItems:'center',padding:'6px 0',
                borderTop:'1px solid rgba(212,149,108,0.06)'}}>
                <span style={{fontFamily:"'Fira Code',monospace",fontSize:10,color:heat(h.materiality??50),textAlign:'right'}}>{h.materiality??'—'}</span>
                {h.url
                  ? <a href={h.url} target="_blank" rel="noopener noreferrer" style={{fontFamily:"'Outfit',sans-serif",fontSize:12,color:'var(--cocoa-dust)',textDecoration:'none'}}>{h.title}</a>
                  : <span style={{fontFamily:"'Outfit',sans-serif",fontSize:12,color:'var(--cocoa-dust)'}}>{h.title}</span>}
                <span style={{fontFamily:"'Fira Code',monospace",fontSize:8.5,color:sentDot(h.sentiment),letterSpacing:1}}>{h.sentiment.slice(0,3).toUpperCase()}</span>
              </div>
            ))}
          </div>
        )}
      </div>


      <div style={{fontFamily:"'Fira Code',monospace",fontSize:9,color:'var(--gold)',letterSpacing:2,marginBottom:8}}>ALL RECENT COVERAGE · sentiment + reasoning</div>
      <div style={{display:'flex',flexDirection:'column',gap:6,marginBottom:18}}>
        {(d.recent_headlines||[]).slice(0,10).map((h,i)=>(
          <div key={i} style={{background:'var(--surface-2)',border:'1px solid rgba(212,149,108,0.12)',borderRadius:8,padding:'9px 12px',borderLeft:`3px solid ${sentDot(h.sentiment)}`}}>
            <div style={{display:'flex',justifyContent:'space-between',gap:10}}>
              {h.url?<a href={h.url} target="_blank" rel="noopener noreferrer" style={{fontSize:12.5,color:'var(--latte)',textDecoration:'none',fontWeight:500,flex:1}}>{h.title}</a>
                :<span style={{fontSize:12.5,color:'var(--latte)',fontWeight:500,flex:1}}>{h.title}</span>}
              <span style={{fontSize:9,color:sentDot(h.sentiment),textTransform:'uppercase',fontWeight:600,flexShrink:0}}>{h.sentiment}</span>
            </div>
            {h.reason && <div style={{fontSize:11,color:'var(--cocoa-dust)',marginTop:4,lineHeight:1.4}}>{h.reason}</div>}
            <div style={{fontSize:9,color:'var(--cocoa)',marginTop:3}}>{h.publisher} · {h.date}</div>
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
          <div style={{background:'var(--surface-2)',border:'1px solid rgba(212,149,108,0.12)',borderRadius:8,padding:'16px 20px',marginBottom:14}}>
            <div style={{fontFamily:"'Fira Code',monospace",fontSize:9,color:'var(--gold)',letterSpacing:2,marginBottom:4}}>LANGUAGE MODEL READ</div>
            <div style={{fontFamily:"'Outfit',sans-serif",fontSize:10.5,color:'var(--cocoa)',marginBottom:14}}>
              FinBERT — a transformer trained on financial text — scores the tone of each article independently of the
              headline ranking above. It reads how coverage is written, not what it reports.
            </div>
            {comp != null && net != null && Math.abs(comp - net) > 0.12 && (
              <div style={{fontFamily:"'Outfit',sans-serif",fontSize:11,color:'var(--latte)',lineHeight:1.55,
                marginBottom:14,paddingLeft:10,borderLeft:'2px solid rgba(212,149,108,0.35)'}}>
                This is the <b>unweighted</b> mean across every article ({comp > 0 ? '+' : ''}{comp.toFixed(2)}).
                The net sentiment above ({net > 0 ? '+' : ''}{net.toFixed(2)}) weights each article by materiality.
                {comp < net
                  ? ' The gap means the more negative writing sits in lower-impact coverage — an unweighted average here would be dominated by filler.'
                  : ' The gap means the more negative writing sits in the highest-impact coverage, which the unweighted average understates.'}
              </div>
            )}
            <div style={{display:'flex',gap:20,alignItems:'center',flexWrap:'wrap'}}>
              <div style={{textAlign:'center',minWidth:90}}>
                <div style={{fontFamily:"'Fira Code',monospace",fontSize:26,fontWeight:800,
                  color: comp > 0.15 ? 'var(--gold)' : comp < -0.15 ? 'var(--bear)' : 'var(--caramel)', lineHeight:1}}>
                  {comp != null ? (comp > 0 ? '+' : '') + comp.toFixed(2) : '—'}
                </div>
                <div style={{fontFamily:"'Fira Code',monospace",fontSize:8,color:'var(--cocoa)',letterSpacing:1,marginTop:3}}>COMPOSITE</div>
                <div style={{fontFamily:"'Outfit',sans-serif",fontSize:10,color:'var(--cocoa-dust)'}}>{data.sentiment.label || ''}</div>
              </div>
              <div style={{flex:1,minWidth:240}}>
                {[{l:'Positive',v:p,c:'var(--gold)'},{l:'Neutral',v:nu,c:'var(--border-2)'},{l:'Negative',v:ng,c:'var(--bear)'}].map(x=>(
                  <div key={x.l} style={{display:'grid',gridTemplateColumns:'70px 1fr 46px',gap:10,alignItems:'center',marginBottom:5}}>
                    <span style={{fontFamily:"'Outfit',sans-serif",fontSize:11,color:'var(--cocoa-dust)'}}>{x.l}</span>
                    <div style={{height:10,background:'var(--surface-3)',borderRadius:2,overflow:'hidden'}}>
                      <div style={{height:'100%',width:`${x.v}%`,background:x.c,opacity:0.55,borderRadius:2}}/>
                    </div>
                    <span style={{fontFamily:"'Fira Code',monospace",fontSize:10.5,color:'var(--latte)',textAlign:'right'}}>{x.v.toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );
      })()}

      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
        <span style={{fontSize:12,color:'var(--cocoa-dust)',letterSpacing:1}}>NEWS SIGNALS · {d.tree.categories.reduce((a,c)=>a+c.n_signals,0)} scored</span>
        <button onClick={()=>{setShowAllSignals(!showAllSignals);const m:Record<string,boolean>={};d.tree.categories.forEach(c=>m[c.id]=!showAllSignals);setExpanded(m);}}
          style={{background:'var(--surface-3)',border:'1px solid rgba(212,149,108,0.12)',color:'var(--cocoa-dust)',borderRadius:8,padding:'5px 12px',fontSize:11,cursor:'pointer'}}>
          {showAllSignals?'Collapse all':'Expand all'}</button>
      </div>
      <div style={{display:'flex',flexDirection:'column',gap:8}}>
        {d.tree.categories.map(cat=>{
          const open=expanded[cat.id];
          return (
            <div key={cat.id} style={{background:'var(--surface-2)',border:'1px solid rgba(212,149,108,0.12)',borderRadius:10,overflow:'hidden'}}>
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
                    return (
                      <div key={s.id} title={s.evidence} style={{display:'flex',alignItems:'center',gap:10,padding:'5px 0',borderBottom:'1px solid #1e1e1e',opacity:pending?0.5:1}}>
                        <span style={{fontSize:12,color:'var(--latte)',flex:1}}>{s.label}</span>
                        <span style={{fontSize:12,color:'var(--cocoa-dust)',width:64,textAlign:'right'}}>{pending?'—':fmtVal(s.id,s.raw_value)}</span>
                        <div style={{width:80,height:6,background:'var(--surface-3)',borderRadius:3,overflow:'hidden'}}>
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
