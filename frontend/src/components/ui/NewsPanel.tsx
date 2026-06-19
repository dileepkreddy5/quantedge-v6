// ============================================================
// QuantEdge v6.0 — News Panel (per-stock news tab)
// Recent articles for a ticker: headline, source, time, sentiment, link.
// ============================================================

import React, { useEffect, useState, useCallback } from 'react';
import { api } from '../../auth/authStore';

const C = {
  bg: '#0a0505', panel: '#1a0f0a', panel2: '#1f130d', border: '#3a2920',
  border2: '#4a3428', gold: '#daa520', text: '#d4c4b0', textDim: '#9d8b7a',
  green: '#22c55e', red: '#ef4444', grey: '#9d8b7a',
};

interface Article {
  title: string; snippet: string; source: string; author: string;
  url: string; published: string; relative_time: string;
  sentiment: 'positive' | 'negative' | 'neutral'; sentiment_score: number | null;
}
interface Summary { count: number; label: string; pos: number; neg: number; neu: number; }
interface NewsData { ticker: string; summary: Summary; articles: Article[]; }

const dot = (s: string) =>
  s === 'positive' ? C.green : s === 'negative' ? C.red : C.grey;

interface Props { data?: any; ticker?: string; compact?: boolean; onSeeAll?: () => void; }

const NewsPanel: React.FC<Props> = ({ ticker: tickerProp, data: analysisData, compact, onSeeAll }) => {
  const ticker = (tickerProp || analysisData?.ticker || analysisData?.symbol || '').toUpperCase();
  const [news, setNews] = useState<NewsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(false);

  const load = useCallback(async () => {
    if (!ticker) { setLoading(false); return; }
    setLoading(true); setErr(false);
    try {
      const res = await api.get(`/api/v6/news/${ticker}`, { params: { limit: compact ? 3 : 30 } });
      setNews(res.data?.data || null);
    } catch { setErr(true); }
    finally { setLoading(false); }
  }, [ticker, compact]);

  useEffect(() => { load(); }, [load]);

  if (!ticker) return <div style={{ color: C.textDim, padding: 20 }}>Analyze a stock to see its news.</div>;
  if (loading) return <div style={{ color: C.textDim, padding: 20 }}>Loading news…</div>;
  if (err) return <div style={{ color: C.textDim, padding: 20 }}>Couldn’t load news for {ticker}.</div>;

  const articles = news?.articles || [];
  const summary = news?.summary;

  if (articles.length === 0)
    return <div style={{ color: C.textDim, padding: 20 }}>No recent news for {ticker}.</div>;

  return (
    <div style={{ color: C.text }}>
      {!compact && summary && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap',
                      padding: '4px 0 14px', borderBottom: `1px solid ${C.border}`, marginBottom: 14 }}>
          <span style={{ color: C.gold, fontWeight: 700, fontSize: 15 }}>{ticker} NEWS</span>
          <span style={{ color: C.textDim, fontSize: 13 }}>
            {summary.count} recent articles · <span style={{ color:
              summary.label.includes('positive') ? C.green :
              summary.label.includes('negative') ? C.red : C.textDim }}>{summary.label}</span>
          </span>
          <span style={{ marginLeft: 'auto', display: 'flex', gap: 10, fontSize: 12 }}>
            <span style={{ color: C.green }}>● {summary.pos}</span>
            <span style={{ color: C.grey }}>● {summary.neu}</span>
            <span style={{ color: C.red }}>● {summary.neg}</span>
          </span>
        </div>
      )}

      {compact && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <span style={{ color: C.gold, fontWeight: 700, fontSize: 13, letterSpacing: 0.5 }}>RECENT NEWS</span>
          {onSeeAll && <button onClick={onSeeAll} style={{ background: 'none', border: 'none', color: C.gold, fontSize: 12, cursor: 'pointer', fontWeight: 600 }}>See all →</button>}
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: compact ? 8 : 2 }}>
        {articles.map((a, i) => (
          <a key={i} href={a.url} target="_blank" rel="noopener noreferrer"
             style={{ textDecoration: 'none', color: 'inherit', display: 'block',
                      padding: compact ? '8px 0' : '12px 4px',
                      borderBottom: compact ? 'none' : `1px solid ${C.border}` }}>
            <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
              <span style={{ marginTop: 6, flexShrink: 0, width: 8, height: 8, borderRadius: '50%', background: dot(a.sentiment) }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ color: C.text, fontSize: compact ? 13 : 14, fontWeight: 600, lineHeight: 1.35 }}>{a.title}</div>
                {!compact && a.snippet && (
                  <div style={{ color: C.textDim, fontSize: 12.5, marginTop: 4, lineHeight: 1.45,
                                display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                    {a.snippet}
                  </div>
                )}
                <div style={{ color: C.textDim, fontSize: 11, marginTop: 4, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <span style={{ color: C.gold }}>{a.source || 'source'}</span>
                  {a.relative_time && <span>· {a.relative_time}</span>}
                </div>
              </div>
            </div>
          </a>
        ))}
      </div>

      {!compact && (
        <div style={{ fontSize: 11, color: C.textDim, marginTop: 14, fontStyle: 'italic' }}>
          Headlines link to original sources. News is recent (data may be delayed ~15 min).
        </div>
      )}
    </div>
  );
};

export default NewsPanel;
