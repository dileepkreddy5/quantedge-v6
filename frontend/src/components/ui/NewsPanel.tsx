// ============================================================
// QuantEdge v6.0 — News Panel (Key Developments + articles)
// AI-synthesized key points from real events + Polygon articles.
// ============================================================

import React, { useEffect, useState, useCallback } from 'react';
import { api } from '../../auth/authStore';

const C = {
  border: '#3a2920', gold: '#daa520', text: '#d4c4b0', textDim: '#9d8b7a',
  green: '#22c55e', red: '#ef4444', panel: '#1a0f0a',
};

interface Article { title: string; one_liner: string; source: string; url: string; relative_time: string; }
interface Events {
  next_earnings_date?: string | null; eps_surprise_pct?: number | null;
  analyst?: { strongBuy: number; buy: number; hold: number; sell: number; strongSell: number; period: string } | null;
}
interface NewsData {
  ticker: string; key_points: string[]; events: Events;
  articles: Article[]; ai_synthesized?: boolean;
}

interface Props { data?: any; ticker?: string; }

const NewsPanel: React.FC<Props> = ({ ticker: tickerProp, data: analysisData }) => {
  const ticker = (tickerProp || analysisData?.ticker || analysisData?.symbol || '').toUpperCase();
  const [news, setNews] = useState<NewsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(false);

  const load = useCallback(async () => {
    if (!ticker) { setLoading(false); return; }
    setLoading(true); setErr(false);
    try {
      const res = await api.get(`/api/v6/news/${ticker}`, { params: { limit: 10 } });
      setNews(res.data?.data || null);
    } catch { setErr(true); }
    finally { setLoading(false); }
  }, [ticker]);

  useEffect(() => { load(); }, [load]);

  if (!ticker) return <div style={{ color: C.textDim, padding: 20 }}>Analyze a stock to see its news.</div>;
  if (loading) return <div style={{ color: C.textDim, padding: 20 }}>Building briefing…</div>;
  if (err || !news) return <div style={{ color: C.textDim, padding: 20 }}>Couldn’t load news for {ticker}.</div>;

  const ev = news.events || {};
  const an = ev.analyst;
  const points = news.key_points || [];
  const articles = news.articles || [];

  return (
    <div style={{ color: C.text }}>
      {/* Event chips */}
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 16 }}>
        {ev.next_earnings_date && (
          <span style={chip}>📅 Earnings: <b style={{ color: C.gold }}>{ev.next_earnings_date}</b></span>
        )}
        {ev.eps_surprise_pct != null && (
          <span style={chip}>
            EPS surprise: <b style={{ color: ev.eps_surprise_pct >= 0 ? C.green : C.red }}>
              {ev.eps_surprise_pct >= 0 ? '+' : ''}{ev.eps_surprise_pct}%</b>
          </span>
        )}
        {an && (an.strongBuy + an.buy + an.hold + an.sell + an.strongSell) > 0 && (
          <span style={chip}>
            Analysts: <b style={{ color: C.green }}>{an.strongBuy + an.buy} buy</b> · {an.hold} hold · <b style={{ color: C.red }}>{an.sell + an.strongSell} sell</b>
          </span>
        )}
      </div>

      {/* Key Developments */}
      {points.length > 0 && (
        <div style={{ marginBottom: 22 }}>
          <div style={{ color: C.gold, fontWeight: 700, fontSize: 14, letterSpacing: 0.5, marginBottom: 10 }}>
            KEY DEVELOPMENTS
            <span style={{ color: C.textDim, fontWeight: 400, fontSize: 11, marginLeft: 8 }}>
              {news.ai_synthesized ? 'synthesized from recent news + data' : 'from available data'}
            </span>
          </div>
          <ol style={{ margin: 0, paddingLeft: 20, display: 'flex', flexDirection: 'column', gap: 7 }}>
            {points.map((p, i) => (
              <li key={i} style={{ fontSize: 13.5, lineHeight: 1.4, color: C.text }}>{p}</li>
            ))}
          </ol>
        </div>
      )}

      {/* Articles */}
      <div style={{ color: C.gold, fontWeight: 700, fontSize: 14, letterSpacing: 0.5, marginBottom: 10 }}>
        RECENT ARTICLES
      </div>
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        {articles.map((a, i) => (
          <a key={i} href={a.url} target="_blank" rel="noopener noreferrer"
             style={{ textDecoration: 'none', color: 'inherit', padding: '11px 4px', borderBottom: `1px solid ${C.border}` }}>
            <div style={{ fontSize: 14, fontWeight: 600, lineHeight: 1.35, color: C.text }}>{a.title}</div>
            {a.one_liner && (
              <div style={{ fontSize: 12.5, color: C.textDim, marginTop: 3, lineHeight: 1.4,
                            display: '-webkit-box', WebkitLineClamp: 1, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                {a.one_liner}
              </div>
            )}
            <div style={{ fontSize: 11, color: C.textDim, marginTop: 3 }}>
              <span style={{ color: C.gold }}>{a.source || 'source'}</span>
              {a.relative_time && <span> · {a.relative_time}</span>}
            </div>
          </a>
        ))}
      </div>

      <div style={{ fontSize: 11, color: C.textDim, marginTop: 14, fontStyle: 'italic' }}>
        Earnings &amp; analyst data via Finnhub. Headlines via Polygon. Key developments AI-summarized from these sources — verify before acting.
      </div>
    </div>
  );
};

const chip: React.CSSProperties = {
  background: '#1f130d', border: '#3a2920', borderWidth: 1, borderStyle: 'solid',
  borderRadius: 6, padding: '5px 10px', fontSize: 12, color: '#d4c4b0',
};

export default NewsPanel;
