// ============================================================
// QuantEdge v6.0 — Research Page
// Read-only view of research-zone results (status: research).
// NOT promoted signals. NOT investment advice.
// ============================================================

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../auth/authStore';

interface Row {
  window: string; supplier: string; customer: string;
  weight: number; customer_shock: number; supplier_next: number;
}
interface CFArtifact {
  unit: string; status: string; generated: string; title: string;
  mechanism: string; n_edges: number; n_observations: number;
  avg_supplier_after_pos_shock: number; avg_supplier_after_neg_shock: number;
  directional_spread: number; rows: Row[];
  honest_caveats: string[]; promotion: string;
}

const GOLD = '#daa520', DIM = '#9d8b7a', BG = '#1a0f0a', BORDER = '#3a2920';
const pct = (x: number) => (x >= 0 ? '+' : '') + (x * 100).toFixed(2) + '%';

export default function Research() {
  const nav = useNavigate();
  const [data, setData] = useState<CFArtifact | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.get('/api/v6/research/cf')
      .then((res) => setData(res.data))
      .catch(() => setErr('Could not load research artifact.'));
  }, []);

  const wrap: React.CSSProperties = {
    minHeight: '100vh', background: '#0d0805', color: '#e8dcc8',
    fontFamily: "'Fira Code', monospace", padding: '24px',
  };
  const card: React.CSSProperties = {
    background: BG, border: `1px solid ${BORDER}`, borderRadius: 6,
    padding: 18, marginBottom: 16,
  };
  const label: React.CSSProperties = {
    color: GOLD, fontSize: 11, letterSpacing: 2, textTransform: 'uppercase',
  };

  return (
    <div style={wrap}>
      <button onClick={() => nav('/')} style={{
        background: 'transparent', border: `1px solid ${BORDER}`, color: DIM,
        padding: '6px 14px', borderRadius: 4, cursor: 'pointer',
        fontFamily: "'Fira Code', monospace", fontSize: 10, marginBottom: 18,
      }}>← HOME</button>

      <h1 style={{ ...label, fontSize: 16 }}>QuantEdge — Research Lab</h1>
      <div style={{
        ...card, borderColor: '#5a4020', background: '#23150a',
        color: '#e0b860', fontSize: 12,
      }}>
        <strong style={label}>STATUS: RESEARCH — NOT INVESTMENT ADVICE</strong>
        <div style={{ marginTop: 8, color: DIM, lineHeight: 1.6 }}>
          This page shows results from the point-in-time research harness. These
          have <strong>not</strong> passed the promotion gate and are not signals
          on the live product. Shown for transparency about how the system is built.
        </div>
      </div>

      {err && <div style={card}>{err}</div>}
      {!data && !err && <div style={{ color: DIM }}>Loading…</div>}

      {data && (
        <>
          <div style={card}>
            <div style={label}>{data.title}</div>
            <div style={{ marginTop: 8, color: DIM, fontSize: 12, lineHeight: 1.6 }}>
              {data.mechanism}
            </div>
            <div style={{ display: 'flex', gap: 28, marginTop: 16, flexWrap: 'wrap' }}>
              <Metric label="Edges" value={String(data.n_edges)} />
              <Metric label="Observations" value={String(data.n_observations)} />
              <Metric label="After + shock" value={pct(data.avg_supplier_after_pos_shock)} />
              <Metric label="After − shock" value={pct(data.avg_supplier_after_neg_shock)} />
              <Metric label="Spread" value={pct(data.directional_spread)} hot />
            </div>
          </div>

          <div style={{ ...card, borderColor: '#5a3020', background: '#1f0f0a' }}>
            <div style={{ ...label, color: '#e08850' }}>Honest caveats — read these</div>
            <ul style={{ marginTop: 10, color: '#d8b89a', fontSize: 12, lineHeight: 1.8 }}>
              {data.honest_caveats.map((c, i) => <li key={i}>{c}</li>)}
            </ul>
            <div style={{ marginTop: 10, color: DIM, fontSize: 11 }}>{data.promotion}</div>
          </div>

          <div style={card}>
            <div style={label}>Observations ({data.rows.length})</div>
            <div style={{ overflowX: 'auto', marginTop: 10 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead><tr style={{ color: GOLD, textAlign: 'left' }}>
                  <th style={{ padding: 6 }}>WINDOW</th><th>SUPPLIER</th><th>CUSTOMER</th>
                  <th>WEIGHT</th><th>CUST SHOCK</th><th>SUPP NEXT MO</th>
                </tr></thead>
                <tbody>
                  {data.rows.map((r, i) => (
                    <tr key={i} style={{ borderTop: `1px solid ${BORDER}`, color: '#cbb89a' }}>
                      <td style={{ padding: 6 }}>{r.window}</td>
                      <td>{r.supplier}</td><td>{r.customer}</td>
                      <td>{(r.weight * 100).toFixed(0)}%</td>
                      <td style={{ color: r.customer_shock >= 0 ? '#6fbf73' : '#cf6b5a' }}>{pct(r.customer_shock)}</td>
                      <td style={{ color: r.supplier_next >= 0 ? '#6fbf73' : '#cf6b5a' }}>{pct(r.supplier_next)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div style={{ color: DIM, fontSize: 10 }}>Generated {data.generated}</div>
        </>
      )}
    </div>
  );
}

function Metric({ label, value, hot }: { label: string; value: string; hot?: boolean }) {
  return (
    <div>
      <div style={{ color: '#9d8b7a', fontSize: 10, letterSpacing: 1 }}>{label.toUpperCase()}</div>
      <div style={{ color: hot ? '#daa520' : '#e8dcc8', fontSize: 18, marginTop: 2 }}>{value}</div>
    </div>
  );
}
