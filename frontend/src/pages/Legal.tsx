// Plain legal & info pages. DRAFTS — have a lawyer review before charging for any feature.
import React from 'react';
import { useNavigate } from 'react-router-dom';
const PAGES: Record<string, { title: string; body: string[] }> = {
  disclaimer: { title: 'Disclaimer', body: [
    'QuantEdge provides research and educational information only. Nothing on this site is investment, financial, legal or tax advice, and nothing is a recommendation to buy, sell or hold any security.',
    'Model outputs, pattern statistics and historical frequencies describe the past. They do not predict or guarantee future results. Where a forecast is marked "not validated," it has not held up on recent data and should not be relied on.',
    'Data may be delayed, incomplete or wrong. Always verify information independently and consider speaking with a licensed financial adviser before making investment decisions. Investing involves risk, including the loss of principal.'] },
  'data-sources': { title: 'Data sources', body: [
    'Market prices, volumes and news headlines: Polygon.io. Intraday figures show the time of the quote received.',
    'Company filings (8-K, Form 4, 10-Q/10-K financial data, 13F institutional holdings): the U.S. Securities and Exchange Commission (SEC EDGAR). Each filing is dated to when it became public.',
    'US indexes and world markets are shown through the exchange-traded funds that track them. World funds trade in dollars during US market hours.',
    'Not available on the current data plan, and never estimated: analyst earnings estimates, company guidance, earnings-call transcripts, options positioning.'] },
  terms: { title: 'Terms of use', body: [
    'Draft — last updated September 28, 2026. By using QuantEdge you agree to use it for personal research and education, not to scrape or resell its data, and not to attempt to disrupt the service.',
    'QuantEdge is provided "as is," without warranties of accuracy, completeness or availability. To the extent permitted by law, QuantEdge and its operator are not liable for decisions made using the site.',
    'Some features may require an account or payment in the future; any such terms will be shown before you are charged.',
    'Questions: dileepkreddy5@gmail.com.'] },
  privacy: { title: 'Privacy policy', body: [
    'Draft — last updated September 28, 2026. QuantEdge does not require an account to research stocks. Searches are used to serve results and are not sold.',
    'Standard server logs (IP address, time, pages requested) are kept to operate and secure the service and are deleted on a rolling basis.',
    'If you create an account, your email and settings are stored to provide the account. Your browser may store small preferences locally.',
    'Requests about your data: dileepkreddy5@gmail.com.'] },
};
const Legal: React.FC<{ page: string }> = ({ page }) => {
  const nav = useNavigate(); const p = PAGES[page];
  return (<div style={{ minHeight: '100vh', background: '#0d0806', color: '#d9c9b4', fontFamily: "'IBM Plex Sans',system-ui,sans-serif" }}>
    <div style={{ maxWidth: 760, margin: '0 auto', padding: '56px 28px' }}>
      <a onClick={() => nav('/')} style={{ cursor: 'pointer', fontFamily: "'IBM Plex Mono',monospace", fontSize: 13, color: '#e0ad3a', letterSpacing: '.3em' }}>← QUANTEDGE</a>
      <h1 style={{ fontFamily: "'Fraunces',Georgia,serif", fontWeight: 300, fontSize: 46, color: '#f6ecdd', margin: '32px 0 24px' }}>{p.title}</h1>
      {p.body.map((t, i) => <p key={i} style={{ fontSize: 16.5, lineHeight: 1.8, color: '#c9b8a4', margin: '0 0 18px' }}>{t}</p>)}
      <p style={{ marginTop: 40, fontFamily: "'IBM Plex Mono',monospace", fontSize: 12, color: '#8a7762' }}>© 2026 Dileep Kumar Reddy Kapu</p>
    </div></div>);
};
export default Legal;
