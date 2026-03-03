# QuantEdge v6.0

> ⚠️ **Deployment in progress.** Market data is being migrated from Yahoo Finance to Polygon API to eliminate rate limits. AWS infrastructure is fully defined and containerized — rollout underway.

QuantEdge is a modular AI platform designed to replicate institutional-grade quantitative decision workflows. I built it because I wanted the kind of analysis that hedge funds use — not the watered-down stuff you get from retail tools like Yahoo Finance or Robinhood. I wanted to type a ticker, hit enter, and get a real institutional-grade report: regime detection, volatility forecasting, options flow, sentiment, risk-adjusted position sizing — all of it, in one place, in under a minute.

Live deployment: **https://quant.dileepkapu.com** . Single user. No SaaS, no subscriptions, no ads.

Built in 2026 by Dileep Kumar Reddy Kapu.
---

## What it does

Type a ticker — AAPL, NVDA, SPY, anything — and QuantEdge runs 8 analysis models simultaneously against 10 years of market data. You get a complete picture of that stock: what regime the market is in, where volatility is heading, what the options market is pricing in, how sentiment looks, and what a risk-adjusted position size would look like — all in one place.

---

## How it works

**Data** is fetched in parallel: 10 years of price history, full fundamentals, the complete options chain across 6 expiry dates, recent news headlines, and Reddit posts from investing communities.

**200+ features** are computed from that data — momentum factors, technical indicators, microstructure signals like VPIN (which estimates how much volume is coming from informed vs uninformed traders), and fractional differentiation to make the series stationary for ML without discarding historical memory.

**Historical labels** use triple-barrier methodology from Marcos Lopez de Prado's Advances in Financial Machine Learning. Each data point is labeled by whichever barrier is hit first — profit target, stop-loss, or time expiry — rather than the naive fixed-horizon approach most models use.

**8 models run in parallel:**

- **GJR-GARCH** — volatility forecasting with asymmetric leverage effect (bad news spikes vol more than equivalent good news)
- **Hidden Markov Model (5 states)** — market regime detection across bull/bear and high/low volatility states, with full transition probability matrix
- **Kalman Filter** — noise-filtered trend extraction and momentum signal
- **Monte Carlo (100,000 paths)** — jump-diffusion simulation giving the full distribution of 1-year outcomes from P10 to P90
- **LSTM** — sequence model predicting returns at 5, 10, 21, 63, and 252 day horizons
- **XGBoost + LightGBM** — gradient boosted ensemble for signal ranking and feature importance
- **Ensemble** — combines all models with confidence weighting and surfaces where they disagree
- **Claude Opus 4** — AI reasoning layer that interprets computed signals when local model files are unavailable

**Risk engine** runs independently of all prediction models. It computes CVaR, applies volatility targeting at 10% annualized, and enforces a drawdown governor — position sizing scales down at -15% drawdown and halts at -20%.

**Portfolio construction** uses Hierarchical Risk Parity (HRP) rather than Markowitz mean-variance optimization. HRP clusters assets by correlation and allocates risk hierarchically, avoiding the matrix inversion instability that makes classic MPT unreliable in practice.

**Governance layer** computes the Deflated Sharpe Ratio to correct for multiple testing across 8 models, monitors information coefficient decay, and runs statistical drift detection on live predictions.

---

## Interface

11 analysis tabs:

| Tab | What it shows |
|-----|---------------|
| Overview | Composite signal, score 0–100, 4-scenario price targets |
| ML Models | Per-model predictions, SHAP feature importance, ensemble consensus |
| Volatility | GARCH forecast, vol term structure, VaR / CVaR |
| Regime | Current HMM state, transition matrix, regime duration stats |
| Options | GEX, gamma flip level, vanna / charm flows, ATM Greeks |
| Sentiment | NLP scores, composite sentiment, top driving headlines |
| Monte Carlo | Full return distribution, downside / upside probabilities |
| Risk | Vol scale factor, drawdown governor status, position sizing |
| Fundamentals | Valuation, margins, growth, balance sheet, dividends |
| Watchlist | Saved tickers with quick re-analysis |
| Scenarios | Bull / base / bear / tail risk with probability weights |

---

## Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.11, FastAPI, Celery |
| ML | PyTorch (LSTM), XGBoost, LightGBM, arch (GARCH), hmmlearn, filterpy |
| Frontend | React 18, TypeScript, Recharts |
| Auth | AWS Cognito + TOTP MFA |
| Cache | Redis (AWS ElastiCache) |
| Database | PostgreSQL (AWS RDS) |
| Hosting | AWS ECS Fargate + CloudFront + S3 |
| Infrastructure | Terraform |
| CI/CD | GitHub Actions |

---

## Security

- MFA required on every login — TOTP enforced at both the Cognito and application layer
- Token revocation via Redis blocklist — logout is immediate, not expiry-based
- Brute force lockout after 5 failed attempts per IP, 30-minute cooldown
- AWS WAF with OWASP Top 10 rules, IP reputation lists, and rate limiting in front of CloudFront

---

## Running locally

```bash
cp .env.template .env
# Fill in required values

docker-compose up -d
curl http://localhost:8000/health

cd frontend
npm install --legacy-peer-deps
npm start
```

Required environment variables: `SECRET_KEY`, `DATABASE_URL`, `REDIS_URL`, `COGNITO_USER_POOL_ID`, `COGNITO_CLIENT_ID`, `AWS_ACCOUNT_ID`. Everything else is optional.

---

© 2026 Dileep Kumar Reddy Kapu. All Rights Reserved.
