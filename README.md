# QuantEdge v6.0
⚠ **Deployment Status**

Production deployment is currently in progress.

The market data layer is being migrated from Yahoo Finance to Polygon API to eliminate rate limits and improve reliability. The full AWS infrastructure (ECS Fargate, Redis, RDS, CloudFront, Cognito, Terraform) is already defined and containerized for production rollout.

This is my personal stock analysis platform. I built it because I wanted the kind of analysis that hedge funds use — not the watered-down stuff you get from retail tools like Yahoo Finance or Robinhood. I wanted to type a ticker, hit enter, and get a real institutional-grade report: regime detection, volatility forecasting, options flow, sentiment, risk-adjusted position sizing — all of it, in one place, in under a minute.

It lives at **https://quant.dileepkapu.com** and it's mine. Single user. No SaaS, no subscriptions, no ads.

Built in 2026 by Dileep Kumar Reddy Kapu.

---

## What it actually does

You type a ticker — AAPL, NVDA, SPY, whatever — and it runs 8 different analysis models simultaneously against 10 years of market data. By the time it's done (usually 15–60 seconds), you have a complete picture of that stock: where it's been, what regime it's in right now, where the models think it's going, how risky it is, and what size position makes sense given current volatility.

Here's what's running under the hood when you hit Analyze:

**It fetches real data first.** Price history going back 10 years, fundamentals (PE ratio, margins, debt, growth rates, all of it), the full options chain across 6 expiry dates, and the latest news headlines plus Reddit posts from r/wallstreetbets, r/investing, and r/stocks. All of this happens in parallel so it does not take forever.

**Then it computes over 200 features.** Things like 12-month minus 1-month momentum (the way quant funds actually measure momentum, not just "it went up"), RSI at 3 different timeframes, MACD, Bollinger Bands, volume-weighted average price deviation, a statistical measure called VPIN that estimates how much of the trading volume is coming from informed traders vs noise, and something called fractional differentiation which makes the price series stationary for ML models while preserving as much historical memory as possible.

**Then it labels the historical data properly.** Most ML models trained on stock data use a naive approach — they say "did the stock go up in the next 30 days?" But that is unrealistic because in real life you would have a stop-loss. So instead this uses triple-barrier labeling from Marcos Lopez de Prado's book. Each historical data point gets labeled based on which barrier was hit first: your profit target, your stop-loss, or time running out. Much more realistic.

**Then 8 models run:**

- **GJR-GARCH** — a volatility model that understands the "leverage effect": bad news causes bigger volatility spikes than good news of the same magnitude. This is just true of markets and most models ignore it.
- **Hidden Markov Model (5 states)** — figures out what market regime you are in right now. Bull low-vol, bull high-vol, mean-reverting, bear low-vol, bear high-vol. Each regime has different expected behavior and the model tells you the probability of transitioning to each other regime.
- **Kalman Filter** — strips out the noise from the price series to find the underlying trend. Tells you if momentum is strengthening or weakening.
- **Monte Carlo simulation (100,000 paths)** — simulates 100,000 possible futures for the stock over the next year using a jump-diffusion model that accounts for sudden crashes, not just smooth random walks. Gives you the full distribution: P10, P25, P50, P75, P90 outcomes.
- **LSTM neural network** — deep learning model trained on sequences of 60 days of features to predict returns at 5, 10, 21, 63, and 252 day horizons.
- **XGBoost + LightGBM** — gradient boosted tree models, which tend to be more interpretable than neural nets and often beat them on tabular financial data.
- **Ensemble** — combines all of the above with confidence scores and shows you where models agree vs disagree.
- **Claude Opus 4 (AI fallback)** — when the local ML models are not available, Claude acts as a reasoning engine. It gets the computed features and regime, and returns structured predictions. Not a magic 8-ball — it is given actual computed signals and reasons from them.

**Then the risk engine runs independently.** This is important: the risk engine is completely separate from the prediction models. It does not care what the alpha models say. It looks at the actual return history and computes CVaR (how bad is the average loss in the worst 5% of scenarios), figures out how much volatility targeting should scale your position size up or down, and checks whether the drawdown governor should kick in. If the stock is down 15% from its recent peak the governor starts reducing recommended position size. At -20% it halts entirely.

**Then portfolio construction.** Uses Hierarchical Risk Parity (HRP). The classic Markowitz approach requires inverting a covariance matrix which becomes numerically unstable and tends to concentrate everything in a few positions. HRP does not need matrix inversion — it clusters assets by correlation, then allocates risk hierarchically across clusters. More robust, especially in crisis periods when correlations all spike toward 1 simultaneously.

**Finally, a governance check.** Computes the Deflated Sharpe Ratio, which corrects the observed Sharpe for the fact that you tested 8 different models. If you try enough models one will look good by chance. DSR corrects for that. If it is positive, the strategy shows genuine alpha.

---

## The interface

It is a dark-themed web app with 11 tabs:

- **Overview** — the summary: signal (STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL), score 0–100, current price, predicted 1-year return, and a 4-scenario breakdown (bull / base / bear / tail risk) with target prices
- **ML Models** — all model predictions: LSTM forecasts at each horizon, XGBoost signal strength, LightGBM rank score, ensemble consensus, and the top features driving the prediction
- **Volatility** — GARCH output: current volatility forecast, the volatility term structure out to 1 year, VaR and CVaR numbers
- **Regime** — HMM analysis: current regime with confidence, how long regimes typically last, transition probabilities between all 5 states
- **Options** — GEX (Gamma Exposure), the gamma flip level, charm and vanna flows, full ATM Greeks
- **Sentiment** — NLP scores on news headlines and Reddit posts, composite sentiment score, top headlines that drove it
- **Monte Carlo** — the 100K path simulation: full distribution of 1-year outcomes, probability of loss, probability of 10%+ gain, probability of 20%+ gain
- **Risk** — risk engine output: vol scale factor, leverage signal, governor status, CVaR breakdown, recommended max position size
- **Fundamentals** — all financial data: valuation multiples, margins, growth rates, balance sheet health, dividend info
- **Watchlist** — your saved tickers, stored in Redis so they survive page refreshes and server restarts
- **Scenarios** — bull / base / bear / tail risk scenarios with full target prices and probabilities

No login required to run analysis. Login (with Google Authenticator MFA) is only needed to save your watchlist.

---

## How it is built

**Backend:** Python with FastAPI. Runs in a Docker container on AWS ECS Fargate — serverless containers, no EC2 instances to manage, no servers to patch.

**Frontend:** React 18 with TypeScript. Served from S3 via CloudFront CDN. Dark brown and gold design using Bebas Neue for headers, Fira Code for data and numbers, and Outfit for body text.

**Cache:** Redis on AWS ElastiCache. Handles analysis results (5 minute TTL), session tokens, login attempt tracking, IP lockout, and watchlist storage (90 day TTL). If the container restarts nothing is lost.

**Database:** PostgreSQL on AWS RDS. Reserved for future structured storage — current analysis data lives in Redis for speed.

**Auth:** AWS Cognito with mandatory MFA. Every login requires a 6-digit TOTP code from Google Authenticator. No way around it. The platform is also locked to a single username at the code level, not just the login screen.

**Infrastructure:** Everything defined in one Terraform file (`infrastructure/terraform/main.tf`). VPC, ECS cluster, load balancer, CloudFront distribution, RDS, ElastiCache, Cognito user pool, TLS certificate, DNS records, WAF rules, SNS alerts, Secrets Manager, IAM roles — all of it. One `terraform apply` command creates everything from scratch.

**CI/CD:** GitHub Actions. Every push to the main branch automatically builds Docker, pushes to ECR, deploys to ECS, builds React, uploads to S3, and clears the CDN cache. Zero manual steps after the first setup.

---

## Security

A few things worth knowing:

**MFA is not optional.** There is no code path that accepts a login without a TOTP challenge. It is enforced at the Cognito pool level and double-checked in the backend.

**Sessions can be instantly revoked.** When you log out, the token gets added to a Redis blocklist. Every single API request checks this list before doing anything. The token is dead immediately — not when it naturally expires in 8 hours.

**Brute force protection.** After 5 failed login attempts from the same IP, that IP is locked out for 30 minutes. An email alert fires immediately via SNS.

**Single owner enforced in code.** Even if someone somehow got a valid Cognito JWT for a different user, the authentication layer checks that the username matches the configured owner. Anyone else gets a 403 Forbidden — no exceptions.

**WAF in front.** AWS Managed Rule Sets covering OWASP Top 10, known malicious IPs, SQL injection, and XSS are attached to the CloudFront distribution. Rate limiting at 2000 requests per IP.

---

## Running it locally

```bash
# Copy and fill in your environment variables
cp .env.template .env

# Start PostgreSQL + Redis + API + Celery worker
docker-compose up -d

# Check the API is up
curl http://localhost:8000/health

# Start the frontend (in a new terminal)
cd frontend
npm install --legacy-peer-deps
npm start
# Opens at http://localhost:3000
```

The minimum required env vars to get running locally are `SECRET_KEY`, `DATABASE_URL`, `REDIS_URL`, `COGNITO_USER_POOL_ID`, `COGNITO_CLIENT_ID`, and `AWS_ACCOUNT_ID`. Everything else is optional and the platform degrades gracefully without them.

---

## Deploying to AWS

The full step-by-step guide with every command is in the **ChatGPT Deployment Guide** document. Short version:

1. Run `terraform apply` in `infrastructure/terraform/` — creates all 16 AWS resources in about 20 minutes
2. Create the Cognito user 'dileep' and enroll Google Authenticator for MFA
3. Build the Docker image and push to ECR
4. Add 5 secrets to GitHub and push to main — CI/CD handles all future deploys automatically
5. Visit https://quant.dileepkapu.com

Takes about 2 hours the first time. Monthly AWS cost is roughly $60–80.

---

## Files to update if you change the copyright year or owner details

The year "2024–2025" appears in these files and should be updated to reflect the correct year:

| File | What to change |
|------|----------------|
| `backend/main_v6.py` | Line 8: `© 2024–2025 Dileep Kumar Reddy Kapu` in the module docstring |
| `backend/core/config.py` | Line 3: `© 2024–2025 Dileep Kumar Reddy Kapu` in the module docstring |
| `backend/tasks.py` | Top of file: copyright comment |
| `backend/ml/data_pipeline/fetch_and_store.py` | Top of file: copyright comment |
| `backend/ml/training/train_lstm.py` | Top of file: copyright comment |
| `backend/ml/training/train_xgboost.py` | Top of file: copyright comment |
| `backend/ml/training/run_training.py` | Top of file: copyright comment |
| `docker-compose.yml` | Line 4: copyright comment |
| `README.md` | Bottom line: copyright footer |

Just open each file, find the line with `2024–2025` and change it to `2026`. That is all.

---

*© 2026 Dileep Kumar Reddy Kapu. All Rights Reserved. Proprietary & Confidential.*
