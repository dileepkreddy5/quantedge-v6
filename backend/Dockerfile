# ============================================================
# QuantEdge v6.0 — Production Dockerfile
# © 2024–2025 Dileep Kumar Reddy Kapu. All Rights Reserved.
# ============================================================
# Build: docker build -t quantedge-api:latest .
# Run:   docker run -p 8000:8000 --env-file .env quantedge-api:latest

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential libpq-dev curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# ── Core Web Framework ────────────────────────────────────────────────────────

RUN pip install --no-cache-dir \
    fastapi==0.111.0 uvicorn[standard]==0.29.0 python-multipart==0.0.9 httpx==0.27.0 \
    requests==2.31.0 aiohttp==3.9.5

# ── Auth & AWS ────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    python-jose[cryptography]==3.3.0 passlib[bcrypt]==1.7.4 \
    boto3==1.34.69 cryptography==42.0.5 pyotp==2.9.0

# ── Database & Cache ──────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    asyncpg==0.29.0 sqlalchemy[asyncio]==2.0.29 alembic==1.13.1 \
    redis==5.0.3 aioredis==2.0.1 psycopg2-binary==2.9.9

# ── Config & Utils ────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    pydantic==2.7.0 pydantic-settings==2.2.1 loguru==0.7.2 \
    tenacity==8.2.3 python-dotenv==1.0.1 celery[redis]==5.3.6 joblib==1.3.2

# ── Data & Market Feeds ───────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    yfinance==0.2.38 pandas==2.2.1 numpy==1.24.4 scipy==1.13.0 \
    fredapi==0.5.1 newsapi-python==0.2.7 praw==7.7.1


# ── ML Core ───────────────────────────────────────────────────────────────────

# Force NumPy 1.x for ML compatibility
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scikit-learn==1.4.2 \
    xgboost==2.0.3 \
    lightgbm==4.3.0 \
    statsmodels==0.14.2 \
    arch==6.3.0 \
    hmmlearn==0.3.2 \
    ta==0.11.0 \
    shap>=0.45.0
# ── Optimization ─────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir cvxpy==1.4.2

# ── NLP / Sentiment ───────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    transformers==4.40.1 tokenizers==0.19.1 \
    vaderSentiment==3.3.2 textblob==0.18.0 nltk==3.8.1

# ── AI API ────────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir anthropic>=0.30.0

# ── PyTorch CPU-only (large layer — keep last before COPY) ────────────────────
RUN pip install --no-cache-dir torch==2.2.2

# ── App Code ─────────────────────────────────────────────────────────────────
COPY . .

RUN mkdir -p /app/ml/saved_models/lstm /app/ml/saved_models/xgboost

RUN useradd --create-home --shell /bin/bash quantedge \
    && chown -R quantedge:quantedge /app
USER quantedge


EXPOSE 8000

CMD ["uvicorn", "main_v6:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
