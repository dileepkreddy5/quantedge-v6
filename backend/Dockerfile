###############################################################################
#  QuantEdge v6.0 Backend — Personal Use Dockerfile
#
#  Changes vs enterprise version:
#    - CMD uses 1 worker (was 2) — enough for personal use, saves 256MB RAM
#    - Redis is NOT installed here — it runs as a sidecar in ECS
#    - All other packages identical — same ML libraries
###############################################################################

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install all Python packages
# NOTE: torch CPU-only to save image size (~1.5GB vs ~3.5GB with CUDA)
# Personal use on Fargate does not have a GPU anyway

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pydantic-settings \
    xgboost \
    lightgbm \
    transformers \
    sentencepiece \
    arch \
    hmmlearn \
    filterpy \
    scipy \
    scikit-learn \
    numpy \
    pandas \
    shap \
    joblib \
    polygon-api-client \
    anthropic \
    asyncpg \
    aioredis \
    loguru \
    "python-jose[cryptography]" \
    "passlib[bcrypt]" \
    apscheduler \
    python-multipart \
    httpx

# Create model directory — trained models persist here across uvicorn restarts
# NOTE: In ECS, this directory is in-container — models are retrained if task restarts
# For personal use this is fine: training takes ~5 minutes and happens once per task lifetime
RUN mkdir -p /app/models

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

COPY --chown=appuser:appuser . .

EXPOSE 8000

# 1 worker — sufficient for personal use (1 person)
# Uses less memory than 2 workers (saves ~256MB)
CMD ["uvicorn", "main_v6:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
