#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# QuantEdge Price Oracle — AWS Deployment Script
# Run this on your EC2 instance or inside your ECS container build
# ═══════════════════════════════════════════════════════════════════════════════

set -e

echo "╔══════════════════════════════════════════════════════╗"
echo "║   QuantEdge Price Oracle — Installing Dependencies  ║"
echo "╚══════════════════════════════════════════════════════╝"

# ── 1. Python dependencies ────────────────────────────────────────────────────
echo ""
echo "▶ Installing Python packages..."

pip install \
    yfinance==0.2.54 \
    arch==7.0.0 \
    scikit-learn==1.4.0 \
    scipy==1.13.0 \
    numpy==1.26.4 \
    pandas==2.2.0 \
    lightgbm==4.3.0 \
    httpx==0.27.0 \
    fastapi==0.111.0 \
    uvicorn==0.29.0 \
    pydantic==2.7.0 \
    ta-lib-easy \
    --upgrade

echo "✓ Dependencies installed"

# ── 2. Verify imports work ────────────────────────────────────────────────────
echo ""
echo "▶ Verifying imports..."
python3 -c "
import yfinance, arch, sklearn, scipy, numpy, pandas, lightgbm, httpx, fastapi
print('✓ yfinance:', yfinance.__version__)
print('✓ arch:', arch.__version__)
print('✓ scikit-learn:', sklearn.__version__)
print('✓ scipy:', scipy.__version__)
print('✓ numpy:', numpy.__version__)
print('✓ pandas:', pandas.__version__)
print('✓ lightgbm:', lightgbm.__version__)
print('✓ All imports OK')
"

# ── 3. Integration into existing FastAPI app ──────────────────────────────────
echo ""
echo "▶ Integration instructions:"
cat << 'INSTRUCTIONS'

Add these 3 lines to your existing backend/main.py:

    from ml.price_oracle.router import router as oracle_router
    app.include_router(oracle_router)

That's it. The endpoint is immediately available at:
    POST /api/v1/oracle/predict
    GET  /api/v1/oracle/health
    GET  /api/v1/oracle/methodology

INSTRUCTIONS

echo "✓ Setup complete"
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Test with:                                         ║"
echo "║   curl -X POST http://localhost:8000/api/v1/oracle/predict  ║"
echo "║        -H 'Content-Type: application/json'           ║"
echo "║        -d '{\"ticker\":\"AAPL\",\"account_size\":25000}'  ║"
echo "╚══════════════════════════════════════════════════════╝"
