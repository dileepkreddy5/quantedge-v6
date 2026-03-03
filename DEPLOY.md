# QuantEdge v5.0 — Complete Deployment Guide
## quant.dileepkapu.com | Owner: Dileep Kumar Reddy Kapu

---

## 📁 Repository Structure

```
quantedge/
├── backend/                    ← FastAPI Python API
│   ├── main.py                 ← App entry point
│   ├── requirements.txt        ← Python dependencies
│   ├── Dockerfile              ← Container build
│   ├── core/config.py          ← All settings (reads from env)
│   ├── auth/cognito_auth.py    ← JWT + Cognito validation
│   ├── routers/
│   │   ├── auth_router.py      ← Login/MFA/logout
│   │   ├── analysis.py         ← Main analysis endpoint
│   │   ├── watchlist.py        ← Personal watchlist
│   │   └── portfolio.py        ← Portfolio tracker
│   ├── ml/
│   │   ├── features/feature_engineering.py  ← 200+ features
│   │   └── models/
│   │       ├── lstm_model.py          ← BiLSTM + Attention
│   │       ├── xgboost_lgbm.py        ← GBDT ensemble + SHAP
│   │       ├── regime_volatility.py   ← HMM + GJR-GARCH + Kalman + MC
│   │       └── nlp_options.py         ← FinBERT + BSM Greeks + GEX
│   └── data/feeds/market_data.py      ← yFinance + Alpha Vantage
│
├── frontend/                   ← React TypeScript SPA
│   ├── src/
│   │   ├── App.tsx             ← Router
│   │   ├── index.tsx           ← Entry point
│   │   ├── auth/authStore.ts   ← Zustand auth + Axios client
│   │   ├── pages/
│   │   │   ├── Login.tsx       ← Login + MFA (chocolate aesthetic)
│   │   │   └── Dashboard.tsx   ← Main dashboard (10 tabs)
│   │   ├── components/
│   │   │   ├── ui/Panels.tsx   ← All 10 analysis panels
│   │   │   └── charts/PriceChart.tsx ← Recharts price chart
│   │   └── styles/globals.css  ← Design tokens (matches dileepkapu.com)
│   └── package.json
│
├── infrastructure/terraform/main.tf  ← Complete AWS infra (IaC)
├── .github/workflows/deploy.yml      ← CI/CD pipeline
└── DEPLOY.md                         ← This file
```

---

## 🚀 Phase 1: DNS Setup (5 minutes — FREE, do this first)

### Add CNAME to your DNS (Namecheap/Cloudflare)
```
Type:  CNAME
Host:  quant
Value: [your-cloudfront-domain].cloudfront.net
TTL:   Automatic
```
**After Terraform runs**, copy the `cloudfront_domain` output and add it here.

---

## 🔧 Phase 2: AWS Prerequisites (30 minutes)

### 1. Install tools
```bash
# macOS (Homebrew)
brew install awscli terraform

# Verify
aws --version     # aws-cli/2.x
terraform --version  # v1.7+
```

### 2. Create AWS IAM user for Terraform
1. AWS Console → IAM → Users → Create User: `quantedge-terraform`
2. Attach policy: `AdministratorAccess` (for initial setup)
3. Create access key → download CSV

### 3. Configure AWS CLI
```bash
aws configure
# AWS Access Key ID: [from CSV]
# AWS Secret Access Key: [from CSV]
# Default region: us-east-1
# Default output: json
```

### 4. Create Terraform state bucket (one time)
```bash
aws s3 mb s3://quantedge-terraform-state-dileep --region us-east-1
aws s3api put-bucket-versioning --bucket quantedge-terraform-state-dileep --versioning-configuration Status=Enabled
aws dynamodb create-table \
  --table-name quantedge-terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

---

## 🏗 Phase 3: Terraform Infrastructure (20 minutes)

```bash
cd infrastructure/terraform

# Initialize
terraform init

# Preview (no changes yet)
terraform plan \
  -var="db_password=YourStrongPassword123!" \
  -var="secret_key=$(openssl rand -hex 32)"

# Apply (creates all AWS resources)
terraform apply \
  -var="db_password=YourStrongPassword123!" \
  -var="secret_key=$(openssl rand -hex 32)" \
  -auto-approve

# Save outputs
terraform output > ../outputs.txt
cat ../outputs.txt
```

**Resources created:**
- VPC, subnets, NAT Gateway
- ALB, ACM SSL certificate
- Cognito User Pool (MFA enforced)
- RDS PostgreSQL, ElastiCache Redis
- ECR repository, ECS Fargate cluster
- S3 buckets (frontend, data lake, models)
- CloudFront distribution
- Route53 DNS record
- WAF, Secrets Manager, SNS, CloudWatch

**Cost estimate:** ~$35-50/month initially

---

## 🔑 Phase 4: Create Your Account (5 minutes)

```bash
# Get Cognito User Pool ID from Terraform output
POOL_ID=$(terraform -chdir=infrastructure/terraform output -raw cognito_user_pool_id)

# Create your user
aws cognito-idp admin-create-user \
  --user-pool-id $POOL_ID \
  --username dileep \
  --user-attributes Name=email,Value=dileep@dileepkapu.com Name=email_verified,Value=true \
  --temporary-password "TempPass123!" \
  --message-action SUPPRESS

# Set permanent password
aws cognito-idp admin-set-user-password \
  --user-pool-id $POOL_ID \
  --username dileep \
  --password "YourRealPassword123!" \
  --permanent

# Set up MFA (TOTP)
aws cognito-idp admin-set-user-mfa-preference \
  --user-pool-id $POOL_ID \
  --username dileep \
  --software-token-mfa-settings Enabled=true,PreferredMfa=true

echo "✅ User created. Now set up Google Authenticator."
```

### Associate Google Authenticator
```bash
# Get MFA secret (scan QR or enter manually in Google Auth)
aws cognito-idp associate-software-token \
  --user-pool-id $POOL_ID \
  --username dileep
# Returns: SecretCode — enter this in Google Authenticator
# App name to use: QuantEdge
```

---

## 🔐 Phase 5: Configure API Secrets (5 minutes)

```bash
# Update secrets in AWS Secrets Manager
aws secretsmanager update-secret \
  --secret-id quantedge/app-secrets \
  --secret-string '{
    "SECRET_KEY": "'$(openssl rand -hex 32)'",
    "ALPHA_VANTAGE_KEY": "YOUR_KEY_FROM_alphavantage.co",
    "FRED_API_KEY": "YOUR_KEY_FROM_fred.stlouisfed.org/docs/api",
    "ANTHROPIC_API_KEY": "sk-ant-YOUR_KEY",
    "REDDIT_CLIENT_ID": "OPTIONAL_REDDIT_APP_ID",
    "REDDIT_SECRET": "OPTIONAL_REDDIT_SECRET",
    "POLYGON_API_KEY": "OPTIONAL_polygon.io_KEY",
    "TRADIER_API_KEY": "OPTIONAL_tradier.com_KEY"
  }'

# Free API keys to get:
# Alpha Vantage: https://www.alphavantage.co/support/#api-key (free, 25 calls/day)
# FRED:          https://fred.stlouisfed.org/docs/api/api_key.html (free)
# Anthropic:     https://console.anthropic.com (pay per use)
# Reddit:        https://www.reddit.com/prefs/apps (free, optional)
```

---

## 🐳 Phase 6: Build & Deploy Backend (15 minutes)

```bash
# Get ECR URL
ECR_URL=$(terraform -chdir=infrastructure/terraform output -raw ecr_repository_url)

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $ECR_URL

# Build image
cd backend
docker build -t quantedge-api:latest .

# Tag and push
docker tag quantedge-api:latest $ECR_URL:latest
docker push $ECR_URL:latest

# Force ECS to pick up new image
aws ecs update-service \
  --cluster quantedge-cluster \
  --service quantedge-api \
  --force-new-deployment

# Watch deployment progress
aws ecs wait services-stable \
  --cluster quantedge-cluster \
  --services quantedge-api

echo "✅ Backend deployed!"
```

---

## ⚛️ Phase 7: Build & Deploy Frontend (10 minutes)

```bash
cd frontend

# Install dependencies
npm install

# Build for production
REACT_APP_API_URL=https://quant.dileepkapu.com npm run build

# Get bucket name
S3_BUCKET=$(terraform -chdir=../infrastructure/terraform output -raw s3_frontend_bucket)
CF_ID=$(terraform -chdir=../infrastructure/terraform output -raw cloudfront_distribution_id 2>/dev/null || echo $CLOUDFRONT_DISTRIBUTION_ID)

# Upload to S3
aws s3 sync build/ s3://$S3_BUCKET/ --delete \
  --cache-control "no-cache" --include "*.html" --exclude "*"
aws s3 sync build/ s3://$S3_BUCKET/ --delete \
  --cache-control "max-age=31536000,immutable" --exclude "*.html"

# Invalidate CloudFront cache
aws cloudfront create-invalidation \
  --distribution-id $CF_ID \
  --paths "/*"

echo "✅ Frontend deployed to https://quant.dileepkapu.com"
```

---

## 🔄 Phase 8: Set Up CI/CD (GitHub Actions)

```bash
# Add GitHub Secrets (Settings → Secrets → Actions):
gh secret set AWS_ACCESS_KEY_ID --body "YOUR_KEY_ID"
gh secret set AWS_SECRET_ACCESS_KEY --body "YOUR_SECRET"
gh secret set CLOUDFRONT_DISTRIBUTION_ID --body "E1234EXAMPLE"
gh secret set SNS_TOPIC_ARN --body "arn:aws:sns:us-east-1:..."
```

After this, every `git push` to `main` auto-deploys both frontend and backend.

---

## ✅ Phase 9: Verify Everything Works

```bash
# 1. Check backend health
curl https://quant.dileepkapu.com/health

# Expected: {"status":"healthy","version":"5.0.0","redis":"connected",...}

# 2. Test login (should get MFA challenge)
curl -X POST https://quant.dileepkapu.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"dileep","password":"YourPassword"}'

# 3. Open browser
open https://quant.dileepkapu.com
# → Should redirect to login page
# → Enter credentials
# → Enter Google Authenticator 6-digit code
# → Dashboard loads!

# 4. Test analysis
# → Type "AAPL" in search bar
# → Click ANALYZE
# → Wait 15-60 seconds
# → Full institutional report appears
```

---

## 🧠 Phase 10: Train Real ML Models on SageMaker (Optional, Week 3-4)

```bash
# Install SageMaker SDK
pip install sagemaker boto3

# Run training scripts (one at a time)
python ml_training/lstm/train_lstm.py
python ml_training/xgboost/train_xgboost.py

# Deploy as SageMaker endpoints
python ml_training/ensemble/deploy_all.py

# Update environment variables in ECS task definition with endpoint names
```

Until SageMaker models are deployed, the system uses **Claude API** as the intelligent ML fallback (fully functional, just slower and pay-per-call).

---

## 💰 Cost Breakdown (Monthly)

| Service | Spec | Cost |
|---------|------|------|
| ECS Fargate | 1 vCPU, 2GB RAM | ~$20 |
| RDS PostgreSQL | db.t3.small | ~$25 |
| ElastiCache Redis | cache.t3.micro | ~$13 |
| ALB | 1 LCU avg | ~$18 |
| CloudFront | ~10GB/mo | ~$1 |
| S3 (all buckets) | ~50GB | ~$2 |
| Cognito | <50K MAU | $0 |
| CloudWatch | 5GB logs | ~$3 |
| WAF | 3 rules | ~$5 |
| Route53 | 1 zone | ~$1 |
| **TOTAL** | | **~$88/month** |

Claude API (fallback): ~$5-20/month depending on usage

SageMaker (when added): +$35-80/month

---

## 🛡 Security Checklist

- ✅ MFA enforced (TOTP, not SMS)
- ✅ JWT in httpOnly cookies (XSS-safe)
- ✅ All API keys in Secrets Manager
- ✅ RDS/Redis in private subnets only
- ✅ WAF rate limiting + IP reputation
- ✅ CloudTrail audit logging
- ✅ TLS 1.3 everywhere
- ✅ ECR image scanning on push
- ✅ Deletion protection on RDS
- ✅ SNS email alerts on login failures

---

## 🔗 Quick Reference URLs

- **Your Platform:** https://quant.dileepkapu.com
- **Portfolio Site:** https://dileepkapu.com
- **AWS Console:** https://console.aws.amazon.com
- **Free API Keys:**
  - Alpha Vantage: https://alphavantage.co
  - FRED: https://fred.stlouisfed.org/docs/api/api_key.html
  - Reddit: https://reddit.com/prefs/apps
- **Anthropic:** https://console.anthropic.com
- **Polygon.io (paid):** https://polygon.io/pricing

---

*Built for Dileep Kumar Reddy Kapu | Senior Data Engineer | AWS & Azure Certified*
*QuantEdge v5.0 · 8 ML Models · 200+ Features · Institutional Grade*
