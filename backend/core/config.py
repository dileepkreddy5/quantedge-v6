"""
QuantEdge v6.0 — Core Configuration
© 2026 Dileep Kumar Reddy Kapu. All Rights Reserved.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List
import os


class Settings(BaseSettings):
    # ─── App ──────────────────────────────────────────────
    APP_NAME: str = "QuantEdge"
    APP_VERSION: str = "6.0.0"
    ENVIRONMENT: str = Field(default="production", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    SECRET_KEY: str = Field(..., env="SECRET_KEY")

    # ─── CORS ─────────────────────────────────────────────
    # Used as settings.CORS_ORIGINS in main_v6.py
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000",
        env="CORS_ORIGINS"
    )

    # ─── AWS ──────────────────────────────────────────────
    AWS_REGION: str = Field(default="us-east-1", env="AWS_REGION")
    AWS_ACCOUNT_ID: str = Field(default="", env="AWS_ACCOUNT_ID")

    # ─── Cognito ──────────────────────────────────────────
    COGNITO_USER_POOL_ID: str = Field(..., env="COGNITO_USER_POOL_ID")
    COGNITO_CLIENT_ID: str = Field(..., env="COGNITO_CLIENT_ID")   # maps from COGNITO_APP_CLIENT_ID in .env
    COGNITO_REGION: str = Field(default="us-east-1", env="COGNITO_REGION")
    JWT_ALGORITHM: str = "RS256"
    ACCESS_TOKEN_EXPIRE_HOURS: int = 8
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # ─── Database ─────────────────────────────────────────
    # DATABASE_URL can be provided directly OR constructed from components
    # (DB_HOST + DB_PORT + DB_NAME + DB_USER + DB_PASSWORD). Component approach
    # is preferred — secrets never appear as a full URL in task definitions.
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")
    DB_HOST: Optional[str] = Field(default=None, env="DB_HOST")
    DB_PORT: str = Field(default="5432", env="DB_PORT")
    DB_NAME: Optional[str] = Field(default=None, env="DB_NAME")
    DB_USER: Optional[str] = Field(default=None, env="DB_USER")
    DB_PASSWORD: Optional[str] = Field(default=None, env="DB_PASSWORD")

    @property
    def effective_database_url(self) -> Optional[str]:
        """Return DATABASE_URL if set, otherwise construct from components.
        Returns None if neither path yields a valid URL."""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        if all([self.DB_HOST, self.DB_NAME, self.DB_USER, self.DB_PASSWORD]):
            return (
                f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
                f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
            )
        return None

    # ─── Redis ────────────────────────────────────────────
    REDIS_URL: str = Field(..., env="REDIS_URL")
    SESSION_TTL_SECONDS: int = 28800
    RATE_LIMIT_PER_MINUTE: int = 200

    # ─── AI / ML APIs ─────────────────────────────────────
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")

    # ─── Data APIs (all optional — graceful fallback) ─────
    ALPHA_VANTAGE_KEY: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_KEY")
    POLYGON_API_KEY: Optional[str] = Field(default=None, env="POLYGON_API_KEY")
    FRED_API_KEY: Optional[str] = Field(default=None, env="FRED_API_KEY")
    NEWSAPI_KEY: Optional[str] = Field(default=None, env="NEWSAPI_KEY")
    REDDIT_CLIENT_ID: Optional[str] = Field(default=None, env="REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET: Optional[str] = Field(default=None, env="REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT: str = Field(default="QuantEdge/6.0 by Dileep", env="REDDIT_USER_AGENT")

    # ─── Owner (single-user platform) ─────────────────────
    OWNER_USERNAME: str = Field(default="dileep", env="OWNER_USERNAME")
    OWNER_EMAIL: str = Field(default="dileep@dileepkapu.com", env="OWNER_EMAIL")

    # ─── SageMaker (optional — used by v5 analysis.py) ────
    SAGEMAKER_LSTM_ENDPOINT: Optional[str] = Field(default=None, env="SAGEMAKER_LSTM_ENDPOINT")
    SAGEMAKER_XGB_ENDPOINT: Optional[str] = Field(default=None, env="SAGEMAKER_XGB_ENDPOINT")
    SAGEMAKER_LGBM_ENDPOINT: Optional[str] = Field(default=None, env="SAGEMAKER_LGBM_ENDPOINT")
    SAGEMAKER_TFT_ENDPOINT: Optional[str] = Field(default=None, env="SAGEMAKER_TFT_ENDPOINT")
    SAGEMAKER_FINBERT_ENDPOINT: Optional[str] = Field(default=None, env="SAGEMAKER_FINBERT_ENDPOINT")
    SAGEMAKER_RUNTIME_REGION: str = Field(default="us-east-1", env="SAGEMAKER_RUNTIME_REGION")
    USE_ANTHROPIC_FALLBACK: bool = Field(default=True, env="USE_ANTHROPIC_FALLBACK")

    # ─── S3 ───────────────────────────────────────────────
    S3_BUCKET_DATA: str = Field(default="quantedge-datalake", env="S3_BUCKET_DATA")
    S3_BUCKET_MODELS: str = Field(default="quantedge-models", env="S3_BUCKET_MODELS")
    S3_BUCKET_FRONTEND: str = Field(default="quantedge-frontend", env="S3_BUCKET_FRONTEND")

    # ─── Alerts (optional) ────────────────────────────────
    SNS_ALERT_TOPIC_ARN: Optional[str] = Field(default=None, env="SNS_ALERT_TOPIC_ARN")

    # ─── Security ─────────────────────────────────────────
    MAX_LOGIN_ATTEMPTS: int = 5
    LOCKOUT_DURATION_SECONDS: int = 1800
    ALLOWED_IPS: Optional[List[str]] = Field(default=None, env="ALLOWED_IPS")

    # ─── ML Config ────────────────────────────────────────
    MODEL_DIR: str = Field(default="/app/ml/saved_models", env="MODEL_DIR")
    MONTE_CARLO_PATHS: int = 10_000
    SEQUENCE_LENGTH: int = 60
    FEATURE_LOOKBACK_DAYS: int = 252

    class Config:
        env_file = ".env"
        case_sensitive = True
        # Allow COGNITO_APP_CLIENT_ID in .env to map to COGNITO_CLIENT_ID
        env_prefix = ""


settings = Settings()

# ─── Logging Setup ────────────────────────────────────────
from loguru import logger
import sys

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO" if not settings.DEBUG else "DEBUG",
)
# Only add file logger if directory exists
log_dir = "/var/log/quantedge"
if os.path.exists(log_dir):
    logger.add(
        f"{log_dir}/app.log",
        rotation="100 MB",
        retention="30 days",
        compression="gz",
        level="INFO",
    )
