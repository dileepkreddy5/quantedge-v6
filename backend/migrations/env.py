"""Alembic environment for QuantEdge v6.0"""
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os

config = context.config

# Override sqlalchemy.url from environment
# Support both DATABASE_URL and DB_* components (same logic as core.config.effective_database_url)
_direct_url = os.environ.get("DATABASE_URL", "")
if _direct_url:
    db_url = _direct_url.replace("postgresql+asyncpg://", "postgresql://")
else:
    _h = os.environ.get("DB_HOST"); _p = os.environ.get("DB_PORT", "5432")
    _n = os.environ.get("DB_NAME"); _u = os.environ.get("DB_USER"); _pw = os.environ.get("DB_PASSWORD")
    db_url = f"postgresql://{_u}:{_pw}@{_h}:{_p}/{_n}" if all([_h, _n, _u, _pw]) else ""
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    connectable = engine_from_config(config.get_section(config.config_ini_section, {}), prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
