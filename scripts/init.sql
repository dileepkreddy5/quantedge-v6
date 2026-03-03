-- QuantEdge v6.0 — PostgreSQL Init Script
-- © 2024–2025 Dileep Kumar Reddy Kapu. All Rights Reserved.
-- Runs automatically on first docker-compose up

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE quantedge TO quantedge;
