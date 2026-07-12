#!/usr/bin/env bash
# QuantEdge — nightly Postgres backup (step-8)
# Dumps the quantedge DB from the postgres container, gzips it, verifies the
# archive, rotates 14 days. Runs from cron at 08:00 UTC (04:00 ET, after the
# 02:00 ET scan jobs). Restore: gunzip -c FILE | docker compose exec -T postgres psql -U quantedge -d quantedge
set -euo pipefail

COMPOSE_DIR=/opt/quantedge
BACKUP_DIR=/root/backups
mkdir -p "$BACKUP_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
FILE="$BACKUP_DIR/quantedge_${STAMP}.sql.gz"

cd "$COMPOSE_DIR"
docker compose exec -T postgres pg_dump -U "${POSTGRES_USER:-quantedge}" -d "${POSTGRES_DB:-quantedge}" | gzip > "$FILE"

# integrity check — a corrupt gzip means the backup is worthless
gunzip -t "$FILE"

# rotate: keep 14 days
find "$BACKUP_DIR" -name "quantedge_*.sql.gz" -mtime +14 -delete

SIZE=$(du -h "$FILE" | cut -f1)
echo "$(date -Is) backup OK: $FILE ($SIZE)" >> "$BACKUP_DIR/backup.log"
echo "backup OK: $FILE ($SIZE)"
