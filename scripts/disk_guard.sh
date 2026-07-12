#!/usr/bin/env bash
# QuantEdge — disk guard (step-10)
# Every 6h via cron: if root disk >80%, prune dangling Docker artifacts
# (safe: never touches running containers, named volumes, or tagged images
# in use). Logs a WARNING if still >85% after pruning — that means the
# EDGAR zip / backups / logs need manual attention.
set -euo pipefail
LOG=/root/backups/disk_guard.log

usage() { df --output=pcent / | tail -1 | tr -dc '0-9'; }

U=$(usage)
if [ "$U" -gt 80 ]; then
    echo "$(date -Is) disk at ${U}% — pruning docker" >> "$LOG"
    docker system prune -f >> "$LOG" 2>&1
    docker builder prune -f --keep-storage 10GB >> "$LOG" 2>&1
    U=$(usage)
fi

if [ "$U" -gt 85 ]; then
    echo "$(date -Is) WARNING: disk still at ${U}% after prune — manual cleanup needed" >> "$LOG"
else
    echo "$(date -Is) disk OK: ${U}%" >> "$LOG"
fi
