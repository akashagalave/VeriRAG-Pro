#!/usr/bin/env bash
set -Eeuo pipefail
cd /opt/verirag/frontend
docker compose --env-file release.env -f frontend-compose.yml ps
for attempt in $(seq 1 24); do
  if curl --fail --silent --show-error --max-time 5 'http://127.0.0.1:8501/_stcore/health' >/dev/null; then
    echo 'frontend passed local health check'
    exit 0
  fi
  sleep 5
done
echo 'frontend did not become healthy' >&2
exit 1
