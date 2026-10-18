#!/usr/bin/env bash
set -Eeuo pipefail
cd /opt/verirag/backend
docker compose --env-file release.env -f backend-compose.yml ps
for attempt in $(seq 1 24); do
  if curl --fail --silent --show-error --max-time 5 'http://127.0.0.1:8000/health' >/dev/null; then
    echo 'backend passed local health check'
    exit 0
  fi
  sleep 5
done
echo 'backend did not become healthy' >&2
exit 1
