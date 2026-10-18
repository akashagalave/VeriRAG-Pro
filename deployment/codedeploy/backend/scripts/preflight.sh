#!/usr/bin/env bash
set -Eeuo pipefail
command -v docker >/dev/null
command -v aws >/dev/null
command -v python3 >/dev/null
docker compose version >/dev/null
systemctl is-active --quiet docker
install -d -m 0755 /opt/verirag/backend
