#!/usr/bin/env bash
set -Eeuo pipefail
cd /opt/verirag/frontend
# GitHub must package release.env with IMAGE_TAG=<40-character commit SHA>.
[[ -f release.env ]] || { echo 'Missing release.env' >&2; exit 1; }
set -a
# Only allow SHA-based image tags from the release manifest.
IMAGE_TAG="$(sed -n 's/^IMAGE_TAG=//p' release.env | head -n1)"
[[ "$IMAGE_TAG" =~ ^[a-f0-9]{40}$ ]] || { echo 'Invalid IMAGE_TAG (expected 40 hex characters)' >&2; exit 1; }
export IMAGE_TAG
set +a
registry='372806411232.dkr.ecr.ap-south-1.amazonaws.com'
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin "$registry" >/dev/null
# Pull and run only the service assigned to this Auto Scaling group.
docker compose --env-file release.env -f frontend-compose.yml pull
docker compose --env-file release.env -f frontend-compose.yml up -d --no-build --remove-orphans
