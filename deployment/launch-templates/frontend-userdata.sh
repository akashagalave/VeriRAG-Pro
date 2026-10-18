#!/usr/bin/env bash
set -Eeuo pipefail
exec > >(tee -a /var/log/verirag-bootstrap.log | logger -t verirag-bootstrap -s 2>/dev/console) 2>&1

# Launch Template user data: Ubuntu 24.04 LTS, frontend EC2 instance.
# CodeDeploy delivers the Streamlit image and runtime configuration.
export DEBIAN_FRONTEND=noninteractive
REGION="ap-south-1"

apt-get update
apt-get install -y --no-install-recommends ca-certificates curl wget docker.io docker-compose-v2 awscli
systemctl enable --now docker
install -d -m 0755 /opt/verirag/frontend

installer_dir="$(mktemp -d)"
cd "$installer_dir"
curl --fail --location --silent --show-error \
  "https://aws-codedeploy-${REGION}.s3.${REGION}.amazonaws.com/latestv2/install" \
  --output install
chmod 700 install
./install auto
systemctl enable --now codedeploy-agent
rm -rf "$installer_dir"

systemctl is-active --quiet docker
systemctl is-active --quiet codedeploy-agent
printf 'VeriRAG frontend bootstrap complete\n'
