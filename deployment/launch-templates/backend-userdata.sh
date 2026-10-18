#!/usr/bin/env bash
set -Eeuo pipefail

exec > >(tee -a /var/log/verirag-bootstrap.log | logger -t verirag-bootstrap -s 2>/dev/console) 2>&1

# ============================================================
# VeriRAG Backend EC2 Bootstrap
# OS: Ubuntu 24.04 LTS (x86_64)
# Purpose: Install Docker, AWS CLI v2 and CodeDeploy agent
# Application deployment is handled separately by CodeDeploy.
# ============================================================

export DEBIAN_FRONTEND=noninteractive
REGION="ap-south-1"

echo "Starting VeriRAG backend bootstrap..."

# ------------------------------------------------------------
# 1. Update Ubuntu and install required packages
# ------------------------------------------------------------

apt-get update

apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  wget \
  unzip \
  docker.io \
  docker-compose-v2

# ------------------------------------------------------------
# 2. Enable Docker
# ------------------------------------------------------------

systemctl enable --now docker

docker --version
docker compose version

# ------------------------------------------------------------
# 3. Install AWS CLI v2
# Ubuntu 24.04 AMI may not provide awscli through APT.
# ------------------------------------------------------------

echo "Installing AWS CLI v2..."

aws_dir="$(mktemp -d)"

curl --fail --location --silent --show-error \
  "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
  --output "$aws_dir/awscliv2.zip"

unzip -q "$aws_dir/awscliv2.zip" -d "$aws_dir"

"$aws_dir/aws/install"

rm -rf "$aws_dir"

aws --version

# ------------------------------------------------------------
# 4. Prepare Backend Deployment Directory
# ------------------------------------------------------------

install -d -m 0755 /opt/verirag/backend

# ------------------------------------------------------------
# 5. Install AWS CodeDeploy Agent v2
# ------------------------------------------------------------

echo "Installing CodeDeploy agent..."

codedeploy_dir="$(mktemp -d)"

curl --fail --location --silent --show-error \
  "https://aws-codedeploy-${REGION}.s3.${REGION}.amazonaws.com/latestv2/install" \
  --output "$codedeploy_dir/install"

chmod 700 "$codedeploy_dir/install"

(
  cd "$codedeploy_dir"
  ./install auto
)

rm -rf "$codedeploy_dir"

systemctl enable --now codedeploy-agent

# ------------------------------------------------------------
# 6. Validate Installed Services
# ------------------------------------------------------------

echo "Validating installed services..."

command -v aws
command -v docker
docker compose version

systemctl is-active --quiet docker
systemctl is-active --quiet codedeploy-agent

echo "Docker: ACTIVE"
echo "CodeDeploy agent: ACTIVE"

echo "VeriRAG backend bootstrap completed successfully."
