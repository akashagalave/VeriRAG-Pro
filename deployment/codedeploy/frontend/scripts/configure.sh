#!/usr/bin/env bash
set -Eeuo pipefail

cd /opt/verirag/frontend
umask 077

REGION="ap-south-1"

# 1. Get backend URL from SSM
url="$(aws ssm get-parameter \
  --name '/verirag/production/backend-url' \
  --region "$REGION" \
  --query 'Parameter.Value' \
  --output text)"

[[ "$url" =~ ^https?://[^[:space:]]+$ ]] || {
  echo "Invalid backend URL" >&2
  exit 1
}

# 2. Get OpenAI API key securely from Secrets Manager
secret_json="$(aws secretsmanager get-secret-value \
  --secret-id api-keys \
  --region "$REGION" \
  --query SecretString \
  --output text)"

openai_key="$(printf '%s' "$secret_json" | python3 -c '
import json, sys
key = json.load(sys.stdin).get("OPENAI_API_KEY", "")
if not isinstance(key, str) or not key or "\n" in key or "\r" in key:
    sys.exit("Missing or invalid OPENAI_API_KEY")
print(key, end="")
')"

unset secret_json

# 3. Validate release.env
grep -Eq '^IMAGE_TAG=[a-f0-9]{40}$' release.env
[[ "$(wc -l < release.env)" -eq 1 ]] || {
  echo "Unexpected release.env format" >&2
  exit 1
}

# 4. Write runtime configuration
printf 'BACKEND_URL=%s\n' "$url" >> release.env
printf 'OPENAI_API_KEY=%s\n' "$openai_key" >> release.env

unset openai_key

chmod 600 release.env

echo "Frontend runtime configuration completed successfully"