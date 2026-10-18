#!/usr/bin/env bash
set -Eeuo pipefail
cd /opt/verirag/backend
# Fetch secrets on instance, not from the source-control repository.
# IAM instance profile needs secretsmanager:GetSecretValue and rds:DescribeDBInstances.
python3 - <<'PY_RUNTIME'
import json, os, subprocess, tempfile
from pathlib import Path
from urllib.parse import quote

def aws(*args):
    return subprocess.check_output(['aws', *args, '--region', 'ap-south-1', '--output', 'text'], text=True).strip()

api = json.loads(aws('secretsmanager', 'get-secret-value', '--secret-id', 'api-keys', '--query', 'SecretString'))
arn = aws('rds', 'describe-db-instances', '--db-instance-identifier', 'verirag-postgres', '--query', 'DBInstances[0].MasterUserSecret.SecretArn')
db = json.loads(aws('secretsmanager', 'get-secret-value', '--secret-id', arn, '--query', 'SecretString'))
address = aws('rds', 'describe-db-instances', '--db-instance-identifier', 'verirag-postgres', '--query', 'DBInstances[0].Endpoint.Address')
user, pwd = quote(db['username'], safe=''), quote(db['password'], safe='')
api['DATABASE_URL'] = f'postgresql://{user}:{pwd}@{address}:5432/verirag?sslmode=require'
api['REDIS_URL'] = 'rediss://master.verirag-redis.c0b7aq.aps1.cache.amazonaws.com:6379/0'
# Docker Compose's env_file parser supports double-quoted values; JSON serialization escapes safely.
p = Path('/opt/verirag/backend/.env')
fd, name = tempfile.mkstemp(dir=p.parent, prefix='.env-tmp-')
try:
    with os.fdopen(fd, 'w') as f:
        for key, value in api.items():
            if not key.isidentifier():
                raise ValueError('Unexpected secret variable name')
            f.write(f'{key}={json.dumps(str(value))}\n')
    os.chmod(name, 0o600)
    os.replace(name, p)
finally:
    if os.path.exists(name):
        os.unlink(name)
print('Backend application environment prepared from Secrets Manager')
PY_RUNTIME
# Fail early if GitHub release bundle doesn't include exact commit tag.
grep -Eq '^IMAGE_TAG=[a-f0-9]{40}$' release.env
