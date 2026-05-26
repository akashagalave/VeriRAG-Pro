Okay, clear picture. You're starting completely fresh — new repo, new infra, everything from zero. Redis is not needed right now, skip it. Here's the exact order to follow, nothing extra.

---

## Where you are right now

```
✅ New GitHub repo created (VeriRAG-Pro)
✅ Project open in VSCode
✅ Git initialized
✅ requirements.txt installing
✅ New Qdrant API key + endpoint
❌ No Redis (skip for now — app works without it)
❌ No Docker images
❌ No ECR
❌ No EKS cluster
❌ No GitHub secrets
```

---

## PHASE 1 — Finish local setup first

**Step 1 — Create your `.env` file in the project root**

```
OPENAI_API_KEY=sk-...
QDRANT_URL=https://your-new-qdrant-endpoint
QDRANT_API_KEY=your-new-qdrant-key
TAVILY_API_KEY=tvly-...
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=VeriRAG-Pro
LANGSMITH_TRACING=true
```

No `REDIS_URL` for now. Leave it out completely.

**Step 2 — Test FastAPI starts without errors**

```bash
uvicorn backend.api:app --port 8000
```

Open browser: `http://localhost:8000/health`

Should return:
```json
{"status": "ok", "providers": [{"provider": "openai-gpt-4o-mini", "circuit_state": "closed"}]}
```

**Step 3 — Test Streamlit works**

```bash
streamlit run app.py
```

Upload a PDF → ask a question → confirm answer comes back.

**Step 4 — Run evaluation and create baseline**

```bash
python evaluate.py
```

This creates `eval_baseline.json` and `goldens.json`. Then:

```bash
git add eval_baseline.json goldens.json
git commit -m "add eval baseline"
```

---

## PHASE 2 — AWS setup (run these in order)

You need AWS CLI installed and configured. Check:

```bash
aws --version
aws sts get-caller-identity   # confirms your credentials work
```

---

**Step 5 — Create ECR repositories (2 repos — one per image)**

```bash
# Set your region
export AWS_REGION=ap-south-1       # change to your region
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "Account: $AWS_ACCOUNT_ID  Region: $AWS_REGION"

# Create backend repo
aws ecr create-repository \
  --repository-name verirag-backend \
  --region $AWS_REGION

# Create frontend repo
aws ecr create-repository \
  --repository-name verirag-frontend \
  --region $AWS_REGION

# Save your ECR registry URL — you'll need this
export ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
echo "ECR Registry: $ECR_REGISTRY"
```

---

**Step 6 — Build and push Docker images to ECR**

```bash
# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $ECR_REGISTRY

# Build backend
docker build -f Dockerfile.backend -t $ECR_REGISTRY/verirag-backend:latest .
docker push $ECR_REGISTRY/verirag-backend:latest

# Build frontend
docker build -f Dockerfile.frontend -t $ECR_REGISTRY/verirag-frontend:latest .
docker push $ECR_REGISTRY/verirag-frontend:latest

echo "Both images pushed to ECR"
```

---

**Step 7 — Create EKS cluster**

This takes 15–20 minutes. Install `eksctl` first if you don't have it:

```bash
# Install eksctl (Mac)
brew tap weaveworks/tap
brew install weaveworks/tap/eksctl

# Install eksctl (Linux)
curl --silent --location \
  "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_Linux_amd64.tar.gz" \
  | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin
```

Create the cluster:

```bash
eksctl create cluster \
  --name verirag-pro \
  --region $AWS_REGION \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 2 \
  --nodes-min 2 \
  --nodes-max 5 \
  --managed
```

Wait for it to finish. Then confirm kubectl is connected:

```bash
kubectl get nodes
# Should show 2 nodes in Ready state
```

---

**Step 8 — Create the verirag namespace**

```bash
kubectl apply -f k8s/namespace.yml
kubectl get namespaces | grep verirag
```

---

**Step 9 — Create Kubernetes secrets**

Open `k8s/secrets-and-config.yml` in VSCode and fill in your actual values. The file has base64-encoded secrets. Easiest way is to let kubectl do the encoding:

```bash
kubectl create secret generic verirag-secrets \
  --namespace verirag \
  --from-literal=OPENAI_API_KEY="sk-..." \
  --from-literal=QDRANT_URL="https://your-qdrant-endpoint" \
  --from-literal=QDRANT_API_KEY="your-qdrant-key" \
  --from-literal=TAVILY_API_KEY="tvly-..." \
  --from-literal=LANGSMITH_API_KEY="lsv2_..." \
  --from-literal=GRAFANA_ADMIN_PASSWORD="verirag-grafana"

kubectl create configmap verirag-config \
  --namespace verirag \
  --from-literal=LANGSMITH_PROJECT="VeriRAG-Pro" \
  --from-literal=LANGSMITH_TRACING="true" \
  --from-literal=BACKEND_URL="http://verirag-fastapi-service:8000"

# Verify
kubectl get secrets -n verirag
kubectl get configmaps -n verirag
```

---

**Step 10 — Update image URLs in k8s deployment files**

Open `k8s/backend-deployment.yml` and `k8s/frontend-deployment.yml` in VSCode.

Find the `image:` line in each file and replace with your actual ECR URLs:

```yaml
# backend-deployment.yml
image: YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/verirag-backend:latest

# frontend-deployment.yml
image: YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/verirag-frontend:latest
```

---

**Step 11 — Deploy the application to EKS**

```bash
kubectl apply -f k8s/services.yml
kubectl apply -f k8s/backend-deployment.yml
kubectl apply -f k8s/frontend-deployment.yml
kubectl apply -f k8s/hpa.yml

# Watch pods come up
kubectl get pods -n verirag -w
# Wait until all show Running
```

---

**Step 12 — Set up AWS Load Balancer Controller (for ingress)**

```bash
# Install AWS Load Balancer Controller via Helm
helm repo add eks https://aws.github.io/eks-charts
helm repo update

# Create IAM policy for the controller
curl -o iam-policy.json \
  https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/main/docs/install/iam_policy.json

aws iam create-policy \
  --policy-name AWSLoadBalancerControllerIAMPolicy \
  --policy-document file://iam-policy.json

# Create service account
eksctl create iamserviceaccount \
  --cluster verirag-pro \
  --namespace kube-system \
  --name aws-load-balancer-controller \
  --attach-policy-arn arn:aws:iam::${AWS_ACCOUNT_ID}:policy/AWSLoadBalancerControllerIAMPolicy \
  --approve

# Install controller
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  --namespace kube-system \
  --set clusterName=verirag-pro \
  --set serviceAccountName=aws-load-balancer-controller

# Apply ingress
kubectl apply -f k8s/ingress.yml

# Get your public URL (takes 2-3 min to provision)
kubectl get ingress -n verirag
```

The ADDRESS column is your public ALB URL.

---

**Step 13 — Install Prometheus + Grafana via Helm**

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword=verirag-grafana \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

# Wait for pods
kubectl get pods -n monitoring -w
# All should show Running (takes 2-3 min)
```

**Connect VeriRAG metrics to Prometheus:**

Create a file `servicemonitor.yml`:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: verirag-backend
  namespace: monitoring
  labels:
    release: prometheus
spec:
  namespaceSelector:
    matchNames:
      - verirag
  selector:
    matchLabels:
      app: verirag-backend
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
```

Apply it:

```bash
kubectl apply -f servicemonitor.yml
```

**Open Grafana:**

```bash
kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring
```

Go to `http://localhost:3000` → login: `admin / verirag-grafana`

Add these 4 panels in a new dashboard:

| Panel | Query |
|---|---|
| p95 Latency | `histogram_quantile(0.95, rate(verirag_request_latency_seconds_bucket[5m]))` |
| Request Rate | `rate(verirag_request_count_total[5m])` |
| Error Rate | `rate(verirag_error_count_total[5m])` |
| Circuit Breaker | `verirag_circuit_breaker_open` |

---

**Step 14 — Set up GitHub Actions secrets**

Go to your GitHub repo → Settings → Secrets and variables → Actions

Add all of these:

```
AWS_ACCESS_KEY_ID          = your AWS access key
AWS_SECRET_ACCESS_KEY      = your AWS secret key
AWS_REGION                 = ap-south-1  (your region)
ECR_REGISTRY               = ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com
EKS_CLUSTER_NAME           = verirag-pro
OPENAI_API_KEY             = sk-...
QDRANT_URL                 = https://your-qdrant-endpoint
QDRANT_API_KEY             = your-qdrant-key
TAVILY_API_KEY             = tvly-...
LANGSMITH_API_KEY          = lsv2_...
```

No `REDIS_URL` for now.

---

**Step 15 — Push code to trigger full CI/CD**

```bash
git add -A
git commit -m "feat: VeriRAG-Pro initial production deployment"
git push origin main
```

Watch Actions tab → "Deploy to EKS" → 3 jobs run: `eval-gate → build → deploy`

---

## Add Redis later (when ready)

Once you have ElastiCache created:

```bash
# 1. Add to k8s secret
kubectl create secret generic verirag-redis \
  --namespace verirag \
  --from-literal=REDIS_URL="redis://your-elasticache-endpoint:6379"

# 2. Add REDIS_URL to GitHub secrets

# 3. Restart backend pods to pick it up
kubectl rollout restart deployment/verirag-backend -n verirag
```

---

## Quick status check at any point

```bash
kubectl get pods -n verirag          # app pods
kubectl get pods -n monitoring        # prometheus + grafana
kubectl get ingress -n verirag        # your public URL
kubectl logs -n verirag deployment/verirag-backend --tail=50  # backend logs
```