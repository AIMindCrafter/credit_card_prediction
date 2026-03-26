# Credit Card Fraud Detection — Enterprise MLOps Platform

![CI](https://github.com/your-org/credit_card_fraud_Detection/actions/workflows/ci.yml/badge.svg)
![CD](https://github.com/your-org/credit_card_fraud_Detection/actions/workflows/cd.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![MLflow](https://img.shields.io/badge/MLflow-2.19-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Kubernetes](https://img.shields.io/badge/Kubernetes-ready-326ce5)
![License](https://img.shields.io/badge/license-MIT-green)

> **Production-ready**, end-to-end credit card fraud detection system with real-time REST API, MLflow experiment tracking, Kubernetes auto-scaling, and full Prometheus/Grafana monitoring.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT / LOAD BALANCER                       │
└─────────────────────┬───────────────────────────────────────────────┘
                      │ HTTPS
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│            NGINX INGRESS  (Rate Limiting + TLS)                     │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │  FastAPI (3 replicas) │  ← HPA: 2–10 pods
          │  /predict             │
          │  /predict/batch       │
          │  /health  /ready      │
          │  /metrics (Prometheus)│
          └───────┬───────────────┘
                  │
        ┌─────────┴──────────┐
        │                    │
        ▼                    ▼
 ┌──────────────┐   ┌─────────────────┐
 │ ML Model     │   │ MLflow Registry │
 │ (joblib)     │   │ + PostgreSQL    │
 │ RandomForest │   │ + MinIO (S3)    │
 └──────────────┘   └─────────────────┘
        │
        ▼
 ┌─────────────────────────────┐
 │  Prometheus → Grafana       │
 │  (Metrics + Dashboards)     │
 └─────────────────────────────┘
```

---

## Quick Start (Docker Compose)

```bash
# 1. Clone the repo
git clone https://github.com/your-org/credit_card_fraud_Detection.git
cd credit_card_fraud_Detection

# 2. Copy environment variables
cp .env.example .env

# 3. Place your trained model (or train a new one — see below)
# models/fraud_pipeline.joblib should already exist.

# 4. Start the full stack
docker compose up -d

# 5. Verify all services
docker compose ps
```

| Service | URL |
|---|---|
| 🚀 **Fraud API** | http://localhost:8000 |
| 📖 **API Docs** | http://localhost:8000/docs |
| 📊 **MLflow UI** | http://localhost:5000 |
| 🪣 **MinIO UI** | http://localhost:9001 |
| 📈 **Prometheus** | http://localhost:9090 |
| 📉 **Grafana** | http://localhost:3000 |

---

## API Endpoints

### `POST /predict` — Single Transaction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0.0,
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
    "V5": -0.3383, "V6": 0.4624, "V7": 0.2396, "V8": 0.0987,
    "V9": 0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
    "V13": -0.9914, "V14": -0.3112, "V15": 1.4682, "V16": -0.4704,
    "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
    "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
    "V25": 0.1285, "V26": -0.1891, "V27": 0.1336, "V28": -0.0211,
    "Amount": 149.62
  }'
```

**Response:**
```json
{
  "request_id": "3f8c1a2b-...",
  "timestamp": "2026-03-25T18:00:00Z",
  "is_fraud": false,
  "fraud_probability": 0.032,
  "confidence": "HIGH",
  "label": "LEGITIMATE",
  "latency_ms": 4.21,
  "model_version": "local"
}
```

### `POST /predict/batch` — Batch Transactions
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [{...}, {...}]}'
```

### `GET /health` — Liveness Probe
```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "1.0.0", ...}
```

### `GET /ready` — Readiness Probe
```bash
curl http://localhost:8000/ready
# {"status": "ready", "model_loaded": true, "mlflow_reachable": true, ...}
```

---

## Train a New Model

```bash
# Install dependencies
make install-dev

# Train with MLflow tracking
make train DATA_FILE=creditcard.csv

# Evaluate and generate reports
make evaluate

# View the experiment in MLflow UI
open http://localhost:5000
```

---

## Development

```bash
# Run dev server (auto-reload)
make serve-dev

# Run tests with coverage
make test

# Lint
make lint

# Type check
make typecheck
```

---

## Cloud Deployment (Kubernetes)

```bash
# 1. Build and push Docker image
make build push REGISTRY=ghcr.io/your-org

# 2. Apply all K8s manifests
kubectl create namespace fraud-detection
kubectl apply -f k8s/

# 3. Check deployment
make k8s-status
```

The deployment includes:
- **3 replicas** with rolling updates (zero-downtime)
- **HPA**: auto-scales 2–10 pods on CPU/memory
- **NGINX Ingress** with TLS + rate limiting
- **Liveness & Readiness** probes

---

## Project Structure

```
credit_card_fraud_Detection/
├── app/                    # FastAPI service
│   ├── main.py             # Application entry point
│   ├── predict.py          # Prediction endpoints
│   ├── health.py           # Health/readiness probes
│   ├── middleware.py       # Logging middleware
│   └── schemas.py          # Pydantic models
├── src/                    # ML pipeline
│   ├── train.py            # Training script
│   ├── preprocessing.py    # Data preprocessing
│   ├── evaluate.py         # Model evaluation
│   └── mlflow_utils.py     # MLflow helpers
├── config/
│   └── settings.py         # Pydantic settings
├── tests/                  # pytest test suite
├── monitoring/
│   └── prometheus.yml      # Scrape config
├── k8s/                    # Kubernetes manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   └── hpa.yaml
├── .github/workflows/
│   ├── ci.yml              # Lint + test + build
│   └── cd.yml              # Push + deploy
├── Dockerfile              # Multi-stage production image
├── Dockerfile.mlflow       # MLflow server image
├── docker-compose.yml      # Full local stack
├── Makefile                # Developer shortcuts
├── requirements.txt
└── requirements-dev.txt
```

---

## MLOps Stack

| Layer | Tool |
|---|---|
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| Artifact Store | MinIO (S3-compatible) |
| Metadata Store | PostgreSQL |
| API Framework | FastAPI + Uvicorn |
| Containerization | Docker (multi-stage) |
| Orchestration | Kubernetes + HPA |
| Monitoring | Prometheus + Grafana |
| CI/CD | GitHub Actions |
| Config | Pydantic Settings |
| Logging | structlog (JSON) |

---

## Dataset

- **Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: ~284K transactions, 31 features (Time, V1–V28 PCA, Amount, Class)
- **Class imbalance**: ~0.17% fraud rate
- **Note**: `creditcard.csv` is excluded from version control (`.gitignore`)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
