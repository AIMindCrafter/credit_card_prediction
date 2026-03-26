.PHONY: help install install-dev train evaluate serve test lint typecheck build push clean

# ─────────────────────────────────────────────────────────────────────────────
# Variables
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_NAME   ?= fraud-detection-api
IMAGE_TAG    ?= latest
REGISTRY     ?= ghcr.io/your-org
PYTHON       ?= python3
DATA_FILE    ?= creditcard.csv
EXPERIMENT   ?= credit-card-fraud-detection
MODEL_TYPE   ?= random_forest

# ─────────────────────────────────────────────────────────────────────────────
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
install:  ## Install production dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev:  ## Install development + test dependencies
	pip install --upgrade pip
	pip install -r requirements-dev.txt
	pre-commit install

# ─────────────────────────────────────────────────────────────────────────────
# ML Pipeline
# ─────────────────────────────────────────────────────────────────────────────
train:  ## Train the fraud detection model
	$(PYTHON) src/train.py \
		--data $(DATA_FILE) \
		--experiment $(EXPERIMENT) \
		--model $(MODEL_TYPE)

train-no-register:  ## Train without MLflow model registry
	$(PYTHON) src/train.py \
		--data $(DATA_FILE) \
		--experiment $(EXPERIMENT) \
		--model $(MODEL_TYPE) \
		--no-register

evaluate:  ## Evaluate saved model and generate reports
	$(PYTHON) src/evaluate.py \
		--model models/fraud_pipeline.joblib \
		--data $(DATA_FILE) \
		--output-dir reports

# ─────────────────────────────────────────────────────────────────────────────
# Development Server
# ─────────────────────────────────────────────────────────────────────────────
serve-dev:  ## Run FastAPI in development mode (auto-reload)
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

serve:  ## Run FastAPI in production mode
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# ─────────────────────────────────────────────────────────────────────────────
# Docker Compose
# ─────────────────────────────────────────────────────────────────────────────
up:  ## Start full MLOps stack (API + MLflow + DB + MinIO + Prometheus + Grafana)
	docker compose up -d

up-build:  ## Build and start full stack
	docker compose up -d --build

down:  ## Stop all services
	docker compose down

down-clean:  ## Stop services and remove volumes
	docker compose down -v --remove-orphans

logs:  ## Tail all service logs
	docker compose logs -f

logs-api:  ## Tail API service logs
	docker compose logs -f api

status:  ## Show service status
	docker compose ps

# ─────────────────────────────────────────────────────────────────────────────
# Docker Image
# ─────────────────────────────────────────────────────────────────────────────
build:  ## Build production Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

push:  ## Push image to container registry
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	docker push $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

# ─────────────────────────────────────────────────────────────────────────────
# Testing & Code Quality
# ─────────────────────────────────────────────────────────────────────────────
test:  ## Run all tests with coverage
	pytest tests/ -v --cov=app --cov=src --cov-report=term-missing --cov-report=html

test-fast:  ## Run tests excluding slow tests
	pytest tests/ -v -m "not slow"

lint:  ## Run ruff linter
	ruff check .

lint-fix:  ## Run ruff and auto-fix issues
	ruff check --fix .

typecheck:  ## Run mypy type checker
	mypy app/ src/ --ignore-missing-imports

format:  ## Format code with ruff
	ruff format .

# ─────────────────────────────────────────────────────────────────────────────
# Kubernetes
# ─────────────────────────────────────────────────────────────────────────────
k8s-apply:  ## Apply all Kubernetes manifests
	kubectl apply -f k8s/

k8s-delete:  ## Delete all Kubernetes resources
	kubectl delete -f k8s/

k8s-status:  ## Check Kubernetes deployment status
	kubectl get pods,svc,ingress -l app=fraud-detection

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
smoke-test:  ## Run API smoke test (requires running API at localhost:8000)
	@echo "Testing /health..."
	curl -sf http://localhost:8000/health | python3 -m json.tool
	@echo "\nTesting /ready..."
	curl -sf http://localhost:8000/ready | python3 -m json.tool

clean:  ## Remove generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage reports/
