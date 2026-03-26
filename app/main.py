"""
main.py
~~~~~~~
FastAPI application entry point — Credit Card Fraud Detection API.

Start with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations

import logging
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    import structlog
    _HAS_STRUCTLOG = True
except ImportError:
    _HAS_STRUCTLOG = False

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False

# Add project root to path so src/ packages resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.health import router as health_router
from app.middleware import RequestLoggingMiddleware
from app.predict import router as predict_router
from config.settings import settings

# ---------------------------------------------------------------------------
# Logging — use structlog if available, else standard logging
# ---------------------------------------------------------------------------
if _HAS_STRUCTLOG:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
    logger = structlog.get_logger(__name__)
else:
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("structlog not installed — using standard logging")

# Candidate paths for the model file (in priority order)
MODEL_CANDIDATES = [
    Path(settings.MODEL_PATH),
    Path("complete_fraud_pipeline.joblib"),          # existing trained model
    Path("models/fraud_pipeline.joblib"),
    Path("../complete_fraud_pipeline.joblib"),
]


def _find_and_load_model():
    """Try each candidate path and return (model, scaler, version)."""
    print(f"\n[MODEL SEARCH] CWD={Path.cwd()}")
    for candidate in MODEL_CANDIDATES:
        exists = candidate.exists()
        print(f"  checking: {candidate.resolve()} → exists={exists}")
        if exists:
            try:
                print(f"  loading:  {candidate} ...")
                artifact = joblib.load(candidate)
                print(f"  loaded:   type={type(artifact).__name__}")
                if isinstance(artifact, dict):
                    model  = artifact.get("model") or artifact.get("classifier")
                    scaler = artifact.get("scaler")
                else:
                    model  = artifact
                    scaler = None
                print(f"  [OK] model={type(model).__name__}")
                return model, scaler, str(candidate)
            except Exception as e:
                print(f"  [FAIL] {candidate}: {e}")
                traceback.print_exc()
    print("  [WARN] No model file successfully loaded.")
    return None, None, None


# ---------------------------------------------------------------------------
# Lifespan — model loading on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    logger.info("[startup] Loading fraud detection model…")

    try:
        model, scaler, path = _find_and_load_model()
        app.state.model         = model
        app.state.scaler        = scaler
        app.state.model_version = path if path else None

        if model is not None:
            logger.info(f"[model_loaded] path={path}")
        elif settings.MODEL_LOAD_FROM_REGISTRY:
            from src.mlflow_utils import load_model_from_registry
            app.state.model = load_model_from_registry(
                model_name=settings.MLFLOW_MODEL_NAME,
                stage=settings.MLFLOW_MODEL_STAGE,
            )
            app.state.scaler        = None
            app.state.model_version = (
                f"{settings.MLFLOW_MODEL_NAME}/{settings.MLFLOW_MODEL_STAGE}"
            )
            logger.info("[model_loaded] source=mlflow_registry")
        else:
            candidates = [str(p) for p in MODEL_CANDIDATES]
            logger.warning(f"[model_not_found] checked={candidates}")
    except Exception as exc:
        logger.error(f"[model_load_failed] error={exc}")
        app.state.model         = None
        app.state.scaler        = None
        app.state.model_version = None

    logger.info(f"[startup_complete] service={settings.APP_NAME} version={settings.APP_VERSION}")
    yield
    logger.info("[shutdown] Fraud Detection API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ── Middleware (order matters — outermost runs first) ──────────────────────
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)

# ── Prometheus metrics (optional) ──────────────────────────────────────
if _HAS_PROMETHEUS:
    Instrumentator(
        should_group_status_codes=False,
        should_respect_env_var=False,
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# ── Routers ──────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(predict_router)

# ── Serve frontend static files ──────────────────────────────────────────
_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")


# ---------------------------------------------------------------------------
# Global Exception Handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"[unhandled_exception] {request.method} {request.url.path} "
        f"error={exc!r} type={type(exc).__name__}"
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred.",
        },
    )


# ---------------------------------------------------------------------------
# Root → serve frontend dashboard
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    """Serve the FraudShield dashboard (frontend/index.html)."""
    html_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    # Fallback JSON when frontend not found
    return JSONResponse({
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "ui": "/ui",
        "health": "/health",
        "metrics": "/metrics",
    })


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.DEBUG,
    )
