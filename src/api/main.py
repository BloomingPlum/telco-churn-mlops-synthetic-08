import warnings
import time
warnings.filterwarnings("ignore", message=".*protected namespace.*", category=UserWarning)

from fastapi import FastAPI, HTTPException, Request
from fastapi import Response
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timezone
import json
import logging
import os
import sys

# HTTP metrics (автоматичні)
from prometheus_fastapi_instrumentator import Instrumentator

# ML метрики (наші власні)
from src.metrics import (
    PREDICTIONS_TOTAL,
    PREDICTION_LATENCY,
    MODEL_LOAD_TIME,
    ACTIVE_MODEL_VERSION,
    NULL_FEATURES_TOTAL,
    PREDICTION_CONFIDENCE,
    HIGH_RISK_PREDICTIONS,
    HIGH_RISK_THRESHOLD,
    MODEL_ACCURACY,
    MODEL_F1_SCORE,
    MODEL_AUC_ROC,
    API_REQUESTS_TOTAL,
    REQUEST_DURATION,
    ACTIVE_CONNECTIONS,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from src.api.models import CustomerFeatures, PredictionResponse
from src.api import predict as predict_module


# ── Structured JSON logging (Promtail parses it, Loki filters by fields) ──
class JSONFormatter(logging.Formatter):
    EXTRA_FIELDS = ("endpoint", "method", "status_code", "latency_ms",
                    "model_version", "outcome", "churn_probability", "contract_type")

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }
        for field in self.EXTRA_FIELDS:
            if hasattr(record, field):
                data[field] = getattr(record, field)
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
logger = logging.getLogger(__name__)
access_logger = logging.getLogger("churn-api.access")

# Значення Contract, які знає модель. Все інше → "Other", щоб довільний
# рядок від клієнта не створював нові time series (high cardinality).
KNOWN_CONTRACTS = {"Month-to-month", "One year", "Two year"}
# Ці ендпоінти викликаються кожні 10–15 с (Prometheus, healthcheck) — не засмічуємо логи
QUIET_ENDPOINTS = {"/metrics", "/health"}

# Версія моделі — буде оновлена при старті
MODEL_VERSION = "v1.0"

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="API для прогнозування відтоку клієнтів (churn prediction)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Автоматичні HTTP метрики (http_requests_total, http_request_duration і т.д.)
Instrumentator().instrument(app).expose(app)


class MetricsMiddleware(BaseHTTPMiddleware):
    """api_requests_total / request_duration_seconds / active_http_connections
    (used by the "API Performance" dashboard) + one JSON log line per request."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        status_code = 500
        ACTIVE_CONNECTIONS.inc()
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            ACTIVE_CONNECTIONS.dec()
            duration = time.perf_counter() - start
            # Route template ("/predict"), not the raw URL → bounded label values
            route = request.scope.get("route")
            endpoint = getattr(route, "path", "unmatched")
            API_REQUESTS_TOTAL.labels(
                method=request.method, endpoint=endpoint, status_code=str(status_code)
            ).inc()
            REQUEST_DURATION.labels(method=request.method, endpoint=endpoint).observe(duration)

            if endpoint in QUIET_ENDPOINTS and status_code < 400:
                level = logging.DEBUG
            elif status_code >= 500:
                level = logging.ERROR
            elif status_code >= 400:
                level = logging.WARNING
            else:
                level = logging.INFO
            access_logger.log(level, f"{request.method} {endpoint} {status_code}", extra={
                "endpoint": endpoint,
                "method": request.method,
                "status_code": status_code,
                "latency_ms": round(duration * 1000, 2),
                "model_version": MODEL_VERSION,
            })


app.add_middleware(MetricsMiddleware)


def publish_model_quality(model_version: str) -> None:
    """Read models/model_metrics.json (written by pipelines/train.py) and expose
    the test-set scores as gauges for the "ML Model Health" dashboard."""
    default_path = os.path.join(os.path.dirname(getattr(predict_module, "MODEL_PATH", "models/churn_model.pkl")),
                                "model_metrics.json")
    path = os.getenv("MODEL_METRICS_PATH", default_path)
    try:
        with open(path) as f:
            scores = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning(f"Model metrics not published ({path}): {e}")
        return
    if "accuracy" in scores:
        MODEL_ACCURACY.labels(model_version=model_version, dataset_split="test").set(scores["accuracy"])
    if "f1" in scores:
        MODEL_F1_SCORE.labels(model_version=model_version).set(scores["f1"])
    if "roc_auc" in scores:
        MODEL_AUC_ROC.labels(model_version=model_version).set(scores["roc_auc"])
    logger.info(f"Model quality published from {path}: accuracy={scores.get('accuracy')}")


@app.on_event("startup")
async def startup_event():
    global MODEL_VERSION

    if predict_module.model is None:
        logger.error("Модель не завантажилася при старті API!")
        return

    logger.info("Модель успішно завантажена при старті API")

    # Визначити версію моделі
    model_source = getattr(predict_module, "model_source", None)
    if model_source and "MLflow" in str(model_source):
        MODEL_VERSION = "mlflow-production"
    else:
        MODEL_VERSION = "v1.0-local"

    # Записати час завантаження (старт — немає таймера, ставимо символічне значення)
    MODEL_LOAD_TIME.labels(model_version=MODEL_VERSION).set(0)

    # Позначити активну версію моделі
    ACTIVE_MODEL_VERSION.labels(
        model_version=MODEL_VERSION,
        stage="production",
        run_id=str(getattr(predict_module, "model_source", "local")),
    ).set(1)

    logger.info(f"Model version label: {MODEL_VERSION}")
    publish_model_quality(MODEL_VERSION)

    # Створити серії лічильників зі значенням 0 одразу при старті: rate()/increase()
    # не бачать перше значення нової серії, тож без цього прогнози до першого
    # scrape "губляться" (а high-risk share рахувався б неправильно), і панелі = No data.
    for contract in KNOWN_CONTRACTS:
        HIGH_RISK_PREDICTIONS.labels(model_version=MODEL_VERSION, contract_type=contract)
        for outcome in ("churn", "no_churn", "error"):
            PREDICTIONS_TOTAL.labels(model_version=MODEL_VERSION, outcome=outcome, contract_type=contract)


@app.get("/health")
def health():
    model_status = "завантажена" if predict_module.model is not None else "НЕ завантажена"
    return {
        "status": "healthy" if predict_module.model is not None else "degraded",
        "service": "churn-prediction-api",
        "timestamp": datetime.utcnow().isoformat(),
        "model_status": model_status,
        "model_path": getattr(predict_module, "MODEL_PATH", "невідомо"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    input_data = features.dict()

    # Перевірити null значення у ключових полях
    key_fields = ["tenure", "MonthlyCharges", "TotalCharges", "Contract"]
    for field in key_fields:
        if input_data.get(field) is None:
            NULL_FEATURES_TOTAL.labels(feature_name=field).inc()

    contract_type = str(input_data.get("Contract", "Unknown"))
    if contract_type not in KNOWN_CONTRACTS:
        contract_type = "Other"

    try:
        # Вимірювати час inference
        with PREDICTION_LATENCY.labels(model_version=MODEL_VERSION).time():
            result = predict_module.predict_churn(input_data)

        if "error" in result:
            # Помилка моделі — зафіксувати
            PREDICTIONS_TOTAL.labels(
                model_version=MODEL_VERSION,
                outcome="error",
                contract_type=contract_type,
            ).inc()
            raise ValueError(result["error"])

        churn_prob = result["churn_probability"]
        outcome    = "churn" if result["churn_prediction"] == 1 else "no_churn"

        # Основний лічильник
        PREDICTIONS_TOTAL.labels(
            model_version=MODEL_VERSION,
            outcome=outcome,
            contract_type=contract_type,
        ).inc()

        # Розподіл впевненості моделі
        PREDICTION_CONFIDENCE.labels(
            model_version=MODEL_VERSION,
            outcome=outcome,
        ).observe(churn_prob)

        # Кастомна метрика: клієнти з високим ризиком відтоку (p > 0.8)
        if churn_prob > HIGH_RISK_THRESHOLD:
            HIGH_RISK_PREDICTIONS.labels(
                model_version=MODEL_VERSION,
                contract_type=contract_type,
            ).inc()
            logger.info("High-risk customer detected", extra={
                "endpoint": "/predict",
                "outcome": outcome,
                "churn_probability": churn_prob,
                "contract_type": contract_type,
                "model_version": MODEL_VERSION,
            })

        return PredictionResponse(
            churn_probability=churn_prob,
            churn_prediction=result["churn_prediction"],
            features_used=result["features_used"],
        )

    except ValueError:
        raise HTTPException(status_code=500, detail=str(result.get("error", "Unknown error")))
    except Exception as e:
        logger.error(f"Помилка під час прогнозу: {str(e)}", exc_info=True)
        PREDICTIONS_TOTAL.labels(
            model_version=MODEL_VERSION,
            outcome="error",
            contract_type=contract_type,
        ).inc()
        raise HTTPException(
            status_code=500,
            detail=f"Помилка обробки запиту: {str(e)}",
        )
