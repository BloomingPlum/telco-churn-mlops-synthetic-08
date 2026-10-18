"""
src/metrics.py
==============
Prometheus metrics для telco-churn-mlops-synthetic-07.

Самодостатній модуль — НЕ залежить від інших файлів репозиторію.

ВСТАНОВЛЕННЯ:
  1. Скопіюй цей файл у директорію src/ репозиторію:
       cp src/metrics.py <repo>/src/metrics.py

  2. Додай у requirements-api.txt:
       prometheus-client==0.19.0
       structlog==23.2.0

  3. Додай в app.py (де є FastAPI):
       from src.metrics import (
           PREDICTIONS_TOTAL, PREDICTION_LATENCY, MODEL_ACCURACY,
           API_REQUESTS_TOTAL, REQUEST_DURATION, ACTIVE_CONNECTIONS,
           NULL_FEATURES_TOTAL, MODEL_LOAD_TIME, ACTIVE_MODEL_VERSION,
           generate_latest, CONTENT_TYPE_LATEST,
       )
       # ... і /metrics endpoint (div. app_patch_example.py)

ПЕРЕВІРКА після інтеграції:
  curl http://localhost:8000/metrics | grep predictions_total
"""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,        # noqa: F401  (re-exported for app.py convenience)
    CONTENT_TYPE_LATEST,    # noqa: F401
)

# ═══════════════════════════════════════════════════════════════
# ML MODEL METRICS
# ═══════════════════════════════════════════════════════════════

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total churn predictions made by the model",
    ["model_version", "outcome", "contract_type"],
    # outcome:       churn | no_churn | error
    # contract_type: Month-to-month | One year | Two year | Unknown
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time to run model.predict() or predict_proba() in seconds",
    ["model_version"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    # Usage: with PREDICTION_LATENCY.labels(model_version="v1.2").time(): ...
)

MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Current model accuracy on evaluation dataset (0.0 – 1.0)",
    ["model_version", "dataset_split"],
    # dataset_split: train | val | test
    # Update periodically: MODEL_ACCURACY.labels("v1.2", "test").set(0.847)
)

MODEL_F1_SCORE = Gauge(
    "model_f1_score",
    "Model F1 score (macro average)",
    ["model_version"],
)

MODEL_AUC_ROC = Gauge(
    "model_auc_roc",
    "Model AUC-ROC score",
    ["model_version"],
)

CHURN_RATE_PREDICTED = Gauge(
    "churn_rate_predicted",
    "Rolling fraction of predictions classified as churn",
    ["time_window"],
    # time_window: 1h | 24h
)

MODEL_LOAD_TIME = Gauge(
    "model_load_time_seconds",
    "Seconds to load model artifact from MLflow at startup",
    ["model_version"],
)

ACTIVE_MODEL_VERSION = Gauge(
    "active_model_version_info",
    "Currently active model (value=1 when active)",
    ["model_version", "stage", "run_id"],
    # Usage: ACTIVE_MODEL_VERSION.labels("v1.2", "Production", run_id).set(1)
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence_score",
    "Distribution of churn probability scores (0.0 – 1.0)",
    ["model_version", "outcome"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ── Custom business metric: high-risk customers ──────────────────
# A prediction is "high risk" when churn_probability > HIGH_RISK_THRESHOLD.
# These are the customers retention should call first, so the business
# wants to see how many there are and how that number moves over time.
HIGH_RISK_THRESHOLD = 0.8

HIGH_RISK_PREDICTIONS = Counter(
    "high_risk_predictions_total",
    f"Predictions with churn probability > {HIGH_RISK_THRESHOLD}",
    ["model_version", "contract_type"],
    # contract_type: Month-to-month | One year | Two year | Other  (low cardinality)
    # Usage: if prob > HIGH_RISK_THRESHOLD: HIGH_RISK_PREDICTIONS.labels(v, ct).inc()
    # PromQL share of high-risk: sum(rate(high_risk_predictions_total[5m]))
    #                           / sum(rate(predictions_total{outcome!="error"}[5m]))
)

# ═══════════════════════════════════════════════════════════════
# DATA QUALITY & DRIFT
# (updated by monitoring/drift_exporter.py via Pushgateway)
# ═══════════════════════════════════════════════════════════════

FEATURE_DRIFT_SCORE = Gauge(
    "feature_drift_score",
    "Kolmogorov-Smirnov statistic: 0=no drift, 1=max drift. Alert >0.3",
    ["feature_name", "model_version"],
)

FEATURE_PSI_SCORE = Gauge(
    "feature_psi_score",
    "PSI for categorical features: <0.1 stable, >0.25 significant",
    ["feature_name", "model_version"],
)

DATA_QUALITY_SCORE = Gauge(
    "data_quality_score",
    "Fraction of valid (non-null) values for this feature (0.0 – 1.0)",
    ["feature_name"],
)

NULL_FEATURES_TOTAL = Counter(
    "null_features_total",
    "Null/missing feature values received in prediction requests",
    ["feature_name"],
    # Usage: if data.tenure is None: NULL_FEATURES_TOTAL.labels("tenure").inc()
)

OUT_OF_RANGE_FEATURES_TOTAL = Counter(
    "out_of_range_features_total",
    "Feature values outside expected range (schema violations)",
    ["feature_name"],
)

# ═══════════════════════════════════════════════════════════════
# HTTP / API METRICS
# (updated automatically by MetricsMiddleware in app.py)
# ═══════════════════════════════════════════════════════════════

API_REQUESTS_TOTAL = Counter(
    "api_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "request_duration_seconds",
    "End-to-end HTTP request duration including serialization",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

ACTIVE_CONNECTIONS = Gauge(
    "active_http_connections",
    "Currently active HTTP connections",
)

RESPONSE_SIZE_BYTES = Histogram(
    "response_size_bytes",
    "HTTP response body size in bytes",
    ["endpoint"],
    buckets=[128, 256, 512, 1024, 4096, 16384],
)

# ═══════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# (updated by Airflow DAG via Pushgateway)
# ═══════════════════════════════════════════════════════════════

MODEL_TRAINING_RUNS = Counter(
    "model_training_runs_total",
    "Model training run outcomes",
    ["status"],   # success | failure | running
)

MODEL_TRAINING_DURATION = Gauge(
    "model_training_duration_seconds",
    "Time to complete model training",
)

MODEL_NEW_ACCURACY = Gauge(
    "model_new_accuracy",
    "Accuracy of the freshly trained model (before promotion)",
    ["model_version"],
)

# ═══════════════════════════════════════════════════════════════
# AGENT / LLM METRICS (Modules 12-13)
# ═══════════════════════════════════════════════════════════════

AGENT_LLM_CALLS = Counter(
    "agent_llm_calls_total",
    "LLM API calls made by agent",
    ["model", "status", "tool_name"],
)

AGENT_TOKENS = Counter(
    "agent_tokens_total",
    "Tokens consumed",
    ["direction", "model"],   # direction: input | output
)

AGENT_TOOL_LATENCY = Histogram(
    "agent_tool_latency_seconds",
    "Agent tool call execution time",
    ["tool_name", "status"],
    buckets=[0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
)

CONV_TURN_COUNT = Histogram(
    "agent_conversation_turns",
    "Turns per conversation session",
    buckets=[1, 2, 3, 5, 10, 20, 50],
)
