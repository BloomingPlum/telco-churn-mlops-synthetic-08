# Module 8 — Monitoring & Observability (Docker Compose)

Homework on top of `telco-churn-mlops-synthetic-08`:

1. Set up the project, create the model, set up monitoring with Docker Compose.
2. Custom metric `HIGH_RISK_PREDICTIONS` — counter of predictions with churn probability > 0.8.

## Quick start

```bash
git clone https://github.com/<your-user>/telco-churn-mlops-synthetic-08.git
cd telco-churn-mlops-synthetic-08

make model    # generate data + train the model inside Docker (same libraries as the API image)
make up       # API + MLflow + Prometheus, Grafana, Loki, Promtail, Alertmanager, node-exporter, cAdvisor, Pushgateway
make high-risk-demo   # send 20 high-risk + 20 low-risk customers, print the counter before/after
```

`make model` is optional: a trained `models/churn_model.pkl` (+ `models/model_metrics.json`) is already committed.

Without `make`:

```bash
docker network create mlops-net
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml --profile train run --rm generator
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml --profile train run --rm trainer
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d --build
python3 scripts/high_risk_demo.py
```

| Service | URL |
|---|---|
| Churn API (Swagger) | http://localhost:8000/docs |
| API metrics | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 (targets: `/targets`, alerts: `/alerts`) |
| Grafana | http://localhost:3000 — `admin` / `mlops_pass`, folder **MLOps** |
| Alertmanager | http://localhost:9093 |
| MLflow | http://localhost:5000 |

## The model

`pipelines/train.py` — RandomForest in a scikit-learn Pipeline (OneHotEncoder for categorical columns).

* trained on exactly the 19 fields the API accepts (`src/api/models.py`); the generated CSV also has
  `RecordDate`, and a model trained with it fails every `/predict` call with *"columns are missing"*;
* `max_depth=10, min_samples_leaf=10`: test accuracy 0.684 / F1 0.728 / ROC-AUC 0.736, model file ≈ 2.5 MB
  (the unlimited forest scored 0.649 and pickled to ≈ 190 MB — over GitHub's 100 MB limit);
* writes `models/model_metrics.json`; the API publishes it as `model_accuracy`, `model_f1_score`, `model_auc_roc`.

## Custom metric `HIGH_RISK_PREDICTIONS`

| | |
|---|---|
| Definition | `src/metrics.py` → `high_risk_predictions_total` (Counter), labels `model_version`, `contract_type` |
| Incremented | `src/api/main.py` → `/predict`, when `churn_probability > 0.8` |
| Recording rules | `job:high_risk_predictions:rate5m`, `job:high_risk_predictions:ratio5m` |
| Alert | `HighRiskPredictionsSpike` — high-risk share > 25 % for 5 min (baseline ≈ 13 %) |
| Dashboard | Grafana → *ML Model Health — Telco Churn* → row **🚨 High-Risk Customers** |
| Logs | each high-risk prediction writes `"High-risk customer detected"` (JSON → Loki) |
| Tests | `tests/test_high_risk_metric.py` (`make test-high-risk`, no Docker needed) |

Useful PromQL:

```promql
high_risk_predictions_total
sum by (contract_type) (rate(high_risk_predictions_total[5m])) * 60          # per minute
sum(rate(high_risk_predictions_total[5m]))
  / sum(rate(predictions_total{outcome!="error"}[5m]))                        # share of traffic
```

## Fixes made to get the stack running

| File | Problem → fix |
|---|---|
| `docker-compose.yml` | monitoring services were duplicated from the overlay → app services only; generator/trainer under profile `train`; healthchecks used `curl`, which `python:3.11-slim` lacks → python `urllib` |
| `mlflow_db/mlflow.db` | DB had an MLflow 3.x schema (revision `1b5f0d9ad7c1`), the server image is 2.11.3 → server refuses to start. The DB was empty, so it is removed; MLflow creates a fresh one |
| `monitoring/prometheus/prometheus.yml` | target `mlflow.int:5000` — Docker DNS does not resolve `hostname:` → `mlflow:5000` |
| `monitoring/grafana/dashboards/*.json` | `${DS_PROMETHEUS}` / `${DS_LOKI}` are only substituted on manual import, not provisioning → provisioned uids `prometheus_telco` / `loki_telco` |
| `src/api/main.py` | `api_requests_total`, `request_duration_seconds`, `active_http_connections` (API Performance dashboard) were never recorded → `MetricsMiddleware`; logs were plain text → JSON, so Loki `\| json` queries work |
| `recording_rules.yml` | `job:model_health:composite_score` added vectors with different labels → always empty |

## Expected alerts

* `ModelAccuracyLow` / `ModelAccuracyCritical` **fire** — the course SLO is accuracy ≥ 0.82, the synthetic
  dataset gives 0.684. That is the monitoring working as intended; raise the model or lower the threshold.
* Slack/e-mail receivers in `alertmanager.yml` are placeholders; alerts are visible at http://localhost:9093.

## Screenshots for the submission

1. `docker compose ... ps` — all containers `Up`.
2. Prometheus `/targets` — `churn-api`, `mlflow`, `prometheus`, `alertmanager`, `node-exporter`, `cadvisor`, `pushgateway`, `grafana` are UP.
3. `curl -s localhost:8000/metrics | grep high_risk` after `make high-risk-demo`.
4. Prometheus graph of `high_risk_predictions_total`.
5. Grafana *ML Model Health* — row **High-Risk Customers**.
6. Prometheus `/alerts` → `HighRiskPredictionsSpike` (fires ~5 min after `make high-risk-demo`).
