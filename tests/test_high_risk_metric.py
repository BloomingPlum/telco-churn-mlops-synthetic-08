"""
tests/test_high_risk_metric.py
==============================
Unit tests for the custom HIGH_RISK_PREDICTIONS counter
(predictions with churn_probability > 0.8).

The model call is stubbed, so the tests are deterministic and do not depend
on how the current model was trained. No running Docker stack is needed:

    MLFLOW_TRACKING_URI= pytest tests/test_high_risk_metric.py -v
"""

import os

os.environ["MLFLOW_TRACKING_URI"] = ""   # load the local model file, never MLflow

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from src.api import main
from src.api import predict as predict_module
from src.metrics import HIGH_RISK_THRESHOLD

CUSTOMER = {
    "tenure": 1, "MonthlyCharges": 105.0, "TotalCharges": 105.0,
    "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:   # "with" runs the startup event
        yield c


def high_risk_count(contract_type: str = "Month-to-month") -> float:
    value = REGISTRY.get_sample_value(
        "high_risk_predictions_total",
        {"model_version": main.MODEL_VERSION, "contract_type": contract_type},
    )
    return value or 0.0


def stub_probability(monkeypatch, prob: float) -> None:
    def fake_predict(features):
        return {
            "churn_probability": prob,
            "churn_prediction": int(prob >= 0.5),
            "features_used": list(features.keys()),
        }
    monkeypatch.setattr(predict_module, "predict_churn", fake_predict)


def test_threshold_is_0_8():
    assert HIGH_RISK_THRESHOLD == 0.8


@pytest.mark.parametrize("prob, expected_increment", [
    (0.95, 1),   # high risk → counted
    (0.81, 1),
    (0.80, 0),   # exactly 0.8 is NOT "> 0.8"
    (0.65, 0),   # churn, but not high risk
    (0.10, 0),
])
def test_counter_increments_only_above_threshold(client, monkeypatch, prob, expected_increment):
    stub_probability(monkeypatch, prob)
    before = high_risk_count()
    resp = client.post("/predict", json=CUSTOMER)
    assert resp.status_code == 200
    assert resp.json()["churn_probability"] == prob
    assert high_risk_count() - before == expected_increment


def test_counter_labelled_by_contract_type(client, monkeypatch):
    stub_probability(monkeypatch, 0.9)
    before = high_risk_count("One year")
    client.post("/predict", json={**CUSTOMER, "Contract": "One year"})
    assert high_risk_count("One year") - before == 1


def test_unknown_contract_does_not_create_new_series(client, monkeypatch):
    stub_probability(monkeypatch, 0.9)
    before = high_risk_count("Other")
    client.post("/predict", json={**CUSTOMER, "Contract": "some-random-value-123"})
    assert high_risk_count("Other") - before == 1
    assert REGISTRY.get_sample_value(
        "high_risk_predictions_total",
        {"model_version": main.MODEL_VERSION, "contract_type": "some-random-value-123"},
    ) is None


def test_failed_prediction_is_not_counted(client, monkeypatch):
    monkeypatch.setattr(predict_module, "predict_churn", lambda f: {"error": "boom"})
    before = high_risk_count()
    resp = client.post("/predict", json=CUSTOMER)
    assert resp.status_code == 500
    assert high_risk_count() == before


def test_metric_exposed_on_metrics_endpoint(client, monkeypatch):
    stub_probability(monkeypatch, 0.99)
    client.post("/predict", json=CUSTOMER)
    body = client.get("/metrics").text
    assert "# TYPE high_risk_predictions_total counter" in body
    assert 'high_risk_predictions_total{contract_type="Month-to-month"' in body


def test_http_metrics_use_route_template(client):
    client.get("/health")
    body = client.get("/metrics").text
    assert 'api_requests_total{endpoint="/health",method="GET",status_code="200"}' in body
