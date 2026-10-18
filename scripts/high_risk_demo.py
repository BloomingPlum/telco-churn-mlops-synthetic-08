"""
scripts/high_risk_demo.py
=========================
Demo for the custom metric HIGH_RISK_PREDICTIONS (high_risk_predictions_total):
sends a mix of high-risk and low-risk customers to /predict and shows how the
counter on /metrics changes. Standard library only — no pip install needed.

    python3 scripts/high_risk_demo.py                 # 20 high-risk + 20 low-risk
    python3 scripts/high_risk_demo.py --high 50 --low 10 --url http://localhost:8000

Then look at:
    Prometheus  http://localhost:9090/graph?g0.expr=high_risk_predictions_total
    Grafana     ML Model Health — Telco Churn → row "High-Risk Customers"
"""

import argparse
import json
import random
import urllib.request

# Month-to-month + fiber + electronic check + new customer → p ≈ 0.95 with the current model
HIGH_RISK = {
    "tenure": 1, "MonthlyCharges": 105.0, "TotalCharges": 105.0,
    "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}
# Two-year contract, long tenure, DSL with support → p ≈ 0.43
LOW_RISK = {
    "tenure": 60, "MonthlyCharges": 45.0, "TotalCharges": 2700.0,
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
    "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)",
}


def post(url: str, payload: dict) -> float:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)["churn_probability"]


def high_risk_total(base_url: str) -> float:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as resp:
        text = resp.read().decode()
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith("high_risk_predictions_total{")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--high", type=int, default=20, help="high-risk requests to send")
    parser.add_argument("--low", type=int, default=20, help="low-risk requests to send")
    args = parser.parse_args()

    before = high_risk_total(args.url)
    batch = [HIGH_RISK] * args.high + [LOW_RISK] * args.low
    random.shuffle(batch)

    above = 0
    for customer in batch:
        payload = dict(customer)
        payload["MonthlyCharges"] = round(payload["MonthlyCharges"] + random.uniform(-3, 3), 2)
        prob = post(f"{args.url}/predict", payload)
        above += prob > 0.8

    after = high_risk_total(args.url)
    print(f"Sent {len(batch)} predictions, {above} of them with churn_probability > 0.8")
    print(f"high_risk_predictions_total: {before:.0f} -> {after:.0f}  (+{after - before:.0f})")
    if after - before != above:
        print("WARNING: counter delta does not match — check the /predict instrumentation")


if __name__ == "__main__":
    main()
