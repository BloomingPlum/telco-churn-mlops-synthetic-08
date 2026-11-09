# labs/part_a/lab_a2_retrain.py
"""
Lab A2: Automated Retrain — Drift Simulation + Trigger
Читає DEMO_DRIFT з env:
  false → "Model healthy"
  true  → симулює дрейф (+40% MonthlyCharges) → trigger

ЗМІНА порівняно з оригіналом:
  ФІКС БАГУ: усі фічі (не тільки числові) передаються в model.predict(),
  інакше ColumnTransformer падає з KeyError на відсутніх категоріальних
  колонках (Contract, PaymentMethod, ...).
"""
import os
import sys
from pathlib import Path
import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import f1_score

sys.path.insert(0, "/app")
from labs.common.env_utils import env_float

load_dotenv()

DATA_DIR     = Path("/app/data")
MODEL_PATH   = Path("/app/models/churn_model.pkl")
RESULTS_DIR  = Path("/app/results")
RESULTS_DIR.mkdir(exist_ok=True)

BASELINE_F1  = env_float("BASELINE_F1", 0.72)
F1_THRESHOLD = env_float("F1_THRESHOLD", 0.05)
DEMO_DRIFT   = os.getenv("DEMO_DRIFT", "false").lower() == "true"


def get_f1(model, csv_path: Path, n: int = 1000) -> float:
    df = pd.read_csv(csv_path).tail(n)
    # ФІКС: усі фічі, а не тільки числові (модель очікує і категоріальні)
    feat = [c for c in df.columns if c not in ("customerID", "Churn")]
    X = df[feat]
    y = df["Churn"].map({"Yes": 1, "No": 0})
    return f1_score(y, model.predict(X), zero_division=0)


def simulate_drift(src: Path) -> Path:
    """Симулює дрейф: MonthlyCharges × 1.4"""
    out = DATA_DIR / "telco_customers_drifted.csv"
    df  = pd.read_csv(src).copy()
    df["MonthlyCharges"] = df["MonthlyCharges"] * 1.4
    df.to_csv(out, index=False)
    return out


def main():
    print("=" * 55)
    mode = "DEMO DRIFT MODE" if DEMO_DRIFT else "NORMAL MODE"
    print(f"LAB A2: Automated Retrain Check  [{mode}]")
    print("=" * 55)

    model    = joblib.load(MODEL_PATH)
    csv_path = DATA_DIR / "telco_customers.csv"

    if DEMO_DRIFT:
        print("\n⚠️  Симулюємо дрейф: MonthlyCharges × 1.4")
        csv_path = simulate_drift(csv_path)
        print(f"   Дрейфований файл: {csv_path}")

    current_f1 = get_f1(model, csv_path)
    delta      = BASELINE_F1 - current_f1
    pct        = delta / BASELINE_F1 * 100

    print(f"\n📊 Baseline F1 : {BASELINE_F1:.4f}")
    print(f"   Current F1  : {current_f1:.4f}")
    print(f"   Delta       : {delta:+.4f}  ({pct:+.1f}%)")
    print(f"   Threshold   : {F1_THRESHOLD}")

    metrics_out = RESULTS_DIR / f"lab_a2_{'drift' if DEMO_DRIFT else 'normal'}.txt"
    metrics_out.write_text(
        f"mode={mode}\n"
        f"baseline_f1={BASELINE_F1:.4f}\n"
        f"current_f1={current_f1:.4f}\n"
        f"delta={delta:+.4f}\n"
        f"triggered={delta > F1_THRESHOLD}\n"
    )

    if delta > F1_THRESHOLD:
        print(f"\n🚨 DEGRADATION DETECTED  (Δ={delta:.4f} > {F1_THRESHOLD})")
        print("   → Initiating retraining pipeline...")
        print("   → In production:")
        print("     1. dvc repro             (оновити дані)")
        print("     2. python pipelines/train.py  (retrain + MLflow log)")
        print("     3. kubectl rollout restart    (canary deploy)")

        with mlflow.start_run(run_name="lab_a2_retrain_trigger"):
            mlflow.log_params({
                "demo_drift":  DEMO_DRIFT,
                "baseline_f1": BASELINE_F1,
                "threshold":   F1_THRESHOLD,
            })
            mlflow.log_metrics({
                "current_f1": current_f1,
                "delta":      delta,
            })
            mlflow.set_tags({
                "lab":            "A2",
                "trigger_reason": "f1_degradation",
                "action":         "retrain_initiated",
                "demo_mode":      str(DEMO_DRIFT),
            })

        print(f"\n   ✅ MLflow: retrain trigger event logged")
        print(f"   ✅ Метрики збережено: {metrics_out}")
        print(f"\n   💡 Щоб побачити різницю: порівняй цей run")
        print(f"      з попереднім у MLflow → http://localhost:5000")

    else:
        print(f"\n✅ Model healthy — no action needed")
        print(f"   F1 delta {delta:+.4f} в межах допустимого ({F1_THRESHOLD})")
        print(f"   Наступна перевірка: завтра 02:00 (cron)")
        print(f"\n   💡 Запусти lab-a2-drift щоб побачити trigger:")
        print(f"      docker compose --profile part-a-drift up lab-a2-drift")

    print(f"\n📁 Метрики: {metrics_out}")


if __name__ == "__main__":
    main()
