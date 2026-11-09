# labs/part_a/lab_a1_ml_eval.py
"""
Lab A1: Повний ML Evaluation на Churn Model
- Threshold sweep з cost matrix
- MLflow logging метрик та параметрів
- Збереження графіків у /app/results/

ЗМІНА порівняно з оригіналом:
  ФІКС БАГУ: `select_dtypes(include="number")` відкидав категоріальні
  колонки (Contract, PaymentMethod, InternetService, ...), яких очікує
  ColumnTransformer моделі за іменем → model.predict_proba() падав з
  KeyError. Тепер передаємо ВСІ фічі (все окрім customerID/Churn) —
  так само, як під час тренування в pipelines/train.py.
"""
import os
import sys
from pathlib import Path
import joblib
import mlflow
import matplotlib
matplotlib.use("Agg")   # без GUI — для Docker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, "/app")
from labs.common.env_utils import env_int

load_dotenv()

COST_FP    = env_int("COST_FP", 10)
COST_FN    = env_int("COST_FN", 300)
N_TEST     = env_int("N_TEST_ROWS", 2000)

DATA_DIR    = Path("/app/data")
MODEL_PATH  = Path("/app/models/churn_model.pkl")
RESULTS_DIR = Path("/app/results")
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 55)
    print("LAB A1: ML Evaluation на Churn Model")
    print(f"        cost_FP=${COST_FP}  cost_FN=${COST_FN}")
    print("=" * 55)

    # ── 1. Завантаження ──────────────────────────────────────────
    model = joblib.load(MODEL_PATH)
    df    = pd.read_csv(DATA_DIR / "telco_customers.csv")
    df    = df.sort_values("RecordDate").reset_index(drop=True)  # temporal split, не random

    # ── ФІКС: усі фічі, включно з категоріальними ─────────────────
    feature_cols = [c for c in df.columns if c not in ("customerID", "Churn")]

    X_test = df.tail(N_TEST)[feature_cols]
    y_test = df.tail(N_TEST)["Churn"].map({"Yes": 1, "No": 0})
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n📂 Test set: {len(y_test)} рядків  "
          f"| Churn rate: {y_test.mean():.1%}")

    # ── 2. Baseline ───────────────────────────────────────────────
    y_base = (y_prob >= 0.5).astype(int)
    print(f"\n📊 Baseline (t=0.50):")
    print(classification_report(
        y_test, y_base, target_names=["Loyal", "Churn"]))
    print(f"   AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

    # ── 3. Threshold sweep ────────────────────────────────────────
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45,
                  0.50, 0.55, 0.60, 0.65, 0.70]
    records = []
    best_t, best_cost = 0.5, float("inf")

    print(f"\n💰 Threshold sweep:")
    print(f"   {'t':>6} {'F1':>7} {'Prec':>7} "
          f"{'Rec':>7} {'BizCost':>10}")
    print("   " + "─" * 45)

    for t in thresholds:
        y_t  = (y_prob >= t).astype(int)
        cm   = confusion_matrix(y_test, y_t)
        tn, fp, fn, tp = (cm.ravel() if cm.size == 4
                          else (0, 0, 0, 0))
        cost = COST_FP * fp + COST_FN * fn
        f1   = f1_score(y_test, y_t, zero_division=0)
        prec = precision_score(y_test, y_t, zero_division=0)
        rec  = recall_score(y_test, y_t, zero_division=0)
        mark = " ◄ OPTIMAL" if cost < best_cost else ""
        print(f"   {t:>6.2f} {f1:>7.4f} {prec:>7.4f} "
              f"{rec:>7.4f} ${cost:>9,.0f}{mark}")
        records.append(dict(t=t, f1=f1, prec=prec,
                            rec=rec, cost=cost, fp=fp, fn=fn))
        if cost < best_cost:
            best_cost, best_t = cost, t

    print(f"\n✅ Оптимальний threshold: {best_t} "
          f"(бізнес-вартість: ${best_cost:,.0f})")

    # ── 4. MLflow logging ─────────────────────────────────────────
    y_opt = (y_prob >= best_t).astype(int)
    with mlflow.start_run(run_name=f"lab_a1_threshold_{best_t}"):
        mlflow.log_params({
            "threshold":  best_t,
            "cost_fp":    COST_FP,
            "cost_fn":    COST_FN,
            "test_size":  len(y_test),
            "model_type": type(model.named_steps["classifier"]).__name__
                          if hasattr(model, "named_steps")
                          else type(model).__name__,
        })
        mlflow.log_metrics({
            "auc_roc":        roc_auc_score(y_test, y_prob),
            "f1_optimal":     f1_score(y_test, y_opt, zero_division=0),
            "precision":      precision_score(y_test, y_opt, zero_division=0),
            "recall":         recall_score(y_test, y_opt, zero_division=0),
            "biz_cost":       best_cost,
            "f1_baseline":    f1_score(y_test, y_base, zero_division=0),
            "biz_cost_improvement": (
                records[thresholds.index(0.5)]["cost"] - best_cost
            ),
        })
        mlflow.set_tags({
            "lab":              "A1",
            "eval_dataset":     "telco_customers_v1",
            "theme":            "9",
        })

    print(f"\n📝 MLflow run logged → {os.getenv('MLFLOW_TRACKING_URI')}")

    # ── 5. Confusion matrix + threshold plot ─────────────────────
    cm_opt = confusion_matrix(y_test, y_opt)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im = axes[0].imshow(cm_opt, cmap="Blues")
    plt.colorbar(im, ax=axes[0])
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            val   = cm_opt[i, j]
            color = "white" if val > cm_opt.max() / 2 else "black"
            axes[0].text(j, i,
                         f"{labels[i][j]}\n{val:,}",
                         ha="center", va="center",
                         color=color, fontsize=14, fontweight="bold")
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Loyal", "Churn"])
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["Loyal", "Churn"])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].set_title(f"Confusion Matrix  (t={best_t})", fontsize=13)

    ts    = [r["t"]    for r in records]
    f1s   = [r["f1"]   for r in records]
    costs = [r["cost"] for r in records]
    ax_f1   = axes[1]
    ax_cost = ax_f1.twinx()
    ax_f1.plot(ts, f1s, "o-", color="steelblue",
               linewidth=2, label="F1 Score")
    ax_cost.plot(ts, [c / 1000 for c in costs],
                 "s--", color="tomato", linewidth=2,
                 label=f"Biz Cost (K$, FP={COST_FP}, FN={COST_FN})")
    ax_f1.axvline(best_t, color="green", linestyle=":",
                  linewidth=2, label=f"Optimal t={best_t}")
    ax_f1.set_xlabel("Threshold"); ax_f1.set_ylabel("F1", color="steelblue")
    ax_cost.set_ylabel("Business Cost (K$)", color="tomato")
    ax_f1.set_title("Threshold vs F1 & Business Cost", fontsize=13)
    ax_f1.legend(loc="upper left"); ax_cost.legend(loc="upper right")

    plt.suptitle("Lab A1 — Churn Model Evaluation", fontsize=15, y=1.01)
    plt.tight_layout()
    out_path = RESULTS_DIR / "lab_a1_results.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"📊 Графік: {out_path}")
    print("   Переглянути: http://localhost:8888/lab_a1_results.png")


if __name__ == "__main__":
    main()
