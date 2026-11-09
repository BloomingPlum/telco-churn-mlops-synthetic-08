# labs/setup/generate_explanations.py
"""
Крок 0: Генерація churn_explanations.json.
Запускається один раз до заняття через:
  docker compose -f docker-compose.yml -f docker-compose.theme9.yml \
    --profile setup run --rm lab-setup

ЗМІНИ порівняно з оригіналом:
  1) ФІКС БАГУ: раніше X будувався через
     `df[feature_cols].select_dtypes(include="number")`, що відкидало ВСІ
     категоріальні колонки (Contract, PaymentMethod, InternetService, ...).
     Модель churn_model.pkl — це sklearn Pipeline з ColumnTransformer,
     який явно очікує ЦІ категоріальні колонки за іменем. Без них
     model.predict_proba(X) впаде з KeyError. Тепер передаємо ВСІ
     фічі (все окрім customerID/Churn), як і під час тренування в
     pipelines/train.py.
  2) Замість жорсткого прив'язання до Anthropic — виклик через
     labs/common/llm_client.py (Anthropic / Gemini / Groq — на вибір).
"""
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, "/app")
from labs.common.llm_client import check_provider_configured, get_text_completion
from labs.common.env_utils import env_int

load_dotenv()

DATA_DIR   = Path("/app/data")
MODEL_PATH = Path("/app/models/churn_model.pkl")
OUT_FILE   = DATA_DIR / "churn_explanations.json"
N_SAMPLES  = env_int("N_EVAL_SAMPLES", 50)

EXPLAIN_PROMPT = """You are a CRM analyst for a telecom company.
Given this customer data, explain WHY they might churn and recommend ONE retention action.
Use ONLY the provided data — no assumptions, no invented facts.

Customer data:
- churn_probability: {churn_prob:.0%}
- monthly_charges: ${monthly_charges}
- tenure_months: {tenure}
- contract_type: {contract}
- payment_method: {payment}

Respond in Ukrainian, max 3 sentences."""


def build_ground_truth(row) -> str:
    """
    Детермінований "еталонний" опис причини відтоку — БЕЗ LLM-виклику,
    безкоштовно й миттєво. Потрібен для метрики RAGAS `context_recall`,
    яка звіряє, що модель retrieval-нула, з тим, що мало б бути знайдено
    (ragas==0.1.14 кидає ValueError без колонки ground_truth в датасеті).

    У реальному проєкті тут був би текст, написаний CRM-експертом
    (людиною) — див. презентацію, слайд 27 ("ground_truth = CRM Expert
    Response"). Для демо на 50 прикладах без ручної розмітки — прості
    бізнес-правила на тих самих структурованих даних, що бачить LLM.
    """
    reasons = []

    contract = str(row.get("Contract", ""))
    if "Month-to-month" in contract:
        reasons.append("короткостроковий контракт (місяць-до-місяця)")

    tenure = row.get("tenure")
    if tenure is not None and tenure < 12:
        reasons.append("невеликий стаж клієнта")

    monthly = row.get("MonthlyCharges")
    if monthly is not None and monthly > 70:
        reasons.append("високий щомісячний тариф")

    payment = str(row.get("PaymentMethod", ""))
    if "Electronic check" in payment:
        reasons.append("оплата електронним чеком")

    if str(row.get("TechSupport", "")) == "No":
        reasons.append("відсутність технічної підтримки")

    if str(row.get("OnlineSecurity", "")) == "No":
        reasons.append("відсутність онлайн-захисту")

    if not reasons:
        reasons = ["загальні ознаки зниженої залученості клієнта"]

    return f"Високий ризик відтоку через: {', '.join(reasons)}."


def main():
    print("=" * 55)
    print("THEME 9 SETUP: Генерація churn пояснень")
    print("=" * 55)

    ok, msg = check_provider_configured()
    print(f"\n🔑 {msg}")
    if not ok:
        print("   → заповни відповідний ключ у .env.theme9 (див. README: 'Безкоштовні LLM').")
        sys.exit(1)

    # Перевірка артефактів
    if not MODEL_PATH.exists():
        print(f"❌ Модель не знайдена: {MODEL_PATH}")
        print("   Спочатку: make -f Makefile.theme9 train-model")
        sys.exit(1)

    csv_path = DATA_DIR / "telco_customers.csv"
    if not csv_path.exists():
        print(f"❌ Дані не знайдені: {csv_path}")
        print("   Спочатку: make -f Makefile.theme9 generate-data")
        sys.exit(1)

    if OUT_FILE.exists():
        existing = json.loads(OUT_FILE.read_text())
        print(f"✅ Вже існує: {OUT_FILE} ({len(existing)} записів)")
        print("   Видалити файл щоб перегенерувати.")
        return

    model = joblib.load(MODEL_PATH)
    df    = pd.read_csv(csv_path)

    # ── ФІКС: усі фічі, а не тільки числові ────────────────────────
    feature_cols = [c for c in df.columns if c not in ("customerID", "Churn")]
    X_all = df[feature_cols]
    df["churn_prob"] = model.predict_proba(X_all)[:, 1]

    at_risk = df.nlargest(N_SAMPLES, "churn_prob").reset_index(drop=True)
    print(f"\n📊 Беремо топ-{N_SAMPLES} at-risk клієнтів")
    print(f"   Діапазон ризику: "
          f"{at_risk['churn_prob'].min():.0%}–"
          f"{at_risk['churn_prob'].max():.0%}")

    explanations = []
    for idx, row in tqdm(at_risk.iterrows(),
                         total=len(at_risk),
                         desc="Генеруємо пояснення"):
        cid = row.get("customerID", f"UA-{idx:05d}")
        try:
            answer = get_text_completion(
                EXPLAIN_PROMPT.format(
                    churn_prob=row["churn_prob"],
                    monthly_charges=row.get("MonthlyCharges", "N/A"),
                    tenure=row.get("tenure", "N/A"),
                    contract=row.get("Contract", "N/A"),
                    payment=row.get("PaymentMethod", "N/A"),
                ),
                max_tokens=300,
            )
        except Exception as e:
            print(f"   ⚠️  {cid}: API error — {e}")
            answer = f"Помилка генерації: {e}"

        explanations.append({
            "customer_id": cid,
            "question":    f"Чому клієнт {cid} може відтекти?",
            "context": (
                f"churn_prob={row['churn_prob']:.2f}, "
                f"MonthlyCharges={row.get('MonthlyCharges','N/A')}, "
                f"tenure={row.get('tenure','N/A')}, "
                f"Contract={row.get('Contract','N/A')}, "
                f"PaymentMethod={row.get('PaymentMethod','N/A')}"
            ),
            "answer":       answer,
            "ground_truth": build_ground_truth(row),
            "churn_prob":   float(row["churn_prob"]),
        })

    OUT_FILE.write_text(
        json.dumps(explanations, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n✅ Збережено: {OUT_FILE} ({len(explanations)} записів)")
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    if provider == "anthropic":
        print(f"   Орієнтовна вартість: ~${len(explanations) * 0.003:.2f}")
    else:
        print(f"   Провайдер '{provider}' — безкоштовний тариф, вартість ~$0.00")


if __name__ == "__main__":
    main()
