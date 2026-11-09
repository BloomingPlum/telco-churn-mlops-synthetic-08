#!/usr/bin/env python3
# scripts/theme9_verify_artifacts.py
"""
Перевіряє, що models/churn_model.pkl та data/telco_customers.csv валідні.

ІСТОРІЯ БАГА: раніше ця перевірка була написана як один "склеєний" рядок
python3 -c "... try: ... except: ..." всередині Makefile.theme9. Make
з'єднує багаторядковий рецепт через backslash в ОДИН логічний рядок shell-команди,
а Python не дозволяє записати try/except через ';' без справжнього
переносу рядка й відступу — звідси `SyntaxError: invalid syntax`.
Рішення: винести перевірку в окремий, нормальний .py-файл.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # корінь репозиторію
MODEL_PATH = ROOT / "models" / "churn_model.pkl"
DATA_PATH = ROOT / "data" / "telco_customers.csv"

REQUIRED_COLUMNS = {
    "customerID", "Churn", "MonthlyCharges",
    "tenure", "Contract", "PaymentMethod", "RecordDate",
}

exit_code = 0

# ── 1. Модель ──────────────────────────────────────────────────────
try:
    import joblib
    model = joblib.load(MODEL_PATH)
    print(f"✅ churn_model.pkl завантажується коректно: {type(model).__name__}")
except Exception as e:
    print(f"❌ churn_model.pkl ПОШКОДЖЕНО: {e}")
    print("   → rm models/churn_model.pkl && make -f Makefile.theme9 train-model")
    exit_code = 1

# ── 2. Дані ────────────────────────────────────────────────────────
try:
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"відсутні колонки: {missing}")
    print(f"✅ telco_customers.csv валідний: {len(df)} рядків, {len(df.columns)} колонок")
except Exception as e:
    print(f"❌ telco_customers.csv проблема: {e}")
    print("   → make -f Makefile.theme9 generate-data")
    exit_code = 1

sys.exit(exit_code)
