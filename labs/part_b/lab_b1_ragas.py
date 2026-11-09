# labs/part_b/lab_b1_ragas.py
"""
Lab B1: RAGAS Evaluation на Churn-Поясненнях
- Читає data/churn_explanations.json (з lab-setup)
- Запускає RAGAS: faithfulness, answer_relevancy, context_recall, context_precision
- Показує worst cases (галюцинації)
- Gate check: passed / failed
- Зберігає графік у /app/results/

ГОЛОВНА ЗМІНА порівняно з оригіналом:
  RAGAS `evaluate()` БЕЗ явних llm=/embeddings= мовчки використовує
  ChatOpenAI + OpenAIEmbeddings → вимагає OPENAI_API_KEY, якого немає.
  Тепер явно передаємо LLM (Anthropic/Gemini/Groq — обирається через
  LLM_PROVIDER) та локальні, безкоштовні sentence-transformers embeddings.
  Жодного OpenAI-ключа більше не потрібно.
"""
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

sys.path.insert(0, "/app")
from labs.common.llm_client import check_provider_configured
from labs.common.ragas_backend import get_ragas_embeddings, get_ragas_llm, get_run_config
from labs.common.env_utils import env_int, env_float

load_dotenv()

# ── Видиме логування retry (замість мовчазних спроб на 0% прогресу) ──
# RAGAS (log_tenacity=True в get_run_config) логує кожну повторну спробу
# на рівні DEBUG через логери з динамічними іменами (ragas.retry.*,
# TENACITYRetry[...]) — за замовчуванням Python показує лише WARNING+,
# тож без цього налаштування retry відбувається "мовчки", і прогрес-бар
# на 0% виглядає як зависання, хоча насправді просто триває повтор.
# Глушимо шумні бібліотеки (httpx/urllib3/...), лишаючи видимими лише
# повідомлення від самого ragas.
logging.basicConfig(level=logging.DEBUG, format="   [%(name)s] %(message)s")
for _noisy in ("httpx", "httpcore", "urllib3", "google", "grpc", "matplotlib",
               "sentence_transformers", "transformers", "filelock", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

DATA_DIR    = Path("/app/data")
RESULTS_DIR = Path("/app/results")
RESULTS_DIR.mkdir(exist_ok=True)

THRESHOLD  = env_float("RAGAS_THRESHOLD", 0.85)
N_SAMPLES  = env_int("N_EVAL_SAMPLES", 50)
EXPL_FILE  = DATA_DIR / "churn_explanations.json"

METRICS = [faithfulness, answer_relevancy,
           context_recall, context_precision]
METRIC_COLS = ["faithfulness", "answer_relevancy",
               "context_recall", "context_precision"]


def main():
    print("=" * 55)
    print(f"LAB B1: RAGAS Eval на Churn-Поясненнях")
    print(f"        samples={N_SAMPLES}  threshold={THRESHOLD}")
    print("=" * 55)

    ok, msg = check_provider_configured()
    print(f"\n🔑 {msg}")
    if not ok:
        print("   → заповни відповідний ключ у .env.theme9 (див. README: 'Безкоштовні LLM').")
        sys.exit(1)

    # ── 1. Завантаження ──────────────────────────────────────────
    if not EXPL_FILE.exists():
        print(f"❌ Файл не знайдено: {EXPL_FILE}")
        print("   Спочатку запусти: --profile setup up lab-setup")
        raise SystemExit(1)

    raw = json.loads(EXPL_FILE.read_text(encoding="utf-8"))[:N_SAMPLES]
    print(f"\n📂 Завантажено {len(raw)} churn-пояснень")

    # Метрика context_recall вимагає колонку ground_truth (ragas==0.1.14
    # кидає ValueError без неї). Якщо файл згенеровано СТАРОЮ версією
    # generate_explanations.py (до фіксу) — цього поля там нема.
    if any("ground_truth" not in r for r in raw):
        print(f"\n❌ У {EXPL_FILE} немає поля 'ground_truth' "
              f"(файл згенеровано старою версією скрипта).")
        print(f"   → rm {EXPL_FILE} && make -f Makefile.theme9 setup-data")
        raise SystemExit(1)

    # ── 2. RAGAS dataset ─────────────────────────────────────────
    dataset = Dataset.from_dict({
        "question":     [r["question"]     for r in raw],
        "answer":       [r["answer"]       for r in raw],
        "contexts":     [[r["context"]]    for r in raw],
        "ground_truth": [r["ground_truth"] for r in raw],
    })

    # ── 3. Eval (без OpenAI!) ───────────────────────────────────
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    print(f"\n⏳ Запускаємо RAGAS через провайдер '{provider}' "
          f"+ локальні embeddings (≈{N_SAMPLES * 3}–{N_SAMPLES * 5}с)...")
    if provider in ("gemini", "groq"):
        print(f"   ℹ️  Безкоштовний тариф має ліміт запитів/хвилину — запити "
              f"серіалізовані (max_workers знижено), тому це помітно повільніше, "
              f"ніж з платним провайдером. Це очікувано, не помилка.")
    result = evaluate(
        dataset,
        metrics=METRICS,
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
        run_config=get_run_config(),
    )
    df     = result.to_pandas()
    df["customer_id"] = [r["customer_id"] for r in raw]
    df["churn_prob"]  = [r["churn_prob"]  for r in raw]

    # ── 4. Summary ────────────────────────────────────────────────
    print("\n📊 Результати RAGAS:")
    print("─" * 50)
    for col in METRIC_COLS:
        mean = df[col].mean()
        flag = "🔴" if mean < 0.80 else ("🟡" if mean < THRESHOLD else "🟢")
        print(f"  {flag}  {col:<28}: {mean:.4f}")
    print("─" * 50)
    overall = df[METRIC_COLS].mean().mean()
    print(f"     {'Overall mean':<28}: {overall:.4f}")

    # ── 5. Worst cases ────────────────────────────────────────────
    print(f"\n🚨 Топ-5 галюцинацій (найнижча faithfulness):")
    worst = df.nsmallest(5, "faithfulness")[
        ["customer_id", "faithfulness", "answer_relevancy", "churn_prob"]
    ]
    print(worst.to_string(index=False))

    w_idx = df["faithfulness"].idxmin()
    w_row = raw[w_idx]
    print(f"\n📋 Найгірше пояснення  "
          f"(faithfulness={df.loc[w_idx,'faithfulness']:.3f}):")
    print(f"   Customer : {w_row['customer_id']}")
    print(f"   Context  : {w_row['context']}")
    print(f"   Answer   :\n   {w_row['answer'][:300]}...")

    # ── 6. Save CSV ───────────────────────────────────────────────
    csv_out = RESULTS_DIR / "lab_b1_results.csv"
    df.to_csv(csv_out, index=False)

    # ── 7. Plots ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df[METRIC_COLS].boxplot(ax=axes[0], patch_artist=True)
    axes[0].axhline(THRESHOLD, color="red", linestyle="--",
                    linewidth=1.5, label=f"threshold={THRESHOLD}")
    axes[0].set_title("RAGAS Metrics Distribution", fontsize=13)
    axes[0].set_ylabel("Score (0–1)")
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend()

    sc = axes[1].scatter(
        df["churn_prob"], df["faithfulness"],
        c=df["faithfulness"], cmap="RdYlGn",
        vmin=0.5, vmax=1.0, s=80, alpha=0.85, edgecolors="gray"
    )
    plt.colorbar(sc, ax=axes[1], label="Faithfulness")
    axes[1].axhline(THRESHOLD, color="red", linestyle="--",
                    linewidth=1.5, label=f"threshold={THRESHOLD}")
    axes[1].set_xlabel("Churn Probability (ML model)")
    axes[1].set_ylabel("Faithfulness (RAGAS)")
    axes[1].set_title("Faithfulness vs Churn Risk", fontsize=13)
    axes[1].legend()

    plt.suptitle("Lab B1 — RAGAS Churn Explanations Eval",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    img_out = RESULTS_DIR / "lab_b1_ragas_results.png"
    plt.savefig(img_out, dpi=120, bbox_inches="tight")

    # ── 8. Gate check ─────────────────────────────────────────────
    avg_faith = df["faithfulness"].mean()
    print("\n" + "=" * 55)
    if avg_faith >= THRESHOLD:
        print(f"✅ EVAL GATE PASSED  "
              f"faithfulness={avg_faith:.4f} ≥ {THRESHOLD}")
    else:
        print(f"❌ EVAL GATE FAILED  "
              f"faithfulness={avg_faith:.4f} < {THRESHOLD}")
        print(f"   → PR було б заблоковано у CI!")
        print(f"   → Worst customers: "
              f"{', '.join(worst['customer_id'].tolist())}")
    print("=" * 55)

    print(f"\n📊 Графік  : http://localhost:8888/lab_b1_ragas_results.png")
    print(f"📁 CSV     : {csv_out}")


if __name__ == "__main__":
    main()
