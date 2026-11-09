# labs/part_b/lab_b2_judge.py
"""
Lab B2: LLM-Judge vs CRM Manager Feedback
- Завантажує churn_explanations.json
- LLM-judge (Anthropic/Gemini/Groq) оцінює N пояснень
- Порівнює з human_score (симульованим або реальним)
- Кореляційний аналіз + аналіз по категоріях
- Зберігає графік у /app/results/

ЗМІНА порівняно з оригіналом: виклик судді йде через
labs/common/llm_client.py — можна обрати провайдера без OpenAI-ключа.
"""
import json
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, "/app")
from labs.common.llm_client import check_provider_configured, get_json_completion
from labs.common.env_utils import env_int, env_float

load_dotenv()

DATA_DIR    = Path("/app/data")
RESULTS_DIR = Path("/app/results")
RESULTS_DIR.mkdir(exist_ok=True)

N_SAMPLES      = env_int("N_JUDGE_SAMPLES", 20)
CORR_THRESHOLD = env_float("CORR_THRESHOLD", 0.75)
EXPL_FILE      = DATA_DIR / "churn_explanations.json"

JUDGE_PROMPT = """You are an expert evaluator for a telecom churn explanation system.

A CRM manager asked: {question}
The system had this customer data: {context}
The AI generated this explanation: {answer}

Evaluate FAITHFULNESS (0.0–1.0):
Does the explanation rely ONLY on the provided customer data?

Rules:
- 0.9–1.0: all claims directly in context data
- 0.7–0.9: minor domain extrapolation, acceptable
- < 0.7: hallucination (invented facts, competitors, programs not in data)

Respond ONLY as valid JSON (no markdown):
{{"score": 0.9, "reason": "one sentence max", "hallucinations": ["any invented facts"]}}"""


def judge(question: str, answer: str, context: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, answer=answer, context=context)
    return get_json_completion(prompt, max_tokens=256)


def simulate_human_score(churn_prob: float, seed: int) -> float:
    """Симуляція human_score якщо labeled файлу немає."""
    rng = random.Random(seed)
    base = 0.85 if churn_prob > 0.7 else 0.78
    return round(min(1.0, max(0.0, base + rng.gauss(0, 0.07))), 3)


def main():
    print("=" * 55)
    print(f"LAB B2: LLM-Judge vs CRM Manager Feedback")
    print(f"        n_samples={N_SAMPLES}  corr_threshold={CORR_THRESHOLD}")
    print("=" * 55)

    ok, msg = check_provider_configured()
    print(f"\n🔑 {msg}")
    if not ok:
        print("   → заповни відповідний ключ у .env.theme9 (див. README: 'Безкоштовні LLM').")
        sys.exit(1)

    if not EXPL_FILE.exists():
        print(f"❌ {EXPL_FILE} не знайдено. Запусти lab-setup спочатку.")
        raise SystemExit(1)

    raw = json.loads(EXPL_FILE.read_text(encoding="utf-8"))[:N_SAMPLES]
    print(f"\n📂 Завантажено {len(raw)} пояснень для оцінки")

    labeled_file = DATA_DIR / "churn_explanations_labeled.json"
    if labeled_file.exists():
        labeled = {r["customer_id"]: r["human_score"]
                   for r in json.loads(labeled_file.read_text())}
        print("   👤 Human scores: реальні (labeled.json)")
    else:
        labeled = {}
        print("   🤖 Human scores: симульовані (labeled.json не знайдено)")

    provider = os.getenv("LLM_PROVIDER", "anthropic")
    print(f"\n⏳ LLM-judge ('{provider}') оцінює {N_SAMPLES} пояснень...\n")

    records = []
    for i, row in enumerate(raw):
        h_score = labeled.get(
            row["customer_id"],
            simulate_human_score(row["churn_prob"], seed=i)
        )
        try:
            result = judge(
                question=row["question"],
                answer=row["answer"],
                context=row["context"],
            )
            llm_score = result["score"]
            reason    = result.get("reason", "")
            halluc    = result.get("hallucinations", [])
        except Exception as e:
            print(f"  ⚠️  [{i+1:02d}] {row['customer_id']}: "
                  f"judge error — {e}")
            llm_score, reason, halluc = 0.5, str(e), []

        delta = llm_score - h_score
        flag  = ("🔴" if llm_score < 0.70 else
                 "🟡" if llm_score < 0.85 else "🟢")
        halluc_str = f" ⚠️ {halluc}" if halluc else ""
        print(f"  {flag} [{i+1:02d}] {row['customer_id']:<15} "
              f"judge={llm_score:.3f}  human={h_score:.3f}  "
              f"Δ={delta:+.3f}  {reason[:50]}{halluc_str}")

        records.append({
            "customer_id":  row["customer_id"],
            "churn_prob":   row["churn_prob"],
            "llm_score":    llm_score,
            "human_score":  h_score,
            "delta":        delta,
            "hallucinations": halluc,
        })

    df = pd.DataFrame(records)

    corr = df[["llm_score", "human_score"]].corr().iloc[0, 1]
    status = ("✅ Judge надійний (r > 0.75)"
              if corr > CORR_THRESHOLD else
              "⚠️  Потрібна калібрація промпту (r ≤ 0.75)")

    print(f"\n📊 Pearson r = {corr:.3f}  →  {status}")

    gaps = df[abs(df["delta"]) > 0.25]
    if not gaps.empty:
        print(f"\n🔍 Значні розбіжності (|Δ| > 0.25): "
              f"{len(gaps)}/{N_SAMPLES}")
        print(gaps[["customer_id", "llm_score",
                     "human_score", "churn_prob"]].to_string(index=False))

    hallucs = df[df["hallucinations"].apply(bool)]
    if not hallucs.empty:
        print(f"\n🚨 Галюцинації виявлено у {len(hallucs)} поясненнях:")
        for _, r in hallucs.iterrows():
            print(f"   {r['customer_id']}: {r['hallucinations']}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    lims = [0.4, 1.05]
    sc = axes[0].scatter(
        df["human_score"], df["llm_score"],
        c=df["churn_prob"], cmap="YlOrRd",
        s=100, alpha=0.85, edgecolors="gray", linewidths=0.5
    )
    plt.colorbar(sc, ax=axes[0], label="Churn Probability")
    axes[0].plot(lims, lims, "k--", alpha=0.4, label="Perfect agreement")
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)
    axes[0].set_xlabel("CRM Manager Score")
    axes[0].set_ylabel("LLM Judge Score")
    axes[0].set_title(f"Judge vs Human  (Pearson r={corr:.2f})", fontsize=13)
    axes[0].legend()

    axes[1].scatter(
        df["churn_prob"], df["delta"],
        c=abs(df["delta"]), cmap="RdYlGn_r",
        s=80, alpha=0.85, edgecolors="gray", linewidths=0.5
    )
    axes[1].axhline(0,     color="black", linewidth=1.0)
    axes[1].axhline(0.25,  color="red",   linestyle="--",
                    alpha=0.6, label="±0.25 disagreement")
    axes[1].axhline(-0.25, color="red",   linestyle="--", alpha=0.6)
    axes[1].set_xlabel("Churn Probability")
    axes[1].set_ylabel("Judge − Human (score delta)")
    axes[1].set_title("Disagreement by Churn Risk", fontsize=13)
    axes[1].legend()

    plt.suptitle("Lab B2 — LLM-Judge vs CRM Manager", fontsize=15, y=1.01)
    plt.tight_layout()
    img_out = RESULTS_DIR / "lab_b2_judge_results.png"
    plt.savefig(img_out, dpi=120, bbox_inches="tight")

    csv_out = RESULTS_DIR / "lab_b2_results.csv"
    df.to_csv(csv_out, index=False)

    print(f"\n{'=' * 55}")
    if corr > CORR_THRESHOLD:
        print("✅ Висновок: judge надійний → можна використовувати у CI")
    else:
        print("⚠️  Висновок: додай few-shot приклади у JUDGE_PROMPT")
        print("   або спробуй іншого провайдера (LLM_PROVIDER) як судді")
    print(f"{'=' * 55}")

    print(f"\n📊 Графік  : http://localhost:8888/lab_b2_judge_results.png")
    print(f"📁 CSV     : {csv_out}")


if __name__ == "__main__":
    main()
