#!/usr/bin/env python3
# scripts/theme9_check_env.py
"""Перевіряє, що LLM_PROVIDER обраний і потрібний ключ заповнений."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env.theme9")

provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
print(f"  LLM_PROVIDER = {provider}")

if provider == "ollama":
    print("  ℹ️  ollama: ключ не потрібен — перевір: make -f Makefile.theme9 ollama-check")
    sys.exit(0)

key_map = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
}
key = key_map.get(provider)

if key is None:
    print(f"  ❌ Невідомий LLM_PROVIDER='{provider}' (anthropic|gemini|groq|ollama)")
    sys.exit(1)

val = os.getenv(key, "").strip()
if val:
    print(f"  ✅ {key} заповнено")
    sys.exit(0)
else:
    print(f"  ❌ {key} НЕ заповнено — заповни у .env.theme9")
    sys.exit(1)
