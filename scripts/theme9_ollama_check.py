#!/usr/bin/env python3
# scripts/theme9_ollama_check.py
"""Перевіряє, що сервіс Ollama піднятий і потрібна модель завантажена."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env.theme9")
os.environ.setdefault("LLM_PROVIDER", "ollama")

try:
    from labs.common.llm_client import check_provider_configured
    ok, msg = check_provider_configured()
    print(("  ✅ " if ok else "  ❌ ") + msg)
    sys.exit(0 if ok else 1)
except Exception as e:
    print(f"  ❌ Не вдалось перевірити Ollama: {e}")
    print("  → make -f Makefile.theme9 ollama-pull")
    sys.exit(1)
