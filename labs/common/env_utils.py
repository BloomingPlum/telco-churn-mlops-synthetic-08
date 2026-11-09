# labs/common/env_utils.py
"""
Безпечне читання числових змінних середовища.

ІСТОРІЯ БАГА: `int(os.getenv("X", "10"))` виглядає безпечно, але
`os.getenv(name, default)` повертає default лише якщо змінної НЕМАЄ
взагалі. Якщо вона Є, але порожня (`X=` у .env-файлі — саме так було
записано в .env.theme9.example для LLM_RPM_LIMIT/RAGAS_MAX_WORKERS,
щоб позначити "не задано, використай дефолт"), `os.getenv` поверне
порожній рядок `''`, і `int('')` впаде з
`ValueError: invalid literal for int() with base 10: ''`.

Це сталось миттєво для ВСІХ ітерацій одразу (0 мережевих викликів,
~3000 it/s) — ознака, що помилка була в парсингу конфігурації, а не
в API-виклику. env_int/env_float трактують порожній рядок як "не
задано" і використовують дефолт, як і малось на увазі.
"""
import os


def env_int(name: str, default: int) -> int:
    val = os.getenv(name, "").strip()
    return int(val) if val else default


def env_float(name: str, default: float) -> float:
    val = os.getenv(name, "").strip()
    return float(val) if val else default
