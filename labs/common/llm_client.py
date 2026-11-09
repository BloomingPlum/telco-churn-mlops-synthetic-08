# labs/common/llm_client.py
"""
Єдина точка виклику LLM для всіх лаб теми 9.

Навіщо цей файл:
  Оригінальні лаби були жорстко прив'язані до Anthropic (generate_explanations.py,
  lab_b2_judge.py) і до OpenAI (RAGAS за замовчуванням). У студентів немає
  OPENAI_API_KEY, а Anthropic-ключ теж не завжди є. Цей модуль дозволяє
  перемикати провайдера ОДНІЄЮ змінною середовища — LLM_PROVIDER.

Підтримані провайдери (обери один через .env.theme9):
  LLM_PROVIDER=anthropic   → потрібен ANTHROPIC_API_KEY   (платно, дешево, ключ вже був у коді)
  LLM_PROVIDER=gemini      → потрібен GOOGLE_API_KEY       (БЕЗКОШТОВНО: aistudio.google.com/apikey)
  LLM_PROVIDER=groq        → потрібен GROQ_API_KEY         (БЕЗКОШТОВНО: console.groq.com, дуже швидко)
  LLM_PROVIDER=ollama      → ключ НЕ потрібен взагалі      (100% офлайн, локальна модель в Docker)

Жоден з цих шляхів НЕ використовує OpenAI.

Про режим ollama:
  Модель виконується у власному Docker-сервісі `ollama` (docker-compose.theme9.yml),
  усередині мережі mlops-net — жодного виходу в інтернет під час заняття не
  потрібно, ЯКЩО модель вже завантажена заздалегідь (`make -f Makefile.theme9
  ollama-pull`, один раз, поки є інтернет). Якість відповідей нижча, ніж у
  хмарних провайдерів (типова модель — 3B параметрів), а швидкість залежить
  від CPU/RAM ноутбука — тому це запасний варіант для аудиторій без мережі,
  а не варіант "за замовчуванням".
"""
import json
import os
import time

from .env_utils import env_int


# ── Rate limiting ──────────────────────────────────────────────────
# Безкоштовні тарифи мають ЖОРСТКІ ліміти запитів/хвилину (RPM), і якщо
# бити в API без пауз, після кількох десятків запитів отримаєш 429
# ResourceExhausted (саме так і сталось: gemini-3.1-flash-lite free tier
# = 15 RPM, а скрипт стріляв запитами настільки швидко, наскільки міг).
# Тут — простий client-side throttle: перед кожним викликом чекаємо,
# щоб інтервал між запитами був не меншим за 60/RPM секунд.
_DEFAULT_RPM = {
    "anthropic": 50,   # платний тариф, ліміт значно вищий
    "gemini": 15,      # free tier gemini-3.1-flash-lite (з повідомлення Google: limit: 15)
    "groq": 28,        # free tier, трохи запасу від типового лімту ~30 RPM
    "ollama": 0,       # 0 = без обмежень (локальна модель, немає rate limit)
}
_last_call_ts = {"t": 0.0}


def _rate_limit_wait(provider: str) -> None:
    rpm = env_int("LLM_RPM_LIMIT", _DEFAULT_RPM.get(provider, 30))
    if rpm <= 0:
        return
    min_interval = 60.0 / rpm
    now = time.monotonic()
    elapsed = now - _last_call_ts["t"]
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_call_ts["t"] = time.monotonic()


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "429" in msg
        or "resourceexhausted" in msg
        or "rate limit" in msg
        or "quota" in msg
        or "too many requests" in msg
    )


def _extract_json(raw: str) -> dict:
    """Прибирає markdown-огорожі ```json ... ``` якщо модель їх додала."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def get_text_completion(prompt: str, max_tokens: int = 300, temperature: float = 0.2) -> str:
    """Повертає сирий текст відповіді моделі (для генерації пояснень).

    Тротлить запити під RPM-ліміт провайдера і, якщо все ж прийшла
    429/ResourceExhausted (наприклад, через паралельний запуск кількох
    лаб одночасно), чекає й повторює до 3 разів — замість того, щоб
    одразу здатись і записати текст помилки як "пояснення".
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    max_attempts = 4
    backoff_seconds = 45  # трохи більше за типовий retry_delay від Google (~37-40с)

    for attempt in range(1, max_attempts + 1):
        _rate_limit_wait(provider)
        try:
            return _call_provider(provider, prompt, max_tokens, temperature)
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < max_attempts:
                print(
                    f"   ⏳ Rate limit ({provider}), спроба {attempt}/{max_attempts} — "
                    f"чекаємо {backoff_seconds}с..."
                )
                time.sleep(backoff_seconds)
                continue
            raise


def _call_provider(provider: str, prompt: str, max_tokens: int, temperature: float) -> str:
    # ВАЖЛИВО: явний timeout на кожен HTTP-клієнт. Без нього запит може
    # "зависнути" на невизначений час (мережевий блип, повільний TLS
    # handshake) — і тоді власний retry вище (get_text_completion) НІКОЛИ
    # не отримає шансу спрацювати, бо виняток просто не виникає. Явний
    # timeout гарантує, що запит ОБОВ'ЯЗКОВО завершиться помилкою за
    # REQUEST_TIMEOUT секунд.
    REQUEST_TIMEOUT = env_int("LLM_REQUEST_TIMEOUT", 60)

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(timeout=REQUEST_TIMEOUT)  # читає ANTHROPIC_API_KEY з env
        msg = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    elif provider == "gemini":
        # ВАЖЛИВО: використовуємо langchain-google-genai (той самий пакет,
        # що і ragas_backend.py), а не пакет `google-generativeai` напряму.
        # Причина: `google-generativeai` жорстко пінує точну версію
        # google-ai-generativelanguage, яка конфліктує з тим, що вимагає
        # langchain-google-genai>=2.0 — тримати обидва пакети одночасно
        # неможливо без конфлікту резолвера. Один пакет — один шлях.
        #
        # max_retries=0: власний retry-цикл вище вже керує повторами з
        # правильним backoff (~45с) — internal retry langchain'а (тільки
        # 2с між спробами) марно "спалює" квоту free tier, не чекаючи
        # стільки, скільки Google реально просить (retry_delay ~37-40с).
        from langchain_google_genai import ChatGoogleGenerativeAI
        chat = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
            google_api_key=os.environ["GOOGLE_API_KEY"],
            temperature=temperature,
            max_output_tokens=max_tokens,
            max_retries=0,
            timeout=REQUEST_TIMEOUT,
        )
        resp = chat.invoke(prompt)
        return resp.content

    elif provider == "groq":
        # Groq має OpenAI-сумісний endpoint — той самий `openai` SDK,
        # просто інший base_url і ключ. Жодного OpenAI-акаунту не потрібно.
        from openai import OpenAI
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
            timeout=REQUEST_TIMEOUT,
        )
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    elif provider == "ollama":
        # 100% локально, без ключа. host — ім'я Docker-сервіса `ollama`
        # у мережі mlops-net (див. docker-compose.theme9.yml), або
        # http://localhost:11434, якщо Ollama піднята поза Docker.
        import ollama as ollama_sdk
        client = ollama_sdk.Client(
            host=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        )
        resp = client.chat(
            model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        # ollama>=0.4 повертає об'єкт ChatResponse (атрибути),
        # старіші версії — звичайний dict. Підтримуємо обидва варіанти.
        msg = getattr(resp, "message", None) or resp["message"]
        return getattr(msg, "content", None) or msg["content"]

    else:
        raise ValueError(
            f"Невідомий LLM_PROVIDER='{provider}'. "
            "Допустимі значення: anthropic | gemini | groq | ollama"
        )


def get_json_completion(prompt: str, max_tokens: int = 300) -> dict:
    """Те саме, але одразу парсить JSON-відповідь (для LLM-judge)."""
    raw = get_text_completion(prompt, max_tokens=max_tokens, temperature=0.0)
    return _extract_json(raw)


def check_provider_configured() -> tuple[bool, str]:
    """Перевірка, що потрібний ключ для обраного провайдера заповнений
    (для ollama — ключ не потрібен, перевіряється натомість, що сервіс
    відповідає і потрібна модель вже завантажена)."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
    }

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        try:
            import ollama as ollama_sdk
            client = ollama_sdk.Client(host=base_url)
            names = [m["model"] for m in client.list().get("models", [])]
            if not any(model.split(":")[0] in n for n in names):
                return False, (
                    f"LLM_PROVIDER='ollama', але модель '{model}' ще не "
                    f"завантажена на {base_url}. Запусти (з інтернетом, один раз): "
                    f"make -f Makefile.theme9 ollama-pull"
                )
            return True, f"OK: провайдер=ollama, модель={model} готова на {base_url}"
        except Exception as e:
            return False, (
                f"LLM_PROVIDER='ollama', але сервіс на {base_url} не відповідає ({e}). "
                f"Перевір: docker compose ... up -d ollama"
            )

    if provider not in key_map:
        return False, f"LLM_PROVIDER='{provider}' не підтримується (anthropic|gemini|groq|ollama)"
    key_name = key_map[provider]
    if not os.getenv(key_name, "").strip():
        return False, f"LLM_PROVIDER='{provider}', але {key_name} не заповнено в .env.theme9"
    return True, f"OK: провайдер={provider}, ключ={key_name}"
