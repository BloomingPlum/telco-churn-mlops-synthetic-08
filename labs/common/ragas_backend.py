# labs/common/ragas_backend.py
"""
RAGAS за замовчуванням (evaluate() без параметрів llm=/embeddings=) мовчки
використовує ChatOpenAI + OpenAIEmbeddings — тобто вимагає OPENAI_API_KEY,
навіть якщо у промпт-коді жодного разу не згадано OpenAI.

Цей модуль підміняє і LLM, і embeddings на безкоштовні альтернативи:
  - LLM для RAGAS-суддів (faithfulness/answer_relevancy/...) — той самий
    провайдер, що і LLM_PROVIDER (anthropic / gemini / groq).
  - Embeddings — ЛОКАЛЬНА модель sentence-transformers (HuggingFace),
    працює повністю офлайн, без жодного API-ключа. Модель ваги
    (~90 МБ) завантажується один раз при першому запуску.

Використання в lab_b1_ragas.py:
    from labs.common.ragas_backend import get_ragas_llm, get_ragas_embeddings
    result = evaluate(dataset, metrics=METRICS,
                       llm=get_ragas_llm(), embeddings=get_ragas_embeddings())
"""
import os

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from .env_utils import env_int
from .llm_client import _rate_limit_wait


def get_ragas_llm():
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    # ВАЖЛИВО: явний timeout на кожен провайдер. Без нього HTTP-запит
    # може "зависнути" на невизначений час (мережевий блип, повільний
    # TLS handshake, тимчасова проблема на боці API) — і тоді ані наш
    # власний retry, ані retry RAGAS навіть НЕ ПОЧНУТЬСЯ, бо запит
    # ніколи не завершується (ні успіхом, ні помилкою). Прогрес-бар
    # стоїть на 0% нескінченно, і це виглядає як "все зламалось",
    # хоча насправді один запит просто ніколи не отримає відповіді.
    # Явний timeout гарантує: запит ОБОВ'ЯЗКОВО завершиться помилкою
    # за N секунд — і ТІЛЬКИ ТОДІ retry-логіка отримає шанс спрацювати.
    REQUEST_TIMEOUT = env_int("LLM_REQUEST_TIMEOUT", 60)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        chat = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            temperature=0,
            default_request_timeout=REQUEST_TIMEOUT,
        )
    elif provider == "gemini":
        # max_retries=0: retry-логіку бере на себе RunConfig ragas
        # (get_run_config, max_wait=90с) — щоб не було двох незалежних
        # шарів retry з різним backoff одночасно.
        #
        # ФІКС: ragas.llms.base.LangchainLLMWrapper.agenerate_text ЗАВЖДИ
        # явно передає temperature= як call-time kwarg до agenerate_prompt,
        # навіть коли модель вже має власний self.temperature з конструктора.
        # langchain_google_genai==1.0.10 не забирає цей "зайвий" temperature
        # з **kwargs перед тим, як переслати решту прямо в низькорівневий
        # GenerativeServiceClient.generate_content() — а той такого
        # параметра взагалі не приймає:
        #   TypeError: generate_content() got an unexpected keyword
        #   argument 'temperature'
        # Значення temperature вже коректно застосоване РАНІШЕ, всередині
        # _prepare_request/_prepare_params (через generation_config) —
        # тож видалення цього дубльованого kwarg нічого не змінює по суті,
        # лише прибирає крах. Підклас нижче — мінімальний, хірургічний фікс.
        #
        # ДРУГИЙ ФІКС (429 ResourceExhausted): параметр max_retries=0,
        # який передається в конструктор нижче, В ЦІЙ ВЕРСІЇ ПАКЕТА
        # НІЧОГО НЕ РОБИТЬ — у джерелах chat_models.py `_create_retry_
        # decorator()` жорстко хардкодить `max_retries = 2` локально,
        # повністю ігноруючи self.max_retries. Це підтверджений баг/
        # недороблення саме цієї версії. Крім того, max_workers=1
        # (get_run_config) серіалізує запити, але САМ ПО СОБІ НЕ
        # гарантує дотримання лімту 15 запитів/хв — якщо кожен запит
        # виконується швидко (1-2с), можна легко перевищити ліміт навіть
        # без паралелізму. Тому додаємо СПРАВЖНІЙ throttle прямо тут —
        # ту саму функцію _rate_limit_wait, яку вже перевірено в
        # llm_client.py (тримає інтервал 60/RPM між послідовними
        # викликами), викликану ПЕРЕД кожним реальним запитом.
        from langchain_google_genai import ChatGoogleGenerativeAI as _BaseChatGoogleGenerativeAI

        class ChatGoogleGenerativeAI(_BaseChatGoogleGenerativeAI):
            def _generate(self, *args, **kwargs):
                kwargs.pop("temperature", None)
                _rate_limit_wait("gemini")
                return super()._generate(*args, **kwargs)

            async def _agenerate(self, *args, **kwargs):
                kwargs.pop("temperature", None)
                _rate_limit_wait("gemini")
                return await super()._agenerate(*args, **kwargs)

        chat = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
            temperature=0,
            google_api_key=os.environ["GOOGLE_API_KEY"],
            max_retries=0,
            timeout=REQUEST_TIMEOUT,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        chat = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,
            api_key=os.environ["GROQ_API_KEY"],
            request_timeout=REQUEST_TIMEOUT,
        )
    elif provider == "ollama":
        # 100% локально, без ключа — модель виконується в сервісі `ollama`
        # (docker-compose.theme9.yml). Має бути заздалегідь завантажена:
        # make -f Makefile.theme9 ollama-pull (потребує інтернет ОДИН РАЗ).
        # ChatOllama не має окремого timeout-поля в pydantic-моделі — inference
        # локальний, тож "мережевого" зависання типу описаного вище тут
        # немає; повільність (слабкий CPU) — інша проблема, не лікується
        # таймаутом запиту.
        from langchain_ollama import ChatOllama
        chat = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
            temperature=0,
        )
    else:
        raise ValueError(f"Невідомий LLM_PROVIDER='{provider}' (anthropic|gemini|groq|ollama)")

    return LangchainLLMWrapper(chat)


# ── Rate limiting для RAGAS ─────────────────────────────────────────
# evaluate() за замовчуванням запускає до 16 ПАРАЛЕЛЬНИХ LLM-викликів
# (RunConfig.max_workers=16). Для платних API це нормально, але
# безкоштовний тариф gemini-3.1-flash-lite дозволяє лише 15 запитів/
# ХВИЛИНУ — 16 одночасних запитів миттєво впираються в 429
# ResourceExhausted. Тому серіалізуємо виклики для rate-limited
# провайдерів через max_workers=1 (RAGAS все одно ретраїть 429 сам,
# з експоненційним backoff до max_wait секунд — тож зменшення
# паралельності майже завжди вирішує проблему без інших змін).
_DEFAULT_MAX_WORKERS = {
    "anthropic": 4,
    "gemini": 1,
    "groq": 2,
    "ollama": 1,   # локальна модель — паралельні запити лише сповільнять CPU inference
}


def get_run_config():
    """RunConfig для evaluate(..., run_config=...), підлаштований під
    RPM-ліміти обраного провайдера. Використання:
        result = evaluate(dataset, metrics=METRICS,
                           llm=get_ragas_llm(), embeddings=get_ragas_embeddings(),
                           run_config=get_run_config())

    log_tenacity=True: без явного timeout на LLM-клієнті (див.
    get_ragas_llm) один зависший HTTP-запит міг чекати нескінченно,
    і прогрес-бар стояв на 0% без жодного пояснення. Тепер: (1) кожен
    запит гарантовано завершується помилкою за LLM_REQUEST_TIMEOUT
    секунд, (2) log_tenacity=True + logging-конфіг у lab_b1_ragas.py
    роблять кожну повторну спробу видимою в консолі, а не мовчазною.
    """
    from ragas.run_config import RunConfig

    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    max_workers = env_int("RAGAS_MAX_WORKERS", _DEFAULT_MAX_WORKERS.get(provider, 2))
    return RunConfig(
        max_workers=max_workers,
        max_wait=90,        # трохи більше за типовий retry_delay від Google (~37-40с)
        max_retries=8,      # з таймаутом 60с на запит — до ~12 хв worst-case на приклад
        timeout=240,
        log_tenacity=True,
    )


def get_ragas_embeddings():
    """
    Embeddings для RAGAS (метрика answer_relevancy).

    За замовчуванням — локальні sentence-transformers (HuggingFace),
    ваги вшиваються в образ під час `docker build` (Dockerfile.labs),
    тож ключ не потрібен НІКОЛИ, а на самому занятті інтернет теж не
    потрібен (ваги вже в образі).

    Якщо LLM_PROVIDER=ollama і хочеш, щоб і embeddings йшли через той
    самий локальний сервіс (наприклад, зовсім без sentence-transformers
    у образі) — встав EMBEDDING_BACKEND=ollama в .env.theme9. Тоді
    потрібно заздалегідь підтягнути модель embeddings:
        make -f Makefile.theme9 ollama-pull-embeddings
    """
    backend = os.getenv("EMBEDDING_BACKEND", "huggingface").lower()

    if backend == "ollama":
        from langchain_ollama import OllamaEmbeddings
        emb = OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
        )
        return LangchainEmbeddingsWrapper(emb)

    from langchain_community.embeddings import HuggingFaceEmbeddings

    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    hf = HuggingFaceEmbeddings(model_name=model_name)
    return LangchainEmbeddingsWrapper(hf)
