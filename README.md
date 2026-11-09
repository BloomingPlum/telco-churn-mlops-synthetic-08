# Тема 9 — Docker Lab Setup
## AI Evaluation, Drift & Continuous Improvement

> **Репо-основа:** `telco-churn-mlops-synthetic-08`
> **Що додається:** лабораторні роботи теми 9 як Docker-сервіси
> **Запуск:** `make -f Makefile.theme9 theme9-all` або по одній лабі через `make -f Makefile.theme9 lab-a1`
> **Модуль 8 (моніторинг):** [docs/MODULE8_MONITORING.md](docs/MODULE8_MONITORING.md) · попередній README — [docs/README_main.md](docs/README_main.md)

---

## ⚠️ Що виправлено в цій версії (важливо прочитати)

Під час перевірки перед додаванням у репозиторій знайдено і виправлено 4 проблеми,
які заблокували би демонстрацію лабораторних:

1. **Немає `data/telco_customers.csv` у чистому клоні репо.** Файл навмисно в
   `.gitignore` (основний репо генерує його сам). Потрібно один раз викликати
   `make -f Makefile.theme9 generate-data` (запускає існуючий сервіс `generator`
   з `docker-compose.yml`).
2. **`models/churn_model.pkl` у поточному репозиторії пошкоджений** — не
   завантажується (`MemoryError` у `joblib.load`, незалежно від версії
   numpy/scikit-learn — файл фізично обірваний, схоже на пошкодження при
   заливанні бінарника на GitHub). Рішення: перетренувати модель локально —
   `make -f Makefile.theme9 train-model` (обгортка над `pipelines/train.py`).
3. **Баг вибору фіч у всіх лабораторних скриптах.** Оригінальний код будував
   `X_test` через `df[feature_cols].select_dtypes(include="number")`, що
   відкидає ВСІ категоріальні колонки (`Contract`, `PaymentMethod`,
   `InternetService` та ін.). Модель — це `sklearn.Pipeline` з
   `ColumnTransformer`, який очікує ці колонки за іменем (`num: passthrough`
   + `cat: OneHotEncoder`), тож `model.predict()` падав з `KeyError`.
   Виправлено у `lab_a1_ml_eval.py`, `lab_a2_retrain.py`,
   `generate_explanations.py` — тепер передаються всі фічі, як і під час
   тренування.
4. **RAGAS мовчки вимагав `OPENAI_API_KEY`.** `evaluate()` без явних
   `llm=`/`embeddings=` за замовчуванням використовує `ChatOpenAI` +
   `OpenAIEmbeddings`. Тепер підключений мультипровайдерний шар — див.
   розділ нижче. **OpenAI більше не потрібен ніде.**

Також відсутній `docker/nginx-results.conf`, на який посилався
`docker-compose.theme9.yml` — файл додано.

---

## 🆓 Безкоштовні LLM-ключі (замість OPENAI_API_KEY)

У жодному зі скриптів тепер немає обов'язкової залежності від OpenAI.
Обери **ОДНОГО** провайдера через `LLM_PROVIDER` у `.env.theme9`:

| Провайдер | `LLM_PROVIDER=` | Вартість | Де взяти ключ | Швидкість | Якість пояснень | Потрібен інтернет на занятті? |
|---|---|---|---|---|---|---|
| **Google Gemini** (рекомендовано для груп) | `gemini` | Безкоштовно (щедрий free tier) | https://aistudio.google.com/apikey — без картки | Середня | Добра | Так |
| **Groq** (рекомендовано, якщо потрібна швидкість) | `groq` | Безкоштовно (free tier, ліміт запитів/хв) | https://console.groq.com/keys | Дуже висока (LPU) | Добра (Llama 3.3 70B) | Так |
| **Anthropic** (код вже був написаний під нього) | `anthropic` | Платно, дуже дешево (~$0.15 на всі лаби групи) | https://console.anthropic.com | Висока | Найкраща | Так |
| **Ollama** (для аудиторії без мережі) | `ollama` | Безкоштовно, без ключа | не потрібен — локальна модель у Docker | Залежить від CPU/RAM (без GPU повільно) | Слабша (модель ~3B) | **Ні** (після одноразового `ollama-pull` з інтернетом) |

**Для embeddings у RAGAS (метрика `answer_relevancy`) ключ не потрібен ніколи** —
використовується локальна модель `sentence-transformers/all-MiniLM-L6-v2`
(HuggingFace), яка працює повністю офлайн після першого завантаження ваг
(~90 МБ, тягнеться автоматично під час `docker build`).

### Швидкий старт з Gemini (найпростіший безкоштовний варіант)

```bash
cp .env.theme9.example .env.theme9
```

У `.env.theme9`:
```bash
LLM_PROVIDER=gemini
GOOGLE_API_KEY=AIza...    # aistudio.google.com/apikey, 30 секунд, без картки
```

### Швидкий старт з Groq (якщо потрібна максимальна швидкість на занятті)

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...      # console.groq.com/keys
```

### Якщо в тебе вже є Anthropic-ключ

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

> 💡 **Для групи студентів:** видай кожному свій безкоштовний Gemini-ключ
> (реєстрація займає хвилину, не потребує картки) — це найпростіший спосіб,
> щоб усі одночасно запускали лабораторні без спільного платного ключа.

### 🔌 Повністю офлайн-варіант: Ollama (без жодного ключа)

Якщо в аудиторії немає інтернету на самому занятті — використай `ollama`.
Модель виконується у власному Docker-сервісі, локально, без виходу в мережу.

**Єдина умова:** модель треба завантажити ЗАЗДАЛЕГІДЬ, поки інтернет є
(наприклад, за день до заняття, у себе вдома чи в офісі):

```bash
# 1. У .env.theme9:
LLM_PROVIDER=ollama
#    (OLLAMA_BASE_URL / OLLAMA_MODEL / OLLAMA_EMBED_MODEL — можна лишити дефолтні)

# 2. Підняти Ollama і завантажити модель (ОДИН РАЗ, з інтернетом):
make -f Makefile.theme9 ollama-pull

# 3. Перевірити, що все готово офлайн:
make -f Makefile.theme9 ollama-check
```

Після цього кроку `ollama-pull` можна вимикати інтернет — усі лаби
(`generate_explanations.py`, `lab_b1_ragas.py`, `lab_b2_judge.py`) працюють
проти локального сервіса `ollama` в тій самій Docker-мережі `mlops-net`.

**Компроміси, які варто знати заздалегідь:**
- Дефолтна модель `llama3.2:3b` — маленька (3 млрд параметрів), тож якість
  churn-пояснень і оцінок LLM-judge помітно слабша за Gemini/Groq/Anthropic.
  Для демонстрації самого **механізму** (RAGAS, LLM-as-judge, CI gate) цього
  достатньо; для "гарних" прикладів пояснень краще хмарний провайдер.
- Швидкість залежить від CPU/RAM ноутбука — без GPU 20-50 пояснень можуть
  зайняти помітно довше, ніж хмарні API. Для демо варто зменшити
  `N_EVAL_SAMPLES` / `N_JUDGE_SAMPLES` (наприклад, до 5–10) і використати
  `make -f Makefile.theme9 lab-b1-fast`.
- Образ `ollama/ollama` + модель `llama3.2:3b` (~2 ГБ) + `nomic-embed-text`
  (~270 МБ) — переконайся, що це влізає на диск і завантажиться до
  заняття, а не в останній момент.
- Якщо хочеш взагалі прибрати залежність від `sentence-transformers` у
  Docker-образі лабораторій — постав `EMBEDDING_BACKEND=ollama` в
  `.env.theme9` (тоді і embeddings для RAGAS йдуть через `nomic-embed-text`
  в Ollama, а не через HuggingFace).

---

## Структура нових файлів

```
telco-churn-mlops-synthetic-08/
│
├── docker-compose.yml              ← ІСНУЮЧИЙ (теми 1–8)
├── docker-compose.theme9.yml       ← НОВИЙ: override для теми 9
│
├── docker/
│   ├── Dockerfile.labs             ← НОВИЙ: базовий образ лабораторій
│   └── nginx-results.conf          ← НОВИЙ: перегляд графіків
│
├── labs/
│   ├── common/
│   │   ├── llm_client.py           ← НОВИЙ: multi-provider LLM (anthropic/gemini/groq)
│   │   └── ragas_backend.py        ← НОВИЙ: RAGAS без OpenAI
│   ├── setup/
│   │   └── generate_explanations.py ← генерація churn пояснень (фікс фіч)
│   ├── part_a/
│   │   ├── lab_a1_ml_eval.py       ← threshold sweep + MLflow (фікс фіч)
│   │   └── lab_a2_retrain.py       ← drift detection + trigger (фікс фіч)
│   └── part_b/
│       ├── lab_b1_ragas.py         ← RAGAS eval на churn даних (без OpenAI)
│       └── lab_b2_judge.py         ← LLM-judge vs менеджер (multi-provider)
│
├── results/                        ← PNG графіки, CSV звіти (генерується автоматично)
│
├── .env.theme9.example             ← НОВИЙ: шаблон env (без OpenAI)
├── Makefile.theme9                 ← НОВИЙ: зручні команди + generate-data/train-model
└── requirements-theme9.txt         ← НОВИЙ: LLM залежності (без OpenAI)
```

---

## Сервіси

| Сервіс | Profile | Порт | Опис |
|--------|---------|------|------|
| `mlflow` | *(існуючий)* | 5000 | MLflow Tracking UI |
| `generator` | *(існуючий)* | — | Генерація `data/telco_customers.csv` |
| `lab-setup` | `setup` | — | Генерація `churn_explanations.json` |
| `lab-a1` | `part-a`, `labs` | — | ML eval + threshold sweep |
| `lab-a2` | `part-a`, `labs` | — | Retrain check (healthy mode) |
| `lab-a2-drift` | `part-a-drift`, `labs` | — | Retrain check (drift demo) |
| `lab-b1` | `part-b`, `labs` | — | RAGAS eval (50 прикладів) |
| `lab-b1-fast` | `part-b-fast` | — | RAGAS eval (10 прикладів, швидко) |
| `lab-b2` | `part-b`, `labs` | — | LLM-judge comparison |
| `lab-results` | `results`, `labs` | **8888** | Nginx: перегляд графіків |
| `ollama` | `ollama`, `labs` | 11434 | Локальний LLM (тільки якщо `LLM_PROVIDER=ollama`) |
| `ollama-pull` | `ollama-setup` | — | Одноразове завантаження моделі (`make ollama-pull`) |

---

## Перший запуск (один раз, у такому порядку)

### 1. Скопіювати .env та обрати безкоштовного провайдера

```bash
make -f Makefile.theme9 init
```

Відкрити `.env.theme9`, встановити `LLM_PROVIDER` і заповнити **один** ключ
(див. розділ "Безкоштовні LLM-ключі" вище).

### 2. Згенерувати дані та натренувати модель

```bash
make -f Makefile.theme9 train-model
```

Це послідовно: генерує `data/telco_customers.csv` (сервіс `generator`,
~50К рядків, ~1-2 хв) і тренує `models/churn_model.pkl` через
`pipelines/train.py`. **Не пропускай цей крок**, навіть якщо
`models/churn_model.pkl` вже лежить у репо — поточна закомічена версія
пошкоджена (див. розділ вище).

### 3. Перевірити артефакти

```bash
make -f Makefile.theme9 verify-artifacts
```

Має вивести два `✅`. Якщо бачиш `❌ churn_model.pkl ПОШКОДЖЕНО` —
`rm models/churn_model.pkl && make -f Makefile.theme9 train-model`.

### 4. Зібрати образ лабораторій

```bash
make -f Makefile.theme9 build
```

### 5. Згенерувати churn-пояснення (один раз, ~2–5 хвилин залежно від провайдера)

```bash
make -f Makefile.theme9 setup-data
```

Запустить `lab-setup` → обраний LLM_PROVIDER → збереже
`data/churn_explanations.json` (50 записів).

---

## Запуск під час заняття

```bash
make -f Makefile.theme9 theme9-up          # MLflow + Results viewer

# Частина A: ML Evaluation
make -f Makefile.theme9 lab-a1             # threshold sweep
make -f Makefile.theme9 lab-a2             # healthy model
make -f Makefile.theme9 lab-a2-drift       # drift demo

# Частина B: LLM Evaluation (без OpenAI)
make -f Makefile.theme9 lab-b1             # RAGAS (50 samples)
make -f Makefile.theme9 lab-b1-fast        # RAGAS (10 samples, швидко)
make -f Makefile.theme9 lab-b2             # LLM-judge
```

Або все одразу: `make -f Makefile.theme9 theme9-all`

---

## URLs після запуску

| URL | Що відкривається |
|-----|-----------------|
| http://localhost:5000 | MLflow UI — runs, metrics, artifacts |
| http://localhost:8888 | Nginx — всі PNG графіки результатів |
| http://localhost:8888/lab_a1_results.png | Lab A1: confusion matrix + threshold sweep |
| http://localhost:8888/lab_b1_ragas_results.png | Lab B1: RAGAS метрики + scatter |
| http://localhost:8888/lab_b2_judge_results.png | Lab B2: judge vs human correlation |

---

## Змінні середовища

| Змінна | За замовчуванням | Опис |
|--------|-----------------|------|
| `LLM_PROVIDER` | `gemini` | `anthropic` \| `gemini` \| `groq` \| `ollama` |
| `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` / `GROQ_API_KEY` | — | Заповнюється лише той, що відповідає провайдеру (`ollama` — без ключа) |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Адреса Docker-сервіса Ollama в мережі `mlops-net` |
| `OLLAMA_MODEL` / `OLLAMA_EMBED_MODEL` | `llama3.2:3b` / `nomic-embed-text` | Моделі, які тягне `make ollama-pull` |
| `EMBEDDING_BACKEND` | `huggingface` | `huggingface` (локально, в образі) \| `ollama` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Локальні embeddings для RAGAS, без ключа |
| `LANGCHAIN_API_KEY` | — | LangSmith трасування (опційно, безкоштовний free tier) |
| `COST_FP` / `COST_FN` | `10` / `300` | Вартість FP/FN ($) для threshold sweep |
| `BASELINE_F1` / `F1_THRESHOLD` | `0.72` / `0.05` | Baseline F1 та допустима деградація для trigger |
| `RAGAS_THRESHOLD` | `0.85` | Мінімальний faithfulness для eval gate |
| `N_EVAL_SAMPLES` / `N_JUDGE_SAMPLES` | `50` / `20` | Кількість прикладів для RAGAS / LLM-judge |
| `DEMO_DRIFT` | `false` | Симуляція дрейфу (lab-a2) |

---

## Типові проблеми

### `churn_model.pkl` не завантажується (`MemoryError`, `UnpicklingError`)

Файл пошкоджений — перетренуй:
```bash
rm models/churn_model.pkl
make -f Makefile.theme9 train-model
```

### `data/telco_customers.csv` не знайдено

```bash
make -f Makefile.theme9 generate-data
```

### `KeyError` при `model.predict()` на власному скрипті

Швидше за все, у X_test передані лише числові колонки. Модель — це
`ColumnTransformer` з окремою гілкою для категоріальних колонок
(`Contract`, `PaymentMethod`, ...) — передавай **усі** фічі, окрім
`customerID` і `Churn`.

### `429 ResourceExhausted` / `quota exceeded` при генерації пояснень

Безкоштовні тарифи мають жорсткий ліміт запитів/хвилину (наприклад,
`gemini-3.1-flash-lite` free tier — лише **15 RPM**). Це вже враховано:
`llm_client.py` сам тротлить запити під цей ліміт, а `lab_b1_ragas.py`
серіалізує паралельні виклики RAGAS (`RAGAS_MAX_WORKERS`). Якщо 429
все ж трапляється:
- це нормально бачити кілька рядків `⏳ Rate limit (...), спроба N/4` —
  скрипт сам почекає ~45с і продовжить;
- якщо падає постійно — можливо, ти запустив кілька лаб (`lab-b1` і
  `lab-b2`) одночасно і вони разом перевищують ліміт; запускай по одній;
- швидке рішення для демо: `make -f Makefile.theme9 lab-b1-fast`
  (10 прикладів замість 50) або тимчасово `LLM_PROVIDER=groq` (вищий RPM).

### `ValueError: metric [context_recall] ... requires ... ['ground_truth']`

`lab_b1_ragas.py` тепер вимагає, щоб `data/churn_explanations.json` містив
поле `ground_truth` для кожного запису (потрібне метриці RAGAS
`context_recall`). Якщо файл згенерований СТАРОЮ версією
`generate_explanations.py` (до цього фіксу) — цього поля там нема, і скрипт
сам це виявить та скаже, що робити:
```bash
rm data/churn_explanations.json
make -f Makefile.theme9 setup-data
```
`ground_truth` тут — не відповідь LLM, а детермінований (безкоштовний,
без API-виклику) шаблон на основі бізнес-правил із тих самих структурованих
даних клієнта (тип контракту, тариф, стаж тощо) — стенд-ін для того, що в
реальному проєкті писала б людина-експерт (CRM-менеджер).

### RAGAS питає про OpenAI / падає з `AuthenticationError`
Перевір `.env.theme9`: `LLM_PROVIDER` має відповідати заповненому ключу.
`make -f Makefile.theme9 check-env` покаже, чого не вистачає.

### `InconsistentVersionWarning` при завантаженні `churn_model.pkl`

```
Trying to unpickle estimator ... from version 1.7.2 when using version 1.5.2
```

`docker/Dockerfile.labs` пінує `scikit-learn==1.7.2` — саме версію, з якою
студенти реально тренують модель локально (`pipelines/train.py`). Якщо
все одно бачиш це попередження — версія scikit-learn на твоїй машині
відрізняється від 1.7.2. Дізнайся свою версію (`pip show scikit-learn`)
і заміни цифру в `docker/Dockerfile.labs` (рядок `"scikit-learn==1.7.2"`)
на свою — так контейнер лабораторних завжди читатиме pickle тим самим
бінарним форматом, яким його записано, і попередження зникне повністю.

Альтернативно — тренуй модель НЕ на своїй машині, а всередині самого
контейнера лабораторних (`docker compose ... run --rm lab-a1 bash`,
далі `python pipelines/train.py`), де версія вже гарантовано збігається
з тим, що зашито в образ.

⚠️ **Якщо ти ТАКОЖ використовуєш `Dockerfile.api`/`requirements-api.txt`
з основного репо** (теми 7-8, продакшн-подібний сервінг моделі) — там
scikit-learn і досі запінований на `1.5.2`. Якщо той сервіс вантажить
цю саму модель (треновану під 1.7.2), онови пін і там теж — інакше
побачиш те саме попередження вже в API-сервісі.

Це попередження саме по собі рідко ламає результат (Lab A1 з таким
warning'ом видав цілком коректні AUC=0.97 і збалансований precision/
recall) — але усунути розбіжність версій **надійніше**, ніж покладатись
на те, що конкретні класи (`OneHotEncoder`, `RandomForestClassifier`,
...) залишаться бінарно сумісними між мінорними релізами scikit-learn
і надалі.

### `404 models/gemini-... is not found for API version v1beta`

Google дуже часто вимикає старі моделі Gemini (усі 1.0, 1.5 і Gemini 2.0
Flash/Flash-Lite вже офіційно вимкнені й повертають 404 на будь-який запит).
`.env.theme9.example` пінує `GEMINI_MODEL=gemini-3.1-flash-lite` (безкоштовна,
стабільна, актуальна на момент написання) — але якщо ти читаєш це через
кілька місяців, ця модель теж може вже бути deprecated. Актуальний список:

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models?key=$GOOGLE_API_KEY" \
  | grep '"name"'
```

або офіційна сторінка https://ai.google.dev/gemini-api/docs/models — знайди
там модель з поміткою "Free tier" і встав її ім'я (без префікса `models/`)
у `GEMINI_MODEL` в `.env.theme9`.

### `LLM_PROVIDER=ollama`, а лаба каже "модель ще не завантажена"

```bash
make -f Makefile.theme9 ollama-pull     # потрібен інтернет, один раз
make -f Makefile.theme9 ollama-check    # має вивести ✅
```

### Прогрес-бар RAGAS стоїть на `0/200` і не рухається (довелось Ctrl+C)

Це не "все зламалось" — імовірно один HTTP-запит завис (мережевий блип),
а retry навіть не почався, бо запит формально ще не завершився помилкою.
Вже виправлено:
- на кожен LLM-клієнт (Anthropic/Gemini/Groq) додано явний `timeout`
  (дефолт 60с, `LLM_REQUEST_TIMEOUT` в `.env.theme9`) — тепер запит
  ГАРАНТОВАНО завершується помилкою за N секунд, а не висне назавжди;
- `log_tenacity=True` + налаштоване логування в `lab_b1_ragas.py` — тепер
  кожна повторна спроба RAGAS друкується в консоль (`[TENACITYRetry...]`),
  а не відбувається мовчки, тож видно, що процес живий, просто повільний.

Якщо після цього фіксу прогрес-бар все ще стоїть довше ~2 хвилин без
жодного логу — це вже привід зупинити (Ctrl+C) і перевірити мережу.

### `TypeError: generate_content() got an unexpected keyword argument 'temperature'`

Реальний баг сумісності між `ragas==0.1.14` і `langchain-google-genai==1.0.10`:
RAGAS завжди явно передає `temperature=` при кожному викликові judge-моделі,
навіть коли вона вже має власний `temperature` з конструктора — а ця версія
`langchain-google-genai` не прибирає цей дублікат перед тим, як переслати
залишок kwargs прямо в низькорівневий Google-клієнт, який такого параметра
не приймає. Вже виправлено в `ragas_backend.py` — тонкий підклас
`ChatGoogleGenerativeAI`, який прибирає лише цей зайвий kwarg (саме
значення temperature вже застосоване раніше, через `generation_config`,
тож поведінка не змінюється, лише зникає крах). Якщо після оновлення
патча ця помилка все ще з'являється — переконайся, що `docker build`
реально підхопив новий `ragas_backend.py` (перезбери образ).

### `429 ResourceExhausted` навіть після фіксу з temperature

Дві окремі причини, обидві вже виправлені в `ragas_backend.py`:
- `max_retries=0`, який передавався в конструктор `ChatGoogleGenerativeAI`,
  **нічого не робив** — у джерелах `langchain_google_genai==1.0.10`
  внутрішній retry-декоратор хардкодить `max_retries = 2` локально,
  повністю ігноруючи це поле (підтверджений баг саме цієї версії пакета);
- `max_workers=1` (get_run_config) лише серіалізує запити, але **не**
  гарантує дотримання ліміту 15/хв — швидкі послідовні запити (1-2с
  кожен) легко перевищують ліміт навіть без паралелізму.

Тепер підклас `ChatGoogleGenerativeAI` у `ragas_backend.py` викликає
той самий перевірений throttle (`_rate_limit_wait`), що і `llm_client.py`,
безпосередньо перед кожним реальним запитом — незалежно від того,
що робить (чи не робить) внутрішній retry бібліотеки.

### RAGAS повільно / timeout
```bash
make -f Makefile.theme9 lab-b1-fast   # 10 прикладів замість 50
```

### MLflow healthcheck ніколи не стає healthy

Використовується `python3 -c "import urllib.request; ..."` замість `curl`,
бо офіційний образ `ghcr.io/mlflow/mlflow` не гарантовано містить `curl`.
Якщо все ще падає:
```bash
docker compose -f docker-compose.yml up -d mlflow
docker compose -f docker-compose.yml logs mlflow
```

### `docker build` тягне гігабайти nvidia-пакетів (CUDA)

Вже виправлено в `docker/Dockerfile.labs`: перед основною установкою окремим
кроком ставиться CPU-only `torch` з офіційного індексу PyTorch
(`download.pytorch.org/whl/cpu`), тому `sentence-transformers` більше не
підмінює його на GPU-версію з ~10 пакетами `nvidia-*`. Якщо в логах бачиш
`WARNING: CPU-only torch index unreachable` — це не помилка, просто той домен
заблокований у твоїй мережі, і білд автоматично відкотився до звичайного
(важчого) шляху без переривання збірки.

### `no space left on device` під час `docker build`

Усі 7 лабораторних сервісів (`lab-setup`, `lab-a1`, `lab-a2`, `lab-a2-drift`,
`lab-b1`, `lab-b1-fast`, `lab-b2`) тепер діляться **одним** image-тегом
(`tc09-theme9-labs:latest` в `docker-compose.theme9.yml`) — раніше кожен мав
власний тег, і Docker намагався експортувати 7 повних копій багатогігабайтного
образу одночасно, вичерпуючи диск. Якщо все одно бракує місця:
```bash
docker builder prune -af   # чистить build-кеш (займає найбільше місця)
docker image prune -af     # прибирає "висячі" образи від невдалих спроб
```

### `docker build` падає з `ReadTimeoutError` / `TimeoutError: read operation timed out`

Це не помилка конфігурації — це тимчасовий обрив мережі під час завантаження
великого файлу (xgboost ~224 МБ, torch, transformers, ...). Вже виправлено
двома способами в `docker/Dockerfile.labs`:
- `PIP_DEFAULT_TIMEOUT=180` / `PIP_RETRIES=10` — щедріший таймаут і більше
  спроб для всіх `pip install` в образі;
- кожен великий пакет — в **окремому** `RUN`-шарі, тож якщо один пакет впаде,
  Docker кешує все, що встигло встановитись раніше, і не тягне все з нуля.

Якщо все одно падає — просто запусти збірку ще раз, вона продовжить з місця
обриву завдяки кешу шарів:
```bash
make -f Makefile.theme9 build
```
Якщо мережа настільки нестабільна, що це не допомагає — спробуй з іншої
мережі/часу доби, або збільш `PIP_DEFAULT_TIMEOUT` ще більше вручну
в `docker/Dockerfile.labs`.

---

## Часовий графік демо (довідка)

| Час | Команда | Що показуємо |
|-----|---------|-------------|
| 0:04 | `make -f Makefile.theme9 lab-a1` | Threshold sweep у терміналі |
| 0:10 | — | MLflow UI → метрики run |
| 0:12 | — | http://localhost:8888/lab_a1_results.png |
| 0:15 | `make -f Makefile.theme9 lab-a2` | "Model healthy" |
| 0:17 | `make -f Makefile.theme9 lab-a2-drift` | Деградація → trigger |
| 0:35 | `make -f Makefile.theme9 lab-b1` | RAGAS eval + worst cases |
| 0:43 | — | Gate: passed / failed |
| 0:45 | — | http://localhost:8888/lab_b1_ragas_results.png |
| 0:48 | `make -f Makefile.theme9 lab-b2` | LLM-judge рядок за рядком |
| 0:58 | — | http://localhost:8888/lab_b2_judge_results.png |

---

*Тема 9 · Modern MLOps / LLMOps / AgentOps in Production*
