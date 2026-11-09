# Аналіз `labs_in_docker.zip` для репозиторію `telco-churn-mlops-synthetic-08`

Перевірив архів **і** сам репозиторій (склонував, завантажив модель, перевірив
схему датасету, спробував пропатчений код). Знайшов 5 реальних проблем, які
заблокували б демонстрацію на занятті, і одну (відсутність OpenAI-ключа),
про яку запитував ти сам. Усе виправлено — нижче список і готові файли.

---

## Що було не так

### 1. 🔴 Критично: `models/churn_model.pkl` у репозиторії пошкоджений
Завантажив модель напряму (`joblib.load`) — падає з `MemoryError` **незалежно
від версії** numpy/scikit-learn (перевірив і на найновіших, і на версіях,
точно узгоджених з `requirements-api.txt` репо). Розібрав pickle-заголовок
руками: поле довжини "фрейму" містить сміттєве число (`2314885530818446407`),
що значно перевищує розмір самого файлу — це ознака фізично обірваного/
пошкодженого бінарника (типова причина — заливання бінарного файлу через
git на Windows з автоконвертацією рядків). **Файл треба перетренувати, а не
просто "поправити версії залежностей".**

### 2. 🔴 Критично: `data/telco_customers.csv` відсутній у чистому клоні
Файл навмисно в `.gitignore` основного репо — генерується скриптом
`src/generate_dataset_ext.py` (сервіс `generator` у `docker-compose.yml`).
Лабораторні з архіву посилаються на нього, вважаючи, що він вже є.

### 3. 🔴 Критично: баг вибору фіч ламає `model.predict()` у трьох скриптах
`generate_explanations.py`, `lab_a1_ml_eval.py`, `lab_a2_retrain.py` будували
`X_test` через:
```python
X_test = df[feature_cols].select_dtypes(include="number")
```
Я завантажив реальний `churn_model.pkl` і подивився на його
`ColumnTransformer` — модель тренована на **всіх** колонках: чотири числові
йдуть `passthrough` (`SeniorCitizen, tenure, MonthlyCharges, TotalCharges`),
а понад десять категоріальних (`Contract`, `PaymentMethod`,
`InternetService`, ...) йдуть через `OneHotEncoder` **за іменем колонки**.
`select_dtypes(include="number")` прибирає категоріальні колонки повністю →
`ColumnTransformer` не знаходить очікувані імена → `KeyError` при першому ж
викликові `model.predict()` / `model.predict_proba()`. Це підтверджено кодом
`pipelines/train.py` в основному репо (`build_pipeline()`), який явно ділить
колонки на `numerical_cols` + `categorical_cols` і передає обидві групи.

**Виправлено:** тепер `X` = усі колонки, окрім `customerID` і `Churn` — так
само, як під час тренування.

### 4. 🟡 RAGAS мовчки вимагав `OPENAI_API_KEY`
`lab_b1_ragas.py` викликав `evaluate(dataset, metrics=METRICS)` без
`llm=`/`embeddings=`. RAGAS у такому разі підставляє `ChatOpenAI` +
`OpenAIEmbeddings` **за замовчуванням**, навіть якщо в промпт-коді жодного
разу не згадано OpenAI. Це і є прихована причина, чому без OpenAI-ключа лаба
падала. `generate_explanations.py` і `lab_b2_judge.py` вже були написані під
Anthropic — але Anthropic-ключа в тебе теж може не бути.

**Виправлено:** доданий шар `labs/common/llm_client.py` +
`labs/common/ragas_backend.py`, який дозволяє обрати провайдера однією
змінною `LLM_PROVIDER=anthropic|gemini|groq`, і **локальні, безкоштовні
embeddings** (`sentence-transformers`, без жодного ключа) для RAGAS. OpenAI
більше ніде не потрібен.

### 5. 🟡 Відсутній `docker/nginx-results.conf`
`docker-compose.theme9.yml` монтує цей файл, але його не було в архіві —
сервіс `lab-results` не піднявся б. Додав мінімальний конфіг зі статикою і
листингом файлів.

### 6. 🟡 Ризик: healthcheck MLflow використовує `curl`
Офіційний образ `ghcr.io/mlflow/mlflow` не гарантовано містить `curl` —
якщо healthcheck ніколи не стає `healthy`, усі лаби з
`depends_on: mlflow: condition: service_healthy` зависнуть у нескінченному
очікуванні. Переписав healthcheck на `python3 -c "import urllib.request..."`,
який точно є в образі (Python-based).

### 7. ⚪ Дрібниця: `lab_a1_ml_eval.py` сортував тест-спліт за `tenure`
Презентація (і сам датасет із колонкою `RecordDate`) явно будує сценарій
**temporal split** (концепт-дрейф у часі, тема 8). Сортування за `tenure`
замість `RecordDate` ламає цю ідею — тестова вибірка мала б бути "останні
записи в часі", а не "клієнти з найбільшим стажем". Виправлено на
`sort_values("RecordDate")`.

### 8. ⚪ Дрібниця: `check-env` у `Makefile.theme9` завжди перевіряв 3 фіксовані
ключі (включно з `OPENAI_API_KEY`), незалежно від того, який провайдер
реально обраний, і не повертав код помилки при провалі. Виправлено — перевіряє
лише ключ, що відповідає `LLM_PROVIDER`.

---

## 🆓 Безкоштовні рішення замість OpenAI (коротко)

| Провайдер | Ключ | Вартість | Де взяти |
|---|---|---|---|
| **Google Gemini** ⭐ рекомендую для групи | `GOOGLE_API_KEY` | Безкоштовно | aistudio.google.com/apikey — 30 сек, без картки |
| **Groq** ⭐ рекомендую, якщо важлива швидкість | `GROQ_API_KEY` | Безкоштовно (free tier) | console.groq.com/keys |
| Anthropic (код вже був написаний під нього) | `ANTHROPIC_API_KEY` | Дешево (~$0.15 на всі лаби) | console.anthropic.com |
| **Ollama** ⭐ якщо в аудиторії немає інтернету | — (без ключа) | Безкоштовно | локальний Docker-сервіс, модель тягнеться заздалегідь |
| Embeddings для RAGAS | — | Завжди безкоштовно | `sentence-transformers` локально (або через Ollama), без ключа |

### Четвертий провайдер — Ollama (повністю офлайн)

Додав `LLM_PROVIDER=ollama` як рівноправну опцію в `labs/common/llm_client.py`
і `labs/common/ragas_backend.py`. Це не HTTP-виклик до хмари, а окремий
Docker-сервіс `ollama/ollama` (у `docker-compose.theme9.yml`, мережа
`mlops-net`), з моделлю `llama3.2:3b` за замовчуванням.

**Важливо зрозуміти компроміс:** інтернет потрібен **один раз** — щоб
завантажити модель (`make -f Makefile.theme9 ollama-pull`, ~2 ГБ). Після
цього все працює офлайн. Але:
- якість відповідей 3B-моделі помітно нижча, ніж у Gemini/Groq/Anthropic —
  для демонстрації самого механізму (RAGAS/LLM-judge/CI gate) достатньо, для
  "гарних" прикладів пояснень — краще хмарний провайдер;
- швидкість залежить від CPU/RAM ноутбука, без GPU може бути повільно —
  для живого демо раджу `N_EVAL_SAMPLES=5-10` і `lab-b1-fast`;
- можна також перевести embeddings для RAGAS на Ollama
  (`EMBEDDING_BACKEND=ollama`, модель `nomic-embed-text`), якщо хочеш
  прибрати `sentence-transformers` із Docker-образу повністю.

Перевірено реальною установкою: пакети `ollama` (0.3.x) і `langchain-ollama`
(0.2.x) існують на PyPI з очікуваним API — `ollama.Client(host=...).chat(...)`,
`langchain_ollama.ChatOllama`, `langchain_ollama.OllamaEmbeddings` — усі
імпорти підтверджені в ізольованому середовищі, так само як і для
Anthropic/Gemini/Groq шляху раніше.

Перемикається однією змінною `LLM_PROVIDER` у `.env.theme9`. Детальніше —
дивись новий `README.md` (розділ "Безкоштовні LLM-ключі").

Для класу з багатьма студентами найпростіше: кожен реєструє свій **Gemini**
ключ (безкоштовно, без картки, хвилина часу) — не потрібен спільний платний
ключ на всю групу.

---

## Що саме додати/замінити в репозиторії

Скопіюй файли з цього патча в корінь `telco-churn-mlops-synthetic-08/`,
зберігаючи структуру:

```
telco-churn-mlops-synthetic-08/
├── README.md                        ← ЗАМІНИТИ (описано зміни + free keys)
├── Makefile.theme9                  ← ЗАМІНИТИ (fix check-env + нові таргети)
├── docker-compose.theme9.yml        ← ЗАМІНИТИ (healthcheck fix + нові env)
├── requirements-theme9.txt          ← ЗАМІНИТИ (без OpenAI-обов'язковості)
├── .env.theme9.example              ← ЗАМІНИТИ (LLM_PROVIDER замість OpenAI)
├── docker/
│   ├── Dockerfile.labs              ← ЗАМІНИТИ
│   └── nginx-results.conf           ← ДОДАТИ (новий файл)
└── labs/
    ├── common/
    │   ├── __init__.py              ← ДОДАТИ
    │   ├── llm_client.py            ← ДОДАТИ
    │   └── ragas_backend.py         ← ДОДАТИ
    ├── setup/
    │   └── generate_explanations.py ← ДОДАТИ (сюди, у підпапку setup/)
    ├── part_a/
    │   ├── lab_a1_ml_eval.py        ← ДОДАТИ (у підпапку part_a/)
    │   └── lab_a2_retrain.py        ← ДОДАТИ (у підпапку part_a/)
    └── part_b/
        ├── lab_b1_ragas.py          ← ДОДАТИ (у підпапку part_b/)
        └── lab_b2_judge.py          ← ДОДАТИ (у підпапку part_b/)
```

**Важливо:** у вихідному архіві всі `.py`-файли лежали пласко в одній
папці — `docker-compose.theme9.yml` і `Dockerfile.labs` очікують саме
підпапки `labs/setup/`, `labs/part_a/`, `labs/part_b/` (і новий
`labs/common/`), інакше `COPY labs/ ./labs/` в Docker-образі не знайде
потрібні файли за шляхами з `command:`.

### Додати в кореневий `.gitignore` (якщо ще нема):
```
.env.theme9
results/
data/telco_customers_drifted.csv
data/churn_explanations.json
```

---

## Порядок дій перед заняттям

```bash
git clone https://github.com/mentorchita/telco-churn-mlops-synthetic-08.git
cd telco-churn-mlops-synthetic-08
# ... скопіювати файли патча сюди ...

make -f Makefile.theme9 init            # .env.theme9 + обрати LLM_PROVIDER
# → відкрити .env.theme9, LLM_PROVIDER=gemini, GOOGLE_API_KEY=...

make -f Makefile.theme9 train-model     # генерує CSV + тренує ВАЛІДНУ модель
make -f Makefile.theme9 verify-artifacts
make -f Makefile.theme9 build
make -f Makefile.theme9 setup-data      # churn_explanations.json

make -f Makefile.theme9 theme9-all      # усі лаби послідовно
```

---

## Що НЕ перевірялося (через відсутність Docker у середовищі аналізу)

Я перевірив: схему даних, структуру моделі (`ColumnTransformer`), цілісність
pickle-файлу, узгодженість версій scikit-learn/numpy, логіку кожного
Python-скрипта, наявність усіх файлів, на які посилається
`docker-compose.theme9.yml`. Я **не** мав змоги реально піднімати
`docker compose` в цьому середовищі (немає Docker-демона), тож рекомендую
один прогін `make -f Makefile.theme9 theme9-all` заздалегідь, за 1-2 дні до
заняття, а не в день заняття — щоб залишити час на непередбачені дрібниці
(наприклад, точну поведінку healthcheck на твоїй версії Docker Desktop).
