# DMV RAG: поиск по документам и генерация ответов

Рабочая Retrieval-Augmented Generation система на документах DMV из MultiDoc2Dial. Пользователь
задаёт вопрос на русском или английском языке, система находит релевантные фрагменты документов,
переранжирует их и передаёт YandexGPT. В ответе показываются номера и тексты использованных
фрагментов.

Текущая версия включает:

- загрузку и нормализацию 149 документов MultiDoc2Dial DMV;
- token chunking `120/0`, выбранный по результатам эксперимента;
- эмбеддинги `sentence-transformers/all-MiniLM-L6-v2`;
- точный индекс `FAISS IndexFlatIP`;
- гибридный retrieval BM25 + FAISS через Reciprocal Rank Fusion;
- cross-encoder reranker `ms-marco-MiniLM-L2-v2` для первых 10 кандидатов;
- генерацию ответа моделью `yandexgpt-5-lite` через Yandex AI Studio;
- нативный Structured Output с JSON Schema и повторной Pydantic-валидацией;
- локальную модель через LM Studio как дополнительный вариант;
- FastAPI backend и Streamlit-интерфейс.

## Архитектура

```text
Streamlit (http://127.0.0.1:8501)
              ↓ HTTP
FastAPI (http://127.0.0.1:8000)
              ↓
     BM25 ───────── FAISS Flat
              ↓
       Reciprocal Rank Fusion
              ↓
      Cross-encoder reranker
              ↓ top-5
  Yandex AI Studio / LM Studio
              ↓
 JSON Schema + Pydantic validation
              ↓
     Ответ и найденные источники
```

Retrieval и генерация разделены. Модель генерации, backend и интерфейс можно менять независимо,
не перестраивая индекс.

## Требования

- Windows 10/11;
- Python 3.13;
- 8 ГБ оперативной памяти минимум, 16 ГБ рекомендуется;
- доступ в интернет и действующий API-ключ Yandex Cloud;
- около 4 ГБ свободного места для Python-зависимостей и retrieval-моделей.

## Установка

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

## Настройка Yandex AI Studio

Скопируйте шаблон конфигурации:

```powershell
Copy-Item .env.example .env
```

Заполните в `.env` API-ключ и идентификатор каталога Yandex Cloud:

```env
RAG_LLM_PROVIDER=yandex
RAG_PROMPT_STRATEGY=structured_output
YANDEX_CLOUD_API_KEY=ваш_api_ключ
YANDEX_CLOUD_FOLDER_ID=идентификатор_каталога
YANDEX_GPT_MODEL=yandexgpt-5-lite
YANDEX_GPT_FALLBACK_MODELS=yandexgpt-5.1,yandexgpt-5-pro
YANDEX_TEMPERATURE=0.1
```

`structured_output` — выбранная стратегия основного pipeline. Yandex получает JSON Schema с
полями `answer`, `confidence` и `source`, а полученный объект дополнительно проверяется Pydantic.
API-ключ хранится только в `.env`; этот файл исключён из Git.

## Запуск приложения

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_backend.ps1
```

Первый запуск может занять одну-две минуты: загружаются embedding-модель и reranker. После
запуска доступны:

- API: `http://127.0.0.1:8000`;
- Swagger: `http://127.0.0.1:8000/docs`;
- health check: `http://127.0.0.1:8000/health`.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_frontend.ps1
```

Откройте `http://127.0.0.1:8501`. 

## Проверка API без Streamlit

Только retrieval, без языковой модели:

```powershell
$body = @{ question = "How can I renew my vehicle registration?" } | ConvertTo-Json
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/retrieve `
  -ContentType "application/json" `
  -Body $body
```

Полный ответ RAG на русском:

```powershell
$body = @{
  question = "How quickly must I report a change of address to DMV?"
  language = "ru"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/ask `
  -ContentType "application/json" `
  -Body $body
```

Допустимые значения `language`: `auto`, `ru`, `en`.

Ответ `/api/v1/ask` дополнительно содержит `prompt_strategy`, `confidence`, `source`,
`structured_parse_success` и `schema_valid`.

## Локальная модель как запасной вариант

Приложение также поддерживает OpenAI-совместимый API LM Studio. Для локального запуска без
облачной генерации установите LM Studio, скачайте `Gemma 3 4B Instruct`, включите сервер в разделе
`Developer` и измените `.env`:

```env
RAG_LLM_PROVIDER=lmstudio
LOCAL_LLM_BASE_URL=http://127.0.0.1:1234/v1
LOCAL_LLM_MODEL=google/gemma-3-4b
```

Если заданная модель отсутствует, приложение выберет первую доступную текстовую модель LM Studio.

## Пересборка данных и индекса

Готовые чанки, эмбеддинги и FAISS-индекс уже находятся в репозитории. Полная пересборка нужна
только при изменении документов, chunking или embedding-модели.

```powershell
.\.venv\Scripts\python.exe src\rag_pipeline\import_multidoc2dial_dmv.py
.\.venv\Scripts\python.exe src\rag_pipeline\build_token_chunks.py
.\.venv\Scripts\python.exe -m rag_pipeline.build_rag_index
```

Результаты сохраняются в:

```text
data/current/prepared/dmv_chunks_token_120_0.jsonl
data/current/indexes/retrieval/chunk_embeddings.npy
data/current/indexes/retrieval/faiss_flat.index
data/current/indexes/retrieval/index_metadata.json
```

## Тесты

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Тесты проверяют загрузку документов, chunking, BM25, cosine, FAISS, RRF, четыре prompt-стратегии,
Pydantic-валидацию, метрики оценки, Yandex fallback, локальный OpenAI-совместимый provider,
FastAPI endpoints и валидацию запросов.

## Повторение экспериментов

Полный прогон создаёт тестовый набор, выполняет 80 сравнений prompt-стратегий и 30 запусков для
сравнения temperature. Используются платные запросы Yandex AI Studio; выполнение занимает около
5–10 минут.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_rag_evaluation.ps1
```

Если выполнение прервалось, продолжить с сохранённых результатов можно командой:

```powershell
.\.venv\Scripts\python.exe -m rag_pipeline.evaluate_rag --resume
```

Проверить только retrieval без обращений к Yandex:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
.\.venv\Scripts\python.exe -m rag_pipeline.evaluate_rag --retrieval-only
```

Результаты сохраняются в `data/experiments/rag/`, отчёты — в
`reports/prompt_engineering.md` и `reports/rag_quality.md`.

## Результаты retrieval-исследования

На test-наборе лучший вариант `hybrid_reranked` получил Recall@5 `0.6106`. Без reranker гибрид
получил `0.5795`, BM25 — `0.5759`, FAISS — `0.5411`. Поэтому runtime использует hybrid retrieval
с reranker и передаёт генератору пять лучших фрагментов.

Подробные результаты:

- `reports/retrieval_comparison.md`;
- `data/experiments/retrieval/retrieval_comparison.csv`;
- `data/experiments/retrieval/retrieval_per_query.jsonl`.

## Результаты Prompt Engineering

На 20 одинаковых вопросах сравнивались обычный prompt, JSON-prompt, prompt с Pydantic-схемой и
нативный Structured Output Yandex Responses API. Retrieval-контекст и temperature `0.1` были
одинаковыми для всех вариантов.

| Вариант | Соблюдение схемы | Ошибки JSON | Correct | Semantic similarity | Валидные citations | Среднее время |
|---|---:|---:|---:|---:|---:|---:|
| Обычный prompt | — | 0 | 80% | 0.765 | 95% | 2521 мс |
| JSON-prompt | 100% | 0 | 70% | 0.785 | 95% | 2700 мс |
| Pydantic в prompt | 95% | 0 | 75% | 0.775 | 90% | 2684 мс |
| Structured Output | 100% | 0 | 75% | 0.771 | 95% | 2820 мс |

В основной pipeline выбран `structured_output`: он обеспечивает настоящий API-контракт, а не
только текстовую просьбу вернуть JSON. Обычный prompt немного лучше прошёл эвристику корректности,
но не гарантирует структуру. У Pydantic-варианта один ответ был JSON-объектом, однако не прошёл
валидацию схемы.

Поле `confidence` оказалось некалиброванным: Structured Output вернул `1.0` для всех вопросов,
включая неполные ответы. Поэтому сейчас оно отображается как диагностическое значение и не
используется для принятия решений.

## Оценка качества текущего RAG

Тестовый набор состоит из 19 самостоятельных validation-вопросов MultiDoc2Dial и одного ручного
регрессионного вопроса о потерянных водительских правах.

| k | Hit Rate | Recall@k | MRR | Среднее время retrieval |
|---:|---:|---:|---:|---:|
| 1 | 55% | 55% | 0.698 | 181 мс |
| 3 | 85% | 85% | 0.698 | 181 мс |
| 5 | 95% | 95% | 0.698 | 181 мс |

Для итогового `structured_output`: Exact Match `0%`, token F1 `0.504`, semantic similarity
`0.771`, автоматический correct rate `75%`. Exact Match равен нулю, потому что модель корректно
переформулирует короткие эталоны; для генеративного ответа эта метрика слишком строгая.

Temperature `0`, `0.1`, `0.3` и `0.7` не повлияла на соблюдение схемы: во всех случаях получено
100%. На подмножестве из 10 вопросов `0.7` показала 100% по автоматической эвристике, однако одного
прогона недостаточно, чтобы считать улучшение устойчивым. В pipeline оставлена `0.1` для более
воспроизводимых ответов.

### Основные типы ошибок

- нужный документ находится в top-5, но в контекст попадает неподходящий чанк;
- частное условие выдаётся вместо общего ответа — пример с формой MV-78B для потерянных прав;
- внутренние ссылки документа вроде `[8]` смешиваются с номерами RAG-фрагментов;
- `confidence` завышен и не отражает полноту ответа;
- часть автоматических ошибок вызвана короткими или неполными gold-ответами: более подробный
  корректный ответ получает низкое сходство.

Ручная проверка показала две содержательные ошибки выбора контекста: вопрос о стоимости learner
permit и вопрос о потерянных правах. Следующий этап оптимизации должен начинаться с query rewriting,
учёта заголовков в BM25/reranker и выбора нескольких чанков внутри найденного документа.

## Структура актуального кода

```text
src/
├── backend/
│   └── main.py                 # FastAPI endpoints
└── rag_pipeline/
    ├── settings.py             # переменные окружения
    ├── schemas.py              # API-контракты Pydantic
    ├── retriever.py            # BM25 + FAISS + RRF + reranker
    ├── generator.py            # LM Studio и Yandex providers
    ├── prompting.py            # четыре prompt-стратегии и схема ответа
    ├── evaluation.py           # метрики retrieval, ответа и citations
    ├── build_rag_test_set.py   # воспроизводимый тестовый набор
    ├── evaluate_rag.py         # эксперименты Prompt Engineering и RAG
    ├── service.py              # единый RAG-сервис
    └── build_rag_index.py      # пересборка runtime-индекса

streamlit_app.py                # пользовательский интерфейс
scripts/run_backend.ps1         # запуск FastAPI
scripts/run_frontend.ps1        # запуск Streamlit
scripts/run_rag_evaluation.ps1  # полный эксперимент
```

## Ограничения базовой версии

- На полном test-наборе Retrieval Recall@5 пока `0.6106`, поэтому часть вопросов не получает
  правильный контекст.
- Диалоговая история и query rewriting ещё не реализованы.
- Выбор соседних и родительских чанков внутри релевантного документа ещё не реализован.
- `confidence` модели не откалиброван.
- Top-p и размер контекста пока не исследованы; temperature проверена только одним прогоном.
- Для генерации через Yandex AI Studio нужны интернет, действующий ключ и доступная квота.
