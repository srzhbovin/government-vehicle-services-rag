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
YANDEX_CLOUD_API_KEY=ваш_api_ключ
YANDEX_CLOUD_FOLDER_ID=идентификатор_каталога
YANDEX_GPT_MODEL=yandexgpt-5-lite
YANDEX_GPT_FALLBACK_MODELS=yandexgpt-5.1,yandexgpt-5-pro
```

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

Тесты проверяют загрузку документов, chunking, BM25, cosine, FAISS, RRF, prompt с источниками,
Yandex fallback, локальный OpenAI-совместимый provider, FastAPI endpoints и валидацию запросов.

## Результаты retrieval-исследования

На test-наборе лучший вариант `hybrid_reranked` получил Recall@5 `0.6106`. Без reranker гибрид
получил `0.5795`, BM25 — `0.5759`, FAISS — `0.5411`. Поэтому runtime использует hybrid retrieval
с reranker и передаёт генератору пять лучших фрагментов.

Подробные результаты:

- `reports/retrieval_comparison.md`;
- `data/experiments/retrieval/retrieval_comparison.csv`;
- `data/experiments/retrieval/retrieval_per_query.jsonl`.

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
    ├── service.py              # единый RAG-сервис
    └── build_rag_index.py      # пересборка runtime-индекса

streamlit_app.py                # пользовательский интерфейс
scripts/run_backend.ps1         # запуск FastAPI
scripts/run_frontend.ps1        # запуск Streamlit
```

## Ограничения базовой версии

- Retrieval Recall@5 пока около 61%, поэтому часть вопросов не получает правильный контекст.
- Диалоговая история и query rewriting ещё не реализованы.
- Prompt пока один базовый; сравнение JSON, Pydantic и Structured Output будет отдельным этапом.
- Параметры temperature, top-p и размер контекста пока не исследованы.
- Для генерации через Yandex AI Studio нужны интернет, действующий ключ и доступная квота.
