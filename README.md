# Government Vehicle Services RAG

Проект реализует RAG-систему по документам DMV из MultiDoc2Dial. Система ищет релевантные фрагменты документов, передаёт их в LLM и возвращает ответ с источниками.

Текущая версия — полноценный исследовательский RAG-прототип: есть backend на FastAPI, интерфейс на Gradio, гибридный retrieval, structured output, диалоговая история, настройка параметров генерации, защита от галлюцинаций через LLM-as-a-judge и priority gateway перед LiteLLM.

## Что сейчас входит в пайплайн

1. Загрузка DMV-документов из MultiDoc2Dial.
2. Очистка и подготовка документов.
3. Разбиение документов на token chunks.
4. Построение эмбеддингов через `sentence-transformers/all-MiniLM-L6-v2`.
5. Индекс FAISS `IndexFlatIP`.
6. BM25-поиск.
7. Гибридный поиск BM25 + FAISS через weighted RRF.
8. Cross-encoder reranker `cross-encoder/ms-marco-MiniLM-L2-v2`.
9. Формирование контекста после retrieval: можно настраивать `top_k`, `context_window`, `intro_chunks` и `max_context_chunks`.
10. Refusal gate останавливает генерацию, если найденные документы недостаточно релевантны.
11. Генерация ответа через Yandex AI Studio API.
12. Structured Output: модель возвращает структурированный объект `answer`, `confidence`, `source`.
13. Проверка ответа через LLM-as-a-judge.
14. Адаптивный повтор с расширенным контекстом, если первый ответ не прошёл проверку.
15. FastAPI отдаёт результат в API.
16. Gradio показывает ответ, параметры, judge-вердикт, найденные фрагменты и поддерживает Chat RAG.
17. При локальном LLM-запуске priority gateway распределяет ограниченные слоты между группами пользователей до передачи запросов в LiteLLM.

## Что изменено в последней итерации

### Исследование стратегий формирования контекста

Добавлен отдельный эксперимент для шага 13. Теперь можно сравнивать, сколько чанков брать после retrieval и нужно ли добавлять соседние/начальные чанки документа.

По результатам проверки текущий дефолт изменён на более компактный:

```env
RAG_TOP_K=3
RAG_CONTEXT_WINDOW=0
RAG_CONTEXT_INTRO_CHUNKS=0
RAG_MAX_CONTEXT_CHUNKS=6
```

Расширение контекста осталось доступным через API и Gradio. Дополнительно включён адаптивный fallback: если LLM-as-a-judge не принимает первый компактный ответ, система делает второй проход с более широким контекстом.

### Сравнение языковых моделей

Добавлен эксперимент для шага 14: сравнение YandexGPT и open-source моделей из Yandex AI Studio на одинаковых retrieval-контекстах. По текущей проверке в runtime выбрана `qwen3.6-35b-a3b`.

### Отказ от ответа

Добавлен refusal gate для шага 15. Если найденные документы слишком слабо связаны с вопросом, генерация не запускается, а система отвечает, что в DMV-документах недостаточно информации.

### Chat RAG

Добавлен endpoint `/api/v1/chat` и диалоговый режим в Gradio. Retrieval получает компактный самостоятельный запрос, а генератор и judge учитывают предыдущие вопросы и чистые ответы ассистента. Это позволяет восстанавливать смысл коротких follow-up вопросов без передачи в backend HTML-разметки интерфейса.

### Защита от галлюцинаций

Добавлен LLM-as-a-judge. После генерации ответа система делает отдельную проверку: действительно ли ответ следует из найденных фрагментов.

Judge возвращает:

```json
{
  "verdict": "grounded",
  "score": 1.0,
  "reason": "The answer is supported by the fragments.",
  "corrected_answer": null,
  "source": "[1], [2]"
}
```

Если ответ не подтверждается контекстом, judge может вернуть исправленный ответ. Тогда API сохранит исходный ответ в `original_answer`, а пользователю покажет исправленную версию.

### Настраиваемые параметры

В API и интерфейс добавлены параметры:

- `top_k` — сколько основных retrieval-кандидатов брать;
- `context_window` — сколько соседних чанков добавлять вокруг найденного чанка;
- `intro_chunks` — сколько первых чанков документа добавлять в контекст;
- `max_context_chunks` — максимальный размер контекста в чанках перед отправкой в LLM;
- `temperature` — насколько свободно модель формулирует ответ;
- `top_p` — nucleus sampling;
- `max_output_tokens` — лимит длины ответа;
- `use_judge` — включать или выключать LLM-as-a-judge для конкретного запроса.

Beam search отдельно не добавлялся: в текущем Yandex Responses API основной рабочий контроль генерации — это `temperature`, `top_p` и лимит токенов. Для RAG важнее качество retrieval-контекста и groundedness-проверка, чем beam search.

### Улучшение retrieval-контекста

Проблема прошлой версии была не только в промпте. Иногда правильный документ находился, но в генерацию попадал не тот кусок документа. Например, для вопроса:

```text
What should I do if I lost my driver license?
```

старая версия могла слишком сильно зацепиться за фрагмент про `MV-78B` или вообще за похожий документ про номерные знаки.

Сейчас добавлено:

- title-aware поиск: заголовок документа участвует в BM25, эмбеддингах и reranker;
- простое query expansion для частых DMV-ситуаций вроде lost/stolen license;
- context expansion: если найден важный чанк документа, в контекст добавляются первые и соседние чанки этого документа.

На проблемном вопросе retrieval теперь сначала поднимает документ `Replace license or permit`, а в контекст попадают способы замены online/by mail/office и fee `$17.50`.

### Интерфейс на Gradio

Streamlit-интерфейс заменён на Gradio. В интерфейсе можно:

- задать вопрос;
- выбрать язык ответа;
- менять `top_k`, `context_window`, `intro_chunks`, `max_context_chunks`, `temperature`, `top_p`, `max_output_tokens`;
- включать/выключать judge;
- видеть итоговый ответ;
- видеть judge verdict, score и причину;
- смотреть найденные фрагменты документов;
- раскрывать карточки источников ответа с `document_id`, `chunk_id`, score и текстом чанка;
- использовать отдельный блок Chat RAG: ответы в чате дополняются источниками, а подробные фрагменты выводятся ниже в раскрывающихся карточках.

### Исследование инференс-движков

Для шага 17 добавлен воспроизводимый стенд сравнения SGLang, vLLM и LMDeploy на одной модели и одной NVIDIA GPU. Финальный прогон выполнен на Kaggle с Tesla T4: протестированы три движка, пять размеров батча и три повтора. На этой конфигурации лучшим оказался LMDeploy/TurboMind.

### Балансировка и приоритизация перед LiteLLM

Для шага 18 реализован отдельный OpenAI-compatible gateway перед LiteLLM. Он определяет группу по уже выданному API-ключу, ограничивает число одновременных генераций и выбирает следующий запрос через priority queue with aging. Жёсткий starvation timeout гарантирует, что фоновый запрос не останется в очереди навсегда. Правила лежат в YAML и меняются без доработки Gradio, RAG backend или других клиентов.

В контролируемом сравнении FIFO и priority+aging выполнено 1080 запросов при 1, 2, 4 и 8 слотах. В финальном прогоне ожидание запросов `critical` уменьшилось на 80.7–91.6%, а максимальная разница общей пропускной способности составила 1.06%. Полная методика и ограничения приведены в `reports/priority_scheduling.md`.

## Соответствие этапам технического задания

| Этап | Что получено | Где находится вывод |
|---:|---|---|
| 1 | Теоретическая база применена при выборе архитектуры и метрик | этот README и отчёты ниже |
| 2 | Сравнены MiniLM и MPNet на одинаковом точном FAISS-поиске | `reports/embedding_model_comparison.md` |
| 3 | Сравнены полный, безопасно сокращённый и агрессивно сокращённый корпус | `reports/audit_steps_1_5.md` |
| 4 | Сравнены 35 конфигураций character, recursive, token и semantic-lite chunking | `reports/chunking_comparison.md` |
| 5 | Сравнены Flat, IVF и HNSW | `reports/faiss_comparison.md` |
| 6 | Сравнены cosine, BM25, FAISS, hybrid и hybrid + reranker | `reports/retrieval_comparison.md` |
| 7 | Собран рабочий сквозной RAG | `reports/rag_baseline.md` |
| 8 | Сравнены четыре стратегии промптинга, выбран Structured Output | `reports/prompt_engineering.md` |
| 9–10 | Посчитаны retrieval/generation-метрики и систематизированы ошибки | `reports/rag_quality.md` |
| 11 | Сохранена последовательная история изменений | Git log репозитория |
| 12 | Архитектура, команды, таблицы и ограничения сведены в документацию | этот README и каталог `reports/` |
| 13 | Исследованы 255 стратегий формирования контекста | `reports/context_strategy_comparison.md` |
| 14 | Сравнены восемь LLM при одинаковом retrieval-контексте | `reports/llm_model_comparison.md` |
| 15 | Сравнены механизмы отказа от ответа | `reports/refusal_mechanisms.md` |
| 16 | Реализован Chat RAG с разрешением коротких follow-up вопросов | раздел `Chat RAG` ниже |
| 17 | Сравнены SGLang, vLLM и LMDeploy на Tesla T4 | `reports/inference_engine_comparison.md` |
| 18 | Реализована очередь перед LiteLLM и исследована приоритизация | `reports/priority_scheduling.md` |

В каждом указанном отчёте отделены методика, численные результаты, выбранное решение и ограничения. Сгенерированные CSV, JSON/JSONL и графики находятся в `data/experiments/`.

## Быстрый запуск

Все команды ниже выполняются из корня клонированного репозитория.

### 1. Установить зависимости

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
```

Для разработки и запуска автоматических проверок установите также тестовые инструменты:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Dev
python -m pytest -q
python -m ruff check src tests gradio_app.py load_documents.py chunk_documents.py
```

Если `.env` ещё нет, скрипт создаст его из `.env.example`. После этого нужно заполнить:

```env
YANDEX_CLOUD_API_KEY=
YANDEX_CLOUD_FOLDER_ID=
```

### 2. Пересобрать чанки и индекс

```powershell
powershell -ExecutionPolicy Bypass -File scripts/rebuild_pipeline.ps1
```

Этот шаг создаёт актуальные чанки, эмбеддинги и FAISS-индекс.

### 3. Запустить backend и Gradio одной командой

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_app.ps1
```

После запуска:

- FastAPI docs: `http://127.0.0.1:8000/docs`;
- Gradio UI: `http://127.0.0.1:7860`.

Остановить сервисы:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/stop_app.ps1
```

### Альтернативный запуск в двух терминалах

Backend:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_backend.ps1
```

Frontend:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_frontend.ps1
```

## Проверка API

Health-check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Пример запроса к RAG:

```powershell
$body = @{
  question = "What should I do if I lost my driver license?"
  language = "en"
  top_k = 3
  context_window = 0
  intro_chunks = 0
  max_context_chunks = 6
  temperature = 0.1
  top_p = 0.9
  max_output_tokens = 700
  use_judge = $true
  use_adaptive_context = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## Запуск эксперимента по стратегиям контекста

Шаг 13 можно воспроизвести отдельной командой:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_context_strategy_experiment.ps1
```

Скрипт сравнивает сетку параметров:

- `top_k`: 3, 5, 7, 10;
- `intro_chunks`: 0, 1, 2, 3, 5;
- `context_window`: 0, 1, 2;
- `max_context_chunks`: 6, 8, 10, 12, 16.

Результаты сохраняются в:

- `reports/context_strategy_comparison.md`;
- `data/experiments/context_strategies/context_strategy_summary.csv`;
- `data/experiments/context_strategies/context_generation_summary.csv`;
- `data/experiments/context_strategies/context_strategy_details.jsonl`;
- `data/experiments/context_strategies/context_generation_results.jsonl`.

## Запуск эксперимента по инференс-движкам

Шаг 17 сравнивает SGLang, vLLM и LMDeploy на одной GPU, одной модели и одном наборе параметров. Этот эксперимент не использует Yandex AI Studio: каждый движок самостоятельно запускает одинаковые открытые веса модели в локальном контейнере.

Финальный прогон выполнен в Kaggle на одной NVIDIA Tesla T4 16 GB. Полная матрица включает три движка, пять уровней concurrency и три независимых повтора — 45 измерений и 720 запросов. Во время измеряемых прогонов ошибок не было. На этой конфигурации победил LMDeploy с backend TurboMind: он показал лучший медианный throughput и минимальные p95 TTFT/E2E при concurrency=1.

Для запуска нужен Linux-хост с NVIDIA GPU не менее 12 GB VRAM, примерно 70 GB свободного места, Docker Engine, NVIDIA Container Toolkit и Compose v2. Сначала выполните проверку окружения:

```bash
bash scripts/inference_engines/preflight.sh
```

Полный эксперимент запускается одной командой из корня проекта:

```bash
bash scripts/inference_engines/run_benchmark.sh all
```

Скрипт последовательно поднимает каждый движок, ждёт готовности API, выполняет прогрев и измеряемые повторы, собирает телеметрию GPU, останавливает контейнер и переходит к следующему движку. Одновременный запуск трёх серверов не используется: иначе они делили бы память и вычислительные ресурсы одной GPU, а результаты нельзя было бы честно сравнивать.

Зафиксированный протокол находится в `deploy/inference_engines/benchmark.env.example`. При необходимости файл можно скопировать в `deploy/inference_engines/benchmark.env` для локальных runtime-настроек вроде номера GPU или порта. Параметры, влияющие на сравнимость результатов, preflight проверяет и не позволяет незаметно изменить.

Основные артефакты эксперимента:

- `reports/inference_engine_comparison.md` — методика, сводная таблица и выводы;
- `data/experiments/inference_engines/kaggle_t4_qwen2_5_3b/summary.csv` — итоговые метрики по движкам и размерам батча;
- `data/experiments/inference_engines/kaggle_t4_qwen2_5_3b/summary.json` — та же сводка в машиночитаемом виде;
- `data/experiments/inference_engines/kaggle_t4_qwen2_5_3b/raw/` — 45 исходных JSONL-записей;
- `data/experiments/inference_engines/kaggle_t4_qwen2_5_3b/telemetry/` — 45 окон GPU-телеметрии;
- `data/experiments/inference_engines/kaggle_t4_qwen2_5_3b/plots/` — графики throughput, TTFT, E2E и VRAM.

В online serving SGLang, vLLM и LMDeploy используют continuous batching. Поэтому `batch size` в этом эксперименте определяется как максимальное число одновременно выполняемых запросов (`max_concurrency`), а не как статическая матрица, один раз переданная модели.

## Запуск Priority Gateway и LiteLLM

Этот вариант нужен для локального или серверного LLM-контура. Сначала должен быть запущен OpenAI-compatible inference endpoint, например выбранный на шаге 17 LMDeploy на `http://127.0.0.1:18000/v1`.

Создайте локальную конфигурацию и замените тестовые секреты:

```powershell
Copy-Item deploy/priority_gateway/.env.example deploy/priority_gateway/.env
notepad deploy/priority_gateway/.env
```

Поднимите gateway и LiteLLM:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_priority_stack.ps1
```

Снаружи открывается только priority gateway: `http://127.0.0.1:4000`. LiteLLM остаётся внутри Docker-сети. Проверка состояния:

```powershell
Invoke-RestMethod http://127.0.0.1:4000/health
```

Чтобы основной RAG ходил через новый контур, укажите в корневом `.env` ключ нужной группы:

```env
RAG_LLM_PROVIDER=litellm
LOCAL_LLM_BASE_URL=http://127.0.0.1:4000/v1
LOCAL_LLM_MODEL=dmv-rag
LOCAL_LLM_API_KEY=sk-critical-change-me
```

Правила находятся в `deploy/priority_gateway/priorities.yaml`. Изменение ключей, сопоставления групп и числовых приоритетов не требует изменения клиентского кода. Не добавляйте реальные секреты в этот YAML: для них предусмотрены переменные окружения.

Контролируемый эксперимент FIFO против priority+aging запускается отдельно и не требует Docker или LLM:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_priority_experiment.ps1
```

Результаты сохраняются в `data/experiments/priority_gateway/baseline/`, подробный разбор — в `reports/priority_scheduling.md`.

Дополнительный Token-Aware эксперимент сравнивает распределение запросов между тремя репликами по количеству запросов и по ожидаемой токенной стоимости:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_token_routing_experiment.ps1
```

Команда собирает компактный Docker-образ, запускает пять стратегий для двух сценариев нагрузки и сохраняет CSV, JSON и SVG-графики в `data/experiments/priority_gateway/token_aware/`. Для запуска без Docker добавьте параметр `-Local`.

Остановка стека:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/stop_priority_stack.ps1
```

## Основные настройки `.env`

```env
RAG_LLM_PROVIDER=yandex
RAG_PROMPT_STRATEGY=structured_output

YANDEX_TEMPERATURE=0.1
YANDEX_TOP_P=0.9
YANDEX_MAX_OUTPUT_TOKENS=700

RAG_TOP_K=3
RAG_CANDIDATE_K=60
RAG_RERANKER_CANDIDATE_K=20
RAG_BM25_WEIGHT=0.5
RAG_RRF_K=60

RAG_CONTEXT_WINDOW=0
RAG_CONTEXT_INTRO_CHUNKS=0
RAG_MAX_CONTEXT_CHUNKS=6
RAG_ENABLE_ADAPTIVE_CONTEXT=true
RAG_ADAPTIVE_TOP_K=5
RAG_ADAPTIVE_CONTEXT_WINDOW=1
RAG_ADAPTIVE_CONTEXT_INTRO_CHUNKS=1
RAG_ADAPTIVE_MAX_CONTEXT_CHUNKS=8
RAG_ENABLE_REFUSAL_GATE=true
RAG_REFUSAL_MIN_TOP_SCORE=-2.0
RAG_REFUSAL_MIN_LEXICAL_OVERLAP=0.0
RAG_USE_QUERY_EXPANSION=true

RAG_ENABLE_JUDGE=true
RAG_JUDGE_MIN_SCORE=0.72
RAG_JUDGE_TEMPERATURE=0.0
RAG_JUDGE_MAX_OUTPUT_TOKENS=500
```

## Результаты экспериментов

### Embedding models

На одинаковых 1152 чанках и 1132 validation-вопросах сравнивались
`all-MiniLM-L6-v2` и `all-mpnet-base-v2`. Для обеих моделей использовались
нормализованные векторы и точный `FAISS IndexFlatIP`.

| Модель | Размерность | Построение, мс | Поиск, мс/запрос | Recall@5 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | 384 | 68330.8 | 7.949 | 0.5230 | 0.4010 |
| `all-mpnet-base-v2` | 768 | 545944.3 | 54.726 | 0.5009 | 0.3816 |

Для текущего корпуса сохранена `all-MiniLM-L6-v2`: она показала лучшее качество и одновременно оказалась существенно дешевле. Эксперимент воспроизводится командой:

```powershell
python -m src.rag_pipeline.compare_embedding_models
```

Подробности находятся в `reports/embedding_model_comparison.md`. Крупные экспериментальные
эмбеддинги, FAISS-индексы и per-query строки пересоздаются командой выше и не хранятся в Git;
сводные CSV/JSON и метаданные входят в репозиторий.

### Retrieval methods

На предыдущем этапе сравнивались cosine similarity, BM25, FAISS, hybrid BM25 + FAISS и hybrid + reranker.

По итогам был выбран `hybrid_reranked`, потому что он дал лучший результат на full test:

- Recall@1: `0.3949`;
- Recall@3: `0.5356`;
- Recall@5: `0.6106`;
- Recall@10: `0.6590`;
- MRR@10: `0.4814`.

Подробности сохранены в `reports/retrieval_comparison.md`.

### Prompt Engineering

Сравнивались:

- обычный plain prompt;
- JSON prompt;
- prompt с Pydantic schema;
- native Structured Output.

Для основного пайплайна выбран `structured_output`. Он не гарантирует, что ответ всегда идеален по смыслу, но делает формат ответа стабильным для backend: модель возвращает объект, который затем проверяется через Pydantic.

Подробности сохранены в `reports/prompt_engineering.md`.

### RAG quality до последней итерации

На диагностическом наборе из 20 самостоятельных вопросов было:

- Hit@1: `55%`;
- Hit@3: `85%`;
- Hit@5: `95%`;
- semantic similarity: `0.771`;
- условно корректные ответы: `75%`;
- Exact Match: `0%`.

Exact Match равен нулю не потому, что все ответы плохие, а потому что генеративная модель формулирует ответ иначе, чем короткий эталон.

### Sanity-check после последней итерации

После title-aware retrieval, query expansion и пересборки индекса быстрый sanity-check на тех же 20 вопросах для чистого retrieval без context expansion дал:

- Hit@1: `65%`;
- Hit@3: `95%`;
- Hit@5: `95%`;
- MRR: `0.775`.

Context expansion отдельно не стоит сравнивать с обычным Hit@5 напрямую: он специально добавляет соседние чанки одного документа, поэтому top-5 может стать менее разнообразным, зато генератор получает более полный контекст.

Ручная проверка проблемного вопроса `What should I do if I lost my driver license?` теперь проходит лучше: система поднимает документ `Replace license or permit`, отвечает про online/by mail/office и fee `$17.50`, а judge помечает ответ как `grounded`.

### Context strategy experiment

На этом шаге проверялись параметры формирования контекста уже после retrieval:

- `top_k` — сколько основных чанков берём после reranker;
- `context_window` — сколько соседних чанков добавляем вокруг найденного;
- `intro_chunks` — сколько первых чанков документа добавляем как вводную часть;
- `max_context_chunks` — сколько чанков максимум отдаём в LLM.

Сначала была прогнана широкая сетка из 255 корректных комбинаций на 20 диагностических вопросах без LLM-вызовов. Потом несколько ключевых конфигураций были проверены уже с генерацией ответа через YandexGPT.

Лучшие результаты по LLM-проверке:

| config | generation score | required answer coverage | avg token F1 | context chunks |
|---|---:|---:|---:|---:|
| `top_k=3, intro=0, window=0, max=6` | `0.8983` | `1.00` | `0.5200` | `3` |
| `top_k=10, intro=5, window=2, max=16` | `0.8154` | `0.80` | `0.4772` | `16` |
| `top_k=7, intro=3, window=1, max=12` | `0.8111` | `0.80` | `0.5053` | `12` |
| `top_k=5, intro=3, window=1, max=12` | `0.8110` | `0.80` | `0.5050` | `12` |
| `top_k=5, intro=2, window=1, max=6` | `0.7702` | `0.60` | `0.4933` | `6` |

Вывод: для текущего корпуса и текущего retriever лучшим дефолтом стал компактный контекст:

```env
RAG_TOP_K=3
RAG_CONTEXT_WINDOW=0
RAG_CONTEXT_INTRO_CHUNKS=0
RAG_MAX_CONTEXT_CHUNKS=6
```

Причина довольно практичная: после BM25 + FAISS + reranker верхние чанки уже достаточно точные, а добавление соседних и вводных чанков часто приносит шум. Широкий режим `top_k=10, intro=5, window=2, max=16` даёт в среднем 16 чанков и примерно 1866 слов контекста. В нём полезная информация не исчезает полностью, но модель чаще видит лишние условия и начинает хуже выделять главный ответ.

Это не значит, что `context_window` и `intro_chunks` бесполезны. Они оставлены в API и Gradio, потому что дальше можно сделать адаптивный режим: по умолчанию держать контекст компактным, а расширять его только если retriever/Judge видит недостаток информации.

Такой адаптивный режим теперь включён:

```env
RAG_ENABLE_ADAPTIVE_CONTEXT=true
RAG_ADAPTIVE_TOP_K=5
RAG_ADAPTIVE_CONTEXT_WINDOW=1
RAG_ADAPTIVE_CONTEXT_INTRO_CHUNKS=1
RAG_ADAPTIVE_MAX_CONTEXT_CHUNKS=8
```

Логика такая: первый ответ строится на компактном контексте. Если LLM-as-a-judge принимает ответ, ничего больше не происходит. Если judge отклоняет ответ как недостаточно grounded, система делает второй проход с более широким контекстом и использует его только если повторная проверка стала успешной или judge смог вернуть исправленный grounded-ответ.

Подробный отчёт сохранён в `reports/context_strategy_comparison.md`.

### LLM model comparison

Для шага 14 сравнивались модели, доступные через текущий Yandex AI Studio Responses API:

- `yandexgpt-5-lite`;
- `yandexgpt-5-pro`;
- `yandexgpt-5.1`.
- `qwen3.6-35b-a3b`;
- `qwen3-235b-a22b-fp8`;
- `gpt-oss-20b`;
- `gpt-oss-120b`;
- `deepseek-v4-flash`.

Важно: retrieval, найденные чанки, structured output, temperature и остальные параметры были одинаковыми. Менялась только языковая модель.

Результат на 8 диагностических вопросах:

| Модель | Score | Correct | Semantic | Token F1 | Schema | Citations | Avg ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen3.6-35b-a3b` | `0.848` | `100%` | `0.851` | `0.603` | `100%` | `100%` | `2691` |
| `yandexgpt-5-lite` | `0.840` | `100%` | `0.829` | `0.525` | `100%` | `100%` | `3059` |
| `deepseek-v4-flash` | `0.763` | `75%` | `0.821` | `0.559` | `100%` | `100%` | `3176` |
| `yandexgpt-5-pro` | `0.745` | `75%` | `0.826` | `0.510` | `100%` | `100%` | `5424` |
| `yandexgpt-5.1` | `0.745` | `75%` | `0.786` | `0.499` | `100%` | `100%` | `4500` |
| `gpt-oss-120b` | `0.743` | `75%` | `0.725` | `0.342` | `100%` | `100%` | `3201` |
| `gpt-oss-20b` | `0.696` | `75%` | `0.749` | `0.439` | `87.5%` | `87.5%` | `3440` |
| `qwen3-235b-a22b-fp8` | `0.670` | `75%` | `0.698` | `0.387` | `87.5%` | `87.5%` | `5408` |

Для текущего пайплайна выбрана `qwen3.6-35b-a3b`: на этом наборе она дала лучший score, лучший token F1 и была быстрее `yandexgpt-5-lite`. `yandexgpt-5-lite` оставлен fallback-моделью.

Запуск эксперимента:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_llm_model_comparison.ps1
```

Подробный отчёт сохранён в `reports/llm_model_comparison.md`.

### Refusal mechanisms

Для шага 15 добавлен механизм отказа от ответа, если в найденных документах недостаточно информации. Это нужно для вопросов вне корпуса: например про погоду, кулинарию, программирование или другие госуслуги не из DMV.

Исследовались подходы:

- threshold по score после reranker;
- threshold по lexical overlap;
- hybrid score + lexical overlap;
- DistilBERT-подобный подход через маленький cross-encoder reranker `cross-encoder/ms-marco-MiniLM-L2-v2`.

На диагностическом наборе из 20 DMV-вопросов и 20 внешних вопросов лучший строгий порог по balanced score был около `top_score >= -0.5`, но для runtime выбран более осторожный к нормальным DMV-вопросам порог:

```env
RAG_ENABLE_REFUSAL_GATE=true
RAG_REFUSAL_MIN_TOP_SCORE=-2.0
RAG_REFUSAL_MIN_LEXICAL_OVERLAP=0.0
```

Почему не максимально строгий порог: лучше пропустить один спорный вопрос дальше к LLM-as-a-judge, чем ошибочно отказать пользователю на вопрос, который всё-таки относится к DMV-документам.

При отказе генерация не запускается. API возвращает обычный ответ со специальным блоком:

```json
{
  "refusal": {
    "enabled": true,
    "refused": true,
    "strategy": "reranker_score_and_lexical_overlap",
    "reason": "top retrieval score -10.901 is below threshold -2.000"
  }
}
```

Запуск эксперимента:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_refusal_experiment.ps1
```

Подробный отчёт сохранён в `reports/refusal_mechanisms.md`.

### Chat RAG

Для шага 16 добавлен диалоговый режим:

- новый endpoint `POST /api/v1/chat`;
- блок Chat RAG в Gradio;
- учёт истории сообщений;
- учёт предыдущих ответов модели;
- разрешение ссылок на прошлые реплики пользователя.

Если пользователь после первого ответа пишет короткий follow-up вроде:

```text
How much does it cost?
```

система строит standalone-запрос с учётом истории:

```text
I lost my driver license. What should I do? How much does it cost?
```

После этого retrieval работает уже не по голой фразе `How much does it cost?`, а по вопросу с восстановленным контекстом.

Пример запроса к API:

```powershell
$body = @{
  messages = @(
    @{ role = "user"; content = "I lost my driver license. What should I do?" },
    @{ role = "assistant"; content = "You can replace it online, by mail, or at a DMV office." },
    @{ role = "user"; content = "How much does it cost?" }
  )
  language = "en"
  use_judge = $true
  use_adaptive_context = $true
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/chat `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

### Inference engine benchmark

Для шага 17 подготовлено воспроизводимое сравнение трёх движков:

- SGLang;
- vLLM;
- LMDeploy с backend TurboMind.

Во всех случаях зафиксированы одинаковые условия:

| Параметр | Значение |
|---|---|
| Модель | `Qwen/Qwen2.5-3B-Instruct` |
| Revision | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| Точность | FP16, без квантования |
| Параллелизм | TP = 1, одна GPU |
| Максимальный контекст | 2048 токенов |
| Длина входа / выхода | 512 / 128 токенов |
| Sampling | `temperature=0`, `top_p=1`, без generation seed, `ignore_eos=false` |
| Размеры батча | 1, 2, 4, 8, 16 одновременных запросов |
| Workload | RAG-like: общий префикс 128 токенов + уникальная часть 384 токена |
| Нагрузка | 16 запросов, 4 warm-up запроса, 3 измеряемых повтора |

Модель 3B выбрана для самого эксперимента по производительности, а не вместо основной модели RAG. Она поддерживается всеми тремя движками и позволяет проверить рост батча на распространённой GPU с 16 GB VRAM без того, чтобы сравнение преждевременно закончилось нехваткой памяти на FP16-весах 7B-модели.

Основная метрика — output throughput в токенах в секунду. Дополнительно измеряются request throughput, TTFT, TPOT, end-to-end latency, error rate, пиковая VRAM, загрузка GPU и средняя мощность. Для каждого показателя сохраняются отдельные повторы; итог строится по их агрегированным значениям, а не по одному удачному запуску. Конфигурации с ошибками не могут стать победителями по throughput.

Эксперимент выполнен в Kaggle на одной NVIDIA Tesla T4 16 GB. Получены следующие медианные значения output throughput:

| Движок | Batch 1 | Batch 2 | Batch 4 | Batch 8 | Batch 16 |
|---|---:|---:|---:|---:|---:|
| LMDeploy | 27.45 | 62.62 | 114.99 | 193.85 | 295.89 |
| vLLM | 24.27 | 59.86 | 107.49 | 177.31 | 264.87 |
| SGLang | 23.33 | 55.17 | 91.96 | 136.17 | 125.31 |

LMDeploy выиграл на всех пяти уровнях нагрузки. При batch=1 он также показал минимальные p95 TTFT — 188.60 ms — и p95 E2E — 4693.53 ms. Максимум throughput достигнут при batch=16, но для интерактивного RAG разумнее начинать с concurrency 2–4: пропускная способность уже растёт, а хвостовая задержка ещё не достигает значений больших батчей.

SGLang на batch=16 деградировал относительно batch=8. Это связано с Tesla T4 (compute capability 7.5): быстрый CuTe DSL путь текущей версии рассчитан на более новые GPU, поэтому для запуска использовались совместимые Triton/CUDA kernels. Результат нельзя автоматически переносить на A100/H100, где соотношение движков может измениться.

LMDeploy добавлен в приложение как поддерживаемый OpenAI-compatible provider. Yandex AI Studio остаётся настройкой по умолчанию: это сохраняет рабочий запуск на Windows без локальной NVIDIA GPU и не смешивает исследование скорости 3B-модели с ранее выбранной по качеству `qwen3.6-35b-a3b`. На Linux/CUDA после запуска LMDeploy переключение выполняется через `.env`:

```env
RAG_LLM_PROVIDER=lmdeploy
LOCAL_LLM_BASE_URL=http://127.0.0.1:18000/v1
LOCAL_LLM_MODEL=qwen2.5-3b-instruct
LOCAL_LLM_API_KEY=local
```

Генератор и LLM-as-a-Judge используют один OpenAI-compatible endpoint. Если для judge требуется более сильная модель, его можно временно отключить через `RAG_ENABLE_JUDGE=false` или поднять отдельный endpoint в следующей итерации.

### Priority scheduling benchmark

Для шага 18 сравнивались обычная FIFO-очередь и priority queue with aging. В каждом прогоне одновременно поступали 45 запросов: 9 `critical`, 21 `standard` и 15 `batch`. Downstream обрабатывал каждый запрос за фиксированные 100 ms, чтобы измерение не зависело от сети или случайной длины генерации. Для каждого режима использованы три повтора.

| Slots | Critical FIFO, ms | Critical priority, ms | Изменение | FIFO throughput, req/s | Priority throughput, req/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 2257.8 | 436.3 | -80.7% | 9.186 | 9.283 |
| 2 | 1082.2 | 207.6 | -80.8% | 18.028 | 18.009 |
| 4 | 564.5 | 87.4 | -84.5% | 34.674 | 34.585 |
| 8 | 264.4 | 22.2 | -91.6% | 69.278 | 69.220 |

Приоритетный scheduler существенно сокращает очередь для интерактивной группы, почти не меняя суммарный throughput: максимальная разница составила 1.06%. Ускорение `critical` достигается за счёт более поздней обработки `batch`, а aging не позволяет низкоприоритетным запросам ждать бесконечно. Абсолютные миллисекунды могут немного меняться между запусками из-за планировщика ОС; устойчивый результат эксперимента — направление и порядок эффекта.

Для текущего прототипа выбрана стартовая конфигурация с четырьмя слотами. Это не универсальный лимит для любой GPU: в реальной среде `max_concurrency` нужно подобрать повторным прогоном через LiteLLM и выбранную модель. Полный отчёт, p95-метрики, ограничения и схема размещения находятся в `reports/priority_scheduling.md`.

### Token-Aware routing benchmark

Дополнительно исследовано распределение запросов между тремя LLM-репликами. Сравнивались Round Robin, Least Requests, оценка по `max_tokens`, оценка будущего ответа через исторический EWMA и контрольный вариант с заранее известной фактической длиной ответа.

Модель стоимости учитывает различную цену prefill и decode:

```text
estimated_service = input_tokens / prefill_speed
                  + expected_output_tokens / decode_speed
```

В score реплики также входят уже назначенная токенная работа, размер очереди и оценка заполнения KV-cache. Проверены burst 12, 36, 72 и 144 запроса, по пять повторов. В сценарии остаточной нагрузки реплики начинали с 15 000, 4 000 и 9 000 незавершённых токенов.

Результат для burst=144:

| Стратегия | p95 E2E, ms | Makespan, ms | Throughput, req/s | Load CV |
|---|---:|---:|---:|---:|
| Round Robin | 93823.6 | 102825.7 | 1.403 | 0.2063 |
| Least Requests | 93823.6 | 102825.7 | 1.403 | 0.2063 |
| Token-Aware max | 83871.8 | 93025.1 | 1.550 | 0.1212 |
| Token-Aware EWMA | 77509.5 | 84216.3 | 1.711 | 0.0362 |
| Exact-estimate control | 78001.7 | 81512.6 | 1.768 | 0.0080 |

Token-Aware EWMA снизил p95 на 17.39%, уменьшил makespan на 18.10%, увеличил throughput на 21.94% и снизил дисбаланс между репликами на 82.45%. Оценка через `max_tokens` оказалась слишком консервативной: WAPE составил 223.4% против 29.3% у EWMA.

EWMA выбран как лучший практически применимый вариант для будущего multi-replica deployment. В текущем однорепличном запуске он не включён: сначала нужны несколько реальных LiteLLM-инстансов и калибровка коэффициентов по их телеметрии. Полный разбор и графики находятся в `reports/priority_scheduling.md`.
