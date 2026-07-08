# Government Vehicle Services RAG

Проект реализует RAG-систему по документам DMV из MultiDoc2Dial. Система ищет релевантные фрагменты документов, передаёт их в LLM и возвращает ответ с источниками.

Текущая версия — базовый, но уже полноценный RAG: есть backend на FastAPI, интерфейс на Gradio, гибридный retrieval, structured output, настройка параметров генерации и защита от галлюцинаций через LLM-as-a-judge.

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
10. Генерация ответа через YandexGPT.
11. Structured Output: модель возвращает структурированный объект `answer`, `confidence`, `source`.
12. Проверка ответа через LLM-as-a-judge.
13. FastAPI отдаёт результат в API.
14. Gradio показывает ответ, параметры, judge-вердикт и найденные фрагменты.

## Что изменено в последней итерации

### 0. Исследование стратегий формирования контекста

Добавлен отдельный эксперимент для шага 13. Теперь можно сравнивать, сколько чанков брать после retrieval и нужно ли добавлять соседние/начальные чанки документа.

По результатам проверки текущий дефолт изменён на более компактный:

```env
RAG_TOP_K=3
RAG_CONTEXT_WINDOW=0
RAG_CONTEXT_INTRO_CHUNKS=0
RAG_MAX_CONTEXT_CHUNKS=6
```

Расширение контекста осталось доступным через API и Gradio. Дополнительно включён адаптивный fallback: если LLM-as-a-judge не принимает первый компактный ответ, система делает второй проход с более широким контекстом.

### 1. Защита от галлюцинаций

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

### 2. Настраиваемые параметры

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

### 3. Улучшение retrieval-контекста

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

### 4. Новый интерфейс на Gradio

Streamlit-интерфейс заменён на Gradio. В интерфейсе можно:

- задать вопрос;
- выбрать язык ответа;
- менять `top_k`, `context_window`, `intro_chunks`, `max_context_chunks`, `temperature`, `top_p`, `max_output_tokens`;
- включать/выключать judge;
- видеть итоговый ответ;
- видеть judge verdict, score и причину;
- смотреть найденные фрагменты документов.

## Быстрый запуск

Команды ниже выполняются из корня проекта:

```powershell
cd C:\Users\Sergey\Desktop\government_vehicle_services_rag\RAG
```

### 1. Установить зависимости

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
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
RAG_USE_QUERY_EXPANSION=true

RAG_ENABLE_JUDGE=true
RAG_JUDGE_MIN_SCORE=0.72
RAG_JUDGE_TEMPERATURE=0.0
RAG_JUDGE_MAX_OUTPUT_TOKENS=500
```

## Результаты экспериментов

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

## Текущие ограничения

- Judge повышает надёжность, но добавляет второй LLM-вызов и увеличивает задержку.
- Adaptive context retry улучшает устойчивость на спорных запросах, но если первый ответ отклонён judge, запрос становится дороже и медленнее: появляется дополнительная генерация и повторная проверка.
- `confidence` от генератора пока нельзя считать строгой вероятностью правильности.
- Слишком широкий контекст может ухудшать ответ: LLM получает больше текста, но доля действительно полезных чанков падает.
- Query expansion пока простой и rule-based. Это лучше, чем ничего, но дальше его стоит заменить на отдельный query rewriting шаг.
- Корпус документов английский, поэтому русские вопросы потенциально стоит переводить/нормализовать перед retrieval.

## Что логично улучшать дальше

1. Сделать отдельный query rewriting: превращать диалоговый или размытый вопрос в самостоятельный поисковый запрос.
2. Добавить document-level retrieval: сначала выбирать документ, потом лучшие чанки внутри него.
3. Разделить retrieval-метрики и generation-context метрики, чтобы context expansion не путал оценку.
4. Добавить chat-RAG для вопросов с историей диалога из MultiDoc2Dial.
5. Оценить judge на большем наборе: сколько ошибок он ловит и сколько раз зря исправляет хороший ответ.
