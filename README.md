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
9. Расширение контекста: к найденным чанкам добавляются начальные и соседние чанки того же документа.
10. Генерация ответа через YandexGPT.
11. Structured Output: модель возвращает структурированный объект `answer`, `confidence`, `source`.
12. Проверка ответа через LLM-as-a-judge.
13. FastAPI отдаёт результат в API.
14. Gradio показывает ответ, параметры, judge-вердикт и найденные фрагменты.

## Что изменено в последней итерации

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
- менять `top_k`, `temperature`, `top_p`, `max_output_tokens`;
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
  top_k = 6
  temperature = 0.1
  top_p = 0.9
  max_output_tokens = 700
  use_judge = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## Основные настройки `.env`

```env
RAG_LLM_PROVIDER=yandex
RAG_PROMPT_STRATEGY=structured_output

YANDEX_TEMPERATURE=0.1
YANDEX_TOP_P=0.9
YANDEX_MAX_OUTPUT_TOKENS=700

RAG_TOP_K=6
RAG_CANDIDATE_K=60
RAG_RERANKER_CANDIDATE_K=20
RAG_BM25_WEIGHT=0.5
RAG_RRF_K=60

RAG_CONTEXT_WINDOW=1
RAG_CONTEXT_INTRO_CHUNKS=3
RAG_MAX_CONTEXT_CHUNKS=12
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

## Текущие ограничения

- Judge повышает надёжность, но добавляет второй LLM-вызов и увеличивает задержку.
- `confidence` от генератора пока нельзя считать строгой вероятностью правильности.
- Context expansion улучшает полноту ответа, но может ухудшать классические retrieval-метрики из-за повторов одного документа.
- Query expansion пока простой и rule-based. Это лучше, чем ничего, но дальше его стоит заменить на отдельный query rewriting шаг.
- Корпус документов английский, поэтому русские вопросы потенциально стоит переводить/нормализовать перед retrieval.

## Что логично улучшать дальше

1. Сделать отдельный query rewriting: превращать диалоговый или размытый вопрос в самостоятельный поисковый запрос.
2. Добавить document-level retrieval: сначала выбирать документ, потом лучшие чанки внутри него.
3. Разделить retrieval-метрики и generation-context метрики, чтобы context expansion не путал оценку.
4. Добавить chat-RAG для вопросов с историей диалога из MultiDoc2Dial.
5. Оценить judge на большем наборе: сколько ошибок он ловит и сколько раз зря исправляет хороший ответ.
