# Базовая RAG-система

На этапе 7 собран полный RAG pipeline для документов MultiDoc2Dial DMV:

```text
документы → подготовка → token chunks → embeddings → FAISS/BM25 retrieval
→ RRF → reranker → top-5 context → YandexGPT → ответ с источниками
```

## Выбранные компоненты

- корпус: 149 документов DMV;
- chunking: token `120/0`, 1152 чанка;
- embedding-модель: `sentence-transformers/all-MiniLM-L6-v2`;
- dense index: `FAISS IndexFlatIP`;
- retrieval: BM25 + FAISS, weighted RRF `0.5/0.5`;
- reranker: `cross-encoder/ms-marco-MiniLM-L2-v2`, top-10;
- контекст генератора: top-5;
- основной generator provider: Yandex AI Studio Responses API;
- проверенная модель: `yandexgpt-5-lite`;
- локальный запасной provider: LM Studio OpenAI-compatible API с `google/gemma-3-4b`.

## Проверка полного pipeline

Контрольный вопрос:

> How quickly must I report a change of address to DMV?

Retrieval поставил на первое место документ `How to change your address#1`. Полный контрольный
запрос через Yandex AI Studio сформировал ответ на русском языке:

- модель: `gpt://<folder_id>/yandexgpt-5-lite`;
- ответ: `Вы должны сообщить об изменении адреса в DMV в течение 10 дней [1], [2], [3].`;
- retrieval: около 200 мс после загрузки моделей;
- генерация: около 2,2 с;
- передано и сгенерировано: 1199 токенов.

Ответ соответствует найденным фрагментам и содержит номера источников.

## Вывод

Этап завершён рабочим сквозным RAG-пайплайном: от загрузки документов и построения
индекса до retrieval, генерации и возврата источников. Компоненты разделены так, чтобы
последующие эксперименты меняли отдельный этап, не переписывая API и интерфейс.

## Точки расширения

Backend, retrieval и генераторы разделены интерфейсами. Следующие исследования можно добавлять
без изменения транспортного слоя:

- prompt strategies и Structured Output;
- top-p, temperature и max tokens;
- query rewriting и история диалога;
- другие локальные и облачные LLM;
- оценка качества готовых ответов;
- кэширование и асинхронная генерация.
