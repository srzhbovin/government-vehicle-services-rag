# Исследование компонентов RAG на документах DMV

Проект поэтапно исследует подготовку документов, chunking, векторные индексы и retrieval для
Retrieval-Augmented Generation. Корпус — англоязычная часть DMV из MultiDoc2Dial: 149 документов
и 8751 диалоговый вопрос в трёх выборках.

На текущем этапе реализовано сравнение четырёх обязательных способов поиска:
 
- полный перебор по косинусному сходству;
- BM25;
- точный поиск через `FAISS IndexFlatIP`;
- гибрид BM25 + FAISS через Reciprocal Rank Fusion (RRF).

Дополнительно реализован опциональный cross-encoder reranker. Он запускается отдельным флагом,
потому что на CPU заметно медленнее основного поиска.

## Текущий результат

Для эксперимента используются token-чанки `120/0` и эмбеддинги
`sentence-transformers/all-MiniLM-L6-v2`. Метод выбирается по validation-набору; test используется
только для независимой проверки после выбора.

| Метод | Validation Recall@5 | Validation Recall@10 | Test Recall@5 | Test Recall@10 |
|---|---:|---:|---:|---:|
| Cosine | 0.5230 | 0.6131 | 0.5411 | 0.6243 |
| BM25 | 0.5769 | 0.6528 | 0.5759 | 0.6664 |
| FAISS Flat | 0.5230 | 0.6131 | 0.5411 | 0.6252 |
| Hybrid RRF | 0.6104 | **0.6935** | 0.5795 | **0.6590** |
| Hybrid RRF + reranker | **0.6237** | **0.6935** | **0.6106** | **0.6590** |

Для итогового RAG выбран `hybrid_reranked`: сначала BM25 и FAISS объединяются через RRF, затем
первые 10 кандидатов переставляет `cross-encoder/ms-marco-MiniLM-L2-v2`. Это лучший вариант по
основной метрике validation Recall@5, и преимущество подтверждается на test. Reranker не изменил
Recall@10, но чаще поднял уже найденный правильный документ в первую пятёрку.

Цена дополнительного качества заметна: гибрид без reranker обрабатывал запрос примерно за
6–7 мс, а вариант с reranker — за 117–403 мс в пакетном CPU-прогоне. Поэтому в будущем RAG
разумно оставить два профиля: `hybrid_reranked` для максимального качества и `hybrid_rrf` для
минимальной задержки. BM25 остаётся быстрым baseline.

Подробный отчёт находится в [reports/retrieval_comparison.md](reports/retrieval_comparison.md).
Проверка шагов 1–5 — в [reports/audit_steps_1_5.md](reports/audit_steps_1_5.md).

## Быстрый запуск на Windows

Рекомендуется Python 3.13. Текущие версии PyTorch и SentenceTransformers устанавливаются на нём
без дополнительных действий.

Из корня `RAG` выполните:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Полный эксперимент с нейросетевыми эмбеддингами:

```powershell
.\.venv\Scripts\python.exe src\rag_pipeline\compare_retrieval_methods.py --device cpu
```

При первом запуске SentenceTransformers скачает модель с Hugging Face. Последующие запуски
используют локальный кэш.

Эксперимент с reranker:

```powershell
.\.venv\Scripts\python.exe src\rag_pipeline\compare_retrieval_methods.py `
  --device cpu `
  --use-reranker `
  --reranker-candidate-k 10 `
  --reranker-batch-size 128
```

Быстрая локальная проверка без PyTorch и скачивания моделей:

```powershell
.\.venv\Scripts\python.exe src\rag_pipeline\compare_retrieval_methods.py `
  --embedding-backend tfidf
```

TF-IDF режим нужен для smoke-теста и отладки. Итоговые выводы выше получены на
`all-MiniLM-L6-v2`, а не на TF-IDF.


## Как устроено сравнение

Всем методам передаются одинаковые 1152 чанка и одинаковые вопросы. Для cosine и FAISS
используются одни и те же L2-нормализованные эмбеддинги. Поэтому точный `IndexFlatIP` должен
давать то же качество, что и прямое косинусное сходство; небольшое отличие возможно только на
равных score.

Гибрид получает первые 60 результатов BM25 и FAISS и объединяет их с помощью weighted RRF:

```text
score(document) = 0.5 / (60 + rank_bm25) + 0.5 / (60 + rank_faiss)
```

Основная метрика выбора — Recall@5: доля вопросов, для которых в первой пятёрке чанков найден
хотя бы один чанк правильного документа. Дополнительно считаются Recall@1/3/10 и MRR@10.


## Тесты

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Проверяются загрузка и нормализация документов, базовый chunking, BM25, TF-IDF-векторизация,
совпадение точного cosine и FAISS, RRF и расчёт retrieval-метрик. На текущей версии проходят
22 теста.

## Воспроизведение предыдущих шагов

Рабочие скрипты находятся в `src/rag_pipeline`.

Импорт MultiDoc2Dial DMV:

```powershell
.\.venv\Scripts\python.exe src\rag_pipeline\import_multidoc2dial_dmv.py
```

Recursive chunking `500/200`:

```powershell
.\.venv\Scripts\python.exe src\rag_pipeline\recursive_chunk_documents.py
```

Сравнение 35 конфигураций chunking:

```powershell
.\.venv\Scripts\python.exe src\rag_pipeline\compare_chunking_strategies.py
```

Построение выбранных token-чанков `120/0`:

```powershell
.\.venv\Scripts\python.exe src\rag_pipeline\build_token_chunks.py
```

Сравнение FAISS Flat, IVF и HNSW:

```powershell
.\.venv\Scripts\python.exe src\rag_pipeline\benchmark_faiss_indexes.py
```

## Состояние шагов 1–5

Загрузка данных, подготовка, chunking и сравнение FAISS-индексов подтверждены кодом, сохранёнными
результатами и тестами. Полный корпус лучше BM25-сжатых вариантов, token chunking `120/0` лучше
исходного recursive `500/200`, а для текущего небольшого корпуса разумнее точный Flat-индекс.

Отдельное сравнение двух embedding-моделей из шага 2 полного ТЗ в репозитории отсутствует.
Использование `all-MiniLM-L6-v2` на этом шаге закрывает потребность retrieval-эксперимента.

## Структура проекта

```text
RAG/
├── src/rag_pipeline/             # актуальные скрипты этапов исследования
├── tests/                        # запускаемые тесты
├── data/current/                 # рабочие подготовленные данные и индексы
├── data/experiments/             # численные результаты экспериментов
├── reports/                      # отчёты и выводы
├── requirements.txt              # зависимости текущего retrieval-этапа
├── requirements-faiss.txt        # минимальные зависимости старого FAISS-этапа
└── .env.example                  # шаблон секретов для будущего шага генерации
```

