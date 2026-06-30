# RAG: 

Актуальная версия проекта теперь лежит в папке src`.

Проект закрывает несколько этапов RAG-пайплайна:

- recursive chunking;
- `CHUNK_SIZE = 500`;
- `CHUNK_OVERLAP = 200`;
- экспериментальное уменьшение исходного текста через BM25-подобный выбор важных
  предложений;
- сравнение разных стратегий chunking по retrieval-метрикам;
- сравнение FAISS-индексов Flat / IVF / HNSW.

## Что добавлено

```text
.
├── scripts/
│   └── ...                         # legacy/protected старые файлы
├── src/
│   └── rag_pipeline/
│       ├── import_multidoc2dial_dmv.py
│       ├── build_token_chunks.py
│       ├── recursive_chunk_documents.py
│       ├── compress_documents_bm25.py
│       ├── evaluate_bm25_retrieval.py
│       ├── compare_chunking_strategies.py
│       └── benchmark_faiss_indexes.py
├── tests_current/
│   ├── test_recursive_chunk_documents.py
│   ├── test_compress_documents_bm25.py
│   ├── test_evaluate_bm25_retrieval.py
│   └── test_compare_chunking_strategies.py
├── tests/                          # legacy/protected старые файлы
├── data/
│   ├── current/
│   │   ├── downloads/
│   │   │   └── multidoc2dial.zip
│   │   ├── indexes/
│   │   │   └── faiss/
│   │   └── prepared/
│   │       ├── dmv_documents.jsonl
│   │       ├── dmv_questions_train.jsonl
│   │       ├── dmv_questions_validation.jsonl
│   │       ├── dmv_questions_test.jsonl
│   │       ├── dmv_chunks_token_120_0.jsonl
│   │       ├── dmv_chunks_recursive.jsonl
│   │       ├── dmv_documents_compact_bm25.jsonl
│   │       ├── dmv_chunks_recursive_compact_bm25.jsonl
│   │       ├── dmv_documents_compact_bm25_safe.jsonl
│   │       └── dmv_chunks_recursive_compact_bm25_safe.jsonl
│   ├── experiments/
│   │   ├── chunking/
│   │   │   ├── chunking_comparison_results.csv
│   │   │   └── chunking_comparison_results.jsonl
│   │   └── faiss/
│   │       ├── faiss_index_results.csv
│   │       └── faiss_index_results.jsonl
│   └── prepared/                  # legacy/protected старые файлы
├── reports/
│   ├── chunking_comparison.md
│   └── faiss_comparison.md
└── README_CURRENT.md
```

## 1. Импорт DMV-датасета

```bash
python3 src/rag_pipeline/import_multidoc2dial_dmv.py
```

Результат:

- 149 DMV-документов;
- 128 953 слова;
- 6525 train QA-пар;
- 1132 validation QA-пары;
- 1094 test QA-пары.

## 2. Recursive chunking

```bash
python3 src/rag_pipeline/recursive_chunk_documents.py
```

Параметры по умолчанию:

```python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
```

Это character-based recursive chunking: текст сначала пытается резаться по более
крупным естественным границам, потом по более мелким:

```text
\n\n → \n → . → ? → ! → ; → : → , → пробел → fallback по символам
```

Результат на полном DMV-корпусе:

- 2435 чанков;
- все 149 документов покрыты;
- средний размер чанка: 414.3 символа;
- максимум: 499 символов;
- пустых чанков: 0;
- чанков без source spans: 0.

## 3. BM25-подобное сжатие документов

Идея эксперимента:

1. документ режется на предложения/небольшие смысловые сегменты;
2. из документа достаются важные термы через `tf * idf`;
3. эти термы используются как запрос;
4. предложения скорятся BM25-подобной формулой;
5. самые сильные предложения собираются обратно в compact-документ;
6. затем compact-документы снова режутся recursive chunking.

Агрессивный режим:

```bash
python3 src/rag_pipeline/compress_documents_bm25.py
python3 src/rag_pipeline/recursive_chunk_documents.py \
  data/current/prepared/dmv_documents_compact_bm25.jsonl \
  --output data/current/prepared/dmv_chunks_recursive_compact_bm25.jsonl
```

Результат:

- документы ужались до 42.9% по символам;
- слова ужались до 42.7%;
- чанки: 1060 вместо 2435;
- файл чанков: 1.9 MB вместо 4.9 MB.

Safe-режим:

```bash
python3 src/rag_pipeline/compress_documents_bm25.py \
  --target-ratio 0.75 \
  --max-segments 60 \
  --output data/current/prepared/dmv_documents_compact_bm25_safe.jsonl

python3 src/rag_pipeline/recursive_chunk_documents.py \
  data/current/prepared/dmv_documents_compact_bm25_safe.jsonl \
  --output data/current/prepared/dmv_chunks_recursive_compact_bm25_safe.jsonl
```

Результат:

- документы ужались до 70.8% по символам;
- слова ужались до 70.8%;
- чанки: 1771 вместо 2435;
- файл чанков: 3.3 MB вместо 4.9 MB.

## 4. Проверка качества BM25 retrieval

Для честной проверки добавлен простой retrieval baseline:

```bash
python3 src/rag_pipeline/evaluate_bm25_retrieval.py \
  --chunks data/current/prepared/dmv_chunks_recursive.jsonl \
  --questions data/current/prepared/dmv_questions_validation.jsonl
```

Метрика считается document-level: вопрос считается найденным, если среди top-k
чанков есть хотя бы один чанк из правильного `gold_document_ids`.

| Корпус | Чанков | Размер chunk-файла | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| Full recursive | 2435 | 4.9 MB | 0.3569 | 0.5636 | 0.6290 | 0.4389 |
| Compact BM25 aggressive | 1060 | 1.9 MB | 0.2641 | 0.4293 | 0.5009 | 0.3345 |
| Compact BM25 safe | 1771 | 3.3 MB | 0.2951 | 0.4938 | 0.5742 | 0.3790 |

## 5. Исследование способов chunking

Добавлен отдельный эксперимент для сравнения разных способов разбиения:

```bash
python3 src/rag_pipeline/compare_chunking_strategies.py
```

Проверяются:

- `CharacterTextSplitter`-style: фиксированные окна по символам;
- `RecursiveCharacterTextSplitter`-style: recursive splitting по естественным
  разделителям;
- `TokenTextSplitter`-style: фиксированные окна по словам/токенам;
- `semantic_lite`: sentence grouping через TF-IDF cosine similarity без внешних
  embedding-моделей.

Всего прогоняется 35 конфигураций с разными `chunk_size` и `chunk_overlap`.
Результаты сохраняются сюда:

- `data/experiments/chunking/chunking_comparison_results.csv`;
- `data/experiments/chunking/chunking_comparison_results.jsonl`;
- `reports/chunking_comparison.md`.

Лучшие результаты по validation-вопросам:

| Rank | Splitter | Size | Overlap | Unit | Chunks | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | token | 120 | 0 | tokens | 1152 | 0.3595 | 0.5707 | 0.6475 | 0.4501 |
| 2 | recursive | 500 | 0 | characters | 1823 | 0.3445 | 0.5601 | 0.6431 | 0.4373 |
| 3 | character | 800 | 0 | characters | 991 | 0.3269 | 0.5627 | 0.6413 | 0.4280 |
| 11 | recursive | 500 | 200 | characters | 2435 | 0.3569 | 0.5636 | 0.6290 | 0.4389 |

Главный вывод: на текущем BM25 retrieval лучший вариант — token splitting
`120/0`. Baseline вариант `recursive 500/200` остаётся рабочим baseline,
но в этой проверке он не лучший. Большой overlap часто не помогал, а иногда
ухудшал качество из-за роста числа похожих соседних чанков.

## 6. Исследование FAISS

Для FAISS сначала материализован лучший chunking-вариант из предыдущего
исследования:

```bash
python3 src/rag_pipeline/build_token_chunks.py
```

Результат:

- `data/current/prepared/dmv_chunks_token_120_0.jsonl`;
- 1152 чанка;
- token splitting `chunk_size=120`, `chunk_overlap=0`.

Затем построены dense TF-IDF-векторы размерности 2048 и поверх них сравнены
несколько FAISS-индексов:

- `IndexFlatIP`;
- `IndexIVFFlat`, `nlist=32`, `nprobe=4`;
- `IndexIVFFlat`, `nlist=32`, `nprobe=16`;
- `IndexHNSWFlat`, `M=16`, `efSearch=32`;
- `IndexHNSWFlat`, `M=32`, `efSearch=64`.

Запуск:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-faiss.txt
.venv/bin/python src/rag_pipeline/benchmark_faiss_indexes.py
```

Результаты сохраняются сюда:

- `data/experiments/faiss/faiss_index_results.csv`;
- `data/experiments/faiss/faiss_index_results.jsonl`;
- `data/current/indexes/faiss/*.index`;
- `reports/faiss_comparison.md`.

Итоговая таблица:

| Index | Build ms | Search ms/query | Index size | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| IVF nprobe=16 | 8.61 | 0.0781 | 9481.4 KB | 0.3136 | 0.5353 | 0.6299 | 0.4095 |
| HNSW M=32 ef=64 | 162.55 | 0.1155 | 9521.7 KB | 0.3163 | 0.5345 | 0.6281 | 0.4109 |
| Flat IP | 1.50 | 0.0034 | 9216.0 KB | 0.3163 | 0.5345 | 0.6228 | 0.4101 |
| IVF nprobe=4 | 7.77 | 0.0213 | 9481.4 KB | 0.3074 | 0.5371 | 0.6228 | 0.4060 |
| HNSW M=16 ef=32 | 134.90 | 0.0597 | 9378.1 KB | 0.3127 | 0.5274 | 0.6184 | 0.4070 |

Вывод по FAISS:

- на маленьком корпусе из 1152 чанков `Flat IP` оказался самым быстрым и самым
  простым вариантом;
- `IVF nprobe=16` дал лучший `Recall@10`, но был медленнее Flat;
- `HNSW` здесь не даёт выигрыша, потому что данных мало, а построение индекса
  дороже;
- для текущего размера корпуса лучше начинать с `IndexFlatIP`;
- IVF/HNSW имеет смысл включать, когда корпус вырастет хотя бы до десятков или
  сотен тысяч чанков.

## Вывод

Recursive chunking с `500/200` сделан и остаётся рабочим
baseline. После отдельного сравнения chunking-стратегий видно, что для текущего
BM25 retrieval сильнее оказался token splitting `120/0`, поэтому финальный выбор
чанкинга лучше делать по метрикам, а не только по корпоративному дефолту.

BM25-сжатие тоже реализовано, но текущий эксперимент показал важную вещь:
место действительно экономится, однако качество retrieval заметно падает.

## Тесты

```bash
python3 -m unittest discover -s tests_current -v
```

Проверяется:

- recursive chunking;
- overlap;
- сохранение source spans;
- BM25 sentence selection;
- pruning spans после сжатия;
- document-level Recall@k evaluator;
- сравнение Character/Recursive/Token/Semantic-lite chunking;
- FAISS benchmark для Flat / IVF / HNSW.
