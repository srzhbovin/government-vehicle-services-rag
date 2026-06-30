
## Что добавлено

```text
├── scripts/
│   ├── import_multidoc2dial_dmv.py
│   ├── recursive_chunk_documents.py
│   ├── compress_documents_bm25.py
│   ├── evaluate_bm25_retrieval.py
│   └── compare_chunking_strategies.py
├── tests/
├── reports/
│   └── chunking_comparison.md
├── data/
│   ├── downloads/
│   │   └── multidoc2dial.zip
│   ├── experiments/
│   │   └── chunking/
│   │       ├── chunking_comparison_results.csv
│   │       └── chunking_comparison_results.jsonl
│   └── prepared/
│       ├── dmv_documents.jsonl
│       ├── dmv_questions_train.jsonl
│       ├── dmv_questions_validation.jsonl
│       ├── dmv_questions_test.jsonl
│       ├── dmv_chunks_recursive.jsonl
│       ├── dmv_documents_compact_bm25.jsonl
│       ├── dmv_chunks_recursive_compact_bm25.jsonl
│       ├── dmv_documents_compact_bm25_safe.jsonl
│       └── dmv_chunks_recursive_compact_bm25_safe.jsonl
└── README.md
```

## 1. Импорт DMV-датасета

```bash
python3 scripts/import_multidoc2dial_dmv.py
```

Результат:

- 149 DMV-документов;
- 128 953 слова;
- 6525 train QA-пар;
- 1132 validation QA-пары;
- 1094 test QA-пары.

## 2. Recursive chunking

```bash
python3 scripts/recursive_chunk_documents.py
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
python3 scripts/compress_documents_bm25.py
python3 scripts/recursive_chunk_documents.py \
  data/prepared/dmv_documents_compact_bm25.jsonl \
  --output data/prepared/dmv_chunks_recursive_compact_bm25.jsonl
```

Результат:

- документы ужались до 42.9% по символам;
- слова ужались до 42.7%;
- чанки: 1060 вместо 2435;
- файл чанков: 1.9 MB вместо 4.9 MB.

Safe-режим:

```bash
python3 scripts/compress_documents_bm25.py \
  --target-ratio 0.75 \
  --max-segments 60 \
  --output data/prepared/dmv_documents_compact_bm25_safe.jsonl

python3 scripts/recursive_chunk_documents.py \
  data/prepared/dmv_documents_compact_bm25_safe.jsonl \
  --output data/prepared/dmv_chunks_recursive_compact_bm25_safe.jsonl
```

Результат:

- документы ужались до 70.8% по символам;
- слова ужались до 70.8%;
- чанки: 1771 вместо 2435;
- файл чанков: 3.3 MB вместо 4.9 MB.

## 4. Проверка качества BM25 retrieval

Для честной проверки добавлен простой retrieval baseline:

```bash
python3 scripts/evaluate_bm25_retrieval.py \
  --chunks data/prepared/dmv_chunks_recursive.jsonl \
  --questions data/prepared/dmv_questions_validation.jsonl
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
python3 scripts/compare_chunking_strategies.py
```

Проверяются:

- `CharacterTextSplitter`-style: фиксированные окна по символам;
- `RecursiveCharacterTextSplitter`-style: recursive splitting по естественным
  разделителям;
- `TokenTextSplitter`-style: фиксированные окна по словам/токенам;
- `semantic_lite`: sentence grouping через TF-IDF cosine similarity без внешних
  embedding-моделей.

Всего прогоняется 35 комбинаций с разными `chunk_size` и `chunk_overlap`.
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
`120/0`. Вариант `recursive 500/200` остаётся рабочим бейзлайном,
но в этой проверке он не лучший. 

## Вывод

После отдельного сравнения chunking-стратегий видно, что для текущего
BM25 retrieval сильнее оказался token splitting `120/0`.

BM25-сжатие тоже реализовано, но текущий эксперимент показал важную вещь:
место действительно экономится, однако качество retrieval заметно падает.
Поэтому compact-вариант лучше держать как экспериментальную ветку, а не как
основной препроцессинг по умолчанию.

## Тесты

```bash
python3 -m unittest discover -s tests -v
```

Проверяется:

- recursive chunking;
- overlap;
- сохранение source spans;
- BM25 sentence selection;
- pruning spans после сжатия;
- document-level Recall@k evaluator;
- сравнение Character/Recursive/Token/Semantic-lite chunking.
