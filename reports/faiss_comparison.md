# FAISS index comparison

Исследование FAISS-индексов для retrieval поверх DMV chunks.

Корпус чанков: `data\current\prepared\dmv_chunks_token_120_0.jsonl`

Validation-вопросы: `data\current\prepared\dmv_questions_validation.jsonl`

Векторы: dense TF-IDF, L2-normalized, dimension=2048. Поиск идёт через
inner product, то есть эквивалент cosine similarity для нормализованных векторов.

## Вывод

- Лучшее качество: `ivf32_nprobe16`, Recall@10=0.6299.
- Самый быстрый поиск: `flat_ip`, 0.0335 ms/query.
- На текущем небольшом корпусе Flat практически оптимален: он точный, простой и быстрый.
- IVF и HNSW полезнее становятся на больших корпусах, где Flat уже слишком дорогой по времени.

## Results

| Index | Build ms | Search ms/query | Index size | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| ivf32_nprobe16 | 35.88 | 0.1627 | 9481.4 KB | 0.3136 | 0.5353 | 0.6299 | 0.4095 |
| hnsw_m32_ef64 | 306.55 | 0.2276 | 9521.7 KB | 0.3163 | 0.5345 | 0.6281 | 0.4109 |
| flat_ip | 3.21 | 0.0335 | 9216.0 KB | 0.3163 | 0.5345 | 0.6228 | 0.4101 |
| ivf32_nprobe4 | 35.06 | 0.0404 | 9481.4 KB | 0.3074 | 0.5371 | 0.6228 | 0.4060 |
| hnsw_m16_ef32 | 201.32 | 0.1081 | 9378.1 KB | 0.3127 | 0.5274 | 0.6184 | 0.4070 |

