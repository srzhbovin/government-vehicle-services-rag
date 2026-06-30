# Chunking comparison

Сравнение способов разбиения DMV-документов для RAG retrieval.

Документы: `C:\Users\Sergey\Desktop\government_vehicle_services_rag\RAG\data\current\prepared\dmv_documents.jsonl`

Вопросы для проверки: `C:\Users\Sergey\Desktop\government_vehicle_services_rag\RAG\data\current\prepared\dmv_questions_validation.jsonl`

Метрика: document-level Recall@k. Вопрос считается найденным, если среди top-k
чанков есть хотя бы один чанк из правильного `gold_document_ids`.

## Главное

- Лучший вариант по Recall@10: `token`, size=120, overlap=0, Recall@10=0.6475.
- Компания-style baseline `recursive 500/200`: Recall@10=0.6290, Recall@5=0.5636, чанков=2435.
- TokenTextSplitter в этом датасете оказался сильным конкурентом, потому что BM25 retrieval тоже работает по словам/термам.
- Semantic-lite реализован без внешних embedding-моделей, через sentence grouping и TF-IDF similarity;

## Top results

| Rank | Splitter | Size | Overlap | Unit | Threshold | Chunks | Avg chars | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | token | 120 | 0 | tokens |  | 1152 | 632.9 | 0.3595 | 0.5707 | 0.6475 | 0.4501 |
| 2 | recursive | 500 | 0 | characters |  | 1823 | 399.6 | 0.3445 | 0.5601 | 0.6431 | 0.4373 |
| 3 | character | 800 | 0 | characters |  | 991 | 736.5 | 0.3269 | 0.5627 | 0.6413 | 0.4280 |
| 4 | token | 160 | 0 | tokens |  | 872 | 836.5 | 0.3436 | 0.5504 | 0.6413 | 0.4343 |
| 5 | character | 500 | 0 | characters |  | 1543 | 472.9 | 0.3534 | 0.5663 | 0.6360 | 0.4433 |
| 6 | recursive | 800 | 0 | characters |  | 1145 | 636.8 | 0.3472 | 0.5601 | 0.6360 | 0.4354 |
| 7 | character | 800 | 100 | characters |  | 1096 | 752.3 | 0.3489 | 0.5627 | 0.6352 | 0.4393 |
| 8 | recursive | 400 | 0 | characters |  | 2326 | 313.0 | 0.3489 | 0.5610 | 0.6352 | 0.4405 |
| 9 | recursive | 500 | 100 | characters |  | 2264 | 410.6 | 0.3569 | 0.5539 | 0.6325 | 0.4393 |
| 10 | token | 160 | 30 | tokens |  | 1033 | 851.1 | 0.3304 | 0.5504 | 0.6307 | 0.4245 |
| 11 | recursive | 500 | 200 | characters |  | 2435 | 414.3 | 0.3569 | 0.5636 | 0.6290 | 0.4389 |
| 12 | token | 80 | 0 | tokens |  | 1677 | 434.5 | 0.3383 | 0.5610 | 0.6290 | 0.4327 |
| 13 | recursive | 800 | 100 | characters |  | 1307 | 649.5 | 0.3525 | 0.5574 | 0.6290 | 0.4368 |
| 14 | character | 400 | 0 | characters |  | 1903 | 383.4 | 0.3392 | 0.5389 | 0.6272 | 0.4261 |
| 15 | recursive | 800 | 200 | characters |  | 1350 | 654.8 | 0.3595 | 0.5521 | 0.6246 | 0.4414 |

## All results

| Rank | Splitter | Size | Overlap | Unit | Threshold | Chunks | Avg chars | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | token | 120 | 0 | tokens |  | 1152 | 632.9 | 0.3595 | 0.5707 | 0.6475 | 0.4501 |
| 2 | recursive | 500 | 0 | characters |  | 1823 | 399.6 | 0.3445 | 0.5601 | 0.6431 | 0.4373 |
| 3 | character | 800 | 0 | characters |  | 991 | 736.5 | 0.3269 | 0.5627 | 0.6413 | 0.4280 |
| 4 | token | 160 | 0 | tokens |  | 872 | 836.5 | 0.3436 | 0.5504 | 0.6413 | 0.4343 |
| 5 | character | 500 | 0 | characters |  | 1543 | 472.9 | 0.3534 | 0.5663 | 0.6360 | 0.4433 |
| 6 | recursive | 800 | 0 | characters |  | 1145 | 636.8 | 0.3472 | 0.5601 | 0.6360 | 0.4354 |
| 7 | character | 800 | 100 | characters |  | 1096 | 752.3 | 0.3489 | 0.5627 | 0.6352 | 0.4393 |
| 8 | recursive | 400 | 0 | characters |  | 2326 | 313.0 | 0.3489 | 0.5610 | 0.6352 | 0.4405 |
| 9 | recursive | 500 | 100 | characters |  | 2264 | 410.6 | 0.3569 | 0.5539 | 0.6325 | 0.4393 |
| 10 | token | 160 | 30 | tokens |  | 1033 | 851.1 | 0.3304 | 0.5504 | 0.6307 | 0.4245 |
| 11 | recursive | 500 | 200 | characters |  | 2435 | 414.3 | 0.3569 | 0.5636 | 0.6290 | 0.4389 |
| 12 | token | 80 | 0 | tokens |  | 1677 | 434.5 | 0.3383 | 0.5610 | 0.6290 | 0.4327 |
| 13 | recursive | 800 | 100 | characters |  | 1307 | 649.5 | 0.3525 | 0.5574 | 0.6290 | 0.4368 |
| 14 | character | 400 | 0 | characters |  | 1903 | 383.4 | 0.3392 | 0.5389 | 0.6272 | 0.4261 |
| 15 | recursive | 800 | 200 | characters |  | 1350 | 654.8 | 0.3595 | 0.5521 | 0.6246 | 0.4414 |
| 16 | token | 120 | 30 | tokens |  | 1453 | 653.8 | 0.3525 | 0.5477 | 0.6246 | 0.4345 |
| 17 | semantic_lite | 800 | 200 | characters | 0.05 | 1719 | 569.1 | 0.3436 | 0.5309 | 0.6246 | 0.4282 |
| 18 | character | 500 | 200 | characters |  | 2411 | 490.2 | 0.3516 | 0.5468 | 0.6219 | 0.4361 |
| 19 | character | 400 | 100 | characters |  | 2460 | 390.4 | 0.3534 | 0.5459 | 0.6219 | 0.4387 |
| 20 | semantic_lite | 800 | 200 | characters | 0.12 | 1850 | 546.3 | 0.3481 | 0.5389 | 0.6210 | 0.4306 |
| 21 | character | 800 | 200 | characters |  | 1248 | 760.9 | 0.3419 | 0.5548 | 0.6201 | 0.4306 |
| 22 | semantic_lite | 800 | 100 | characters | 0.05 | 1606 | 559.7 | 0.3463 | 0.5424 | 0.6201 | 0.4314 |
| 23 | semantic_lite | 500 | 100 | characters | 0.12 | 2752 | 355.9 | 0.3436 | 0.5398 | 0.6193 | 0.4281 |
| 24 | semantic_lite | 800 | 100 | characters | 0.12 | 1718 | 540.8 | 0.3383 | 0.5398 | 0.6193 | 0.4250 |
| 25 | token | 80 | 30 | tokens |  | 2565 | 443.5 | 0.3525 | 0.5380 | 0.6184 | 0.4327 |
| 26 | character | 500 | 100 | characters |  | 1859 | 484.4 | 0.3419 | 0.5415 | 0.6175 | 0.4252 |
| 27 | recursive | 400 | 100 | characters |  | 2949 | 326.9 | 0.3436 | 0.5486 | 0.6148 | 0.4280 |
| 28 | semantic_lite | 500 | 100 | characters | 0.05 | 2591 | 366.9 | 0.3463 | 0.5442 | 0.6131 | 0.4272 |
| 29 | token | 160 | 60 | tokens |  | 1272 | 872.0 | 0.3392 | 0.5380 | 0.6087 | 0.4210 |
| 30 | token | 120 | 60 | tokens |  | 2076 | 665.6 | 0.3542 | 0.5353 | 0.6078 | 0.4264 |
| 31 | character | 400 | 200 | characters |  | 3581 | 395.2 | 0.3428 | 0.5292 | 0.6051 | 0.4225 |
| 32 | recursive | 400 | 200 | characters |  | 3258 | 332.6 | 0.3569 | 0.5495 | 0.6042 | 0.4347 |
| 33 | semantic_lite | 500 | 200 | characters | 0.05 | 2862 | 372.7 | 0.3569 | 0.5283 | 0.6042 | 0.4317 |
| 34 | semantic_lite | 500 | 200 | characters | 0.12 | 3054 | 360.7 | 0.3534 | 0.5239 | 0.5989 | 0.4268 |
| 35 | token | 80 | 60 | tokens |  | 6072 | 450.1 | 0.3525 | 0.4859 | 0.5654 | 0.4054 |
