# Исследование механизмов отказа от ответа

Проверка выполнена на 20 вопросах из DMV-корпуса и 20 вопросах вне корпуса.

Сравнивались три подхода:

- порог по score после reranker;
- порог по lexical overlap между вопросом и найденными фрагментами;
- гибрид score + overlap;
- роль DistilBERT-подобного классификатора в текущей архитектуре выполняет маленький cross-encoder reranker `cross-encoder/ms-marco-MiniLM-L2-v2`: он получает пару query/document и выдаёт relevance score.

| Механизм | Конфигурация | Balanced | Accuracy | Recall answerable | Specificity unanswerable | False accept | False refusal |
|---|---|---:|---:|---:|---:|---:|---:|
| `reranker_score_threshold` | `top_score>=-2` | 0.975 | 0.975 | 1.000 | 0.950 | 0.050 | 0.000 |
| `reranker_score_threshold` | `top_score>=-0.5` | 0.975 | 0.975 | 0.950 | 1.000 | 0.000 | 0.050 |
| `hybrid_reranker_and_lexical` | `top_score>=-2; lexical>=0` | 0.975 | 0.975 | 1.000 | 0.950 | 0.050 | 0.000 |
| `hybrid_reranker_and_lexical` | `top_score>=-2; lexical>=0.05` | 0.975 | 0.975 | 1.000 | 0.950 | 0.050 | 0.000 |
| `reranker_score_threshold` | `top_score>=-1.5` | 0.950 | 0.950 | 0.950 | 0.950 | 0.050 | 0.050 |
| `reranker_score_threshold` | `top_score>=-1` | 0.950 | 0.950 | 0.950 | 0.950 | 0.050 | 0.050 |
| `reranker_score_threshold` | `top_score>=0` | 0.950 | 0.950 | 0.900 | 1.000 | 0.000 | 0.100 |
| `hybrid_reranker_and_lexical` | `top_score>=0; lexical>=0` | 0.950 | 0.950 | 0.900 | 1.000 | 0.000 | 0.100 |
| `hybrid_reranker_and_lexical` | `top_score>=0; lexical>=0.05` | 0.950 | 0.950 | 0.900 | 1.000 | 0.000 | 0.100 |
| `reranker_score_threshold` | `top_score>=-3` | 0.925 | 0.925 | 1.000 | 0.850 | 0.150 | 0.000 |
| `reranker_score_threshold` | `top_score>=1` | 0.925 | 0.925 | 0.850 | 1.000 | 0.000 | 0.150 |
| `hybrid_reranker_and_lexical` | `top_score>=1; lexical>=0` | 0.925 | 0.925 | 0.850 | 1.000 | 0.000 | 0.150 |

## Вывод

Лучший вариант на этом диагностическом наборе: `reranker_score_threshold` с конфигурацией `top_score>=-0.5`.

В runtime выбран более осторожный к нормальным DMV-вопросам gate `top_score >= -2`. Он может пропустить чуть больше спорных запросов к LLM-as-a-judge, зато на текущем наборе не отказывает на answerable-вопросах.

Если top score ниже `RAG_REFUSAL_MIN_TOP_SCORE`, генерация не запускается и система честно сообщает, что в документах недостаточно информации. Это защищает от вопросов вне DMV-корпуса и экономит LLM-вызовы.

Полный набор признаков и таблица сравнения сохранены в `data/experiments/refusal/`.
