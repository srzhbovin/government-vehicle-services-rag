# Контрольные вопросы DMV

Это небольшая читаемая выборка из validation-части MultiDoc2Dial. Полный набор
из 1132 validation-вопросов находится в
`data/raw/multidoc2dial/dmv_questions_validation.jsonl`.

| № | Вопрос пользователя | Эталонный документ |
|---|---|---|
| 1 | My insurance ended, so what should I do? | `Top 5 DMV Mistakes and How to Avoid Them#3_0` |
| 2 | Do I need insurance to register my vehicle? | `New York State Insurance Requirements#3_0` |
| 3 | How do I pay DIAL-IN search account fees? | `DIAL-IN search accounts#3_0` |
| 4 | How do I surrender New York State plates to the DMV? | `Surrender (return or turn in) your plates to the DMV#3_0` |
| 5 | What can I do to prepare for my road test? | `Prepare for your road test#3_0` |
| 6 | What do I need to know about renewing my non-driver ID card? | `Renew non-driver ID card#1_0` |

Вопросы в датасете разговорные: часть из них зависит от предыдущих реплик.
Поэтому в JSONL вместе с вопросом хранится поле `history`. На первом baseline
можно начать с вопросов без истории, а затем проверить retrieval в контексте
полного диалога.

## Как использовать дальше

Для каждого вопроса поисковик должен вернуть `gold_document_ids` хотя бы в
первых `k` результатах. Например, если правильный документ оказался в top-5,
запрос считается найденным для метрики `Recall@5`.
