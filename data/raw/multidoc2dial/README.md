# MultiDoc2Dial: DMV

Основной корпус проекта — домен DMV из IBM MultiDoc2Dial.

- источник: <https://github.com/IBM/multidoc2dial>
- официальный архив: <https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip>
- статья: <https://arxiv.org/abs/2109.12595>
- язык: английский
- предметная область: услуги DMV

Локальные файлы создаются скриптом:

```bash
python3 scripts/import_multidoc2dial.py
```

Состав:

- `dmv_documents.jsonl` — документы базы знаний;
- `dmv_questions_train.jsonl` — обучающие вопросы и эталонные ссылки;
- `dmv_questions_validation.jsonl` — вопросы для настройки;
- `dmv_questions_test.jsonl` — вопросы для итоговой проверки;
- `manifest.json` — источник и количество записей.

У каждого вопроса сохранены эталонный ответ, правильные документы и конкретные
фрагменты. Это позволит позже измерять качество поиска, а не оценивать ответы
только «на глаз».
