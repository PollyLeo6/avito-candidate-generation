# Модели и результаты

Итоговое решение хранит модель и отчёты в [improved/](improved/).
Файлы в корне этой папки `results/` сохраняют исходную версию и историю её проверки.
Часть исходной версии используется итоговым решением, поэтому обе модели нужны
при воспроизведении.

## Какой ответ и какие модели использовать

| Файл | Назначение |
| --- | --- |
| [../answer.csv](../answer.csv) | Итоговый ответ и эталон для `check.ipynb` |
| [improved/answer.csv](improved/answer.csv) | Копия итогового ответа рядом с выбранной моделью |
| [improved/ranker.cbm](improved/ranker.cbm) | Итоговый CatBoostRanker: 38 признаков, 80 деревьев |
| [ranker.cbm](ranker.cbm) | Фиксированная исходная модель с 31 признаком; отбирает сложные отрицательные примеры и дополняет финальную выдачу |
| [answer.csv](answer.csv) | Ответ первой версии, оставленный для истории эксперимента |
| [ranker_full.cbm](ranker_full.cbm) | Полная модель первой версии до выбора числа деревьев; основной проверке не требуется |

`check.ipynb` загружает обе используемые модели, заново выполняет поиск и считает
признаки. Он не читает готовые ответы как предсказания: корневой CSV нужен только
для сравнения в последней ячейке.

## Где проверить текущий результат

| Файл в `improved/` | Что он показывает |
| --- | --- |
| [selected.json](improved/selected.json) | Выбор на dev: PairLogit, 80 деревьев, поиск рядом и первые 40 результатов новой модели |
| [selection_curve.json](improved/selection_curve.json) | Качество двух функций потерь при разном числе деревьев |
| [feature_names.json](improved/feature_names.json) | Все 38 признаков в порядке подачи в модель |
| [fresh_metrics.json](improved/fresh_metrics.json) | Recall@50 на отдельном holdout из 1 000 запросов и сравнение с исходным методом |
| [fresh_per_query.parquet](improved/fresh_per_query.parquet) | Число положительных, покрытие кандидатов и Recall каждого проверочного запроса |
| [fresh_reservation.json](improved/fresh_reservation.json) | Размер и контрольная сумма заранее отложенного holdout |
| [history_audit.json](improved/history_audit.json) | Размер истории и правила исключения проверочных групп |
| [leakage_audit.json](improved/leakage_audit.json) | Сверка обучающих меток и отсутствие проверочных групп в истории |
| [submission_check.json](improved/submission_check.json) | Формат CSV, число строк и объявлений, контрольные суммы ответа и модели |
| [frozen.json](improved/frozen.json) | Зафиксированные код, данные, история и модель для проверки воспроизведения |

Файлы `train_candidates.json`, `dev_candidates.json`, `dev_expanded_candidates.json`,
`fresh_candidates.json` и `benchmark_candidates.json` описывают поиск на каждом
этапе: размер пула, время и, где есть разметка, покрытие положительных объявлений.
`dev_expanded` содержит те же запросы, что и `dev`, с дополнительным поиском
в радиусе 50 км. Это вариант поиска для сравнения, а не новая выборка запросов.

## Отчёты о воспроизводимости

- [reproduction_check.json](improved/reproduction_check.json) описывает выполненные
  проверки текущего файла. Подробности и ограничения перечислены в
  [документации по воспроизводимости](../docs/reproduction.md).
- [reproduction_before_crlf.json](improved/reproduction_before_crlf.json) сохраняет
  полный запуск поиска и предсказаний в чистом окружении до уточнения переводов строк.
- [submitted_file_check.json](improved/submitted_file_check.json) подтверждает
  побайтовое совпадение повторного предсказания с отправленным ответом.
- [feature_equivalence.json](improved/feature_equivalence.json) проверяет совпадение
  признаков вынесенных модулей с исходным расчётом.
- [documentation_check.json](improved/documentation_check.json) подтверждает,
  что добавление комментариев и докстрингов не изменило вычисления или CSV.

## Где сохраняется новый запуск

Быстрый ноутбук пишет новый ответ и отчёт в `work/quality/`.
Этап `export` полного ноутбука обновляет выбранную модель и основные отчёты
в `results/improved/`, после чего копирует ответ в корень репозитория.
Логи запуска находятся в `results/improved/logs/`.

Отчёты проверки соответствуют тому запуску, в котором были получены.
При изменении алгоритма старые отчёты о воспроизводимости не подтверждают
изменённую версию автоматически.

## История разработки

Корневые `dev_metrics.json`, `holdout_metrics.json`, `training.json`,
`full_reproduction.json` и `logs/` относятся к исходному решению.
Его метод описан в [baseline_approach.md](../docs/baseline_approach.md).

В [experiments/](experiments/) сохранены результаты символьного поиска
и проверки связей локаций. Они разобраны в [анализе ошибок](../docs/error_analysis.md)
и не участвуют в формировании итогового `answer.csv`.
