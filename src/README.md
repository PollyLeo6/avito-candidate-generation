# Где находится код решения

В `src/` три пакета и 32 Python-файла, включая три `__init__.py`.
Текущий ответ собирает `avito_improved`. Он использует обработку текста и проверку
CSV из `avito_retrieval`, а поиск кандидатов и первые 31 признак из `avito_ranker`.

## С чего начать чтение

1. [check.ipynb](../notebooks/check.ipynb) показывает пересчёт ответа с готовой моделью.
   [solution.ipynb](../notebooks/solution.ipynb) содержит полный порядок работы;
   обучение выполняется при `rebuild = True`.
2. Оба ноутбука вызывают `run_stage(stage, config_path)` из
   [avito_improved/run.py](avito_improved/run.py). Эта функция запускает отдельный
   процесс и сохраняет его вывод в лог. `main()` выбирает обработчик этапа.
3. Для основного алгоритма читайте
   [ExtendedSearch.build](avito_improved/retrieval.py), затем
   [create_dataset](avito_improved/dataset.py),
   [train_and_select](avito_improved/model.py) и
   [predict](avito_improved/evaluation.py).

Порядок запуска и промежуточные файлы описаны в [схеме расчёта](../docs/pipeline.md).
Ниже приведена карта модулей, чтобы было проще перейти от ячейки ноутбука к реализации.

## Текущее решение: `avito_improved`

| Файл | Что искать в коде |
| --- | --- |
| [run.py](avito_improved/run.py) | `run_stage`, `main`: запуск этапов, ограничение потоков и логи. Построение BM25 передаётся общему коду `avito_ranker`. |
| [protocol.py](avito_improved/protocol.py) | `prepare_experiment` готовит данные, историю и центры локаций. `reserve_holdout` откладывает новые запросы; `freeze` и `check_frozen` фиксируют и проверяют эксперимент. |
| [history.py](avito_improved/history.py) | `build_history` сохраняет пары только из обучающих групп. `QueryHistory.search` ищет похожие запросы и исключает собственную группу при подготовке train. |
| [geo.py](avito_improved/geo.py) | `build_centers` считает медианные координаты локаций; `distance_km` и `nearby_candidates` добавляют текстово подходящие объявления в радиусе 50 км. |
| [retrieval.py](avito_improved/retrieval.py) | `ExtendedSearch.build` объединяет базовый поиск, историю, фильтры и географию. Возвращает пул объявлений, матрицу 38 признаков, их имена и исходный пул. |
| [dataset.py](avito_improved/dataset.py) | `create_dataset` вызывает поиск для каждого запроса, добавляет метки и пишет Parquet. На train сохраняет найденные positives, сложные и случайные отрицательные примеры. |
| [model.py](avito_improved/model.py) | `train_and_select` обучает CatBoostRanker с PairLogit и QuerySoftMax; функция `recall` внутри него сравнивает число деревьев по Recall@50 на dev. |
| [selection.py](avito_improved/selection.py) | `select_blend` выбирает вариант пула и долю новой выдачи по dev. `top_indices` объединяет две выдачи и пропускает повторные объявления. |
| [evaluation.py](avito_improved/evaluation.py) | `score` применяет модель; `evaluate` считает Recall и bootstrap-интервал на новом holdout; `predict` переводит номера документов в `item_id` и проверяет CSV. |
| [audit.py](avito_improved/audit.py) | `audit_training` проверяет разделение групп и то, что метки обучающих кандидатов совпадают с положительными парами. |
| [artifacts.py](avito_improved/artifacts.py) | `restore` загружает сохранённые артефакты для быстрой проверки; `export` переносит результаты расчёта из рабочей папки в `results/improved/`. |

[__init__.py](avito_improved/__init__.py) обозначает пакет и не запускает расчёт.

## Исходное ранжирование и общие компоненты: `avito_ranker`

Этот пакет сохранён для сравнения с исходной моделью. Его подготовка данных,
BM25-поиск, признаки и настройки также используются текущим решением.

| Файл | Что искать в коде |
| --- | --- |
| [config.py](avito_ranker/config.py) | `load_config` читает TOML и разрешает относительные пути от каталога файла настроек. |
| [prepare.py](avito_ranker/prepare.py) | `strict_group`, `assign_fold`, `sample_contexts` задают разбиение. `prepare` создаёт запросы, разметку и два корпуса: локальный и benchmark. |
| [workflow.py](avito_ranker/workflow.py) | `run_stage` запускает процессы исходного решения. При построении индексов последовательно обрабатывает заголовки, параметры и описания. |
| [stage.py](avito_ranker/stage.py) | `main` выбирает конкретный обработчик исходного решения, в том числе построение индекса одного поля. |
| [candidates.py](avito_ranker/candidates.py) | `CandidateSearch.retrieve` собирает BM25-кандидатов по всему корпусу и отдельно по локации запроса. |
| [features.py](avito_ranker/features.py) | `FeatureBuilder.build` считает первые 31 признак. `rank_values` сопоставляет позиции по `doc_id`, `token_coverage` измеряет покрытие слов запроса. |
| [dataset.py](avito_ranker/dataset.py) | `create_dataset` и `sample_training` создают таблицы и обучающую выборку исходной версии. У текущей версии свой `avito_improved/dataset.py`. |
| [model.py](avito_ranker/model.py) | `train_model`, `select_model`, `evaluate_model` относятся к исходной модели. `per_query_recall` сохраняет в знаменателе positives, не найденные поиском. |
| [predict.py](avito_ranker/predict.py) | `predict` сохраняет ответ исходной версии. Текущий CSV создаёт `avito_improved/evaluation.py`. |
| [freeze.py](avito_ranker/freeze.py) | `fingerprint`, `freeze`, `check_frozen` защищают исходный эксперимент. `artifact_hash` сравнивает содержание JSON независимо от форматирования и используется также текущим решением. |
| [verify.py](avito_ranker/verify.py) | `audit_training` проверяет исходное обучение; `verify` независимо пересчитывает Recall по сохранённым ID и проверяет формат ответа. |

[__init__.py](avito_ranker/__init__.py) обозначает пакет и не запускает расчёт.

## Обработка данных и BM25: `avito_retrieval`

| Файл | Что искать в коде |
| --- | --- |
| [common.py](avito_retrieval/common.py) | `normalize`, `tokenize`, `stem` обрабатывают текст; `stable_topk` разрешает равные оценки; `rrf_scores` объединяет выдачи; `recall_at` считает метрику одного запроса. |
| [data_check.py](avito_retrieval/data_check.py) | `inspect_data` проверяет три Parquet, число строк и обязательные поля. `check_ids` проверяет формат и уникальность ID. |
| [split.py](avito_retrieval/split.py) | `make_contexts` создаёт контексты поиска. Остальные функции готовят и проверяют разбиение первого BM25-эксперимента; текущее строгое разбиение находится в `avito_ranker/prepare.py`. |
| [corpus.py](avito_retrieval/corpus.py) | `write_texts` переносит признаки объявлений порциями. `choose_items` и `microcategory_distance` относятся к корпусу первого BM25-эксперимента; текущее решение берёт все доступные уникальные объявления для локальной оценки. |
| [build_index.py](avito_retrieval/build_index.py) | `build` токенизирует одно текстовое поле, строит BM25 и сохраняет параметры с хешами корпуса. |
| [retrieval.py](avito_retrieval/retrieval.py) | `field_scores` считает BM25. `load_indices`, `candidates`, `combine` содержат варианты первого BM25-эксперимента и объединение RRF. |
| [submission.py](avito_retrieval/submission.py) | `validate_answer` проверяет колонки CSV, полный состав запросов, формат и допустимость `item_id`, повторы и ограничение топ-50. |

[__init__.py](avito_retrieval/__init__.py) обозначает пакет и не запускает расчёт.

## Как связаны переменные

| Переменная | Что означает |
| --- | --- |
| `query` | Один запрос с полями `search_*`: текстом, локацией, фильтрами, категорией и флагом доставки. |
| `context_id` | В локальной выборке это хеш всех пяти признаков поиска. Для benchmark в эту колонку временно переименовывается исходный `query_id`, без изменения его значения. |
| `group`, `strict_group` | `group` сохраняет последовательность нормализованных слов. `strict_group` содержит отсортированный набор стемов и определяет текущее разбиение и исключение собственной группы из истории. |
| `query_number` | Номер запроса внутри конкретной таблицы запросов. Связывает строки кандидатов с итоговой статистикой и задаёт группы при обучении CatBoost. |
| `item_id`, `doc_id` | `item_id` берётся из исходных данных. `doc_id` обозначает позицию объявления в манифесте текущего корпуса; между локальным корпусом и benchmark номера могут отличаться. |
| `pool`, `matrix`, `names` | `pool` содержит `doc_id` кандидатов; строки `matrix` идут в том же порядке; `names` задаёт порядок колонок признаков. |
| `qrels`, `label`, `positives` | `qrels` хранит известные положительные пары. `label` равен 1 для найденной положительной пары и 0 для остальных кандидатов. `positives` хранит полное число известных положительных объявлений, включая не попавшие в пул. |
| `old_score`, `original_candidate` | Оценка исходной модели и признак принадлежности её пулу. Нужны для выбора сложных отрицательных примеров и объединения выдач. |

В модель передаются только колонки из `feature_names.json`. Метки и служебные ID
не входят в эти 38 признаков: `label` передаётся отдельно как целевая переменная,
а `query_number` как идентификатор группы. Нулевая метка означает отсутствие
наблюдаемой положительной пары, а не доказанную нерелевантность объявления.

[Символьный поиск](../experiments/char_retrieval.py) вынесен в отдельный эксперимент.
Он сравнивает покрытие кандидатов и не вызывается при сборке текущего `answer.csv`.
Проверки компонентов перечислены в [tests/README.md](../tests/README.md).
