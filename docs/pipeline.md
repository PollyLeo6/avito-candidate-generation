# Карта расчёта: от Parquet до answer.csv

Этот документ описывает версию, которая формирует приложенный `answer.csv`:
38 признаков, CatBoostRanker с PairLogit и 80 деревьями, 40 результатов новой
модели с дополнением исходной выдачей. Параметры записаны в
[selected.json](../results/improved/selected.json).
Символьный поиск из `experiments/` в этот расчёт не входит.

Установка и команды запуска приведены в [README](../README.md).
Здесь разобраны данные, промежуточные файлы и последовательность вычислений.

## 1. Входные данные и обозначения

| Файл в `data/` | Содержимое | Как используется |
| --- | --- | --- |
| `train.parquet` | 497 673 пары «поисковый контекст - выбранное объявление» | Разбиение запросов, положительные пары для обучения и оценки, история запросов, признаки объявлений локального корпуса |
| `benchmark_queries.parquet` | 2 452 запроса без меток | Запросы, для которых формируется ответ |
| `benchmark_items.parquet` | 189 212 объявлений | Единственный допустимый корпус при формировании ответа |

Одна строка train показывает известный выбор пользователя. Несколько строк
могут относиться к одному поисковому контексту. Повторы одной пары не увеличивают
число положительных объявлений при расчёте Recall.

| Обозначение | Что означает |
| --- | --- |
| `query_id` | Исходный идентификатор запроса benchmark. Сохраняется в CSV без изменения регистра |
| `context_id` | Внутренний идентификатор сочетания пяти поисковых полей. При расчёте benchmark колонка `query_id` временно переименовывается в `context_id` |
| `group` | Текст запроса после нормализации регистра, пробелов, пунктуации и буквы ё. Порядок слов сохраняется |
| `strict_group` | Отсортированный набор уникальных токенов после русского Snowball-стеммера. Используется для разбиения и исключения собственной группы из истории |
| `query_number` | Номер строки в конкретной таблице выбранных запросов. Нужен для группировки кандидатов внутри расчёта |
| `item_id` | Исходный идентификатор объявления. Именно он попадает в ответ |
| `doc_id` | Номер объявления внутри конкретного корпуса и его индексов. Не переносится между корпусами и не попадает в CSV |
| `qrels` | Таблица известных положительных пар `context_id, item_id` |
| `pool` | Набор кандидатов до выбора итоговых 50 объявлений |

Контекст включает `search_query`, `search_location_id`,
`search_is_delivery_search`, `search_infm_params_text` и `search_category`.
Один текст в разных локациях даёт разные контексты, но одну текстовую группу.
Поэтому разбиение выполняется по `strict_group`, а положительные пары собираются
соединением по всем пяти полям контекста.

Код: [common.py](../src/avito_retrieval/common.py),
[split.py](../src/avito_retrieval/split.py),
[prepare.py](../src/avito_ranker/prepare.py).

## 2. Какие выборки и корпуса создаёт prepare

Сначала `prepare` проверяет исходные таблицы, создаёт контексты и назначает
строгим группам train, dev или holdout по стабильному хешу. Группы, которые
просматривались в ранней оценке, закрепляются за dev. В каждой части выбирается
по одному контексту на группу; выбор представителя и самих групп тоже задаётся
стабильными хешами, без использования меток для отбора.

| Набор | Запросов | Назначение и путь |
| --- | ---: | --- |
| Train для модели | 8 000 | `work/validation/train_queries.parquet` |
| Dev | 2 000 | `work/validation/dev_queries.parquet`; выбор модели и поиска |
| Прежний holdout | 2 000 | `work/validation/holdout_queries.parquet`; исключается при резервировании новой оценки |
| Новый holdout, в коде `fresh` | 1 000 | `work/quality/reserved/fresh_queries.parquet`; итоговая оценка после выбора по dev |

У каждого набора есть соответствующий `*_qrels.parquet`. Новый holdout
резервируется уже на этапе `prepare`, до обучения и подбора. Он выбирается
из оставшихся holdout-групп, которые не пересекаются с прежними 2 000 запросами,
train и dev. Число `holdout_queries = 2000` в конфиге относится к прежнему
набору; дополнительные 1 000 задаются в `reserve_holdout()`.

Для поиска создаются два корпуса:

- **Локальный корпус: 515 895 объявлений.** Все уникальные `item_id` из train
  и benchmark. При повторе ID используются признаки из benchmark. Его файлы
  находятся в `work/validation/`, индексы - в `work/validation_indices/`.
- **Корпус ответа: 189 212 объявлений.** Только `benchmark_items.parquet`.
  Его файлы находятся в `work/benchmark/`, индексы - в `work/benchmark_indices/`.

Корпус не подбирается под положительные пары оцениваемого запроса. Метки dev
и holdout не используются в обучающей истории. Координаты и тексты объявлений
локального корпуса используются как признаки без этих меток.

Каждый `corpus_manifest.parquet` содержит соответствие `doc_id -> item_id`
и категории с локациями. Порядок `doc_id` задаётся сортировкой по `item_id`.
Файл `corpus.parquet` содержит тексты и остальные признаки. Перед загрузкой
BM25 проверяется, что индекс построен для нужного manifest.

Код: [prepare.py](../src/avito_ranker/prepare.py),
[protocol.py](../src/avito_improved/protocol.py),
[corpus.py](../src/avito_retrieval/corpus.py).

## 3. История запросов и география

История использует **все группы train**, а не только 8 000 запросов для модели.
В сохранённом расчёте это 50 737 строгих групп и 291 192 уникальные пары
«группа - объявление». Размеры приведены в
[history_audit.json](../results/improved/history_audit.json).

По токенам групп строится разреженная TF-IDF-матрица. Для нового запроса
выбираются до 40 ближайших групп с косинусным сходством не ниже 0,35.
Вес группы равен четвёртой степени сходства. Она даёт:

- прямые кандидаты из объявлений, которые выбирали по похожему запросу;
- признаки сходства с историей;
- веса подкатегорий для дополнительного поиска.

При создании train-признаков из истории исключается **вся собственная строгая
группа запроса**. Так положительная пара не может напрямую подсказать свою
метку. Для dev, fresh и benchmark история остаётся той же обучающей историей;
дополнения из проверочных меток или переобучения на всём train.parquet нет.

Центр локации считается как медиана координат её объявлений в локальном корпусе.
Эти же центры используются для benchmark. Это приблизительный центр локации,
а не координаты пользователя. Если центр отсутствует, расстояние получает -1,
а канал поиска в радиусе 50 км не добавляет кандидатов.

Код: [history.py](../src/avito_improved/history.py),
[geo.py](../src/avito_improved/geo.py).

## 4. Как собираются кандидаты

Все каналы работают до присоединения меток. Объявления объединяются по `doc_id`
без повторов; пропущенные положительные объявления принудительно не добавляются.

| Канал | Что добавляет |
| --- | --- |
| BM25 по заголовку, описанию и параметрам | До 300 результатов каждого поля во всём корпусе и до 150 в точной локации запроса |
| Исходный RRF | Объединяет позиции трёх глобальных BM25-выдач; его топ-50 также входит в исходный пул |
| BM25 по поисковым фильтрам | До 300 результатов во всём корпусе и до 500 в точной локации запроса; сравнивает фильтры с параметрами объявления |
| История похожих запросов | До 200 объявлений с наибольшим весом во всём корпусе и до 200 в локации; нулевые веса отбрасываются |
| Подкатегории из истории | До 200 местных кандидатов по сочетанию веса подкатегории и нормированных текстовых оценок |
| Дополнительный поиск по расстоянию | До 300 результатов каждого текстового поля среди объявлений в пределах 50 км от центра локации |

Локальные BM25-выдачи расширяют пул и дают признаки позиций, но не входят
в исходную сумму RRF. У разных каналов возможны пересечения, поэтому сумма
лимитов не равна размеру итогового пула.

### Когда включён радиус 50 км

| Часть | Дополнительные кандидаты в радиусе 50 км |
| --- | --- |
| `train` | Нет |
| `dev` | Нет; на этом пуле выбираются функция потерь и число деревьев |
| `dev_expanded` | Да; это те же 2 000 dev-запросов с расширенным пулом |
| `fresh`, `benchmark` | Согласно `nearby_candidates` из `selected.json`; у приложенной модели значение `true` |

Признак расстояния рассчитывается во всех частях. Отсутствие дополнительных
кандидатов в train не означает, что географический признак отсутствует.
Расширение поиска выбирается на dev после обучения модели.

Код: [candidates.py](../src/avito_ranker/candidates.py),
[retrieval.py](../src/avito_improved/retrieval.py),
[geo.py](../src/avito_improved/geo.py).

## 5. Какие 38 признаков видит модель

Первые 31 признак совместимы с исходной моделью; следующие семь добавлены
в `ExtendedSearch`. Порядок зафиксирован в
[feature_names.json](../results/improved/feature_names.json).

| Число | Точные имена | Содержание |
| ---: | --- | --- |
| 4 | `title_score`, `title_relative`, `title_rr`, `title_local_rr` | BM25 заголовка, доля от максимальной оценки по корпусу, обратная позиция глобального и местного поиска |
| 4 | `description_score`, `description_relative`, `description_rr`, `description_local_rr` | Те же оценки для описания |
| 4 | `params_score`, `params_relative`, `params_rr`, `params_local_rr` | Те же оценки для параметров |
| 3 | `filter_score`, `filter_relative`, `rrf` | BM25 фильтров, его нормированная оценка и исходная сумма RRF |
| 3 | `same_location`, `same_category`, `delivery` | Совпадение локации и категории; флаг поиска с доставкой |
| 2 | `query_words`, `filter_words` | Число уникальных токенов запроса и фильтров |
| 3 | `title_length`, `title_coverage`, `title_jaccard` | Число уникальных токенов заголовка, покрытие токенов запроса и Jaccard |
| 3 | `params_length`, `params_coverage`, `filter_coverage` | Длина параметров в символах, покрытие токенов запроса и фильтров |
| 5 | `item_price`, `item_rating`, `item_rating_reviews_count`, `item_is_phone_hidden`, `item_is_message_forbidden` | Свойства объявления; цена и число отзывов преобразуются через `log1p` после обнуления отрицательных значений |
| 2 | `history_score`, `history_microcat` | Вес объявления и его подкатегории из истории похожих запросов |
| 2 | `filter_global_rr`, `filter_local_rr` | Обратные позиции в дополнительном поиске по фильтрам |
| 3 | `description_coverage`, `query_in_title`, `log_distance_km` | Покрытие запроса описанием, полное вхождение нормализованного запроса в заголовок, `log1p` расстояния или -1 при неизвестной географии |

Обратная позиция равна `1 / position`, где первое место имеет `position = 1`.
У объявления вне соответствующей выдачи значение равно нулю.
`query_id`, `context_id`, `query_number`, `item_id`, `doc_id` и `label`
не входят в список признаков.

Код: [features.py](../src/avito_ranker/features.py),
[retrieval.py](../src/avito_improved/retrieval.py).

## 6. Обучение и выбор итоговой выдачи

Для каждого из 8 000 train-запросов сначала рассчитывается полный пул.
Если в нём нет известной положительной пары, запрос пропускается при обучении.
В сохранённом расчёте осталось 7 385 запросов и 1 185 743 пары кандидатов.
Для каждого запроса берутся:

1. Все найденные положительные объявления.
2. До 80 отрицательных с наибольшей оценкой исходной модели.
3. До 80 случайных отрицательных из оставшихся.

Отрицательным считается кандидат без известной положительной пары этого
контекста. Положительные объявления вне пула остаются в знаменателе метрик.

### Почему в репозитории две модели

| Файл | Роль |
| --- | --- |
| [`results/ranker.cbm`](../results/ranker.cbm) | Фиксированная исходная модель на 31 признаке: выбирает сложные отрицательные пары, служит базой сравнения и дополняет итоговый топ-50 |
| [`results/improved/ranker.cbm`](../results/improved/ranker.cbm) | Выбранная модель на 38 признаках; восстанавливается при быстрой проверке |

`solution.ipynb` с `rebuild = True` заново обучает модель на 38 признаках.
Исходная модель `results/ranker.cbm` остаётся готовой зависимостью этого расчёта
и в данном ноутбуке не переобучается. Её контрольная сумма входит в снимок
эксперимента.

Обучаются PairLogit с `max_pairs=160` и QuerySoftMax: по 400 деревьев, глубина 6,
`learning_rate=0.05`, `l2_leaf_reg=12`, два потока. На обычном dev через каждые
20 деревьев считается Recall@50. Лучший вариант - PairLogit с 80 деревьями.

Затем `select` сравнивает обычный и расширенный dev-пул и значения `new_head`
из списка `0, 30, 40, 45, 50`. Для приложенной модели выбраны расширенный пул
и `new_head=40`: первые 40 новых результатов дополняются исходной выдачей
без повторов до 50. Если исходной выдачи не хватает, используются оставшиеся
результаты новой модели. Исходная выдача строится только по её исходному пулу.

После этого выполняется `freeze`. Новый holdout не участвует в выборе потерь,
числа деревьев, географии или долей моделей. Финального переобучения на dev
или holdout перед benchmark нет.

Код: [dataset.py](../src/avito_improved/dataset.py),
[model.py](../src/avito_improved/model.py),
[selection.py](../src/avito_improved/selection.py).

## 7. Карта этапов и файлов

Все имена этапов обрабатывает [run.py](../src/avito_improved/run.py).
Ниже `quality/` означает `work/quality/`, а `saved/` - `results/improved/`
при стандартном [конфиге](../configs/improved.toml).
Пути из TOML разрешаются относительно папки `configs/`, а не текущей папки терминала.

| Этап | Вход | Действие | Основной выход | Код |
| --- | --- | --- | --- | --- |
| `prepare` | Три исходных Parquet | Проверка данных, разбиение, корпуса, история и резервирование fresh | `work/validation/`, `work/benchmark/`, `quality/history/`, `quality/reserved/`, `quality/fresh_reservation.json` | [protocol.py](../src/avito_improved/protocol.py), [prepare.py](../src/avito_ranker/prepare.py), [history.py](../src/avito_improved/history.py), [geo.py](../src/avito_improved/geo.py) |
| `restore` | Готовая модель и отчёты в `saved/`; результат `prepare` | Копирование модели и параметров, проверка всех отпечатков | `quality/ranker.cbm`, `selected.json`, `feature_names.json`, `leakage_audit.json`, `frozen.json` | [artifacts.py](../src/avito_improved/artifacts.py) |
| `validation_indices` | `work/validation/` | Три BM25-индекса локального корпуса, по одному за процесс | `work/validation_indices/title/`, `params/`, `description/` | [workflow.py](../src/avito_ranker/workflow.py), [build_index.py](../src/avito_retrieval/build_index.py) |
| `train_features` | Train-запросы, qrels, индексы, история, исходная модель | Поиск без 50 км, исключение собственной группы, выбор отрицательных | `quality/train.parquet`, `train_queries.parquet`, `train_candidates.json`, `feature_names.json` | [dataset.py](../src/avito_improved/dataset.py), [retrieval.py](../src/avito_improved/retrieval.py) |
| `audit` | Обучающая таблица, исходные qrels, группы и история | Независимая проверка меток и состава истории | `quality/leakage_audit.json` | [audit.py](../src/avito_improved/audit.py) |
| `dev_features` | 2 000 dev-запросов, локальный корпус и история | Все кандидаты и признаки, без выборки отрицательных и без 50 км | `quality/dev.parquet`, `dev_queries.parquet`, `dev_candidates.json` | [dataset.py](../src/avito_improved/dataset.py) |
| `train` | `quality/train.parquet`, `dev.parquet` и их служебные таблицы | Обучение двух моделей и выбор числа деревьев на обычном dev | `quality/pairlogit_full.cbm`, `softmax_full.cbm`, `ranker.cbm`, `selected.json`, `selection_curve.json`, `dev_scores.npy` | [model.py](../src/avito_improved/model.py) |
| `dev_expanded_features` | Те же dev-запросы | Повторный поиск с дополнительными кандидатами в радиусе 50 км | `quality/dev_expanded.parquet`, `dev_expanded_queries.parquet`, `dev_expanded_candidates.json` | [dataset.py](../src/avito_improved/dataset.py), [geo.py](../src/avito_improved/geo.py) |
| `select` | Выбранная модель, обычный и расширенный dev | Выбор расширения поиска и состава топ-50 | Обновлённый `quality/selected.json` | [selection.py](../src/avito_improved/selection.py) |
| `freeze` | Код, исходные данные, модель, история, выбранные параметры и отчёты | Фиксация контрольных сумм до итоговой оценки | `quality/frozen.json` | [protocol.py](../src/avito_improved/protocol.py) |
| `fresh_features` | Зарезервированные 1 000 запросов, локальный корпус, выбранные параметры | Проверка снимка, поиск и признаки с выбранной географией | `quality/fresh.parquet`, `fresh_queries.parquet`, `fresh_candidates.json` | [run.py](../src/avito_improved/run.py), [dataset.py](../src/avito_improved/dataset.py) |
| `evaluate` | Fresh-признаки и выбранная модель | Recall@50 трёх методов, Recall пула, парный bootstrap прироста | `quality/fresh_metrics.json`, `fresh_per_query.parquet` | [evaluation.py](../src/avito_improved/evaluation.py) |
| `benchmark_indices` | `work/benchmark/` | Три BM25-индекса корпуса ответа | `work/benchmark_indices/title/`, `params/`, `description/` | [workflow.py](../src/avito_ranker/workflow.py), [build_index.py](../src/avito_retrieval/build_index.py) |
| `benchmark_features` | Benchmark-запросы, его индексы, история, выбранные параметры | Поиск и признаки без разметки | `quality/benchmark.parquet`, `benchmark_queries.parquet`, `benchmark_candidates.json` | [dataset.py](../src/avito_improved/dataset.py) |
| `predict` | Benchmark-признаки, выбранная модель и сохранённые оценки исходной | Выбор топ-50, возврат исходных ID, проверка CSV | `quality/answer.csv`, `submission_check.json` | [evaluation.py](../src/avito_improved/evaluation.py), [submission.py](../src/avito_retrieval/submission.py) |
| `export` | Рассчитанные артефакты в `quality/` | Копирование модели, CSV и итоговых отчётов | Файлы в `saved/`; затем ноутбук копирует ответ в корень | [artifacts.py](../src/avito_improved/artifacts.py) |

Таблица `quality/<fold>_queries.parquet` содержит сводку кандидатов:
`query_number`, `context_id`, `positives`, `pool_hits`, `pool_size`.
Это не копия исходной таблицы с текстами запросов. Тексты и фильтры остаются
в `work/validation/*_queries.parquet`, а для fresh - в `quality/reserved/`.

`quality/<fold>.parquet` содержит одну строку на кандидата, 38 признаков,
`query_number`, `doc_id`, `label`, `old_score` и `original_candidate`.
В benchmark `label` технически равен нулю, поскольку меток нет; эта колонка
не является признаком модели и не используется для оценки benchmark.

## 8. Что выполняют два ноутбука

Оба ноутбука вызывают `run_stage(stage, config_path)`. Каждый тяжёлый этап
работает в отдельном процессе, поэтому после него освобождается память.
Полные логи сохраняются в `results/improved/logs/`; в ячейку выводится конец лога.
При ошибке этапа следующий расчёт не продолжается.

### Быстрая проверка: check.ipynb, 6 кодовых ячеек

| Ячейка | Действие |
| ---: | --- |
| 1 | Читает конфиг, ограничивает потоки, запускает 27 тестов |
| 2 | `prepare`, затем `restore` готовой модели с проверкой отпечатков |
| 3 | `benchmark_indices` |
| 4 | `benchmark_features` |
| 5 | `predict` |
| 6 | Проверка формата и побайтовое сравнение с корневым `answer.csv`; запись `work/quality/check_report.json` |

Обучения, подбора по dev и повторной оценки fresh здесь нет. Поиск и признаки
benchmark пересчитываются. Train.parquet нужен для восстановления обучающей
истории и общих центров локаций. Успешное завершение требует
`status: passed` и `answer_identical: true`.

### Полный расчёт: solution.ipynb, 13 кодовых ячеек

| Ячейка | При `rebuild = True` |
| ---: | --- |
| 1 | Конфиг и флаг режима |
| 2 | 27 тестов |
| 3 | `prepare` |
| 4 | `validation_indices` |
| 5 | `train_features`, `audit` |
| 6 | `dev_features` |
| 7 | `train` |
| 8 | `dev_expanded_features`, `select`; вывод выбранных параметров |
| 9 | `freeze` |
| 10 | `fresh_features`, `evaluate`; вывод метрик |
| 11 | `benchmark_indices`, `benchmark_features` |
| 12 | `predict` |
| 13 | `export`, копирование рассчитанного CSV в корень, вывод проверки ответа |

По умолчанию стоит `rebuild = False`. В этом режиме запускаются тесты
и показываются сохранённые параметры, метрики и проверка CSV. Он не выполняет
поиск, обучение или пересчёт ответа. Для полного расчёта нужно явно поставить
`True`. Результаты полного запуска записываются в `work/` и `results/improved/`.

Ноутбуки: [check.ipynb](../notebooks/check.ipynb),
[solution.ipynb](../notebooks/solution.ipynb).

## 9. Где искать подтверждения качества

Для каждого запроса считаем долю найденных положительных объявлений среди
всех его известных положительных объявлений, затем среднее по запросам.
Пропуск на этапе поиска уменьшает итоговый Recall, даже если ранжирование
оставшихся кандидатов было правильным.

| Вопрос | Отчёт |
| --- | --- |
| Сколько запросов и пар использовано для обучения? | [leakage_audit.json](../results/improved/leakage_audit.json) |
| Сколько групп и пар вошло в историю? | [history_audit.json](../results/improved/history_audit.json) |
| Как отложен отдельный holdout? | [fresh_reservation.json](../results/improved/fresh_reservation.json) |
| Какая модель и смесь выбраны на dev? | [selected.json](../results/improved/selected.json), [selection_curve.json](../results/improved/selection_curve.json) |
| Каков Recall пула до ранжирования? | [dev_candidates.json](../results/improved/dev_candidates.json), [dev_expanded_candidates.json](../results/improved/dev_expanded_candidates.json), [fresh_candidates.json](../results/improved/fresh_candidates.json) |
| Каково итоговое качество на fresh? | [fresh_metrics.json](../results/improved/fresh_metrics.json) и [fresh_per_query.parquet](../results/improved/fresh_per_query.parquet) |
| Какие ошибки разбирались? | [error_analysis.md](error_analysis.md) |
| Соответствует ли CSV требованиям? | [submission_check.json](../results/improved/submission_check.json) |
| Что зафиксировано до оценки? | [frozen.json](../results/improved/frozen.json) |
| Какие запуски воспроизведены? | [reproduction.md](reproduction.md) |

На новом holdout Recall пула равен 0,935, а итоговый Recall@50 - 0,784833.
Исходная модель на тех же запросах получила 0,735833. В
`fresh_per_query.parquet` находятся обе оценки каждого запроса, поэтому
разницу можно проверить без усреднения по числу кандидатов.

Снимок содержит отпечатки Python-кода, трёх входных файлов, истории, выбранных
параметров и моделей. Для выбранной модели сравнивается её функция предсказания
без служебного времени обучения и GUID. `restore` проверяет совпадение с
сохранённым снимком; `freeze` не перезаписывает уже существующий несовместимый
снимок. Это позволяет заметить изменение расчёта перед оценкой.

## 10. Как проследить один запрос

После полного расчёта можно выбрать любой `query_number` из таблицы dev.
Ниже используется первая строка; номер можно заменить. Это проверка по
локальным известным меткам, без подбора ответа для конкретного benchmark ID.
Код выполняется из корня репозитория в установленном окружении.

```python
import json
from pathlib import Path

import polars as pl
from catboost import CatBoostRanker
from avito_improved.selection import top_indices

project = Path.cwd()
work = project / "work"
quality = work / "quality"
query_number = 0

# Выбираем контекст и рассчитанные для него признаки кандидатов.
queries = pl.read_parquet(work / "validation/dev_queries.parquet")
query = queries.row(query_number, named=True)
frame = (pl.scan_parquet(quality / "dev_expanded.parquet")
         .filter(pl.col("query_number") == query_number).collect())
names = json.loads((quality / "feature_names.json").read_text("utf-8"))
settings = json.loads((quality / "selected.json").read_text("utf-8"))
# Повторяем оценку выбранной моделью и объединение двух выдач.
model = CatBoostRanker().load_model(str(quality / "ranker.cbm"))
scores = model.predict(frame.select(names).to_numpy(), thread_count=2)
chosen = top_indices(scores, frame["old_score"].to_numpy(),
                     frame["original_candidate"].to_numpy(), settings["new_head"])

# Возвращаем исходные item_id и берём все известные положительные пары.
manifest = pl.read_parquet(work / "validation/corpus_manifest.parquet")
gold = (pl.read_parquet(work / "validation/dev_qrels.parquet")
        .filter(pl.col("context_id") == query["context_id"]))
candidates = frame.join(manifest.select("doc_id", "item_id"), on="doc_id")
answer = frame[chosen.tolist()].join(manifest.select("doc_id", "item_id"), on="doc_id")
# Отделяем пропуски поиска от ошибок выбора итоговых 50 объявлений.
not_retrieved = gold.join(candidates.select("item_id"), on="item_id", how="anti")
not_selected = (gold.join(candidates.select("item_id"), on="item_id")
               .join(answer.select("item_id"), on="item_id", how="anti"))
```

- `query` показывает текст, локацию и фильтры.
- `gold` содержит все известные положительные объявления контекста.
- `not_retrieved` показывает пропуски поиска.
- `not_selected` показывает найденные положительные, не вошедшие в итоговый топ-50.
- `frame` содержит признаки, по которым можно сравнить пропущенные объявления
  с выбранными. Тексты присоединяются из `work/validation/corpus.parquet` по `item_id`.

Пример использует расширенный dev-пул, потому что именно он выбран в приложенном
`selected.json`. Если в отдельном эксперименте выбран `nearby_candidates=false`,
для такого разбора нужен `dev.parquet`.

## 11. Время и границы выполненных проверок

Проверенный полный пересчёт benchmark с готовой моделью занял **501,32 с**
в новом окружении Python 3.12.10 на Windows, без времени установки зависимостей.
До запуска отсутствовали рабочие индексы и признаки. Это замер `check.ipynb`
из [reproduction_before_crlf.json](../results/improved/reproduction_before_crlf.json).

После перехода на явные окончания строк CRLF повторное предсказание по уже
проверенным признакам заняло **4,42 с**. Это только применение модели и сохранение
CSV, не весь быстрый запуск. Отчёт:
[reproduction_check.json](../results/improved/reproduction_check.json).
Полный быстрый запуск после смены переводов строк отдельно не повторялся.

Полное обучение текущей модели в чистом окружении отдельно не повторялось:
оно выполнялось этапами с ранее построенными индексами. Поэтому точного
замера всего `solution.ipynb` с `rebuild = True` в чистом окружении нет.
Для редакторских изменений комментариев дополнительно сохранена проверка
неизменности вычислительного AST и модели в
[documentation_check.json](../results/improved/documentation_check.json).
