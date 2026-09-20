# Candidate generation для поиска услуг Авито

BM25 находит кандидатов по тексту, CatBoostRanker выбирает итоговые 50 объявлений.
Готовый ответ находится в [answer.csv](answer.csv).

## Быстрая проверка кода

Основной файл для проверки - **[check.ipynb](check.ipynb)**.
Он заново строит BM25-индексы, ищет кандидатов для всех 2 452 запросов,
рассчитывает признаки и получает новый CSV с сохранённой моделью.
В конце выполняется побайтовое сравнение с приложенным `answer.csv`.

Проверенный запуск занял **4,0 мин** без установки зависимостей.
Использовались все 189 212 объявлений benchmark; готовые индексы и признаки
не использовались. Обучение модели в этот запуск не входит.

### 1. Подготовить файлы

Нужен Python 3.12.10. Распаковать репозиторий и положить два файла из задания в `data/`:

```text
data/benchmark_queries.parquet
data/benchmark_items.parquet
```

Открыть терминал в корне репозитория, где находятся `check.ipynb` и `check.toml`.
Если данные лежат в другой папке, указать её через `data_dir` в `check.toml`.

### 2. Установить зависимости и открыть notebook

**Windows / PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name avito-ranking --display-name "Avito ranking"
.\.venv\Scripts\python.exe -m jupyterlab check.ipynb
```

**Linux / macOS:**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebook.txt
.venv/bin/python -m ipykernel install --sys-prefix --name avito-ranking --display-name "Avito ranking"
.venv/bin/python -m jupyterlab check.ipynb
```

Установка нужна один раз. При следующих запусках достаточно последней команды.

### 3. Выполнить проверку

В Jupyter выбрать ядро **Avito ranking**, затем **Run → Run All Cells**.
Менять параметры в ячейках не нужно. Notebook выполнит семь этапов:

1. Проверит целостность кода, модели и параметров.
2. Запустит 20 тестов.
3. Подготовит корпус benchmark.
4. Заново построит три BM25-индекса.
5. Найдёт кандидатов и рассчитает признаки.
6. Получит предсказания и проверит формат CSV.
7. Сравнит новый ответ с приложенным.

При успешном завершении последняя ячейка покажет `"status": "passed"`
и `"answer_identical": true`.

Новый ответ: `work/check/results/answer.csv`.
Отчёт: `work/check/results/check_report.json`. Логи: `work/check/results/logs/`.
Приложенные модель, отчёты обучения и исходный `answer.csv` не перезаписываются.

Замер выполнен на Windows с 8 ГБ RAM и двумя CPU-потоками. На другой машине
время может отличаться; Linux и macOS отдельно не проверялись. GPU не требуется.

## Как устроено решение

1. **Подготовка.** Тексты нормализуются и обрабатываются русским Snowball-стеммером.
   Группы нормализованных запросов разделяются на train, dev и holdout без пересечений.
2. **Кандидаты.** BM25 ищет по заголовку, описанию и параметрам объявления.
   По каждому полю берутся до 300 общих кандидатов и до 150 из локации запроса.
3. **Признаки.** Для пары «запрос - объявление» рассчитывается 31 признак:
   оценки и позиции BM25, совпадения слов, локации и категории, цена, рейтинг и другие.
   Идентификаторы в признаки не входят.
4. **Обучение и выбор.** CatBoostRanker обучается на train с PairLogit.
   Количество деревьев выбирается по dev, затем модель фиксируется и оценивается на holdout.
5. **Ответ.** Выбранная модель с 50 деревьями формирует топ-50 по корпусу benchmark.

| Метод | Dev Recall@50 | Holdout Recall@50 |
| --- | ---: | ---: |
| BM25 + RRF | 28,96% | 27,76% |
| RRF с учётом локации | 61,27% | 58,70% |
| BM25 + CatBoostRanker | **74,10%** | **72,45%** |

Методы сравниваются на одинаковом локальном корпусе из 515 895 объявлений
и по 2 000 запросов dev и holdout. Подробнее: [метод](docs/approach.md)
и [разбор результатов](docs/results.md).

## Почему в notebook короткие вызовы

```python
run_stage("benchmark_features", config_path)
```

Это вызов функции Python: первый аргумент задаёт этап, второй - файл настроек.
Сам код находится в `avito_ranker/` и `avito_retrieval/`.
Notebook содержит порядок действий, пояснения и результаты.
Тяжёлые этапы запускаются в отдельных процессах и освобождают память после завершения.

## Полный пересчёт с обучением

Добавить `data/train.parquet`, открыть [solution.ipynb](solution.ipynb),
установить `rebuild = True` и выполнить все ячейки. Пути задаются в `ranking.toml`.
При `rebuild = False` этот notebook только показывает сохранённые результаты и выполняет проверки.

Полный пересчёт занимает около 44 минут на проверенной машине и перезаписывает
отчёты эксперимента. Подтверждение воспроизводимости: [docs/reproduction.md](docs/reproduction.md).

## Тесты отдельно

Из корня репозитория после установки зависимостей:

**Windows:**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

**Linux / macOS:**

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Ожидаемый результат - 20 пройденных тестов и `OK`. Исходные Parquet для тестов не нужны.

## Использованные библиотеки и подходы

- [BM25S](https://github.com/xhluca/bm25s) - текстовый поиск.
- [snowballstemmer](https://snowballstem.org/) - русский стемминг.
- [Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) - объединение выдач.
- [CatBoostRanker](https://catboost.ai/docs/en/concepts/python-reference_catboostranker) - ранжирование.
- NumPy и SciPy - вычисления; Polars и PyArrow - таблицы и Parquet.
- JupyterLab и ipykernel - notebook; Matplotlib - графики.

Версии зафиксированы в `requirements-lock.txt` и `requirements-notebook.txt`.
Вычисления выполняются локально, без внешних API и загрузки моделей.
