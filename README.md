<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/avito-dark.svg">
    <img src="docs/assets/avito.svg" alt="Авито" width="180">
  </picture>
</p>

# Candidate generation для поиска услуг Авито

Для каждого поискового запроса решение выбирает 50 объявлений из заданного корпуса.
BM25 находит кандидатов, CatBoostRanker выбирает итоговую выдачу.
Ответ для 2 452 запросов сохранён в `answer.csv`.

<p align="center">
  <a href="notebooks/check.ipynb">Быстрая проверка</a> ·
  <a href="notebooks/solution.ipynb">Полный запуск</a> ·
  <a href="answer.csv">Готовый answer.csv</a>
</p>

## Состав репозитория

```text
avito-candidate-generation/
├── notebooks/
│   ├── check.ipynb          проверка с готовой моделью, около 5 минут
│   └── solution.ipynb       полный цикл с обучением, около 44 минут
├── src/
│   ├── avito_retrieval/     обработка текста, BM25 и проверка CSV
│   └── avito_ranker/        кандидаты, признаки, обучение и предсказания
├── configs/                пути и параметры запуска
├── requirements/           зафиксированные зависимости
├── results/                модели, метрики и отчёты эксперимента
├── tests/                  22 автоматических теста
├── docs/                   описание метода, разбор ошибок и графика README
├── data/                   папка для исходных Parquet, данные не включены
├── pyproject.toml          установка модулей из src как Python-пакета
├── answer.csv              готовый файл для отправки
└── README.md
```

Папка `work/` создаётся при запуске и хранит промежуточные файлы.

<details>
<summary><b>Модули, которые вызывают оба ноутбука</b></summary>

Оба ноутбука используют общий код поиска и предсказания. `notebooks/solution.ipynb`
дополнительно запускает подготовку разбиения, обучение и оценку модели.

| Модуль | Назначение |
| --- | --- |
| `src/avito_ranker/workflow.py`, `stage.py` | Запуск этапов в отдельных процессах и сохранение логов |
| `src/avito_ranker/config.py` | Чтение путей и параметров из TOML |
| `src/avito_retrieval/common.py`, `build_index.py` | Обработка текста и построение BM25-индексов |
| `src/avito_ranker/candidates.py` | Поиск и объединение кандидатов |
| `src/avito_ranker/features.py`, `dataset.py` | Признаки и таблицы для модели |
| `src/avito_ranker/model.py` | Обучение, выбор по dev и оценка; чтение признаков |
| `src/avito_ranker/predict.py`, `src/avito_retrieval/submission.py` | Формирование топ-50 и проверка CSV |
| `src/avito_ranker/prepare.py` | Корпус и разбиение запросов для полного эксперимента |
| `src/avito_ranker/freeze.py`, `verify.py` | Контроль целостности, проверка меток и метрик |

</details>

## Как устроено решение

1. **Подготовка.** Тексты нормализуются и обрабатываются русским Snowball-стеммером.
   Запросы разделяются на train, dev и holdout по группам нормализованного текста.
   Группы между частями не пересекаются.
2. **Поиск.** BM25 ищет по заголовкам, описаниям и параметрам объявлений.
   По каждому полю выбираются до 300 кандидатов из всего корпуса и до 150 из локации запроса.
   Списки объединяются без повторов. Baseline объединяет выдачи методом RRF.
3. **Ранжирование.** CatBoostRanker обучается на train с PairLogit и 31 признаком:
   оценки и позиции BM25, совпадения слов, локации и категории, цена, рейтинг и другие.
   Идентификаторы в признаки не входят. Количество деревьев выбирается по dev;
   модель фиксируется до оценки на holdout.
4. **Ответ.** Выбранная модель с 50 деревьями формирует топ-50 из объявлений benchmark.
   Проверяются формат CSV, покрытие запросов, уникальность ID и их принадлежность корпусу.

Ячейки notebook вызывают функции из модулей, например
`run_stage("benchmark_features", config_path)`: первый аргумент задаёт этап,
второй - файл настроек. Так в notebook остаются последовательность действий,
пояснения и результаты, а реализация каждого этапа доступна отдельно.

## Результаты

| Метод | Dev Recall@50 | Holdout Recall@50 |
| --- | ---: | ---: |
| BM25 + RRF | 28,96% | 27,76% |
| RRF с учётом локации | 61,27% | 58,70% |
| BM25 + CatBoostRanker | **74,10%** | **72,45%** |

Методы проверены на одном локальном корпусе из 515 895 объявлений
и по 2 000 запросов dev и holdout. Recall@50 усредняется по запросам;
положительные объявления, пропущенные на этапе поиска кандидатов, также учитываются.
Для итогового ответа используется только корпус benchmark из 189 212 объявлений.

## 1. Быстрая проверка решения

**Что проверяется:** поиск и признаки пересчитываются с нуля для всего benchmark,
сохранённая модель формирует новый CSV, затем он сравнивается с приложенным ответом.
Также запускаются 22 теста. Обучение в эту проверку не входит.

**Шаг 1.** Установить Python 3.12.10, распаковать репозиторий и положить в `data/`:

```text
benchmark_queries.parquet
benchmark_items.parquet
```

**Шаг 2.** Открыть терминал в корне репозитория и выполнить команды для своей ОС.

**Windows / PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/notebook.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name avito-ranking --display-name "Avito ranking"
.\.venv\Scripts\python.exe -m jupyterlab notebooks/check.ipynb
```

**Linux / macOS:**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/notebook.txt
.venv/bin/python -m ipykernel install --sys-prefix --name avito-ranking --display-name "Avito ranking"
.venv/bin/python -m jupyterlab notebooks/check.ipynb
```

Команда установки также подключает код из `src/` к окружению Python.
Установка выполняется один раз. Для повторного запуска достаточно последней команды.

**Шаг 3.** В Jupyter выбрать ядро **Avito ranking** и нажать **Run → Run All Cells**.
Менять код не нужно. Успешный результат в последней ячейке:

```json
{"status": "passed", "answer_identical": true}
```

Эти поля означают, что проверки пройдены, а новый ответ совпал с приложенным побайтово.
Новый CSV находится в `work/check/results/answer.csv`, отчёт - в `work/check/results/check_report.json`.

## 2. Полный запуск решения

**Что выполняется:** подготовка данных, построение индексов, обучение,
выбор модели по dev, оценка на holdout и формирование `answer.csv`.

1. Подготовить окружение по инструкции выше. В `data/` должны находиться все три
   исходных файла: `train.parquet`, `benchmark_queries.parquet`, `benchmark_items.parquet`.
2. В открытом Jupyter выбрать `notebooks/solution.ipynb` и ядро **Avito ranking**.
3. В первой кодовой ячейке установить `rebuild = True`.
4. Нажать **Run → Run All Cells**. Готовый `answer.csv` появится в корне репозитория.

При `rebuild = False` ноутбук показывает сохранённые результаты без повторного обучения.
Полный запуск перезаписывает отчёты эксперимента. Если данные находятся в другой папке,
изменить `data_dir` в `configs/ranking.toml`; для быстрой проверки - в `configs/check.toml`.

Проверенное время без установки зависимостей: **около 5 минут для проверки, 44 минуты для полного расчёта**.
Замеры выполнены на Windows с 8 ГБ RAM и двумя CPU-потоками. GPU не требуется;
для рабочих файлов нужно несколько ГБ свободного места. Linux и macOS отдельно не проверялись.

## Использованные библиотеки и подходы

| Библиотека | Роль в решении |
| --- | --- |
| <img src="docs/assets/catboost.png" height="26" alt="CatBoost"> [CatBoost](https://catboost.ai/) | Обучение модели ранжирования |
| <img src="docs/assets/numpy.svg" width="24" alt="NumPy"> [NumPy](https://numpy.org/) | Численные признаки и массивы |
| <img src="docs/assets/scipy.svg" width="24" alt="SciPy"> [SciPy](https://scipy.org/) | Работа с разреженными матрицами |
| <img src="docs/assets/polars.svg" width="24" alt="Polars"> [Polars](https://pola.rs/) | Подготовка и обработка таблиц |
| <img src="docs/assets/apachearrow.svg" width="24" alt="Apache Arrow"> [PyArrow](https://arrow.apache.org/docs/python/) | Чтение и запись Parquet |
| <img src="docs/assets/jupyter.svg" width="24" alt="Jupyter"> [JupyterLab](https://jupyter.org/) и ipykernel | Пошаговый запуск и просмотр результатов |
| <img src="docs/assets/matplotlib.svg" height="24" alt="Matplotlib"> [Matplotlib](https://matplotlib.org/) | Графики качества |
| <img src="docs/assets/python.svg" width="24" alt="Python"> [BM25S](https://github.com/xhluca/bm25s) и [snowballstemmer](https://snowballstem.org/) | Поиск кандидатов и русский стемминг |

Для объединения выдач используется [Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf).
Версии библиотек зафиксированы в `requirements/lock.txt` и `requirements/notebook.txt`.
Вычисления выполняются локально, без внешних API.
Источники логотипов: [docs/assets/README.md](docs/assets/README.md).
