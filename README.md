<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/avito-dark.svg">
    <img src="docs/assets/avito.svg" alt="Авито" width="180">
  </picture>
</p>

# Candidate generation для поиска услуг Авито

Решение выбирает 50 объявлений для каждого запроса: BM25 и похожие запросы из train
находят кандидатов, CatBoostRanker формирует итоговую выдачу.
Готовый ответ для 2 452 запросов находится в `answer.csv`.
GPU и внешние API не нужны.

## Состав репозитория

```text
notebooks/
  check.ipynb             пересчёт benchmark с готовой моделью
  solution.ipynb          подготовка, обучение, оценка и итоговый CSV
src/
  avito_retrieval/        обработка текста, BM25, метрика и проверка CSV
  avito_ranker/           исходный поиск, признаки и модель для сравнения
  avito_improved/         история запросов, география, обучение и выбор выдачи
configs/                 пути к данным и параметры запуска
requirements/            зафиксированные версии библиотек
results/
  improved/              текущая модель, метрики и проверки
  ranker.cbm             фиксированная исходная модель
tests/                   27 автоматических тестов
docs/                    описание метода и выполненных проверок
data/                    сюда нужно положить исходные Parquet
answer.csv               файл для отправки
```

Оба ноутбука вызывают этапы через `src/avito_improved/run.py`.
`history.py` строит историю запросов; `retrieval.py` и `geo.py` ищут кандидатов;
`dataset.py` считает признаки; `model.py` обучает модель; `selection.py` выбирает
выдачу; `evaluation.py` считает Recall и сохраняет CSV.
`protocol.py`, `audit.py` и `artifacts.py` отвечают за разбиение, проверки и загрузку модели.
Промежуточные файлы создаются в `work/` и в репозиторий не входят.

## Как устроено решение

1. **Разбиение.** Нормализуем тексты и применяем русский Snowball-стеммер.
   Группы близких запросов целиком разделяются между train, dev и holdout.
   Для итоговой оценки заранее отложены 1 000 ранее не использовавшихся запросов.
2. **Кандидаты.** BM25 ищет по заголовку, описанию и параметрам во всём корпусе
   и в локации запроса. Дополнительные кандидаты находятся по фильтрам,
   похожим обучающим запросам и в радиусе 50 км от центра локации.
   История использует только train; при подготовке обучающего примера из неё
   исключается вся группа этого запроса.
3. **Модель.** CatBoostRanker использует 38 признаков: текстовые совпадения,
   оценки BM25, подкатегории из истории, расстояние, фильтры и свойства объявления.
   Сравниваются PairLogit и QuerySoftMax. По dev выбран PairLogit с 80 деревьями.
   Исходная модель из `results/ranker.cbm` помогает выбирать сложные отрицательные примеры.
4. **Топ-50.** Берём 40 объявлений новой модели и дополняем выдачей исходной,
   пропуская повторы. Способ поиска и доля каждой выдачи выбраны по dev.
   Перед оценкой на новом holdout фиксируются модель, код и история.
5. **Ответ.** Для benchmark поиск выполняется только среди его 189 212 объявлений.
   Проверяются все query_id, принадлежность item_id корпусу, уникальность и формат CSV.

В ноутбуках используются вызовы вида `run_stage("benchmark_features", config_path)`.
Первый аргумент - этап расчёта, второй - настройки. Ячейки задают порядок работы,
а реализация вынесена в небольшие модули, общие для обоих ноутбуков.

## Результаты

| Метод | Dev, 2 000 запросов | Новый holdout, 1 000 запросов |
| --- | ---: | ---: |
| BM25 + RRF | 28,96% | 27,88% |
| Исходный BM25 + CatBoost | 74,10% | 73,58% |
| История запросов + география + CatBoost | **78,32%** | **78,48%** |

Все методы в таблице проверены на одинаковом локальном корпусе из 515 895 объявлений.
Recall@50 усредняется по запросам; пропущенные при поиске положительные объявления
остаются в знаменателе. Отчёт: [fresh_metrics.json](results/improved/fresh_metrics.json).
Исходный вариант получил на платформе Авито **0,770104**.
Отчёты первой версии сохранены в `results/`, текущей - в `results/improved/`.

## 1. Быстрая проверка решения

Пересчитывает поиск и признаки для всех запросов benchmark, применяет готовую модель
и сравнивает новый CSV с приложенным побайтово. Также запускает 27 тестов.

**1.** Установить Python 3.12.10, скачать репозиторий и положить в `data/`:

```text
train.parquet
benchmark_queries.parquet
benchmark_items.parquet
```

**2.** Открыть терминал в корне репозитория и выполнить команды своей ОС.

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

**3.** Выбрать ядро **Avito ranking** и нажать **Run → Run All Cells**.
Последняя ячейка должна показать `status: passed` и `answer_identical: true`.
Новый файл появится в `work/quality/answer.csv`, отчёт - в `work/quality/check_report.json`.

Ориентир для проверки: **8.4 минуты**, без установки зависимостей.
Обучение здесь не повторяется. `train.parquet` нужен для восстановления истории
запросов и центров локаций. Установку окружения достаточно выполнить один раз.

## 2. Полный запуск решения

1. Подготовить окружение и три файла данных по инструкции выше.
2. Открыть `notebooks/solution.ipynb` и выбрать ядро **Avito ranking**.
3. В первой кодовой ячейке поставить `rebuild = True`.
4. Нажать **Run → Run All Cells**. Итоговый `answer.csv` появится в корне репозитория.

Ноутбук строит индексы, готовит обучающие пары, сравнивает две функции потерь,
выбирает модель по dev, проверяет её на holdout и формирует ответ.
При `rebuild = False` он показывает сохранённые результаты сразу, без обучения.
Пути меняются в `configs/improved.toml`; отчёты полного запуска перезаписываются.

Полный цикл существенно дольше быстрой проверки: только обучение двух моделей
в проведённом эксперименте заняло около 34 минут. Индексы и признаки считаются отдельно.
Вычисления проверялись на Windows, CPU, 8 ГБ RAM; используются два потока.
Linux и macOS отдельно не проверялись. Подробности: [docs/reproduction.md](docs/reproduction.md).

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
