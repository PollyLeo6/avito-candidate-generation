# Candidate generation для поиска услуг Авито

Решение тестового задания: для каждого поискового запроса выбрать до 50 объявлений
из заданного корпуса. BM25 собирает кандидатов по тексту и локации, CatBoostRanker
выбирает итоговые 50 по текстовым и структурированным признакам.

**Готовый ответ:** [answer.csv](answer.csv). **Основной файл для изучения и запуска:**
[solution.ipynb](solution.ipynb). Вычисления выполняются локально на CPU, без внешних API.

## Результат

| Метод | Dev Recall@50 | Holdout Recall@50 |
| --- | ---: | ---: |
| BM25 + RRF | 28,96% | 27,76% |
| RRF с учётом локации | 61,27% | 58,70% |
| BM25 + CatBoost | **74,10%** | **72,45%** |

Все методы проверены на одинаковых запросах и корпусе из 515 895 объявлений.
Dev и holdout содержат по 2 000 запросов. Модель выбрана по dev и зафиксирована
до оценки holdout. Это локальная оценка по известным положительным парам;
качество на закрытой разметке Авито неизвестно.

## С чего начать

1. Прочитать [описание метода](docs/approach.md): данные, кандидаты, признаки и обучение.
2. Открыть `solution.ipynb` и пройти ячейки сверху вниз. Сохранённая
   [HTML-версия](docs/solution.html) открывается в браузере без Python после скачивания.
3. Посмотреть [метрики и ошибки](docs/results.md).
4. Для сдачи загрузить `answer.csv`, а этот репозиторий приложить как код решения.

## Структура

```text
answer.csv              готовый файл для загрузки
solution.ipynb          пошаговое выполнение и результаты
ranking.toml           пути к данным и параметры
avito_retrieval/        BM25, обработка текста, проверка CSV
avito_ranker/           кандидаты, признаки, CatBoost, оценка
tests/                 проверки метрики, разбиения и формата
results/               модель и отчёты выполненного эксперимента
docs/                  описание метода, результаты, HTML-версия
data/                  место для трёх исходных Parquet
```

`work/` создаётся при вычислениях и хранит индексы и таблицы признаков.
Данные, `work/`, окружение Python и кэши исключены из Git. Модель и отчёты включены,
чтобы можно было изучить результат до повторного обучения.

## Установка

Нужен Python 3.12. Команды выполняются из корня репозитория.

**Windows / PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name avito-ranking --display-name "Avito ranking"
.\.venv\Scripts\python.exe -m jupyterlab solution.ipynb
```

**Linux / macOS:**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebook.txt
.venv/bin/python -m ipykernel install --sys-prefix --name avito-ranking --display-name "Avito ranking"
.venv/bin/python -m jupyterlab solution.ipynb
```

В Jupyter выберите ядро **Avito ranking**. Интернет нужен при установке библиотек;
вычисления не обращаются к API и не загружают модели.

`requirements.txt` содержит основные вычислительные библиотеки,
`requirements-lock.txt` — проверенные версии вместе с зависимостями,
`requirements-notebook.txt` добавляет Jupyter.

## Два режима ноутбука

**Просмотр результатов:** оставьте `rebuild = False` в первой ячейке.
Ячейки показывают сохранённые отчёты, выполняют тесты и проверяют целостность
зафиксированных кода и модели. Исходные Parquet для этого режима не нужны.
Это явно обозначенный просмотр: повторного обучения в нём нет.

**Полный пересчёт:** положите в `data/` исходные файлы:

```text
data/train.parquet
data/benchmark_queries.parquet
data/benchmark_items.parquet
```

В `ranking.toml` можно указать другую папку через `data_dir`. Относительные пути
считаются от этого файла. Поставьте `rebuild = True` и выполните все ячейки сверху
вниз: от подготовки данных до экспорта `answer.csv` в корень репозитория.
Результат предсказания также сохраняется в `results/answer.csv`.

Настройки и модель соответствуют уже оценённому эксперименту. Для его повторения
сохраните алгоритмические параметры. Изменение признаков, модели или настроек
требует нового эксперимента с отдельной оценкой: проверка фиксации не позволит
выдать изменённый вариант за прежний.

Расчёт настроен на два CPU-потока и был выполнен на ноутбуке с 8 ГБ RAM.
Рабочие файлы занимают несколько ГБ; GPU не требуется. Полный пересчёт перезаписывает
отчёты и рабочие файлы, поэтому сохранённый результат удобнее изучать в режиме просмотра.

## Проверки

В Windows:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

В Linux/macOS используйте `.venv/bin/python` вместо пути к Python в Windows.
Тестам не нужны исходные таблицы. Проверяются Recall, группировка запросов,
отбор обучающих кандидатов, ранги и ограничения CSV.

В `answer.csv` проверены UTF-8, две колонки `query_id,answer`, полное покрытие
2 452 запросов, уникальность и принадлежность всех ID корпусу. Отчёт:
[results/requirements_check.json](results/requirements_check.json).

В чистом окружении с готовыми индексами и моделью повторены поиск кандидатов
и предсказания для всего benchmark: CSV совпал побайтово. Полное обучение с нуля
во втором окружении не повторялось. Ячейки сохранённого notebook выполнены
в режиме просмотра результатов.

## Использованные инструменты

[BM25S](https://github.com/xhluca/bm25s),
[Snowball](https://snowballstem.org/algorithms/russian/stemmer.html),
[RRF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf),
[CatBoostRanker](https://catboost.ai/docs/en/concepts/python-reference_catboostranker),
NumPy, SciPy, Polars, PyArrow, Jupyter и Matplotlib.
CatBoost обучен на предоставленных данных; предобученные языковые модели не использовались.
