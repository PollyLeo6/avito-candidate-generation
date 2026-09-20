# Исходные данные

Для проверки через `check.ipynb` нужны два файла из задания:

- `benchmark_queries.parquet` - 2 452 запроса;
- `benchmark_items.parquet` - 189 212 объявлений.

Для полного обучения через `solution.ipynb` дополнительно нужен `train.parquet`.

Архив данных: [Яндекс Диск](https://disk.yandex.ru/d/sNhfo0YOjGtufg).
Данные не включены в Git и не скачиваются автоматически.

Если файлы лежат в другой папке, измените `data_dir` в `check.toml`
для быстрой проверки или в `ranking.toml` для полного запуска.
