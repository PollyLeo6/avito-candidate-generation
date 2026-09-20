# Исходные данные

Для обоих ноутбуков нужны три исходных файла из задания:

- `train.parquet` - 497 673 положительные пары;
- `benchmark_queries.parquet` - 2 452 запроса;
- `benchmark_items.parquet` - 189 212 объявлений.

В быстрой проверке `train.parquet` нужен для истории запросов и центров локаций.
Обучение CatBoost в `notebooks/check.ipynb` не повторяется.

Архив данных: [Яндекс Диск](https://disk.yandex.ru/d/sNhfo0YOjGtufg).
Данные не включены в Git и не скачиваются автоматически.

Если файлы лежат в другой папке, измените `data_dir` в `configs/improved.toml`.
