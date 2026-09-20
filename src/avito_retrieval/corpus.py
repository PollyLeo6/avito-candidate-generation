"""Корпус заданного размера с полным покрытием validation positives."""

import polars as pl
import pyarrow.parquet as pq

from .common import log, stable_hash


def choose_items(train_items, benchmark_items, positive_ids, size):
    """Выбирает корпус заданного размера, сохраняя все positives локальной оценки."""
    # При повторном item_id берём признаки из benchmark_items.
    all_items = pl.concat([benchmark_items, train_items]).unique(
        "item_id", keep="first", maintain_order=True
    )
    if len(positive_ids) > size:
        raise ValueError("Positives не помещаются в корпус")
    selected = set(positive_ids)
    # Квоты приближают распределение подкатегорий к корпусу бенчмарка.
    quotas = dict(benchmark_items.group_by("item_microcat_id").len().iter_rows())
    if size != benchmark_items.height:
        quotas = {key: int(value * size / benchmark_items.height) for key, value in quotas.items()}
    # Остальные объявления идут в порядке хеша, одинаковом при каждом запуске.
    remaining = all_items.filter(~pl.col("item_id").is_in(list(positive_ids)))
    remaining = remaining.with_columns(
        pl.col("item_id")
        .map_elements(lambda value: stable_hash(value, "corpus"), return_dtype=pl.String)
        .alias("sample")
    ).sort("sample")
    # Уже включённые positives тоже занимают места в квотах своих подкатегорий.
    current = dict(
        all_items.filter(pl.col("item_id").is_in(list(positive_ids)))
        .group_by("item_microcat_id")
        .len()
        .iter_rows()
    )
    for item, microcat in remaining.select(["item_id", "item_microcat_id"]).iter_rows():
        if len(selected) >= size:
            break
        if current.get(microcat, 0) < quotas.get(microcat, 0):
            selected.add(item)
            current[microcat] = current.get(microcat, 0) + 1
    # Если квоты недостижимы, оставшиеся места заполняются в том же порядке.
    for item in remaining["item_id"]:
        if len(selected) >= size:
            break
        selected.add(item)
    # Нельзя оценивать поиск, если нужного объявления заведомо нет в корпусе.
    if len(selected) != size or not positive_ids <= selected:
        raise ValueError("Неверный размер или неполное покрытие корпуса")
    # Сортировка по item_id задаёт стабильные номера строк для всех индексов.
    return (
        all_items.filter(pl.col("item_id").is_in(list(selected)))
        .sort("item_id")
        .with_row_index("doc_id")
    )


def write_texts(data_root, manifest, destination):
    """Записывает признаки выбранных объявлений, не загружая весь train в память."""
    # Схема benchmark_items отделяет поля объявления от полей запроса в train.
    columns = pq.read_schema(data_root / "benchmark_items.parquet").names
    missing = set(manifest["item_id"])
    writer = None
    try:
        # Сначала используем записи бенчмарка, затем дополняем недостающие ID из train.
        for source in ["benchmark_items.parquet", "train.parquet"]:
            batches = pq.ParquetFile(data_root / source).iter_batches(
                batch_size=4096, columns=columns, use_threads=False
            )
            for batch in batches:
                data = pl.from_arrow(batch).filter(pl.col("item_id").is_in(list(missing)))
                data = data.unique("item_id", maintain_order=True)
                if not data.height:
                    continue
                # Убираем найденные ID, чтобы один item_id не записался дважды.
                missing.difference_update(data["item_id"])
                table = data.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
                writer.write_table(table)
            log("Источник обработан:", source, "осталось items:", len(missing))
    finally:
        # Закрываем Parquet даже при ошибке чтения очередной порции.
        if writer is not None:
            writer.close()
    # Неполный файл нельзя передавать дальше на индексацию.
    if missing:
        raise ValueError("Не для всех ID найден текст")


def microcategory_distance(corpus, benchmark):
    """Считает расстояние полной вариации между долями подкатегорий двух корпусов."""
    # Полное объединение учитывает подкатегории, встречающиеся только с одной стороны.
    actual = corpus.group_by("item_microcat_id").len(name="actual")
    expected = benchmark.group_by("item_microcat_id").len(name="expected")
    counts = actual.join(expected, on="item_microcat_id", how="full", coalesce=True).fill_null(0)
    # Значение 0 означает одинаковые распределения; верхняя граница равна 1.
    return counts.select(
        ((pl.col("actual") / corpus.height - pl.col("expected") / benchmark.height).abs().sum() / 2)
    ).item()
