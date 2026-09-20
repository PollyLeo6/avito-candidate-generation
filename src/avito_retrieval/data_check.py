"""Проверка исходных файлов до дорогих вычислений."""

import importlib.metadata
import platform

import polars as pl
import pyarrow.parquet as pq

from .common import file_hash, query_columns, save_json, text_fields


def check_ids(values, pattern, unique=True):
    if values.null_count() or not values.str.contains(pattern).all():
        raise ValueError(f"Неверный формат {values.name}")
    if unique and values.n_unique() != len(values):
        raise ValueError(f"Повторные {values.name}")


def inspect_data(data_dir, results_dir):
    item_columns = ["item_id", "item_microcat_id", "item_category_id", "item_location_id"]
    specifications = {
        "train.parquet": (497673, query_columns + item_columns + list(text_fields.values())),
        "benchmark_queries.parquet": (2452, ["query_id"] + query_columns),
        "benchmark_items.parquet": (189212, item_columns + list(text_fields.values())),
    }
    summary = {}
    for name, (expected_rows, required) in specifications.items():
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Не найден {path}. Проверьте data_dir в config.toml")
        parquet = pq.ParquetFile(path)
        if missing := set(required) - set(parquet.schema_arrow.names):
            raise ValueError(f"{name}: отсутствуют колонки {sorted(missing)}")
        if parquet.metadata.num_rows != expected_rows:
            raise ValueError(f"{name}: число строк отличается от условия задания")
        id_column = "query_id" if "queries" in name else "item_id"
        ids = pl.read_parquet(path, columns=[id_column])[id_column]
        pattern = r"^.{16}$" if id_column == "query_id" else r"^[0-9a-f]{16}$"
        check_ids(ids, pattern, unique=name != "train.parquet")
        summary[name] = {
            "rows": len(ids),
            "unique_ids": ids.n_unique(),
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
            "schema": {field.name: str(field.type) for field in parquet.schema_arrow},
        }
    save_json(results_dir / "input_check.json", summary)
    dependencies = ["numpy", "scipy", "polars", "pyarrow", "bm25s", "snowballstemmer"]
    save_json(
        results_dir / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in dependencies},
        },
    )
    return summary
