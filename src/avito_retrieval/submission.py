"""Контракт answer.csv проверяется отдельно от поиска."""

import csv
import re
from pathlib import Path

import polars as pl

from .common import file_hash
from .data_check import check_ids


def validate_answer(path, query_path, item_path):
    queries = pl.read_parquet(query_path, columns=["query_id"])["query_id"]
    items = pl.read_parquet(item_path, columns=["item_id"])["item_id"]
    check_ids(queries, r"^.{16}$")
    check_ids(items, r"^[0-9a-f]{16}$")
    expected = set(queries)
    allowed = set(items)
    seen = set()
    lengths = []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != ["query_id", "answer"]:
            raise ValueError("Нужны ровно две колонки: query_id,answer")
        for line, row in enumerate(reader, 2):
            if len(row) != 2:
                raise ValueError(f"Строка {line}: неверное число колонок")
            query_id, answer = row
            if query_id not in expected or query_id in seen or len(query_id) != 16:
                raise ValueError(f"Строка {line}: неизвестный или повторный query_id")
            seen.add(query_id)
            ids = answer.split(" ") if answer else []
            if len(ids) > 50 or len(ids) != len(set(ids)):
                raise ValueError(f"Строка {line}: больше 50 item_id или есть повторы")
            if any(not re.fullmatch(r"[0-9a-f]{16}", item) or item not in allowed for item in ids):
                raise ValueError(f"Строка {line}: неверный item_id")
            lengths.append(len(ids))
    if seen != expected:
        raise ValueError("В answer.csv пропущены запросы")
    return {
        "valid": True,
        "rows": len(seen),
        "min_items": min(lengths),
        "max_items": max(lengths),
        "sha256": file_hash(path),
    }
