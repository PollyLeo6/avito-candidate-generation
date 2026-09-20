"""Независимая проверка Recall по строковым ID и формата ответа."""

import importlib.metadata
import json
import platform

import numpy as np
import polars as pl

from avito_retrieval.common import save_json
from avito_retrieval.submission import validate_answer
from .freeze import check_frozen


def audit_training(config):
    work = config["work_dir"]
    queries = {fold: pl.read_parquet(work / f"validation/{fold}_queries.parquet")
               for fold in ["train", "dev", "holdout"]}
    groups = {fold: set(frame["strict_group"]) for fold, frame in queries.items()}
    if groups["train"] & (groups["dev"] | groups["holdout"]) or groups["dev"] & groups["holdout"]:
        raise ValueError("Пересеклись группы разных частей")
    train_queries = queries["train"].with_row_index("query_number")
    labels = pl.read_parquet(work / "features/train.parquet", columns=["query_number", "doc_id", "label"])
    if not set(labels["query_number"]) <= set(train_queries["query_number"]):
        raise ValueError("В обучении есть неизвестный запрос")
    items = pl.read_parquet(work / "validation/corpus_manifest.parquet", columns=["doc_id", "item_id"])
    mapped = labels.join(train_queries.select("query_number", "context_id"), on="query_number")
    mapped = mapped.join(items, on="doc_id")
    gold = pl.read_parquet(work / "validation/train_qrels.parquet").with_columns(
        pl.lit(1).alias("expected_label")
    )
    mapped = mapped.join(gold, on=["context_id", "item_id"], how="left")
    wrong = mapped.filter(pl.col("label") != pl.col("expected_label").fill_null(0)).height
    if wrong or mapped.height != labels.height:
        raise ValueError("Метки обучения не совпали с независимым соединением train-пар")
    train_columns = set(pl.read_parquet_schema(work / "features/train.parquet"))
    names = json.loads((work / "features/feature_names.json").read_text("utf-8"))
    forbidden = {"query_number", "doc_id", "label", "query_id", "context_id", "item_id"}
    if forbidden & set(names) or not set(names) <= train_columns:
        raise ValueError("Ошибка списка признаков")
    report = {"groups_disjoint": True, "independently_checked_training_rows": labels.height,
              "wrong_training_labels": wrong, "training_queries": labels["query_number"].n_unique(),
              "ids_and_labels_excluded_from_features": True,
              "checked_without_reading_dev_or_holdout_positive_pairs": True}
    save_json(config["results_dir"] / "leakage_audit.json", report)
    print(report, flush=True)


def verify(config):
    check_frozen(config)
    report = {}
    corpus = set(pl.read_parquet(config["work_dir"] / "validation/corpus_manifest.parquet")["item_id"])
    for fold in ["dev", "holdout"]:
        path = config["work_dir"] / f"validation/{fold}_qrels.parquet"
        gold = dict(pl.read_parquet(path).group_by("context_id").agg(pl.col("item_id")).iter_rows())
        predicted = pl.read_parquet(config["results_dir"] / f"{fold}_predictions.parquet")
        if set(predicted["context_id"]) != set(gold) or predicted["context_id"].n_unique() != len(gold):
            raise ValueError("Набор запросов не совпал с разметкой")
        values = []
        for row in predicted.iter_rows(named=True):
            ids = row["item_id"]
            if len(ids) != 50 or len(set(ids)) != 50 or not set(ids) <= corpus:
                raise ValueError("Неверные объявления в предсказаниях")
            positives = set(gold[row["context_id"]])
            values.append(len(set(ids) & positives) / len(positives))
        observed = float(np.mean(values))
        metrics = json.loads((config["results_dir"] / f"{fold}_metrics.json").read_text("utf-8"))
        if not np.isclose(observed, metrics["ranker_recall_50"], atol=1e-12, rtol=0):
            raise ValueError("Recall не совпал при независимом пересчёте")
        report[fold] = {"queries": len(values), "independent_recall_50": observed}
    report["submission"] = validate_answer(config["results_dir"] / "answer.csv",
                                            config["data_dir"] / "benchmark_queries.parquet",
                                            config["data_dir"] / "benchmark_items.parquet")
    report["environment"] = {"python": platform.python_version(), "packages": {
        name: importlib.metadata.version(name) for name in
        ["numpy", "scipy", "polars", "pyarrow", "bm25s", "snowballstemmer", "catboost"]}}
    save_json(config["results_dir"] / "verification.json", report)
    print(report, flush=True)
