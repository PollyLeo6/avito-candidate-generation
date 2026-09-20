"""В ответ попадают исходные item_id, а не внутренние номера документов."""

import json

from catboost import CatBoostRanker, Pool
import polars as pl

from avito_retrieval.common import file_hash, save_json
from avito_retrieval.submission import validate_answer
from .model import read_features


def predict(config):
    """Строит топ-50 для benchmark и сохраняет CSV только после проверки формата."""
    # Признаки должны идти в порядке, сохранённом при обучении модели.
    frame, names = read_features(config, "benchmark")
    model = CatBoostRanker().load_model(str(config["results_dir"] / "ranker.cbm"))
    scores = model.predict(Pool(frame.select(names).to_numpy(), feature_names=names),
                           thread_count=config["threads"])
    # Используем метод, выбранный на dev, без нового выбора по benchmark.
    selected = json.loads((config["results_dir"] / "selected.json").read_text("utf-8"))
    if selected["method"] == "geo_rrf":
        scores = frame["rrf"].to_numpy() + selected["geo_rrf"]["weight"] * frame["same_location"].to_numpy()
    # При равных оценках doc_id обеспечивает повторяемый порядок кандидатов.
    ranked = frame.select("query_number", "doc_id").with_columns(pl.Series("score", scores))
    ranked = ranked.sort(["query_number", "score", "doc_id"], descending=[False, True, False])
    top = ranked.group_by("query_number", maintain_order=True).head(50)
    # В ответ возвращаются исходные ID, а не номера строк внутренних таблиц.
    items = pl.read_parquet(config["work_dir"] / "benchmark/corpus_manifest.parquet")
    queries = pl.read_parquet(config["work_dir"] / "features/benchmark_queries.parquet")
    top = top.join(items.select("doc_id", "item_id"), on="doc_id", how="left", maintain_order="left")
    grouped = top.group_by("query_number", maintain_order=True).agg(pl.col("item_id").str.join(" ").alias("answer"))
    answer = queries.join(grouped, on="query_number").select(pl.col("context_id").alias("query_id"), "answer").sort("query_id")
    # Невалидный или недописанный файл не должен заменить готовый ответ.
    path = config["results_dir"] / "answer.csv"
    temporary = path.with_suffix(".partial.csv")
    answer.write_csv(temporary)
    check = validate_answer(temporary, config["data_dir"] / "benchmark_queries.parquet",
                            config["data_dir"] / "benchmark_items.parquet")
    temporary.replace(path)
    # Связываем проверенный ответ с конкретным сохранённым файлом модели.
    check["model_sha256"] = file_hash(config["results_dir"] / "ranker.cbm")
    save_json(config["results_dir"] / "submission_check.json", check)
    print(check, flush=True)
