"""Модель обучается на train. Число деревьев выбирается только по dev."""

import json

from catboost import CatBoostRanker, Pool
import numpy as np
import polars as pl

from avito_retrieval.common import file_hash, save_json


def read_features(config, fold):
    """Читает таблицу выбранной части и зафиксированный порядок её признаков."""
    directory = config["work_dir"] / "features"
    names = json.loads((directory / "feature_names.json").read_text("utf-8"))
    frame = pl.read_parquet(directory / f"{fold}.parquet")
    return frame, names


def per_query_recall(frame, predictions, summary):
    """Считает Recall@50 каждого запроса по всем его известным положительным парам."""
    # doc_id разрешает равенство оценок воспроизводимо, без случайного порядка.
    ranked = frame.select("query_number", "doc_id", "label").with_columns(
        pl.Series("score", predictions)
    ).sort(["query_number", "score", "doc_id"], descending=[False, True, False])
    # Сначала отбираем 50 внутри каждого запроса, затем считаем попадания.
    hits = ranked.group_by("query_number", maintain_order=True).head(50).group_by(
        "query_number"
    ).agg(pl.col("label").sum().alias("hits"))
    # В summary остаются и положительные объявления, которых нет в пуле кандидатов.
    return summary.join(hits, on="query_number", how="left").with_columns(
        (pl.col("hits").fill_null(0) / pl.col("positives")).alias("recall_50")
    ).sort("query_number")


def train_model(config):
    """Обучает полную PairLogit-модель на train и сохраняет сведения об обучении."""
    frame, names = read_features(config, "train")
    # Идентификаторы групп нужны для организации данных, но не как входы модели.
    forbidden = {"query_number", "doc_id", "context_id", "query_id", "item_id", "label"}
    if forbidden.intersection(names):
        raise ValueError("В признаки попали идентификаторы или целевая метка")
    # PairLogit сравнивает объявления только внутри одной группы query_number.
    pool = Pool(frame.select(names).to_numpy(), label=frame["label"].to_numpy(),
                group_id=frame["query_number"].to_numpy(), feature_names=names)
    model = CatBoostRanker(
        loss_function="PairLogit:max_pairs=128", iterations=config["iterations"],
        depth=config["depth"], learning_rate=config["learning_rate"], l2_leaf_reg=8,
        random_seed=config["seed"], thread_count=config["threads"],
        allow_writing_files=False, verbose=100,
    )
    # Число деревьев здесь максимальное; подходящий префикс позже выберет dev.
    model.fit(pool)
    model.save_model(str(config["results_dir"] / "ranker_full.cbm"))
    # Хеш таблицы и параметры связывают модель с конкретным обучающим набором.
    save_json(config["results_dir"] / "training.json", {
        "rows": frame.height, "queries": frame["query_number"].n_unique(),
        "positives": int(frame["label"].sum()), "features": names,
        "parameters": model.get_all_params(),
        "training_features_sha256": file_hash(config["work_dir"] / "features/train.parquet"),
        "no_query_or_item_ids_in_features": True,
    })


def select_model(config):
    """Выбирает число деревьев и сравнивает модель с географической эвристикой на dev."""
    frame, names = read_features(config, "dev")
    pool = Pool(frame.select(names).to_numpy(), feature_names=names)
    summary = pl.read_parquet(config["work_dir"] / "features/dev_queries.parquet")
    # Небольшая прибавка за локацию служит простым вариантом для сравнения с моделью.
    geo_variants = []
    for weight in [0.01, 0.03, 0.06]:
        heuristic = frame["rrf"].to_numpy() + weight * frame["same_location"].to_numpy()
        recall = per_query_recall(frame, heuristic, summary)["recall_50"].mean()
        geo_variants.append({"weight": weight, "recall_50": recall})
    geo = max(geo_variants, key=lambda row: (row["recall_50"], -row["weight"]))
    # Одна обученная модель даёт оценки на нескольких этапах, без повторного обучения.
    model = CatBoostRanker().load_model(str(config["results_dir"] / "ranker_full.cbm"))
    candidates = []
    checkpoints = {10, 20, 30, 50, 80, 100, 150, 200, 300, 400, 500, 600, 700}
    for trees, predictions in zip(range(10, config["iterations"] + 1, 10),
                                  model.staged_predict(pool, eval_period=10,
                                                       thread_count=config["threads"])):
        if trees not in checkpoints:
            continue
        # Macro Recall даёт одинаковый вес каждому запросу независимо от числа его пар.
        metrics = per_query_recall(frame, predictions, summary)
        recall = metrics["recall_50"].mean()
        candidates.append({"trees": trees, "recall_50": recall})
        print("dev", trees, recall, flush=True)
    # При равном качестве оставляем меньше деревьев для более быстрого предсказания.
    best = max(candidates, key=lambda row: (row["recall_50"], -row["trees"]))
    model.shrink(best["trees"])
    model.save_model(str(config["results_dir"] / "ranker.cbm"))
    # Сохраняем все проверенные варианты, чтобы было видно основание выбора.
    baseline = summary.select((pl.col("baseline_hits") / pl.col("positives")).mean()).item()
    selected = {"method": "catboost_pairlogit", **best, "baseline_recall_50": baseline,
                "candidates": candidates, "selection_metric": "dev macro Recall@50",
                "geo_rrf": geo, "geo_variants": geo_variants}
    # Итоговый метод определяется только по dev; holdout здесь не читается.
    if geo["recall_50"] > best["recall_50"]:
        selected["method"] = "geo_rrf"
        selected["recall_50"] = geo["recall_50"]
    save_json(config["results_dir"] / "selected.json", selected)
    # Важность помогает разбирать модель, но не используется для выбора по holdout.
    values = model.get_feature_importance(type="PredictionValuesChange")
    save_json(config["results_dir"] / "feature_importance.json",
              sorted(zip(names, values), key=lambda row: -row[1]))
    print(selected, flush=True)


def evaluate_model(config, fold):
    """Оценивает сохранённую модель и пишет предсказания, метрики и bootstrap-интервалы."""
    # После фиксации менять модель перед оценкой holdout нельзя.
    if fold == "holdout":
        from .freeze import check_frozen
        check_frozen(config)
    # Порядок столбцов берём из того же списка, который использовался при обучении.
    frame, names = read_features(config, fold)
    summary = pl.read_parquet(config["work_dir"] / f"features/{fold}_queries.parquet")
    model = CatBoostRanker().load_model(str(config["results_dir"] / "ranker.cbm"))
    predictions = model.predict(Pool(frame.select(names).to_numpy(), feature_names=names),
                                thread_count=config["threads"])
    metrics = per_query_recall(frame, predictions, summary)
    metrics.write_parquet(config["results_dir"] / f"{fold}_per_query.parquet")
    # Сохраняем реальные item_id, чтобы метрику можно было независимо пересчитать.
    ranked = frame.select("query_number", "doc_id").with_columns(pl.Series("score", predictions))
    top = ranked.sort(["query_number", "score", "doc_id"], descending=[False, True, False])
    top = top.group_by("query_number", maintain_order=True).head(50)
    manifest = pl.read_parquet(config["work_dir"] / "validation/corpus_manifest.parquet")
    top = top.join(manifest.select("doc_id", "item_id"), on="doc_id", how="left", maintain_order="left")
    top = top.group_by("query_number", maintain_order=True).agg(pl.col("item_id"))
    summary.select("query_number", "context_id").join(top, on="query_number").write_parquet(
        config["results_dir"] / f"{fold}_predictions.parquet"
    )
    # Все методы сравниваются на одних запросах и с одинаковыми знаменателями.
    baseline = metrics["baseline_hits"].to_numpy() / metrics["positives"].to_numpy()
    improved = metrics["recall_50"].to_numpy()
    selected = json.loads((config["results_dir"] / "selected.json").read_text("utf-8"))
    heuristic = frame["rrf"].to_numpy() + selected["geo_rrf"]["weight"] * frame["same_location"].to_numpy()
    geo = per_query_recall(frame, heuristic, summary)["recall_50"].to_numpy()
    # Парный bootstrap переотбирает запросы вместе с оценками обоих методов.
    rng = np.random.default_rng(config["seed"])
    samples = rng.integers(0, len(improved), size=(2000, len(improved)))
    report = {"queries": len(improved), "baseline_recall_50": float(baseline.mean()),
              "ranker_recall_50": float(improved.mean()),
              "geo_rrf_recall_50": float(geo.mean()),
              "selected_method": selected["method"],
              "selected_recall_50": float((geo if selected["method"] == "geo_rrf" else improved).mean()),
              "delta": float((improved - baseline).mean()),
              "ranker_ci95": np.quantile(improved[samples].mean(1), [0.025, 0.975]).tolist(),
              "delta_ci95": np.quantile((improved - baseline)[samples].mean(1), [0.025, 0.975]).tolist(),
              "model_sha256": file_hash(config["results_dir"] / "ranker.cbm")}
    # Сохраняем и среднее качество, и неопределённость оценки прироста.
    save_json(config["results_dir"] / f"{fold}_metrics.json", report)
    print(report, flush=True)
