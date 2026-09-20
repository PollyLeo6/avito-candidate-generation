"""Кандидаты, признаки и метки хранятся отдельно от исходных текстов."""

import json
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from avito_retrieval.common import file_hash, save_json
from .candidates import CandidateSearch
from .features import FeatureBuilder


def sample_training(pool, labels, matrix, names, limit, rng):
    """Оставляет все найденные положительные примеры и ограничивает отрицательные."""
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(labels == 0)
    # Половина сложных отрицательных примеров, половина случайных из того же пула.
    heuristic = matrix[:, names.index("rrf")] + 0.04 * matrix[:, names.index("same_location")]
    order = negative[np.lexsort((pool[negative], -heuristic[negative]))]
    hard = order[:limit // 2]
    remaining = np.setdiff1d(negative, hard)
    random = rng.choice(remaining, min(limit - len(hard), len(remaining)), replace=False)
    # Сохраняем порядок документов внутри запроса после отбора.
    return np.sort(np.concatenate([positive, hard, random]))


def create_dataset(config, fold):
    """Записывает признаки кандидатов и отдельную сводку для расчёта Recall."""
    # Holdout открывается только после фиксации модели и настроек.
    if fold == "holdout":
        from .freeze import check_frozen
        check_frozen(config)
    work = config["work_dir"]
    benchmark = fold == "benchmark"
    corpus_dir = work / ("benchmark" if benchmark else "validation")
    index_dir = work / ("benchmark_indices" if benchmark else "validation_indices")
    # У benchmark нет меток; остальные части получают их из сохранённых qrels.
    if benchmark:
        queries = pl.read_parquet(config["data_dir"] / "benchmark_queries.parquet")
        queries = queries.rename({"query_id": "context_id"}).sort("context_id")
        qrels = {}
    else:
        queries = pl.read_parquet(corpus_dir / f"{fold}_queries.parquet")
        frame = pl.read_parquet(corpus_dir / f"{fold}_qrels.parquet")
        qrels = dict(frame.group_by("context_id").agg(pl.col("item_id")).iter_rows())
    # Поиск и признаки используют один порядок doc_id из манифеста корпуса.
    search = CandidateSearch(corpus_dir, index_dir, config["global_depth"], config["local_depth"])
    builder = FeatureBuilder(corpus_dir, search.manifest, search.indices)
    item_ids = search.manifest["item_id"].to_numpy()
    by_id = {value: i for i, value in enumerate(item_ids)}
    # Пишем частями во временный файл, чтобы сбой не оставил готовый с виду датасет.
    folder = work / "features"
    folder.mkdir(exist_ok=True)
    destination = folder / f"{fold}.parquet"
    temporary = folder / f"{fold}.partial.parquet"
    writer, pieces, report, names = None, [], [], None
    rng = np.random.default_rng(config["seed"])
    started = time.perf_counter()
    try:
        for number, query in enumerate(queries.iter_rows(named=True)):
            # Сначала поиск без разметки, затем метки только для найденных кандидатов.
            pool, scores, filters, channels, baseline, top50 = search.retrieve(query)
            matrix, names = builder.build(query, pool, scores, filters, channels, baseline)
            positives = {by_id[item] for item in qrels.get(query["context_id"], [])}
            labels = np.isin(pool, list(positives)).astype(np.uint8)
            # Пропущенные поиском положительные пары остаются в знаменателе метрики.
            denominator = len(positives)
            report.append({
                "context_id": query["context_id"], "query_number": number,
                "positives": denominator, "pool_size": len(pool), "pool_hits": int(labels.sum()),
                "baseline_hits": len(set(top50) & positives),
            })
            if fold == "train":
                # Без положительных кандидатов сравнивать пары внутри запроса нельзя.
                if not labels.any():
                    continue
                selected = sample_training(pool, labels, matrix, names, config["train_negatives"], rng)
                pool, matrix, labels = pool[selected], matrix[selected], labels[selected]
            # Строки одного запроса идут подряд: этого требует обучение ранжирования.
            data = {"query_number": np.full(len(pool), number, dtype=np.uint32),
                    "doc_id": pool, "label": labels}
            data.update({name: matrix[:, i] for i, name in enumerate(names)})
            pieces.append(pl.DataFrame(data))
            # Небольшие порции ограничивают расход памяти на большой таблице пар.
            if len(pieces) >= 40 or number == queries.height - 1:
                table = pl.concat(pieces).to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
                pieces.clear()
            if (number + 1) % 200 == 0:
                print(f"{fold}: {number + 1}/{queries.height}, {time.perf_counter()-started:.0f} s", flush=True)
        # Последний train-запрос мог быть пропущен, поэтому дописываем остаток отдельно.
        if pieces:
            table = pl.concat(pieces).to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer:
            writer.close()
    # Только полностью закрытый Parquet получает итоговое имя.
    temporary.replace(destination)
    summary = pl.DataFrame(report)
    summary.write_parquet(folder / f"{fold}_queries.parquet")
    save_json(folder / "feature_names.json", names)
    # Сводка хранит все запросы, включая те, для которых поиск ничего не нашёл.
    checks = {"fold": fold, "queries": queries.height,
              "rows": pq.ParquetFile(destination).metadata.num_rows,
              "features": names, "seconds": time.perf_counter() - started,
              "feature_file_sha256": file_hash(destination)}
    # Усредняем доли по запросам, а не суммарное число попаданий по всем парам.
    if not benchmark:
        checks["baseline_recall_50"] = summary.select((pl.col("baseline_hits") / pl.col("positives")).mean()).item()
        checks["pool_recall"] = summary.select((pl.col("pool_hits") / pl.col("positives")).mean()).item()
        checks["queries_without_positive_in_pool"] = summary.filter(pl.col("pool_hits") == 0).height
    save_json(config["results_dir"] / f"{fold}_candidates.json", checks)
    print(json.dumps(checks, ensure_ascii=False), flush=True)
