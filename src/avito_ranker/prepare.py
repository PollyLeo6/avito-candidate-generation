"""Группы запросов и корпус, который не зависит от положительных меток."""

import itertools

import polars as pl

from avito_retrieval.common import file_hash, query_columns, save_json, stable_hash, tokenize
from avito_retrieval.corpus import write_texts
from avito_retrieval.data_check import inspect_data
from avito_retrieval.split import choose_queries, make_contexts


def strict_group(text):
    """Объединяет запросы с одинаковым набором нормализованных корней слов."""
    # Перестановка слов и русские окончания не разделяют близкие запросы.
    return " ".join(sorted(set(tokenize(text))))


def assign_fold(group, previously_seen):
    """Закрепляет группу за одной частью данных; просмотренные группы идут в dev."""
    if group in previously_seen:
        return "dev"
    # Стабильный хеш воспроизводит разбиение без зависимости от порядка строк.
    value = int(stable_hash(group, "ranking_split_v1")[:8], 16) % 100
    return "train" if value < 80 else ("dev" if value < 90 else "holdout")


def sample_contexts(contexts, fold, count):
    """Выбирает по одному контексту на группу и воспроизводимую выборку групп."""
    # Сначала выбираем представителя группы, затем сами группы.
    pool = contexts.filter(pl.col("fold") == fold).with_columns(
        pl.col("context_id").map_elements(
            lambda value: stable_hash(value, "ranking_context_v1"), return_dtype=pl.String
        ).alias("order")
    )
    pool = pool.sort("order").unique("strict_group", keep="first", maintain_order=True)
    pool = pool.with_columns(pl.col("strict_group").map_elements(
        lambda value: stable_hash(value, "ranking_sample_v1"), return_dtype=pl.String
    ).alias("order")).sort("order")
    # Молча уменьшать выборку нельзя: изменится протокол сравнения моделей.
    if pool.height < count:
        raise ValueError(f"В {fold} только {pool.height} независимых групп, нужно {count}")
    return pool.head(count).drop("order")


def prepare(config):
    """Создаёт разбиение запросов, положительные пары и два корпуса объявлений."""
    # Проверяем исходные файлы до создания производных данных.
    data_dir, work = config["data_dir"], config["work_dir"]
    inspect_data(data_dir, config["results_dir"])
    validation = work / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    item_columns = ["item_id", "item_location_id", "item_category_id", "item_microcat_id"]
    train = pl.read_parquet(data_dir / "train.parquet", columns=query_columns + item_columns)
    # Ранее просмотренные запросы не могут стать независимым holdout.
    contexts = make_contexts(train)
    old_evaluation = choose_queries(contexts, 2452)
    seen = {strict_group(text) for text in old_evaluation["search_query"]}
    contexts = contexts.with_columns(pl.col("search_query").map_elements(
        strict_group, return_dtype=pl.String
    ).alias("strict_group"))
    contexts = contexts.with_columns(pl.col("strict_group").map_elements(
        lambda group: assign_fold(group, seen), return_dtype=pl.String
    ).alias("fold"))
    # Отбор запросов зависит от групп и хешей, а не от числа положительных пар.
    selected = {
        fold: sample_contexts(contexts, fold, config[fold + "_queries"])
        for fold in ["train", "dev", "holdout"]
    }
    # Проверяем границы выборок до извлечения их разметки.
    groups = {fold: set(frame["strict_group"]) for fold, frame in selected.items()}
    for left, right in itertools.combinations(groups, 2):
        if groups[left] & groups[right]:
            raise ValueError(f"Пересечение {left} и {right}")
    if groups["holdout"] & seen:
        raise ValueError("В holdout попали ранее просмотренные группы")
    # Контекст включает фильтры и локацию: одного текста для соединения недостаточно.
    for fold, queries in selected.items():
        qrels = train.join(queries.select(query_columns + ["context_id"]), on=query_columns)
        qrels = qrels.select("context_id", "item_id").unique().sort("context_id", "item_id")
        if set(queries["context_id"]) != set(qrels["context_id"]):
            raise ValueError("Потеряна разметка запроса")
        queries.sort("context_id").write_parquet(validation / f"{fold}_queries.parquet")
        qrels.write_parquet(validation / f"{fold}_qrels.parquet")
    # Полный список групп нужен для истории запросов с тем же разбиением.
    contexts.select("group", "strict_group", "fold").unique().write_parquet(
        validation / "all_groups.parquet"
    )
    # Берём все доступные объявления. Разметка не решает, кто попадёт в корпус.
    benchmark = pl.read_parquet(data_dir / "benchmark_items.parquet", columns=item_columns)
    manifest = pl.concat([benchmark, train.select(item_columns)]).unique(
        "item_id", keep="first", maintain_order=True
    ).sort("item_id").with_row_index("doc_id")
    manifest.write_parquet(validation / "corpus_manifest.parquet")
    write_texts(data_dir, manifest, validation / "corpus.parquet")
    # Для отправки разрешён только корпус benchmark, без объявлений из train.
    benchmark_dir = work / "benchmark"
    benchmark_dir.mkdir(exist_ok=True)
    benchmark.sort("item_id").with_row_index("doc_id").write_parquet(
        benchmark_dir / "corpus_manifest.parquet"
    )
    # Исходный порядок строк может отличаться от doc_id; индекс сопоставляет по item_id.
    import shutil
    shutil.copyfile(data_dir / "benchmark_items.parquet", benchmark_dir / "corpus.parquet")
    # Хеши связывают отчёт с точными входными файлами и выбранными запросами.
    sources = {name: file_hash(data_dir / name) for name in
               ["train.parquet", "benchmark_queries.parquet", "benchmark_items.parquet"]}
    report = {
        "corpus_items": manifest.height,
        "corpus_rule": "all unique items; benchmark features take precedence on duplicates",
        "strict_group": "sorted unique Snowball tokens; punctuation/case/word order ignored",
        "groups_disjoint": True,
        "holdout_previously_seen_groups": 0,
        "query_selection_uses_labels": False,
        "corpus_selection_uses_labels": False,
        "query_counts": {fold: frame.height for fold, frame in selected.items()},
        "all_group_counts": contexts.select("strict_group", "fold").unique()
            .group_by("fold").len().sort("fold").to_dicts(),
        "sources": sources,
        "split_hashes": {path.name: file_hash(path) for path in validation.glob("*_queries.parquet")},
        "limitations": ["observed positives only", "semantic paraphrases can cross groups",
                        "515895-item local corpus is larger than benchmark; scores are not comparable to v1"],
    }
    save_json(config["results_dir"] / "split_audit.json", report)
    print(report, flush=True)
