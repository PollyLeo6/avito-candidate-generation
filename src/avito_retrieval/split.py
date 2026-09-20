"""Разбиение запросов без пересечения одинаковых формулировок."""

import polars as pl

from .common import fold_for_group, group_key, query_columns, query_key, stable_hash


def make_contexts(train):
    """Выделяет поисковые контексты и назначает часть выборки по текстовой группе."""
    # Повторные строки с одинаковыми признаками поиска остаются одним контекстом.
    contexts = train.select(query_columns).unique().sort(query_columns)
    rows = []
    # Один текст с разными фильтрами получает разные ID, но общую часть выборки.
    for row in contexts.iter_rows(named=True):
        group = group_key(row["search_query"])
        rows.append(
            {
                **row,
                "context_id": query_key(row),
                "group": group,
                "fold": fold_for_group(group),
            }
        )
    return pl.DataFrame(rows)


def choose_queries(contexts, count):
    """Выбирает по count независимых текстовых групп для dev и holdout."""
    selected = []
    for fold in ["dev", "holdout"]:
        # Хеш задаёт постоянный порядок отбора, не зависящий от порядка строк train.
        pool = contexts.filter(pl.col("fold") == fold)
        pool = pool.with_columns(
            pl.col("context_id")
            .map_elements(
                lambda value: stable_hash(value, "context_choice"),
                return_dtype=pl.String,
            )
            .alias("choice")
        )
        # Сначала один контекст на формулировку, затем выбор самих формулировок.
        pool = pool.sort("choice").unique("group", keep="first", maintain_order=True)
        pool = pool.with_columns(
            pl.col("group")
            .map_elements(lambda value: stable_hash(value, "sample"), return_dtype=pl.String)
            .alias("sample")
        )
        # Не заменяем недостающие группы повторными контекстами того же запроса.
        if pool.height < count:
            raise ValueError("Недостаточно независимых групп запросов")
        selected.append(pool.sort("sample").head(count).drop("choice", "sample"))
    return pl.concat(selected).sort(["fold", "context_id"])


def make_qrels(train, queries):
    """Собирает известные положительные пары для выбранных контекстов поиска."""
    # Соединяем по всему контексту, чтобы не смешать разные локации и фильтры.
    keys = queries.select(query_columns + ["context_id", "fold"])
    pairs = train.join(keys, on=query_columns, how="inner")
    # Повторный выбор одного item_id не увеличивает число релевантных объявлений.
    return (
        pairs.unique(["context_id", "item_id"])
        .select(["fold", "context_id", "item_id", "item_location_id", "item_microcat_id"])
        .sort(["fold", "context_id", "item_id"])
    )


def check_split(contexts, queries, qrels, count):
    """Проверяет независимость групп, наличие разметки и отсутствие повторных пар."""
    # Каждая текстовая группа должна встречаться в оценке только один раз.
    if queries["group"].n_unique() != 2 * count:
        raise ValueError("В оценку попали повторные группы запросов")
    # Группа из train не должна одновременно участвовать в локальной оценке.
    training = set(contexts.filter(pl.col("fold") == "train")["group"])
    if training & set(queries["group"]):
        raise ValueError("Пересечение train и evaluation")
    # Для Recall нужны непустые множества positives без повторов одного item_id.
    if set(queries["context_id"]) != set(qrels["context_id"]):
        raise ValueError("Есть запрос без разметки")
    if qrels.unique(["context_id", "item_id"]).height != qrels.height:
        raise ValueError("Повторные positives")


def describe_queries(queries, qrels):
    """Суммирует размеры частей, число positives и свойства запросов."""
    result = {}
    for fold in ["dev", "holdout"]:
        # Число positives показывает, сколько объявлений учитывается в Recall каждого запроса.
        subset = queries.filter(pl.col("fold") == fold)
        positives = qrels.filter(pl.col("fold") == fold)
        counts = positives.group_by("context_id").len()
        result[fold] = {
            "queries": subset.height,
            "positive_pairs": positives.height,
            "positive_items": positives["item_id"].n_unique(),
            "mean_positives": counts["len"].mean(),
            "max_positives": counts["len"].max(),
            "single_positive_queries": counts.filter(pl.col("len") == 1).height,
            "empty_filters_fraction": subset.filter(pl.col("search_infm_params_text") == "").height
            / subset.height,
            "mean_query_words": subset["search_query"].str.count_matches(r"\S+").mean(),
            "category_counts": subset.group_by("search_category")
            .len()
            .sort("search_category")
            .to_dicts(),
            "coverage": 1.0,
        }
    return result
