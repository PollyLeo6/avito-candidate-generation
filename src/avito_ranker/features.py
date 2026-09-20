"""Признаки вычисляются без доступа к положительным парам."""

import numpy as np
import polars as pl

from avito_retrieval.common import tokenize


def token_sets(series):
    """Токенизирует повторяющиеся тексты один раз и возвращает наборы слов."""
    unique = {text: frozenset(tokenize(text)) for text in series.unique()}
    return [unique[text] for text in series]


def token_coverage(index, tokens, pool):
    """Считает долю слов запроса, найденных в поле каждого кандидата."""
    # Обратный индекс сразу даёт документы с нужным токеном без чтения всех текстов.
    counts = np.zeros(index.scores["num_docs"], dtype=np.float32)
    pointers = index.scores["indptr"]
    documents = index.scores["indices"]
    for token in index.get_tokens_ids(list(tokens)):
        counts[documents[pointers[token]:pointers[token + 1]]] += 1
    # Пустой запрос получает нулевое покрытие без деления на ноль.
    return counts[pool] / max(len(tokens), 1)


def rank_values(pool, channel):
    """Переносит обратные позиции канала в отсортированный общий пул."""
    result = np.zeros(len(pool), dtype=np.float32)
    if len(channel):
        # Для кандидата вне канала остаётся ноль; первое место получает единицу.
        locations = np.searchsorted(pool, channel)
        result[locations] = 1.0 / (1.0 + np.arange(len(channel)))
    return result


class FeatureBuilder:
    """Готовит признаки пары запрос-объявление без доступа к её метке."""

    def __init__(self, corpus_dir, manifest, indices):
        """Загружает свойства объявлений в том же порядке, что и поисковые индексы."""
        columns = ["item_id", "item_title_raw", "item_infm_params_text", "item_price",
                   "item_rating", "item_rating_reviews_count", "item_is_phone_hidden",
                   "item_is_message_forbidden"]
        frame = pl.read_parquet(corpus_dir / "corpus.parquet", columns=columns)
        # Исходный Parquet может иметь другой порядок строк, поэтому соединяем по ID.
        frame = manifest.join(frame, on="item_id", how="left").sort("doc_id")
        self.location = frame["item_location_id"].to_numpy()
        self.category = frame["item_category_id"].to_numpy()
        # Пропуски численных полей приводим к одному конечному значению.
        self.numeric = {}
        for name in ["item_price", "item_rating", "item_rating_reviews_count",
                     "item_is_phone_hidden", "item_is_message_forbidden"]:
            self.numeric[name] = frame[name].cast(pl.Float32).fill_null(-1).fill_nan(-1).to_numpy()
        # Эти свойства не зависят от запроса и рассчитываются один раз.
        self.title_tokens = token_sets(frame["item_title_raw"])
        self.params_length = frame["item_infm_params_text"].str.len_chars().fill_null(0).to_numpy()
        self.params_index = indices["params"]

    def build(self, query, pool, scores, filter_scores, channels, baseline):
        """Возвращает численную матрицу и имена столбцов в неизменном порядке."""
        size = len(pool)
        values = {}
        # Оценки показывают силу совпадения, позиции позволяют сравнивать разные поля.
        for name, score in scores.items():
            values[name + "_score"] = score[pool]
            values[name + "_relative"] = score[pool] / max(float(score.max()), 1e-6)
            values[name + "_rr"] = rank_values(pool, channels[name])
            values[name + "_local_rr"] = rank_values(pool, channels[name + "_local"])
        # Фильтры оцениваются отдельно, чтобы уточнения не терялись за основным текстом.
        values["filter_score"] = filter_scores[pool]
        values["filter_relative"] = filter_scores[pool] / max(float(filter_scores.max()), 1e-6)
        values["rrf"] = baseline[pool]
        # География и категория остаются признаками, а не жёсткими запретами.
        values["same_location"] = self.location[pool] == query["search_location_id"]
        values["same_category"] = self.category[pool] == query["search_category"]
        values["delivery"] = np.full(size, query["search_is_delivery_search"])
        # Наборы токенов не дают повторению слова искусственно повысить покрытие.
        tokens = set(tokenize(query["search_query"]))
        filters = set(tokenize(query["search_infm_params_text"]))
        values["query_words"] = np.full(size, len(tokens))
        values["filter_words"] = np.full(size, len(filters))
        # Покрытие измеряет найденную долю запроса, Jaccard также учитывает лишние слова.
        for field, documents in [("title", self.title_tokens)]:
            lengths = np.fromiter((len(documents[i]) for i in pool), dtype=np.float32)
            overlap = np.fromiter((len(tokens & documents[i]) for i in pool), dtype=np.float32)
            values[field + "_length"] = lengths
            values[field + "_coverage"] = overlap / max(len(tokens), 1)
            values[field + "_jaccard"] = overlap / np.maximum(lengths + len(tokens) - overlap, 1)
        # Параметры могут содержать услугу, которой нет в заголовке.
        values["params_length"] = self.params_length[pool]
        values["params_coverage"] = token_coverage(self.params_index, tokens, pool)
        values["filter_coverage"] = token_coverage(self.params_index, filters, pool)
        # Логарифм уменьшает разброс цены и числа отзывов; отрицательные значения обнуляем.
        for name, array in self.numeric.items():
            current = array[pool]
            values[name] = np.log1p(np.maximum(current, 0)) if name in [
                "item_price", "item_rating_reviews_count"
            ] else current
        # Порядок словаря задаёт одинаковый порядок признаков при обучении и предсказании.
        matrix = np.column_stack(list(values.values())).astype(np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("В признаках есть nan или infinity")
        return matrix, list(values)
