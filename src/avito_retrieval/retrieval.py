"""BM25 по полям и несколько заранее заданных способов их объединить."""

import numpy as np

from .common import bm25s, rrf_scores, stable_topk, tokenize

# Эти варианты фиксируются до первого расчёта метрик.
variants = {
    "title_raw": {"title_raw": 1.0},
    "title_stem": {"title": 1.0},
    "rrf_equal": {"title": 1.0, "description": 1.0, "params": 1.0},
    "rrf_title": {"title": 2.0, "description": 1.0, "params": 0.5},
    "rrf_filters": {"title": 2.0, "description": 1.0, "params": 0.5, "filters": 0.25},
}


def load_indices(directory):
    """Открывает индексы четырёх текстовых каналов через отображение файлов в память."""
    return {
        name: bm25s.BM25.load(str(directory / name), mmap=True)
        for name in ["title_raw", "title", "description", "params"]
    }


def field_scores(index, text, stemming=True):
    """Возвращает BM25-оценку каждого объявления для одного текста запроса."""
    # Повтор одного слова в запросе не должен давать ему дополнительный вес.
    tokens = list(dict.fromkeys(tokenize(text, stemming=stemming)))
    ids = index.get_tokens_ids(tokens)
    # Неизвестные словарю токены не создают случайных совпадений.
    if not ids:
        return np.zeros(index.scores["num_docs"], dtype=np.float32)
    return index.get_scores_from_ids(ids)


def candidates(indices, query, depth=300):
    """Собирает оценки и выдачи отдельных полей до их объединения."""
    # Текст запроса сравниваем с каждым полем, а фильтры только с параметрами объявления.
    scores = {
        name: field_scores(index, query["search_query"], name != "title_raw")
        for name, index in indices.items()
    }
    scores["filters"] = field_scores(indices["params"], query["search_infm_params_text"])
    ranked = {name: stable_topk(score, depth) for name, score in scores.items()}
    # Нулевые совпадения не получают голос в RRF.
    channels = {name: ids[scores[name][ids] > 0] for name, ids in ranked.items()}
    return scores, ranked, channels


def combine(name, scores, ranked, channels):
    """Возвращает выдачу выбранного поля или взвешенное объединение RRF."""
    # Однополевые варианты нужны для сравнения с более сложным объединением.
    if name in ["title_raw", "title_stem"]:
        field = "title_raw" if name == "title_raw" else "title"
        return ranked[field]
    # Для RRF учитываем позиции в каналах, затем выбираем общий топ-300.
    n_docs = len(scores["title"])
    fused = rrf_scores(channels, variants[name], n_docs)
    return stable_topk(fused, 300)
