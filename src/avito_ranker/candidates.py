"""Общий поиск и отдельная ветка для локации запроса."""

import numpy as np
import polars as pl
import json

from avito_retrieval.common import bm25s, file_hash, rrf_scores, stable_topk
from avito_retrieval.retrieval import field_scores


class CandidateSearch:
    """Собирает кандидатов из общего и локального поиска по трём полям."""

    def __init__(self, corpus_dir, index_dir, global_depth=300, local_depth=150):
        """Проверяет совместимость индексов и готовит списки документов по локациям."""
        self.manifest = pl.read_parquet(corpus_dir / "corpus_manifest.parquet").sort("doc_id")
        # doc_id имеет смысл только внутри корпуса, на котором построен индекс.
        manifest_hash = file_hash(corpus_dir / "corpus_manifest.parquet")
        for name in ["title", "description", "params"]:
            metadata = json.loads((index_dir / name / "build_metadata.json").read_text("utf-8"))
            if metadata["corpus_manifest_sha256"] != manifest_hash:
                raise ValueError("Индекс относится к другому корпусу")
        # Отображение файлов в память позволяет не загружать три индекса целиком.
        self.indices = {name: bm25s.BM25.load(str(index_dir / name), mmap=True)
                        for name in ["title", "description", "params"]}
        self.n_docs = self.manifest.height
        self.locations = self.manifest["item_location_id"].to_numpy()
        self.global_depth = global_depth
        self.local_depth = local_depth
        # Эти позиции совпадают с порядком документов в массивах BM25.
        self.by_location = {}
        for location in np.unique(self.locations):
            self.by_location[int(location)] = np.flatnonzero(self.locations == location)

    def retrieve(self, query):
        """Возвращает общий пул, оценки каналов и исходный топ-50 без меток."""
        scores = {name: field_scores(index, query["search_query"])
                  for name, index in self.indices.items()}
        channels = {}
        local = self.by_location.get(int(query["search_location_id"]), np.empty(0, dtype=int))
        # Локальная выдача сохраняет кандидатов, потерявшихся в общем топе.
        for name, score in scores.items():
            ids = stable_topk(score, self.global_depth)
            channels[name] = ids[score[ids] > 0]
            ids = local[stable_topk(score[local], self.local_depth)]
            channels[name + "_local"] = ids[score[ids] > 0]
        # RRF объединяет позиции, не требуя одинакового масштаба BM25 по полям.
        baseline = rrf_scores(channels, {name: 1.0 for name in scores}, self.n_docs)
        baseline_top = stable_topk(baseline, 50)
        # Baseline входит в пул даже при полном отсутствии текстовых совпадений.
        pool = np.unique(np.concatenate([baseline_top, *channels.values()])).astype(np.int32)
        # Фильтры сравниваются с параметрами объявлений отдельно от текста запроса.
        filter_scores = field_scores(self.indices["params"], query["search_infm_params_text"])
        return pool, scores, filter_scores, channels, baseline, baseline_top
