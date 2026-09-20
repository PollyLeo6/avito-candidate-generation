"""Дополнительные кандидаты и признаки для второго эксперимента."""

import numpy as np
import polars as pl

from avito_retrieval.common import normalize, stable_topk, tokenize
from avito_ranker.candidates import CandidateSearch
from avito_ranker.features import FeatureBuilder, rank_values, token_coverage
from .history import QueryHistory


class ExtendedSearch:
    def __init__(self, corpus, index, history):
        self.base = CandidateSearch(corpus, index)
        self.builder = FeatureBuilder(corpus, self.base.manifest, self.base.indices)
        self.history = QueryHistory(history, self.base.manifest)
        self.microcats = self.base.manifest['item_microcat_id'].to_numpy()
        self.microcat_values, self.microcat_codes = np.unique(self.microcats, return_inverse=True)
        columns = ['item_id', 'item_title_raw', 'item_latitude', 'item_longitude']
        frame = self.base.manifest.join(pl.read_parquet(corpus / 'corpus.parquet', columns=columns),
                                        on='item_id', how='left').sort('doc_id')
        self.titles = [normalize(text) for text in frame['item_title_raw']]
        self.latitude = frame['item_latitude'].cast(pl.Float64).fill_null(float('nan')).to_numpy()
        self.longitude = frame['item_longitude'].cast(pl.Float64).fill_null(float('nan')).to_numpy()
        centers = pl.read_parquet(history / 'location_centers.parquet')
        self.centers = {location: (lat, lon) for location, lat, lon in centers.iter_rows()}

    def build(self, query, training=False):
        pool, scores, filters, channels, baseline, top = self.base.retrieve(query)
        original_pool = pool
        history_ids, history_score, microcats = self.history.search(query, exclude_self=training)
        local = self.base.by_location.get(int(query['search_location_id']), np.empty(0, dtype=int))
        filter_global = stable_topk(filters, 300)
        filter_local = local[stable_topk(filters[local], 500)]
        filter_global = filter_global[filters[filter_global] > 0]
        filter_local = filter_local[filters[filter_local] > 0]
        weights = np.array([microcats.get(int(value), 0) for value in self.microcat_values], dtype=np.float32)
        prior = weights[self.microcat_codes]
        text_score = sum(score / max(float(score.max()), 1e-6) for score in scores.values())
        guided = prior * (1 + text_score)
        guided_local = local[stable_topk(guided[local], 200)]
        guided_local = guided_local[guided[guided_local] > 0]
        pool = np.unique(np.concatenate([pool, history_ids, filter_global, filter_local, guided_local])).astype(np.int32)
        matrix, names = self.builder.build(query, pool, scores, filters, channels, baseline)
        extra = {
            'history_score': history_score[pool],
            'history_microcat': prior[pool],
            'filter_global_rr': rank_values(pool, filter_global),
            'filter_local_rr': rank_values(pool, filter_local),
            'description_coverage': token_coverage(self.base.indices['description'],
                                                   set(tokenize(query['search_query'])), pool),
        }
        text = normalize(query['search_query'])
        extra['query_in_title'] = np.array([float(bool(text) and text in self.titles[i]) for i in pool])
        center = self.centers.get(query['search_location_id'], (None, None))
        lat, lon = center
        distance = np.full(len(pool), -1, dtype=np.float32)
        if lat is not None and lon is not None:
            first = np.radians(lat)
            second = np.radians(self.latitude[pool])
            delta = np.radians(self.longitude[pool] - lon)
            angular = np.sin((first - second) / 2) ** 2 + np.cos(first) * np.cos(second) * np.sin(delta / 2) ** 2
            distance = 6371 * 2 * np.arcsin(np.sqrt(np.clip(angular, 0, 1)))
        extra['log_distance_km'] = np.where(np.isfinite(distance) & (distance >= 0),
                                           np.log1p(np.maximum(distance, 0)), -1)
        return pool, np.column_stack([matrix, *extra.values()]).astype(np.float32), names + list(extra), original_pool
