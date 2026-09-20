"""Кандидаты BM25, история запросов и 38 признаков модели ранжирования."""

import numpy as np
import polars as pl

from avito_retrieval.common import normalize, stable_topk, tokenize
from avito_ranker.candidates import CandidateSearch
from avito_ranker.features import FeatureBuilder, rank_values, token_coverage
from .history import QueryHistory
from .geo import distance_km, nearby_candidates


class ExtendedSearch:
    """Дополняет BM25 историей запросов, фильтрами и близкими локациями."""
    def __init__(self, corpus, index, history, nearby=False):
        """Загружает индексы и выравнивает признаки по порядку объявлений корпуса."""
        self.nearby = nearby
        # Исходный поиск и его 31 признак остаются общей основой обеих моделей.
        self.base = CandidateSearch(corpus, index)
        self.builder = FeatureBuilder(corpus, self.base.manifest, self.base.indices)
        self.history = QueryHistory(history, self.base.manifest)
        # Коды подкатегорий позволяют быстро перенести веса истории на весь корпус.
        self.microcats = self.base.manifest['item_microcat_id'].to_numpy()
        self.microcat_values, self.microcat_codes = np.unique(self.microcats, return_inverse=True)
        # Все массивы выравниваются по doc_id, чтобы признаки относились к нужным объявлениям.
        columns = ['item_id', 'item_title_raw', 'item_latitude', 'item_longitude']
        frame = self.base.manifest.join(pl.read_parquet(corpus / 'corpus.parquet', columns=columns),
                                        on='item_id', how='left').sort('doc_id')
        self.titles = [normalize(text) for text in frame['item_title_raw']]
        self.latitude = frame['item_latitude'].cast(pl.Float64).fill_null(float('nan')).to_numpy()
        self.longitude = frame['item_longitude'].cast(pl.Float64).fill_null(float('nan')).to_numpy()
        # Одинаковые центры используются при локальной оценке и итоговом поиске.
        centers = pl.read_parquet(history / 'location_centers.parquet')
        self.centers = {location: (lat, lon) for location, lat, lon in centers.iter_rows()}

    def build(self, query, training=False):
        """Возвращает кандидатов, 38 признаков и исходный пул для сравнения."""
        pool, scores, filters, channels, baseline, top = self.base.retrieve(query)
        # Сохраняем исходный пул отдельно для честного сравнения и смешивания выдач.
        original_pool = pool
        # При обучении история не видит положительные пары собственной группы.
        history_ids, history_score, microcats = self.history.search(query, exclude_self=training)
        local = self.base.by_location.get(int(query['search_location_id']), np.empty(0, dtype=int))
        # Фильтры добавляют подходящие объявления, даже если текст запроса слишком общий.
        filter_global = stable_topk(filters, 300)
        filter_local = local[stable_topk(filters[local], 500)]
        filter_global = filter_global[filters[filter_global] > 0]
        filter_local = filter_local[filters[filter_local] > 0]
        # Подкатегории похожих запросов направляют дополнительный поиск в той же локации.
        weights = np.array([microcats.get(int(value), 0) for value in self.microcat_values], dtype=np.float32)
        prior = weights[self.microcat_codes]
        # Нормировка не даёт одному текстовому полю подавить остальные своей шкалой.
        text_score = sum(score / max(float(score.max()), 1e-6) for score in scores.values())
        guided = prior * (1 + text_score)
        guided_local = local[stable_topk(guided[local], 200)]
        guided_local = guided_local[guided[guided_local] > 0]
        # Кандидаты из всех каналов объединяются без повторов.
        pool = np.unique(np.concatenate([pool, history_ids, filter_global, filter_local, guided_local])).astype(np.int32)
        center = self.centers.get(query['search_location_id'], (None, None))
        # Расширение по расстоянию применяется только для выбранного варианта поиска.
        if self.nearby:
            extra_ids = nearby_candidates(scores, self.latitude, self.longitude, center)
            pool = np.union1d(pool, extra_ids).astype(np.int32)
        # К 31 исходному признаку добавляются семь признаков истории, фильтров и расстояния.
        matrix, names = self.builder.build(query, pool, scores, filters, channels, baseline)
        extra = {
            'history_score': history_score[pool],
            'history_microcat': prior[pool],
            'filter_global_rr': rank_values(pool, filter_global),
            'filter_local_rr': rank_values(pool, filter_local),
            'description_coverage': token_coverage(self.base.indices['description'],
                                                   set(tokenize(query['search_query'])), pool),
        }
        # Точное вхождение дополняет пословные совпадения.
        text = normalize(query['search_query'])
        extra['query_in_title'] = np.array([float(bool(text) and text in self.titles[i]) for i in pool])
        # Логарифм сжимает диапазон расстояний; -1 отмечает неизвестную географию.
        distance = distance_km(self.latitude[pool], self.longitude[pool], center)
        extra['log_distance_km'] = np.where(np.isfinite(distance) & (distance >= 0),
                                           np.log1p(np.maximum(distance, 0)), -1)
        return pool, np.column_stack([matrix, *extra.values()]).astype(np.float32), names + list(extra), original_pool
