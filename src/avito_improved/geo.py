"""Одинаковые центры локаций для локальной оценки и benchmark."""

import polars as pl
import numpy as np

from avito_retrieval.common import stable_topk


def distance_km(latitude, longitude, center):
    lat, lon = center
    if lat is None or lon is None:
        return np.full(len(latitude), -1, dtype=np.float32)
    first = np.radians(lat)
    second = np.radians(latitude)
    delta = np.radians(longitude - lon)
    angular = np.sin((first - second) / 2) ** 2 + np.cos(first) * np.cos(second) * np.sin(delta / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(angular, 0, 1)))


def nearby_candidates(scores, latitude, longitude, center):
    distance = distance_km(latitude, longitude, center)
    local = np.flatnonzero((distance >= 0) & (distance <= 50))
    pieces = []
    for values in scores.values():
        chosen = local[stable_topk(values[local], 300)]
        pieces.append(chosen[values[chosen] > 0])
    return np.unique(np.concatenate(pieces)).astype(np.int32)


def build_centers(corpus, destination):
    locations = pl.read_parquet(corpus / 'corpus_manifest.parquet', columns=['item_id', 'item_location_id'])
    coordinates = pl.read_parquet(corpus / 'corpus.parquet', columns=['item_id', 'item_latitude', 'item_longitude'])
    frame = locations.join(coordinates, on='item_id', how='left')
    centers = frame.group_by('item_location_id').agg(pl.col('item_latitude').median(),
                                                    pl.col('item_longitude').median())
    centers.sort('item_location_id').write_parquet(destination / 'location_centers.parquet')
