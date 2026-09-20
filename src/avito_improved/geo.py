"""Одинаковые центры локаций для локальной оценки и benchmark."""

import polars as pl
import numpy as np

from avito_retrieval.common import stable_topk


def distance_km(latitude, longitude, center):
    """Считает расстояние по сфере до центра; без центра возвращает -1."""
    lat, lon = center
    # Неизвестный центр нельзя считать нулевым расстоянием.
    if lat is None or lon is None:
        return np.full(len(latitude), -1, dtype=np.float32)
    # Формула гаверсинуса переводит координаты в расстояние по поверхности Земли.
    first = np.radians(lat)
    second = np.radians(latitude)
    delta = np.radians(longitude - lon)
    angular = np.sin((first - second) / 2) ** 2 + np.cos(first) * np.cos(second) * np.sin(delta / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(angular, 0, 1)))


def nearby_candidates(scores, latitude, longitude, center):
    """Выбирает текстово подходящие объявления в пределах 50 км от центра."""
    distance = distance_km(latitude, longitude, center)
    # Сначала ограничиваем радиус, затем выбираем лучшие текстовые совпадения.
    local = np.flatnonzero((distance >= 0) & (distance <= 50))
    pieces = []
    # Каждое поле даёт свои кандидаты; после объединения убираем повторы.
    for values in scores.values():
        chosen = local[stable_topk(values[local], 300)]
        pieces.append(chosen[values[chosen] > 0])
    return np.unique(np.concatenate(pieces)).astype(np.int32)


def build_centers(corpus, destination):
    """Сохраняет медианные координаты локаций по признакам объявлений."""
    # География строится по признакам корпуса, без меток запросов.
    locations = pl.read_parquet(corpus / 'corpus_manifest.parquet', columns=['item_id', 'item_location_id'])
    coordinates = pl.read_parquet(corpus / 'corpus.parquet', columns=['item_id', 'item_latitude', 'item_longitude'])
    frame = locations.join(coordinates, on='item_id', how='left')
    # Медиана уменьшает влияние отдельных удалённых точек внутри локации.
    centers = frame.group_by('item_location_id').agg(pl.col('item_latitude').median(),
                                                    pl.col('item_longitude').median())
    centers.sort('item_location_id').write_parquet(destination / 'location_centers.parquet')
