"""Одинаковые центры локаций для локальной оценки и benchmark."""

import polars as pl


def build_centers(corpus, destination):
    locations = pl.read_parquet(corpus / 'corpus_manifest.parquet', columns=['item_id', 'item_location_id'])
    coordinates = pl.read_parquet(corpus / 'corpus.parquet', columns=['item_id', 'item_latitude', 'item_longitude'])
    frame = locations.join(coordinates, on='item_id', how='left')
    centers = frame.group_by('item_location_id').agg(pl.col('item_latitude').median(),
                                                    pl.col('item_longitude').median())
    centers.sort('item_location_id').write_parquet(destination / 'location_centers.parquet')
