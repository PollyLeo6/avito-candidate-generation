"""Поиск похожих запросов по положительным парам обучающей части."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse

from avito_retrieval.common import group_key, tokenize, stable_topk


def build_history(data_dir, validation_dir, destination):
    groups = pl.read_parquet(validation_dir / 'all_groups.parquet')
    allowed = groups.filter(pl.col('fold') == 'train').select('group', 'strict_group').unique()
    pairs = pl.read_parquet(data_dir / 'train.parquet', columns=['search_query', 'item_id'])
    texts = pairs.select('search_query').unique().with_columns(
        pl.col('search_query').map_elements(group_key, return_dtype=pl.String).alias('group'))
    texts = texts.join(allowed, on='group', how='inner').select('search_query', 'strict_group')
    pairs = pairs.join(texts, on='search_query').select('strict_group', 'item_id').unique()
    rows = pairs.group_by('strict_group').agg(pl.col('item_id').sort()).sort('strict_group')
    token_counts = Counter(token for text in rows['strict_group'] for token in text.split())
    vocabulary = {token: i for i, token in enumerate(sorted(token_counts))}
    idf = np.array([np.log((1 + len(rows)) / (1 + token_counts[token])) + 1
                    for token in vocabulary], dtype=np.float32)
    row_ids, col_ids = [], []
    for i, text in enumerate(rows['strict_group']):
        tokens = text.split()
        row_ids.extend([i] * len(tokens))
        col_ids.extend(vocabulary[token] for token in tokens)
    matrix = sparse.csr_matrix((idf[col_ids], (row_ids, col_ids)),
                               shape=(len(rows), len(vocabulary)), dtype=np.float32)
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    matrix = sparse.diags(1 / np.maximum(norms, 1e-6)).dot(matrix).tocsr()
    destination.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(destination / 'pairs.parquet')
    categories = pl.read_parquet(validation_dir / 'corpus_manifest.parquet',
                                  columns=['item_id', 'item_microcat_id'])
    categories.filter(pl.col('item_id').is_in(pairs['item_id'].unique().implode())).sort('item_id').write_parquet(
        destination / 'item_categories.parquet')
    sparse.save_npz(destination / 'queries.npz', matrix)
    np.save(destination / 'idf.npy', idf)
    (destination / 'vocabulary.json').write_text(json.dumps(vocabulary, ensure_ascii=False), encoding='utf-8')
    audit = {'training_groups': len(rows), 'unique_pairs': len(pairs),
             'dev_groups_used': 0, 'holdout_groups_used': 0,
             'exclude_current_query_group_during_training': True}
    (destination / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(audit, flush=True)


class QueryHistory:
    def __init__(self, path, manifest):
        frame = pl.read_parquet(path / 'pairs.parquet')
        self.matrix = sparse.load_npz(path / 'queries.npz').tocsc()
        self.idf = np.load(path / 'idf.npy')
        self.vocabulary = json.loads((path / 'vocabulary.json').read_text('utf-8'))
        self.groups = frame['strict_group'].to_list()
        self.by_group = {value: i for i, value in enumerate(self.groups)}
        by_item = {value: i for i, value in enumerate(manifest['item_id'])}
        self.documents = [np.array([by_item[item] for item in items if item in by_item], dtype=np.int32)
                          for items in frame['item_id']]
        categories = dict(pl.read_parquet(path / 'item_categories.parquet').iter_rows())
        self.categories = [np.array([categories[item] for item in items]) for items in frame['item_id']]
        self.locations = manifest['item_location_id'].to_numpy()
        self.microcats = manifest['item_microcat_id'].to_numpy()
        self.n_docs = len(manifest)

    def search(self, query, exclude_self=True):
        tokens = sorted(set(tokenize(query['search_query'])))
        cols = [self.vocabulary[token] for token in tokens if token in self.vocabulary]
        direct = np.zeros(self.n_docs, dtype=np.float32)
        microcat_scores = {}
        if not cols:
            return np.empty(0, dtype=np.int32), direct, microcat_scores
        weights = self.idf[cols]
        weights = weights / max(np.linalg.norm(weights), 1e-6)
        similarity = np.asarray(self.matrix[:, cols].dot(weights)).ravel()
        own_group = self.by_group.get(' '.join(tokens))
        if exclude_self and own_group is not None:
            similarity[own_group] = 0
        neighbors = stable_topk(similarity, 40)
        for number in neighbors:
            score = float(similarity[number])
            if score < 0.35:
                continue
            documents = self.documents[number]
            weight = score ** 4
            np.maximum.at(direct, documents, weight)
            categories, counts = np.unique(self.categories[number], return_counts=True)
            for category, count in zip(categories, counts):
                microcat_scores[int(category)] = microcat_scores.get(int(category), 0) + weight * count / len(self.categories[number])
        if microcat_scores:
            maximum = max(microcat_scores.values())
            microcat_scores = {key: value / maximum for key, value in microcat_scores.items()}
        global_ids = stable_topk(direct, 200)
        local = np.flatnonzero(self.locations == query['search_location_id'])
        local_ids = local[stable_topk(direct[local], 200)]
        pool = np.union1d(global_ids, local_ids)
        return pool[direct[pool] > 0], direct, microcat_scores
