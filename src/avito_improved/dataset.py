"""Признаки для train, dev, нового holdout и итоговых запросов."""

import json
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from catboost import CatBoostRanker

from avito_retrieval.common import stable_topk
from .retrieval import ExtendedSearch


def create_dataset(config, fold):
    base = config['work_dir']
    work = config.get('quality_dir', base / 'quality')
    work.mkdir(parents=True, exist_ok=True)
    nearby = fold == 'dev_expanded'
    source_fold = 'dev' if nearby else fold
    if fold in ['fresh', 'benchmark'] and (work / 'selected.json').exists():
        nearby = json.loads((work / 'selected.json').read_text('utf-8')).get('nearby_candidates', False)
    kind = 'benchmark' if fold == 'benchmark' else 'validation'
    search = ExtendedSearch(base / kind, base / (kind + '_indices'), work / 'history', nearby=nearby)
    if fold == 'benchmark':
        queries = pl.read_parquet(config['data_dir'] / 'benchmark_queries.parquet').rename({'query_id': 'context_id'}).sort('context_id')
        qrels = {}
    else:
        directory = work / 'reserved' if fold == 'fresh' else base / 'validation'
        queries = pl.read_parquet(directory / f'{source_fold}_queries.parquet')
        gold = pl.read_parquet(directory / f'{source_fold}_qrels.parquet')
        qrels = dict(gold.group_by('context_id').agg(pl.col('item_id')).iter_rows())
    by_id = {value: i for i, value in enumerate(search.base.manifest['item_id'])}
    old_model = CatBoostRanker().load_model(str(config['project_dir'] / 'results/ranker.cbm'))
    pieces, summary = [], []
    writer = None
    rng = np.random.default_rng(20260920)
    started = time.perf_counter()
    destination = work / f'{fold}.parquet'
    temporary = destination.with_suffix('.partial.parquet')
    try:
        for number, query in enumerate(queries.iter_rows(named=True)):
            pool, matrix, names, original_pool = search.build(query, training=(fold == 'train'))
            positive = {by_id[item] for item in qrels.get(query['context_id'], [])}
            labels = np.isin(pool, list(positive)).astype(np.uint8)
            old_scores = old_model.predict(matrix[:, :31], thread_count=2)
            summary.append({'query_number': number, 'context_id': query['context_id'],
                            'positives': len(positive), 'pool_hits': int(labels.sum()), 'pool_size': len(pool)})
            if fold == 'train':
                if not labels.any():
                    continue
                negatives = np.flatnonzero(labels == 0)
                hard = negatives[stable_topk(old_scores[negatives], 80)]
                other = np.setdiff1d(negatives, hard)
                random = rng.choice(other, min(80, len(other)), replace=False)
                selected = np.sort(np.concatenate([np.flatnonzero(labels), hard, random]))
                pool, matrix, labels, old_scores = pool[selected], matrix[selected], labels[selected], old_scores[selected]
            data = {'query_number': np.full(len(pool), number, dtype=np.uint32), 'doc_id': pool,
                    'label': labels, 'old_score': old_scores.astype(np.float32),
                    'original_candidate': np.isin(pool, original_pool)}
            data.update({name: matrix[:, i] for i, name in enumerate(names)})
            pieces.append(pl.DataFrame(data))
            if len(pieces) >= 40:
                table = pl.concat(pieces).to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
                writer.write_table(table)
                pieces.clear()
            if (number + 1) % 200 == 0:
                print(fold, number + 1, round(time.perf_counter() - started, 1), flush=True)
        if pieces:
            table = pl.concat(pieces).to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
            writer.write_table(table)
    finally:
        if writer:
            writer.close()
    temporary.replace(destination)
    summary = pl.DataFrame(summary)
    summary.write_parquet(work / f'{fold}_queries.parquet')
    (work / 'feature_names.json').write_text(json.dumps(names), encoding='utf-8')
    report = {'fold': fold, 'queries': len(queries), 'seconds': time.perf_counter() - started,
              'pool_recall': summary.select((pl.col('pool_hits') / pl.col('positives')).mean()).item() if qrels else None,
              'average_pool_size': summary['pool_size'].mean()}
    (work / f'{fold}_candidates.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
