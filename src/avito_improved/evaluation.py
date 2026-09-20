"""Одна итоговая оценка на новом holdout и формирование ответа."""

import json

import numpy as np
import polars as pl
from catboost import CatBoostRanker, Pool

from avito_retrieval.common import file_hash, save_json, stable_topk
from avito_retrieval.submission import validate_answer
from .protocol import check_frozen
from .selection import top_indices


def score(config, fold):
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    frame = pl.read_parquet(work / f'{fold}.parquet')
    names = json.loads((work / 'feature_names.json').read_text('utf-8'))
    model = CatBoostRanker().load_model(str(work / 'ranker.cbm'))
    predictions = model.predict(Pool(frame.select(names).to_numpy(), feature_names=names), thread_count=2)
    summary = pl.read_parquet(work / f'{fold}_queries.parquet')
    return frame, predictions, summary


def evaluate(config):
    check_frozen(config)
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    frame, predictions, summary = score(config, 'fresh')
    numbers = frame['query_number'].to_numpy()
    labels = frame['label'].to_numpy()
    legacy = frame['original_candidate'].to_numpy()
    old_scores = frame['old_score'].to_numpy()
    lexical_scores = frame['rrf'].to_numpy()
    head = json.loads((work / 'selected.json').read_text('utf-8')).get('new_head', 50)
    boundaries = np.r_[0, np.flatnonzero(np.diff(numbers)) + 1, len(frame)]
    denominators = summary['positives'].to_numpy()
    original = np.zeros(len(summary))
    improved = np.zeros(len(summary))
    lexical = np.zeros(len(summary))
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        number = numbers[start]
        chosen = top_indices(predictions[start:stop], old_scores[start:stop], legacy[start:stop], head) + start
        improved[number] = labels[chosen].sum() / denominators[number]
        old_rows = np.flatnonzero(legacy[start:stop]) + start
        chosen = old_rows[stable_topk(old_scores[old_rows], 50)]
        original[number] = labels[chosen].sum() / denominators[number]
        chosen = stable_topk(lexical_scores[start:stop], 50) + start
        lexical[number] = labels[chosen].sum() / denominators[number]
    rng = np.random.default_rng(20260920)
    samples = rng.integers(0, len(summary), size=(2000, len(summary)))
    delta = improved - original
    report = {'queries': len(summary), 'original_recall_50': float(original.mean()),
              'bm25_rrf_recall_50': float(lexical.mean()),
              'improved_recall_50': float(improved.mean()), 'delta': float(delta.mean()),
              'delta_ci95': np.quantile(delta[samples].mean(axis=1), [0.025, 0.975]).tolist(),
              'pool_recall': summary.select((pl.col('pool_hits') / pl.col('positives')).mean()).item(),
              'model_sha256': file_hash(work / 'ranker.cbm'),
              'selected_before_fresh_evaluation': True}
    save_json(work / 'fresh_metrics.json', report)
    summary.with_columns(pl.Series('original_recall', original), pl.Series('improved_recall', improved)).write_parquet(
        work / 'fresh_per_query.parquet')
    print(json.dumps(report, indent=2), flush=True)
    return report


def predict(config):
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    frame, predictions, summary = score(config, 'benchmark')
    head = json.loads((work / 'selected.json').read_text('utf-8')).get('new_head', 50)
    numbers = frame['query_number'].to_numpy()
    old_scores = frame['old_score'].to_numpy()
    original = frame['original_candidate'].to_numpy()
    boundaries = np.r_[0, np.flatnonzero(np.diff(numbers)) + 1, len(frame)]
    selected = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        selected.extend((top_indices(predictions[start:stop], old_scores[start:stop], original[start:stop], head) + start).tolist())
    top = frame.select('query_number', 'doc_id')[selected]
    items = pl.read_parquet(config['work_dir'] / 'benchmark/corpus_manifest.parquet')
    top = top.join(items.select('doc_id', 'item_id'), on='doc_id', how='left', maintain_order='left')
    answer = top.group_by('query_number', maintain_order=True).agg(pl.col('item_id').str.join(' ').alias('answer'))
    answer = summary.join(answer, on='query_number').select(pl.col('context_id').alias('query_id'), 'answer').sort('query_id')
    path = work / 'answer.csv'
    temporary = path.with_suffix('.partial.csv')
    answer.write_csv(temporary, line_terminator="\r\n")
    report = validate_answer(temporary, config['data_dir'] / 'benchmark_queries.parquet',
                              config['data_dir'] / 'benchmark_items.parquet')
    temporary.replace(path)
    report['model_sha256'] = file_hash(work / 'ranker.cbm')
    save_json(work / 'submission_check.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report
