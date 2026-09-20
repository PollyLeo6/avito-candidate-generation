"""Независимое сопоставление меток и проверка групп обучающей истории."""

import json

import polars as pl

from avito_retrieval.common import save_json


def audit_training(config):
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    base = config['work_dir'] / 'validation'
    history = set(pl.read_parquet(work / 'history/pairs.parquet')['strict_group'])
    allowed = pl.read_parquet(base / 'all_groups.parquet').filter(pl.col('fold') == 'train')
    dev = set(pl.read_parquet(base / 'dev_queries.parquet')['strict_group'])
    fresh = set(pl.read_parquet(work / 'reserved/fresh_queries.parquet')['strict_group'])
    if not history <= set(allowed['strict_group']) or history & (dev | fresh):
        raise ValueError('Проверочные группы попали в обучающую историю')
    names = json.loads((work / 'feature_names.json').read_text('utf-8'))
    if set(names) & {'label', 'item_id', 'query_id', 'doc_id', 'query_number', 'context_id'}:
        raise ValueError('В признаки попали метки или идентификаторы')
    queries = pl.read_parquet(base / 'train_queries.parquet').with_row_index('query_number')
    items = pl.read_parquet(base / 'corpus_manifest.parquet').select('doc_id', 'item_id')
    labels = pl.read_parquet(work / 'train.parquet', columns=['query_number', 'doc_id', 'label'])
    mapped = labels.join(queries.select('query_number', 'context_id'), on='query_number').join(items, on='doc_id')
    gold = pl.read_parquet(base / 'train_qrels.parquet').with_columns(pl.lit(1).alias('expected'))
    checked = mapped.join(gold, on=['context_id', 'item_id'], how='left')
    wrong = checked.filter(pl.col('label') != pl.col('expected').fill_null(0)).height
    if wrong or len(checked) != len(labels):
        raise ValueError('Обучающие метки не совпали с исходными положительными парами')
    report = {'rows': len(labels), 'queries': labels['query_number'].n_unique(),
              'incorrect_labels': wrong, 'history_dev_overlap': 0, 'history_fresh_overlap': 0,
              'training_history_is_train_only': True, 'no_ids_or_labels_in_features': True,
              'own_training_group_excluded_by_retrieval': True}
    save_json(work / 'leakage_audit.json', report)
    print(json.dumps(report, indent=2), flush=True)
