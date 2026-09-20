"""Подготовка обучающей истории и отдельного holdout до выбора модели."""

import json
import hashlib
import tempfile

from catboost import CatBoostRanker

import polars as pl

from avito_ranker.prepare import prepare, sample_contexts
from avito_ranker.freeze import artifact_hash
from avito_retrieval.common import query_columns, file_hash, save_json
from avito_retrieval.split import make_contexts
from .history import build_history
from .geo import build_centers


def prepare_experiment(config):
    prepare(config)
    base = config['work_dir'] / 'validation'
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    work.mkdir(parents=True, exist_ok=True)
    build_history(config['data_dir'], base, work / 'history')
    build_centers(base, work / 'history')
    reserve_holdout(config)


def reserve_holdout(config):
    base = config['work_dir'] / 'validation'
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    reserved = work / 'reserved'
    reserved.mkdir(parents=True, exist_ok=True)
    old = pl.read_parquet(base / 'holdout_queries.parquet')['strict_group']
    groups = pl.read_parquet(base / 'all_groups.parquet').filter(pl.col('fold') == 'holdout')
    groups = groups.filter(~pl.col('strict_group').is_in(old.implode()))
    train = pl.read_parquet(config['data_dir'] / 'train.parquet', columns=query_columns + ['item_id'])
    contexts = make_contexts(train).drop('fold').join(groups, on='group')
    queries = sample_contexts(contexts, 'holdout', 1000).sort('context_id')
    training = pl.read_parquet(work / 'history/pairs.parquet')['strict_group']
    dev = pl.read_parquet(base / 'dev_queries.parquet')['strict_group']
    if set(queries['strict_group']) & (set(old) | set(training) | set(dev)):
        raise ValueError('Новый holdout пересекается с обучением или прежней оценкой')
    queries.write_parquet(reserved / 'fresh_queries.parquet')
    qrels = train.join(queries.select(query_columns + ['context_id']), on=query_columns)
    qrels.select('context_id', 'item_id').unique().sort('context_id', 'item_id').write_parquet(reserved / 'fresh_qrels.parquet')
    save_json(work / 'fresh_reservation.json', {
        'queries': len(queries), 'previously_evaluated_groups': 0,
        'history_training_overlap': 0, 'ranker_training_overlap': 0,
        'selection_uses_labels': False, 'queries_sha256': file_hash(reserved / 'fresh_queries.parquet')})


def fingerprints(config):
    root = config['project_dir']
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    files = [work / 'ranker.cbm', work / 'selected.json', work / 'feature_names.json',
             work / 'fresh_reservation.json', work / 'leakage_audit.json', root / 'results/ranker.cbm']
    files += sorted((work / 'history').glob('*'))
    files += sorted((root / 'src').glob('*/*.py'))
    hashes = {str(path.relative_to(root)).replace('\\', '/') if path.is_relative_to(root) else path.name:
              artifact_hash(path) for path in files if path.is_file() and path != work / 'ranker.cbm'}
    model = CatBoostRanker().load_model(str(work / 'ranker.cbm'))
    with tempfile.TemporaryDirectory() as directory:
        from pathlib import Path
        path = Path(directory) / 'model.json'
        model.save_model(str(path), format='json')
        values = json.loads(path.read_text('utf-8'))
    values.pop('model_info', None)
    hashes['selected_model_function'] = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    for name in ['train.parquet', 'benchmark_queries.parquet', 'benchmark_items.parquet']:
        hashes['input/' + name] = file_hash(config['data_dir'] / name)
    return hashes


def freeze(config):
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    path = work / 'frozen.json'
    current = fingerprints(config)
    if path.exists() and json.loads(path.read_text('utf-8'))['files'] != current:
        raise ValueError('Зафиксированный эксперимент изменился; повторный подбор по holdout запрещён')
    save_json(path, {'files': current, 'selection_uses': 'dev only', 'fresh_holdout_queries': 1000})


def check_frozen(config):
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    path = work / 'frozen.json'
    if not path.exists() or json.loads(path.read_text('utf-8'))['files'] != fingerprints(config):
        raise ValueError('Перед оценкой нужен неизменный зафиксированный эксперимент')
