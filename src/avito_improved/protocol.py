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
    """Готовит разбиение, историю, географию и отдельный holdout."""
    # Общее разбиение и корпус готовятся тем же кодом, что и для исходной модели.
    prepare(config)
    base = config['work_dir'] / 'validation'
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    work.mkdir(parents=True, exist_ok=True)
    # История использует train, центры - только координаты объявлений корпуса.
    build_history(config['data_dir'], base, work / 'history')
    build_centers(base, work / 'history')
    reserve_holdout(config)


def reserve_holdout(config):
    """Откладывает 1 000 запросов из ещё не использованных групп holdout."""
    base = config['work_dir'] / 'validation'
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    reserved = work / 'reserved'
    reserved.mkdir(parents=True, exist_ok=True)
    # Убираем группы из прежней оценки, прежде чем отложить новый holdout.
    old = pl.read_parquet(base / 'holdout_queries.parquet')['strict_group']
    groups = pl.read_parquet(base / 'all_groups.parquet').filter(pl.col('fold') == 'holdout')
    groups = groups.filter(~pl.col('strict_group').is_in(old.implode()))
    # Отбор запросов определяется контекстом и группой, а не успешностью предсказаний.
    train = pl.read_parquet(config['data_dir'] / 'train.parquet', columns=query_columns + ['item_id'])
    contexts = make_contexts(train).drop('fold').join(groups, on='group')
    queries = sample_contexts(contexts, 'holdout', 1000).sort('context_id')
    # Дополнительно проверяем отсутствие пересечений с train, dev и прежним holdout.
    training = pl.read_parquet(work / 'history/pairs.parquet')['strict_group']
    dev = pl.read_parquet(base / 'dev_queries.parquet')['strict_group']
    if set(queries['strict_group']) & (set(old) | set(training) | set(dev)):
        raise ValueError('Новый holdout пересекается с обучением или прежней оценкой')
    queries.write_parquet(reserved / 'fresh_queries.parquet')
    # После выбора запросов собираем все их положительные пары для знаменателя Recall.
    qrels = train.join(queries.select(query_columns + ['context_id']), on=query_columns)
    qrels.select('context_id', 'item_id').unique().sort('context_id', 'item_id').write_parquet(reserved / 'fresh_qrels.parquet')
    # Контрольная сумма фиксирует состав отложенных запросов до подбора модели.
    save_json(work / 'fresh_reservation.json', {
        'queries': len(queries), 'previously_evaluated_groups': 0,
        'history_training_overlap': 0, 'ranker_training_overlap': 0,
        'selection_uses_labels': False, 'queries_sha256': file_hash(reserved / 'fresh_queries.parquet')})


def fingerprints(config):
    """Считает контрольные суммы кода, данных, истории и содержимого модели."""
    root = config['project_dir']
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    # Проверка охватывает параметры, историю, исходную модель и весь код расчёта.
    files = [work / 'ranker.cbm', work / 'selected.json', work / 'feature_names.json',
             work / 'fresh_reservation.json', work / 'leakage_audit.json', root / 'results/ranker.cbm']
    files += sorted((work / 'history').glob('*'))
    files += sorted((root / 'src').glob('*/*.py'))
    # Для JSON учитываем содержимое, а не различия форматирования между ОС.
    hashes = {str(path.relative_to(root)).replace('\\', '/') if path.is_relative_to(root) else path.name:
              artifact_hash(path) for path in files if path.is_file() and path != work / 'ranker.cbm'}
    # CatBoost добавляет служебные метаданные; сравниваем сами деревья и границы признаков.
    model = CatBoostRanker().load_model(str(work / 'ranker.cbm'))
    with tempfile.TemporaryDirectory() as directory:
        from pathlib import Path
        path = Path(directory) / 'model.json'
        model.save_model(str(path), format='json')
        values = json.loads(path.read_text('utf-8'))
    # Время обучения и служебный GUID не меняют функцию предсказания.
    values.pop('model_info', None)
    hashes['selected_model_function'] = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    # Подмена исходных данных также должна останавливать повторную оценку.
    for name in ['train.parquet', 'benchmark_queries.parquet', 'benchmark_items.parquet']:
        hashes['input/' + name] = file_hash(config['data_dir'] / name)
    return hashes


def freeze(config):
    """Фиксирует эксперимент после выбора по dev и до оценки на holdout."""
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    path = work / 'frozen.json'
    # Существующий снимок нельзя перезаписать изменённым экспериментом.
    current = fingerprints(config)
    if path.exists() and json.loads(path.read_text('utf-8'))['files'] != current:
        raise ValueError('Зафиксированный эксперимент изменился; повторный подбор по holdout запрещён')
    save_json(path, {'files': current, 'selection_uses': 'dev only', 'fresh_holdout_queries': 1000})


def check_frozen(config):
    """Останавливает оценку, если зафиксированный эксперимент изменился."""
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    path = work / 'frozen.json'
    # Сравниваем все сохранённые отпечатки перед обращением к итоговому holdout.
    if not path.exists() or json.loads(path.read_text('utf-8'))['files'] != fingerprints(config):
        raise ValueError('Перед оценкой нужен неизменный зафиксированный эксперимент')
