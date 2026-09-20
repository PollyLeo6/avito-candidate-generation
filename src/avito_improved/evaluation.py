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
    """Применяет сохранённую модель к признакам и возвращает оценки вместе с таблицами."""
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    frame = pl.read_parquet(work / f'{fold}.parquet')
    names = json.loads((work / 'feature_names.json').read_text('utf-8'))
    model = CatBoostRanker().load_model(str(work / 'ranker.cbm'))
    # Порядок колонок берём из сохранённого списка признаков модели.
    predictions = model.predict(Pool(frame.select(names).to_numpy(), feature_names=names), thread_count=2)
    summary = pl.read_parquet(work / f'{fold}_queries.parquet')
    return frame, predictions, summary


def evaluate(config):
    """Сравнивает зафиксированную модель с исходной на отдельном holdout."""
    # Не оцениваем holdout, если после выбора по dev менялись код или модель.
    check_frozen(config)
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    frame, predictions, summary = score(config, 'fresh')
    numbers = frame['query_number'].to_numpy()
    labels = frame['label'].to_numpy()
    legacy = frame['original_candidate'].to_numpy()
    old_scores = frame['old_score'].to_numpy()
    lexical_scores = frame['rrf'].to_numpy()
    head = json.loads((work / 'selected.json').read_text('utf-8')).get('new_head', 50)
    # Каждый запрос занимает непрерывный блок строк в таблице кандидатов.
    boundaries = np.r_[0, np.flatnonzero(np.diff(numbers)) + 1, len(frame)]
    # В знаменателе остаются и положительные объявления вне пула.
    denominators = summary['positives'].to_numpy()
    original = np.zeros(len(summary))
    improved = np.zeros(len(summary))
    lexical = np.zeros(len(summary))
    # Все варианты оцениваются на одних запросах с одинаковым знаменателем.
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        number = numbers[start]
        chosen = top_indices(predictions[start:stop], old_scores[start:stop], legacy[start:stop], head) + start
        improved[number] = labels[chosen].sum() / denominators[number]
        # Исходная модель получает только свой пул, без новых кандидатов.
        old_rows = np.flatnonzero(legacy[start:stop]) + start
        chosen = old_rows[stable_topk(old_scores[old_rows], 50)]
        original[number] = labels[chosen].sum() / denominators[number]
        chosen = stable_topk(lexical_scores[start:stop], 50) + start
        lexical[number] = labels[chosen].sum() / denominators[number]
    # Парный bootstrap пересэмплирует запросы и оценивает неопределённость прироста.
    rng = np.random.default_rng(20260920)
    samples = rng.integers(0, len(summary), size=(2000, len(summary)))
    delta = improved - original
    # Сохраняем средние метрики и оценки по запросам для разбора ошибок.
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
    """Собирает топ-50 для benchmark и сохраняет проверенный answer.csv."""
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    frame, predictions, summary = score(config, 'benchmark')
    head = json.loads((work / 'selected.json').read_text('utf-8')).get('new_head', 50)
    numbers = frame['query_number'].to_numpy()
    old_scores = frame['old_score'].to_numpy()
    original = frame['original_candidate'].to_numpy()
    boundaries = np.r_[0, np.flatnonzero(np.diff(numbers)) + 1, len(frame)]
    # Для каждого запроса применяем зафиксированное смешивание двух выдач.
    selected = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        selected.extend((top_indices(predictions[start:stop], old_scores[start:stop], original[start:stop], head) + start).tolist())
    # Преобразуем номера строк обратно в исходные item_id и сохраняем порядок выдачи.
    top = frame.select('query_number', 'doc_id')[selected]
    items = pl.read_parquet(config['work_dir'] / 'benchmark/corpus_manifest.parquet')
    top = top.join(items.select('doc_id', 'item_id'), on='doc_id', how='left', maintain_order='left')
    answer = top.group_by('query_number', maintain_order=True).agg(pl.col('item_id').str.join(' ').alias('answer'))
    answer = summary.join(answer, on='query_number').select(pl.col('context_id').alias('query_id'), 'answer').sort('query_id')
    # Сначала пишем временный CSV; замена итогового файла идёт только после проверки.
    path = work / 'answer.csv'
    temporary = path.with_suffix('.partial.csv')
    # Явные CRLF дают одинаковые байты файла на разных операционных системах.
    answer.write_csv(temporary, line_terminator="\r\n")
    # Проверяем состав query_id, допустимые item_id, повторы и число колонок.
    report = validate_answer(temporary, config['data_dir'] / 'benchmark_queries.parquet',
                              config['data_dir'] / 'benchmark_items.parquet')
    temporary.replace(path)
    report['model_sha256'] = file_hash(work / 'ranker.cbm')
    save_json(work / 'submission_check.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report
