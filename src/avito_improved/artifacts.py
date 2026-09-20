"""Сохранение модели и восстановление для быстрой проверки."""

import json
import shutil

from .protocol import fingerprints


def restore(config):
    """Загружает готовую модель, если код и восстановленная история совпали со снимком."""
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    saved = config['results_dir']
    work.mkdir(parents=True, exist_ok=True)
    # Готовые параметры загружаются до пересчёта отпечатков.
    for name in ['ranker.cbm', 'selected.json', 'feature_names.json', 'leakage_audit.json']:
        shutil.copyfile(saved / name, work / name)
    # Сверяем не только модель, но и код, входные данные и историю.
    expected = json.loads((saved / 'frozen.json').read_text('utf-8'))
    if expected['files'] != fingerprints(config):
        raise ValueError('Код, модель или обучающая история отличаются от сохранённого эксперимента')
    shutil.copyfile(saved / 'frozen.json', work / 'frozen.json')
    print('Модель и контрольные суммы проверены', flush=True)


def export(config):
    """Копирует модель, отчёты и рассчитанный ответ из work в results."""
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    saved = config['results_dir']
    saved.mkdir(parents=True, exist_ok=True)
    # Сохраняем весь набор, который нужен для проверки и быстрого запуска.
    names = ['ranker.cbm', 'selected.json', 'feature_names.json', 'selection_curve.json',
             'fresh_metrics.json', 'fresh_per_query.parquet', 'frozen.json', 'fresh_reservation.json',
             'train_candidates.json', 'dev_candidates.json', 'dev_expanded_candidates.json', 'fresh_candidates.json',
             'benchmark_candidates.json', 'submission_check.json', 'leakage_audit.json', 'answer.csv']
    for name in names:
        shutil.copyfile(work / name, saved / name)
    # Отчёт об истории хранится рядом с моделью, сами индексы остаются в work.
    shutil.copyfile(work / 'history/audit.json', saved / 'history_audit.json')
    print('Модель, ответ и отчёты сохранены:', saved, flush=True)
