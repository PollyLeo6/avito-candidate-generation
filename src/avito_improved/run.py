"""Этапы запускаются отдельными процессами, чтобы освобождать память."""

import argparse
import os
import subprocess
import sys

from avito_ranker.config import load_config


def run_stage(stage, config_path):
    config = load_config(config_path)
    logs = config['results_dir'] / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({'PYTHONIOENCODING': 'utf-8', 'PYTHONHASHSEED': '0',
                        'POLARS_MAX_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2'})
    path = logs / (stage + '.log')
    print('Начат этап:', stage, flush=True)
    with path.open('w', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, '-m', 'avito_improved.run', stage,
                                 '--config', str(config['config_path'])], cwd=config['project_dir'],
                                env=environment, stdout=stream, stderr=subprocess.STDOUT)
    print(path.read_text('utf-8')[-2500:], flush=True)
    if result.returncode:
        raise RuntimeError(f'Ошибка этапа {stage}. Лог: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage')
    parser.add_argument('--config', default='configs/improved.toml')
    args = parser.parse_args()
    config = load_config(args.config)
    config['results_dir'].mkdir(parents=True, exist_ok=True)
    stage = args.stage
    if stage == 'prepare':
        from .protocol import prepare_experiment
        prepare_experiment(config)
    elif stage.endswith('_indices'):
        from avito_ranker.workflow import run_stage as run_original
        run_original(stage, args.config)
    elif stage.endswith('_features'):
        from .dataset import create_dataset
        fold = stage.removesuffix('_features')
        if fold == 'fresh':
            from .protocol import check_frozen
            check_frozen(config)
        create_dataset(config, fold)
    elif stage == 'train':
        from .model import train_and_select
        train_and_select(config)
    elif stage == 'audit':
        from .audit import audit_training
        audit_training(config)
    elif stage == 'freeze':
        from .protocol import freeze
        freeze(config)
    elif stage == 'select':
        from .selection import select_blend
        select_blend(config)
    elif stage == 'evaluate':
        from .evaluation import evaluate
        evaluate(config)
    elif stage == 'predict':
        from .evaluation import predict
        predict(config)
    elif stage == 'restore':
        from .artifacts import restore
        restore(config)
    elif stage == 'export':
        from .artifacts import export
        export(config)
    else:
        raise ValueError(f'Неизвестный этап: {stage}')


if __name__ == '__main__':
    main()
