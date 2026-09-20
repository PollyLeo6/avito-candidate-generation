"""Обучение и выбор функции потерь и числа деревьев только по dev."""

import gc
import json
import time

import numpy as np
import polars as pl
from catboost import CatBoostRanker, Pool

from avito_retrieval.common import stable_topk


def train_and_select(config):
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    names = json.loads((work / 'feature_names.json').read_text('utf-8'))
    train = pl.read_parquet(work / 'train.parquet')
    train_pool = Pool(train.select(names).to_numpy(), label=train['label'].to_numpy(),
                       group_id=train['query_number'].to_numpy(), feature_names=names)
    print('training', len(train), train['query_number'].n_unique(), flush=True)
    del train
    gc.collect()
    dev = pl.read_parquet(work / 'dev.parquet')
    summary = pl.read_parquet(work / 'dev_queries.parquet')
    dev_pool = Pool(dev.select(names).to_numpy(), feature_names=names)
    labels = dev['label'].to_numpy()
    query_ids = dev['query_number'].to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(query_ids)) + 1, len(dev)]
    denominators = summary['positives'].to_numpy()


    def recall(predictions):
        result = np.zeros(len(summary))
        for begin, end in zip(starts[:-1], starts[1:]):
            chosen = stable_topk(predictions[begin:end], 50) + begin
            number = query_ids[begin]
            result[number] = labels[chosen].sum() / denominators[number]
        return float(result.mean())


    print('old model with wider pool', recall(dev['old_score'].to_numpy()), flush=True)
    del dev
    gc.collect()
    results = []
    best = -1
    started = time.perf_counter()
    for name, loss in [('pairlogit', 'PairLogit:max_pairs=160'), ('softmax', 'QuerySoftMax')]:
        model = CatBoostRanker(loss_function=loss, iterations=400, depth=6, learning_rate=0.05,
                               l2_leaf_reg=12, random_seed=20260920, thread_count=2,
                               allow_writing_files=False, verbose=100)
        model.fit(train_pool)
        model.save_model(str(work / f'{name}_full.cbm'))
        for trees, predictions in zip(range(20, 401, 20), model.staged_predict(dev_pool, eval_period=20, thread_count=2)):
            value = recall(predictions)
            row = {'method': name, 'trees': trees, 'recall_50': value}
            results.append(row)
            print(json.dumps(row), flush=True)
            if value > best:
                selected = model.copy()
                selected.shrink(trees)
                selected.save_model(str(work / 'ranker.cbm'))
                np.save(work / 'dev_scores.npy', predictions)
                best = value
                (work / 'selected.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
        del model
        gc.collect()
    (work / 'selection_curve.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('selected', (work / 'selected.json').read_text('utf-8'), 'seconds', time.perf_counter() - started, flush=True)
