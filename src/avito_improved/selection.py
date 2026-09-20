"""Проверяем на dev, помогает ли добавить выдачу исходной модели."""

import json

import numpy as np
import polars as pl
from catboost import CatBoostRanker

from avito_retrieval.common import save_json, stable_topk


def top_indices(predictions, old_scores, original, new_head=50):
    """Дополняет первые new_head результатов новой модели исходной выдачей."""
    new = stable_topk(predictions, 50)
    # Без смешивания достаточно топ-50 новой модели.
    if new_head == 50:
        return new
    # Исходной модели доступны только кандидаты её собственного поиска.
    available = np.flatnonzero(original)
    old = available[stable_topk(old_scores[available], 50)]
    result = list(new[:new_head])
    seen = set(result)
    # Сначала дополняем старой выдачей, затем оставшимися новыми, пропуская повторы.
    for row in np.r_[old, new[new_head:]]:
        if row not in seen:
            result.append(int(row))
            seen.add(row)
        if len(result) == 50:
            break
    return np.array(result, dtype=np.int32)


def select_blend(config):
    """Выбирает расширение по расстоянию и состав топ-50 только на dev."""
    work = config.get('quality_dir', config['work_dir'] / 'quality')
    names = json.loads((work / 'feature_names.json').read_text('utf-8'))
    model = CatBoostRanker().load_model(str(work / 'ranker.cbm'))
    # Знаменатель Recall одинаков для обычного и расширенного пула dev.
    summary = pl.read_parquet(work / 'dev_queries.parquet')
    denominators = summary['positives'].to_numpy()
    options = []
    # Расширение географии сравнивается на тех же запросах, без чтения holdout.
    for fold in ['dev', 'dev_expanded']:
        if not (work / f'{fold}.parquet').exists():
            continue
        frame = pl.read_parquet(work / f'{fold}.parquet')
        predictions = model.predict(frame.select(names).to_numpy(), thread_count=2)
        numbers = frame['query_number'].to_numpy()
        labels = frame['label'].to_numpy()
        old_scores = frame['old_score'].to_numpy()
        original = frame['original_candidate'].to_numpy()
        # Таблица хранит отдельный непрерывный блок кандидатов каждого запроса.
        boundaries = np.r_[0, np.flatnonzero(np.diff(numbers)) + 1, len(frame)]
        # Проверяем заранее заданные доли новой выдачи, сохраняя лимит в 50 объявлений.
        for head in [0, 30, 40, 45, 50]:
            recall = np.zeros(len(summary))
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                chosen = top_indices(predictions[start:stop], old_scores[start:stop], original[start:stop], head) + start
                recall[numbers[start]] = labels[chosen].sum() / denominators[numbers[start]]
            options.append({'nearby_candidates': fold == 'dev_expanded', 'new_head': head,
                            'recall_50': float(recall.mean())})
        del frame
    # При равном Recall предпочитаем больше новых результатов, затем более простой пул.
    best = max(options, key=lambda row: (row['recall_50'], row['new_head'], not row['nearby_candidates']))
    selected = json.loads((work / 'selected.json').read_text('utf-8'))
    # Сохраняем отдельно качество модели и качество выбранного смешивания.
    selected['ranker_recall_50'] = selected.get('ranker_recall_50', selected['recall_50'])
    selected.update(best)
    selected['blend_options'] = options
    save_json(work / 'selected.json', selected)
    print(json.dumps(selected, indent=2), flush=True)
