"""Проверяем, сколько пропусков поиска закрывают символьные n-граммы."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Ограничиваем нагрузку на CPU и подключаем код репозитория при прямом запуске файла.
os.environ.setdefault('POLARS_MAX_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

from avito_retrieval.common import normalize, stable_topk


def evaluate(validation, features, output, top_k=300):
    """Измеряет, сколько известных positives добавляет символьный поиск к готовому пулу."""
    started = time.monotonic()
    # Порядок манифеста связывает строки TF-IDF с doc_id из сохранённых кандидатов.
    manifest = pl.read_parquet(validation / 'corpus_manifest.parquet', columns=['item_id'])
    texts = pl.read_parquet(validation / 'corpus.parquet', columns=['item_id', 'item_title_raw'])
    titles = manifest.join(texts, on='item_id', how='left', maintain_order='left')
    item_ids = titles['item_id'].to_list()
    by_item = {value: i for i, value in enumerate(item_ids)}
    # Разметка dev используется только для подсчёта попаданий, не для построения индекса.
    queries = pl.read_parquet(validation / 'dev_queries.parquet')
    qrels = dict(pl.read_parquet(validation / 'dev_qrels.parquet').group_by('context_id').agg('item_id').iter_rows())
    pools = dict(pl.read_parquet(features, columns=['query_number', 'doc_id']).group_by('query_number').agg('doc_id').iter_rows())

    # Стемминг здесь не нужен: сохраняем исходные части слов.
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                                preprocessor=normalize, lowercase=False,
                                min_df=2, sublinear_tf=True, dtype=np.float32)
    matrix = vectorizer.fit_transform(titles['item_title_raw'].fill_null('').to_list())
    query_matrix = vectorizer.transform(queries['search_query'].fill_null('').to_list())
    print(f'index ready: {matrix.shape}, {time.monotonic() - started:.1f}s', flush=True)
    # Транспонированный индекс позволяет проходить по n-граммам запроса.
    baseline, extended, character = [], [], []
    recovered = []
    index = matrix.T.tocsr()
    for number, query in enumerate(queries.iter_rows(named=True)):
        # Cosine similarity по TF-IDF сохраняет совпадения при небольших опечатках.
        scores = (query_matrix[number] @ index).toarray().ravel()
        candidates = stable_topk(scores, top_k)
        candidates = candidates[scores[candidates] > 0]
        # Сравниваем исходный пул, его расширение и отдельную символьную выдачу.
        pool = set(pools[number])
        char_pool = set(candidates.tolist())
        gold = {by_item[item] for item in qrels[query['context_id']]}
        baseline.append(len(gold & pool) / len(gold))
        extended.append(len(gold & (pool | char_pool)) / len(gold))
        character.append(len(gold & char_pool) / len(gold))
        # Сохраняем конкретные восстановленные пары для разбора ошибок.
        for doc_id in sorted((gold - pool) & char_pool):
            recovered.append({'query_number': number, 'query': query['search_query'],
                              'item_id': item_ids[doc_id],
                              'title': titles['item_title_raw'][doc_id],
                              'char_rank': int(np.flatnonzero(candidates == doc_id)[0]) + 1})
        if (number + 1) % 500 == 0:
            print(f'queries: {number + 1}/{len(queries)}', flush=True)
    # Усредняем покрытие по запросам. Это ещё не Recall@50 после ранжирования.
    report = {'split': 'dev', 'queries': len(queries), 'corpus_items': len(titles),
              'field': 'item_title_raw', 'analyzer': 'char_wb', 'ngram_range': [3, 5],
              'top_k': top_k, 'min_df': 2, 'sublinear_tf': True,
              'original_pool_recall': float(np.mean(baseline)),
              'extended_pool_recall': float(np.mean(extended)),
              'char_pool_recall': float(np.mean(character)),
              'recovered_queries': len({row['query_number'] for row in recovered}),
              'recovered_items': len(recovered), 'examples': recovered[:20],
              'seconds': round(time.monotonic() - started, 2),
              'note': 'Candidate coverage only; final Recall@50 was not measured.'}
    # Полный отчёт хранит примеры, в консоль выводим только сводные значения.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'examples'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    # Пути и размер символьного пула задаются аргументами, без правок кода.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('work/char_retrieval.json'))
    parser.add_argument('--top-k', type=int, default=300)
    args = parser.parse_args()
    evaluate(args.validation, args.features, args.output, args.top_k)
