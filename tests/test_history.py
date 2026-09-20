"""Своя разметка и проверочные группы не должны попадать в историю."""

import json
from pathlib import Path
import tempfile
import unittest

import polars as pl

from avito_improved.history import build_history, QueryHistory


class HistoryTests(unittest.TestCase):
    def test_history_keeps_only_training_groups_and_excludes_self(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = pl.DataFrame({'search_query': ['ремонт', 'уборка', 'доставка'],
                                  'item_id': ['a', 'b', 'c']})
            pairs.write_parquet(root / 'train.parquet')
            pl.DataFrame({'group': ['ремонт', 'уборка', 'доставка'],
                          'strict_group': ['ремонт', 'уборк', 'доставк'],
                          'fold': ['train', 'dev', 'holdout']}).write_parquet(root / 'all_groups.parquet')
            pl.DataFrame({'item_id': ['a', 'b', 'c'], 'item_microcat_id': [10, 20, 30]}).write_parquet(root / 'corpus_manifest.parquet')
            build_history(root, root, root / 'history')
            stored = pl.read_parquet(root / 'history/pairs.parquet')
            self.assertEqual(stored['item_id'].to_list(), [['a']])
            manifest = pl.DataFrame({'item_id': ['a', 'b', 'c'], 'item_location_id': [1, 1, 1],
                                     'item_microcat_id': [10, 20, 30]})
            search = QueryHistory(root / 'history', manifest)
            query = {'search_query': 'ремонт', 'search_location_id': 1}
            pool, scores, microcats = search.search(query)
            self.assertEqual(len(pool), 0)
            self.assertEqual(float(scores.sum()), 0)
            self.assertEqual(microcats, {})
            pool, scores, microcats = search.search(query, exclude_self=False)
            self.assertEqual(pool.tolist(), [0])
            reduced = QueryHistory(root / 'history', manifest.filter(pl.col('item_id') == 'b'))
            pool, scores, reduced_microcats = reduced.search(query, exclude_self=False)
            self.assertEqual(len(pool), 0)
            self.assertEqual(reduced_microcats, microcats)

    def test_unknown_tokens_return_empty_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pl.DataFrame({'search_query': ['ремонт'], 'item_id': ['a']}).write_parquet(root / 'train.parquet')
            pl.DataFrame({'group': ['ремонт'], 'strict_group': ['ремонт'], 'fold': ['train']}).write_parquet(root / 'all_groups.parquet')
            pl.DataFrame({'item_id': ['a'], 'item_microcat_id': [10]}).write_parquet(root / 'corpus_manifest.parquet')
            build_history(root, root, root / 'history')
            search = QueryHistory(root / 'history', pl.DataFrame(
                {'item_id': ['a'], 'item_location_id': [1], 'item_microcat_id': [10]}))
            pool, scores, microcats = search.search({'search_query': 'abcdef', 'search_location_id': 99})
            self.assertEqual(len(pool), 0)
            self.assertEqual(float(scores.sum()), 0)


if __name__ == '__main__':
    unittest.main()
