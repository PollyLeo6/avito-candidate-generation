"""Проверки разбиения и мест, где легко случайно завысить метрику."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from avito_ranker.dataset import sample_training
from avito_ranker.features import rank_values
from avito_ranker.freeze import check_frozen
from avito_ranker.model import per_query_recall
from avito_ranker.prepare import assign_fold, strict_group


class RankingTests(unittest.TestCase):
    def test_word_order_and_inflections_stay_together(self):
        self.assertEqual(strict_group("Ремонт квартир!"), strict_group("квартиры ремонт"))

    def test_previously_seen_group_cannot_be_holdout_or_train(self):
        self.assertEqual(assign_fold("example", {"example"}), "dev")

    def test_missing_positive_is_in_metric_denominator(self):
        candidates = pl.DataFrame({"query_number": [0, 0, 1], "doc_id": [1, 2, 3], "label": [1, 0, 0]})
        summary = pl.DataFrame({"query_number": [0, 1], "positives": [2, 1]})
        result = per_query_recall(candidates, [1.0, 0.0, 2.0], summary)
        self.assertEqual(result["recall_50"].to_list(), [0.5, 0.0])
        self.assertEqual(result["recall_50"].mean(), 0.25)

    def test_training_sampling_keeps_positives_without_adding_them(self):
        pool = np.arange(10)
        labels = np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 0])
        matrix = np.column_stack([np.arange(10) / 10, np.ones(10)])
        selected = sample_training(pool, labels, matrix, ["rrf", "same_location"], 4, np.random.default_rng(1))
        self.assertEqual(len(selected), 6)
        self.assertTrue({1, 8}.issubset(selected))
        self.assertEqual(len(set(selected)), len(selected))

    def test_rank_features_respect_document_numbers(self):
        values = rank_values(np.array([2, 5, 8]), np.array([8, 2]))
        np.testing.assert_array_equal(values, [0.5, 0, 1])

    def test_holdout_is_closed_until_model_is_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "freeze"):
                check_frozen({"results_dir": Path(directory)})


if __name__ == "__main__":
    unittest.main()
