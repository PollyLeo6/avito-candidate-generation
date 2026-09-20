"""Объединение двух выдач сохраняет 50 разных объявлений."""

import unittest

import numpy as np

from avito_improved.selection import top_indices


class SelectionTests(unittest.TestCase):
    def test_overlap_is_skipped_without_losing_places(self):
        predictions = np.arange(80, 0, -1, dtype=float)
        old_scores = np.roll(predictions, 20)
        original = np.ones(80, dtype=bool)
        result = top_indices(predictions, old_scores, original, new_head=40)
        self.assertEqual(result[:40].tolist(), list(range(40)))
        self.assertEqual(len(result), 50)
        self.assertEqual(len(set(result)), 50)
        without_old = top_indices(predictions, old_scores, ~original, new_head=40)
        self.assertEqual(without_old.tolist(), list(range(50)))


if __name__ == '__main__':
    unittest.main()
