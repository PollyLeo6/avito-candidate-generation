"""Проверки метрики, разбиения и совпадения BM25 с формулой."""

import unittest

import numpy as np
from avito_retrieval.common import (
    bm25s,
    fold_for_group,
    group_key,
    recall_at,
    rrf_scores,
    stable_topk,
    tokenize,
)
from avito_retrieval.retrieval import field_scores


class CoreTests(unittest.TestCase):
    """Проверяем общие правила, от которых зависит качество оценки."""

    def test_query_variants_stay_together(self):
        """Регистр, пунктуация и буква ё не меняют группу запроса."""
        texts = ["Приём  металлолома", "прием металлолома", "приём-металлолома!"]
        groups = [group_key(text) for text in texts]
        self.assertEqual(len(set(groups)), 1)
        self.assertEqual(len({fold_for_group(group) for group in groups}), 1)

    def test_context_is_not_the_split_group(self):
        """Разные услуги не объединяются только из-за общих слов."""
        self.assertNotEqual(group_key("ремонт двери"), group_key("установка двери"))

    def test_recall_uses_sets_and_positive_denominator(self):
        """Повторы не дают лишних попаданий, а число положительных сохраняется."""
        self.assertEqual(recall_at(["a", "a", "x"], ["a", "b"], 3), 0.5)
        self.assertEqual(recall_at(["x", "a"], ["a"], 1), 0)
        self.assertEqual(recall_at(["a", "x"], ["a"], 2), 1)
        with self.assertRaises(ValueError):
            recall_at(["a"], [], 50)

    def test_ties_are_deterministic(self):
        """При равных оценках меньший номер документа выбирается первым."""
        result = stable_topk(np.array([1.0, 2.0, 2.0, 2.0, 0.0]), 2)
        self.assertEqual(result.tolist(), [1, 2])
        self.assertEqual(stable_topk(np.zeros(4), 3).tolist(), [0, 1, 2])

    def test_rrf_counts_each_source(self):
        """Объявление получает вклад от каждого поискового канала."""
        channels = {"a": np.array([2, 0]), "b": np.array([1, 2])}
        scores = rrf_scores(channels, {"a": 1.0, "b": 1.0}, 3)
        self.assertAlmostEqual(float(scores[2]), 1 / 61 + 1 / 62, places=7)
        self.assertEqual(stable_topk(scores, 1).tolist(), [2])

    def test_stemming_keeps_numbers_and_negation(self):
        """Стемминг сближает формы слов и сохраняет смысловые уточнения."""
        self.assertEqual(tokenize("Установка двери"), tokenize("установки дверей"))
        self.assertIn("не", tokenize("не ремонт 220v 3d"))
        self.assertIn("220v", tokenize("не ремонт 220v 3d"))

    def test_bm25_matches_formula(self):
        """Сравниваем оценки библиотеки с ручным расчётом на трёх документах."""
        # В маленьком корпусе известны частоты слов и длины документов.
        documents = [["ремонт", "ремонт", "двери"], ["ремонт"], ["двери", "окна"]]
        index = bm25s.BM25(k1=1.5, b=0.75, method="lucene", idf_method="lucene")
        index.index(documents, show_progress=False)
        scores = index.get_scores(["ремонт"])
        # Формула повторяет используемый вариант Lucene BM25.
        lengths = np.array([3.0, 1.0, 2.0])
        frequency = np.array([2.0, 1.0, 0.0])
        idf = np.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
        expected = (
            idf * frequency / (frequency + 1.5 * (1 - 0.75 + 0.75 * lengths / lengths.mean()))
        )
        np.testing.assert_allclose(scores, expected, rtol=1e-6)
        # Пустой запрос и неизвестное слово не должны находить совпадения.
        np.testing.assert_equal(field_scores(index, "", False), np.zeros(3))
        np.testing.assert_equal(field_scores(index, "нетвсловаре", False), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
