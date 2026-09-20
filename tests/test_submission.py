"""Проверяем формат ответа на небольшом искусственном корпусе."""

import csv
import tempfile
import unittest
from pathlib import Path

import polars as pl

from avito_retrieval.submission import validate_answer


class SubmissionTests(unittest.TestCase):
    """Правильный CSV принимается, нарушения требований отклоняются."""

    def setUp(self):
        """Создаём два запроса и 51 объявление для проверки лимита топ-50."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.query_ids = ["Ab" + "0" * 14, "Cd" + "1" * 14]
        self.item_ids = [f"{number:016x}" for number in range(51)]
        pl.DataFrame({"query_id": self.query_ids}).write_parquet(self.root / "queries.parquet")
        pl.DataFrame({"item_id": self.item_ids}).write_parquet(self.root / "items.parquet")

    def check(self, rows, header=None):
        """Записываем переданные строки и запускаем общий валидатор ответа."""
        path = self.root / "answer.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header or ["query_id", "answer"])
            writer.writerows(rows)
        return validate_answer(path, self.root / "queries.parquet", self.root / "items.parquet")

    def valid_rows(self):
        """Возвращаем корректный ответ для каждого тестового запроса."""
        return [[query, " ".join(self.item_ids[:50])] for query in self.query_ids]

    def test_valid_and_empty_answers(self):
        """Проверяем корректные списки и допустимый пустой ответ."""
        self.assertEqual(self.check(self.valid_rows())["rows"], 2)
        rows = self.valid_rows()
        rows[0][1] = ""
        self.assertEqual(self.check(rows)["min_items"], 0)

    def test_wrong_queries_are_rejected(self):
        """Повтор, неизвестный ID, другой регистр и пропущенный запрос недопустимы."""
        for bad_id in [self.query_ids[1], self.query_ids[0].lower(), "unknown"]:
            rows = self.valid_rows()
            rows[0][0] = bad_id
            with self.subTest(bad_id=bad_id), self.assertRaises(ValueError):
                self.check(rows)
        with self.assertRaises(ValueError):
            self.check(self.valid_rows()[:1])

    def test_wrong_answers_are_rejected(self):
        """Проверяем лимит, повторы, принадлежность корпусу и формат item_id."""
        # Варианты покрывают ошибки длины списка, идентификаторов и разделителей.
        invalid = [
            " ".join(self.item_ids),
            self.item_ids[0] + " " + self.item_ids[0],
            "f" * 16,
            "F" * 16,
            "123",
            self.item_ids[0] + "  " + self.item_ids[1],
        ]
        for answer in invalid:
            rows = self.valid_rows()
            rows[0][1] = answer
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                self.check(rows)

    def test_extra_column_is_rejected(self):
        """В ответе не должно быть колонки с индексом или других лишних полей."""
        with self.assertRaises(ValueError):
            self.check(self.valid_rows(), ["index", "query_id", "answer"])
