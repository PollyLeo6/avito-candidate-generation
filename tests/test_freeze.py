"""Формат JSON не должен мешать повторению расчёта на другой ОС."""

from pathlib import Path
import tempfile
import unittest

from avito_ranker.freeze import artifact_hash


class FreezeTests(unittest.TestCase):
    """Отделяем изменения данных от различий в оформлении JSON."""

    def test_json_formatting_does_not_change_fingerprint(self):
        """Пробелы, порядок ключей и переводы строк не меняют снимок JSON."""
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left.json", Path(directory) / "right.json"
            left.write_bytes(b'{\r\n  "a": 1,\r\n  "b": [2, 3]\r\n}\r\n')
            right.write_bytes(b'{"b":[2,3],"a":1}\n')
            self.assertEqual(artifact_hash(left), artifact_hash(right))

    def test_changed_json_value_changes_fingerprint(self):
        """Замена параметра модели должна менять контрольную сумму."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            path.write_text('{"trees":50}', encoding="utf-8")
            original = artifact_hash(path)
            path.write_text('{"trees":80}', encoding="utf-8")
            self.assertNotEqual(original, artifact_hash(path))

    def test_binary_artifact_is_compared_byte_for_byte(self):
        """Для бинарной модели существенен каждый байт файла."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.cbm"
            path.write_bytes(b"model\r\n")
            original = artifact_hash(path)
            path.write_bytes(b"model\n")
            self.assertNotEqual(original, artifact_hash(path))


if __name__ == "__main__":
    unittest.main()
