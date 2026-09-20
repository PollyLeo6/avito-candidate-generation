"""Пути не должны зависеть от папки, из которой открыт notebook."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from avito_ranker.config import load_config
from avito_ranker.freeze import fingerprint


class LayoutTests(unittest.TestCase):
    def test_config_resolves_paths_from_configs_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "configs/ranking.toml"
            path.parent.mkdir()
            path.write_text('data_dir = "../data"\nwork_dir = "../work"\n'
                            'results_dir = "../results"\nthreads = 2\n', encoding="utf-8")
            config = load_config(path)
            self.assertEqual(config["project_dir"], root)
            for name in ["data", "work", "results"]:
                self.assertEqual(config[name + "_dir"], root / name)

    def test_fingerprint_detects_changes_in_src(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            results = root / "results"
            results.mkdir()
            for name in ["selected", "training", "split_audit", "input_check", "leakage_audit"]:
                (results / (name + ".json")).write_text("{}", encoding="utf-8")
            (results / "ranker.cbm").write_bytes(b"model")
            for name in ["avito_ranker", "avito_retrieval"]:
                folder = root / "src" / name
                folder.mkdir(parents=True)
                (folder / "__init__.py").write_text("", encoding="utf-8")
            source = root / "src/avito_ranker/model.py"
            source.write_text("value = 1\n", encoding="utf-8")
            config = {"project_dir": root, "config_path": root / "configs/ranking.toml",
                      "results_dir": results, "threads": 2}
            with patch("avito_ranker.freeze.model_function_hash", return_value="model"):
                before = fingerprint(config)
                source.write_text("value = 2\n", encoding="utf-8")
                after = fingerprint(config)
            self.assertNotEqual(before["src/avito_ranker/model.py"], after["src/avito_ranker/model.py"])
            self.assertEqual(before["parameters"], after["parameters"])


if __name__ == "__main__":
    unittest.main()
