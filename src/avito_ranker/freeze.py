"""Перед первым чтением holdout фиксируем модель, признаки и настройки."""

import json
import hashlib
from pathlib import Path
import tempfile

from avito_retrieval.common import file_hash, save_json


def artifact_hash(path):
    if path.suffix != ".json":
        return file_hash(path)
    # Windows и Linux могут записать одинаковый JSON с разными переводами строк.
    content = json.loads(path.read_text("utf-8"))
    canonical = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def model_function_hash(config):
    from catboost import CatBoostRanker
    model = CatBoostRanker().load_model(str(config["results_dir"] / "ranker.cbm"))
    with tempfile.NamedTemporaryFile(dir=config["results_dir"], suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        model.save_model(str(temporary), format="json")
        content = json.loads(temporary.read_text("utf-8"))
        content.pop("model_info", None)
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    finally:
        temporary.unlink()


def fingerprint(config):
    root = config["project_dir"]
    files = [config["results_dir"] / "ranker.cbm",
             config["results_dir"] / "selected.json", config["results_dir"] / "training.json",
             config["results_dir"] / "split_audit.json", config["results_dir"] / "input_check.json",
             config["results_dir"] / "leakage_audit.json"]
    files += sorted((root / "src/avito_ranker").glob("*.py"))
    files += sorted((root / "src/avito_retrieval").glob("*.py"))
    hashes = {path.relative_to(root).as_posix(): artifact_hash(path) for path in files}
    parameters = {key: value for key, value in config.items()
                  if key not in {"config_path", "project_dir", "data_dir", "work_dir", "results_dir"}}
    hashes["parameters"] = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
    hashes["model_function"] = model_function_hash(config)
    return hashes


def freeze(config):
    path = config["results_dir"] / "frozen_experiment.json"
    current = fingerprint(config)
    reproduced = False
    provenance = {}
    if path.exists():
        record = json.loads(path.read_text("utf-8"))
        previous = record["files"]
        provenance = {key: record[key] for key in ["original_snapshot", "compatibility_change", "layout_change"]
                      if key in record}
        changed = {name for name in previous.keys() | current.keys() if previous.get(name) != current.get(name)}
        model_key = (config["results_dir"] / "ranker.cbm").relative_to(config["project_dir"]).as_posix()
        # CatBoost пишет в файл время обучения и случайный GUID модели.
        # Сами деревья и границы признаков при таком повторе обязаны совпасть.
        if changed - {model_key}:
            raise RuntimeError("Эксперимент уже зафиксирован. Новый подбор требует отдельной оценки.")
        reproduced = bool(changed)
    save_json(path, {"format_version": 2, "files": current, "selection_uses": "dev only",
                     "holdout_used_for_training_or_selection": False,
                     "reproduction_of_fixed_experiment": reproduced, **provenance})
    print("Модель и код зафиксированы перед holdout", flush=True)


def check_frozen(config):
    path = config["results_dir"] / "frozen_experiment.json"
    if not path.exists():
        raise RuntimeError("Сначала зафиксируйте модель этапом freeze")
    expected = json.loads(path.read_text("utf-8"))["files"]
    if expected != fingerprint(config):
        raise RuntimeError("После фиксации изменились модель, код или настройки")
