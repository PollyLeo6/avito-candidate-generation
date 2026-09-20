"""Настройки одинаковы для notebook и командной строки."""

from pathlib import Path
import tomllib


def load_config(path="ranking.toml"):
    path = Path(path).resolve()
    config = tomllib.loads(path.read_text("utf-8"))
    config["config_path"] = path
    for name in ["data_dir", "work_dir", "results_dir"]:
        config[name] = (path.parent / config[name]).resolve()
    return config
