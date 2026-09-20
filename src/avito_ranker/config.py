"""Настройки одинаковы для notebook и командной строки."""

from pathlib import Path
import tomllib


def load_config(path="configs/ranking.toml"):
    """Читает TOML и разрешает пути относительно файла настроек."""
    path = Path(path).resolve()
    config = tomllib.loads(path.read_text("utf-8"))
    config["config_path"] = path
    # Корень проекта нужен и при запуске из папки notebooks.
    config["project_dir"] = path.parent.parent if path.parent.name == "configs" else path.parent
    # Одинаковый конфиг не должен зависеть от текущего каталога терминала.
    for name in ["data_dir", "work_dir", "results_dir"]:
        config[name] = (path.parent / config[name]).resolve()
    return config
