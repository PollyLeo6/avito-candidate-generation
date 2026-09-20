"""Тяжёлые этапы запускаются отдельно и освобождают память после завершения."""

import json
import os
import subprocess
import sys
import time

from .config import load_config


def run_stage(stage, config_path="configs/ranking.toml"):
    """Запускает этап в отдельных процессах и сохраняет логи и длительность."""
    config = load_config(config_path)
    logs = config["results_dir"] / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    temp = config["work_dir"] / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    # Ограничиваем потоки и задаём рабочую папку временных файлов для дочерних процессов.
    environment = os.environ.copy()
    environment.update({"PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1",
                        "POLARS_MAX_THREADS": str(config["threads"]),
                        "OPENBLAS_NUM_THREADS": str(config["threads"]),
                        "OMP_NUM_THREADS": str(config["threads"]),
                        "TEMP": str(temp), "TMP": str(temp)})
    # Три индекса считаются по очереди, чтобы они не занимали память одновременно.
    jobs = [(stage, None)]
    if stage.endswith("_indices"):
        kind = stage.removesuffix("_indices")
        jobs = [(kind + "_index", field) for field in ["title", "params", "description"]]
    started = time.perf_counter()
    # Используем тот же Python, что в ноутбуке или текущем окружении терминала.
    for task, field in jobs:
        label = task + ("_" + field if field else "")
        command = [sys.executable, "-m", "avito_ranker.stage", task, "--config",
                   str(config["config_path"])]
        if field:
            command += ["--field", field]
        print("Начат этап:", label, flush=True)
        path = logs / (label + ".log")
        # Полный вывод остаётся в логе; в ноутбук возвращается только его конец.
        with path.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=config["project_dir"],
                                    env=environment, stdout=stream, stderr=subprocess.STDOUT)
        print(path.read_text("utf-8")[-2500:], flush=True)
        # Следующие шаги не должны работать с неполными результатами упавшего этапа.
        if result.returncode:
            raise RuntimeError(f"Ошибка этапа {label}. Подробности: {path}")
    # Обновляем время текущего этапа, сохраняя результаты остальных измерений.
    path = config["results_dir"] / "stage_times.json"
    timings = json.loads(path.read_text("utf-8")) if path.exists() else {}
    timings[stage] = round(time.perf_counter() - started, 2)
    path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print("Этап завершён:", stage, timings[stage], "с", flush=True)
