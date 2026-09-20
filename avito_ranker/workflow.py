"""Тяжёлые этапы запускаются отдельно и освобождают память после завершения."""

import json
import os
import subprocess
import sys
import time

from .config import load_config


def run_stage(stage, config_path="ranking.toml"):
    config = load_config(config_path)
    logs = config["results_dir"] / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    temp = config["work_dir"] / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1",
                        "POLARS_MAX_THREADS": str(config["threads"]),
                        "OPENBLAS_NUM_THREADS": str(config["threads"]),
                        "OMP_NUM_THREADS": str(config["threads"]),
                        "TEMP": str(temp), "TMP": str(temp)})
    jobs = [(stage, None)]
    if stage.endswith("_indices"):
        kind = stage.removesuffix("_indices")
        jobs = [(kind + "_index", field) for field in ["title", "params", "description"]]
    started = time.perf_counter()
    for task, field in jobs:
        label = task + ("_" + field if field else "")
        command = [sys.executable, "-m", "avito_ranker.stage", task, "--config",
                   str(config["config_path"])]
        if field:
            command += ["--field", field]
        print("Начат этап:", label, flush=True)
        path = logs / (label + ".log")
        with path.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=config["config_path"].parent,
                                    env=environment, stdout=stream, stderr=subprocess.STDOUT)
        print(path.read_text("utf-8")[-2500:], flush=True)
        if result.returncode:
            raise RuntimeError(f"Ошибка этапа {label}. Подробности: {path}")
    path = config["results_dir"] / "stage_times.json"
    timings = json.loads(path.read_text("utf-8")) if path.exists() else {}
    timings[stage] = round(time.perf_counter() - started, 2)
    path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print("Этап завершён:", stage, timings[stage], "с", flush=True)
