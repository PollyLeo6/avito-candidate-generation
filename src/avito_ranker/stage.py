"""Внутренний запуск одного этапа."""

import argparse

from .config import load_config


def main():
    """Выбирает один этап по аргументам командной строки и запускает его."""
    # Все этапы читают один формат настроек, независимо от способа запуска.
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("--config", default="configs/ranking.toml")
    parser.add_argument("--field")
    args = parser.parse_args()
    config = load_config(args.config)
    config["results_dir"].mkdir(parents=True, exist_ok=True)
    # Импортируем только нужный этап, чтобы не загружать лишние зависимости в процесс.
    if args.stage == "prepare":
        from .prepare import prepare
        prepare(config)
    elif args.stage.endswith("_index"):
        # Для каждого текстового поля строится отдельный индекс своего корпуса.
        from avito_retrieval.build_index import build
        kind = args.stage.removesuffix("_index")
        build(config["work_dir"] / kind,
              config["work_dir"] / (kind + "_indices") / args.field, args.field)
    elif args.stage.endswith("_features"):
        # Префикс имени задаёт часть данных: train, dev, holdout или benchmark.
        from .dataset import create_dataset
        create_dataset(config, args.stage.removesuffix("_features"))
    elif args.stage == "train":
        # Обучение и выбор по dev разделены, чтобы holdout оставался закрытым.
        from .model import train_model
        train_model(config)
    elif args.stage == "select":
        from .model import select_model
        select_model(config)
    elif args.stage.endswith("_evaluate"):
        # Проверка фиксации holdout выполняется внутри функции оценки.
        from .model import evaluate_model
        evaluate_model(config, args.stage.removesuffix("_evaluate"))
    elif args.stage == "predict":
        from .predict import predict
        predict(config)
    elif args.stage == "freeze":
        from .freeze import freeze
        freeze(config)
    elif args.stage == "check":
        # Независимый пересчёт проверяет уже сохранённые предсказания и CSV.
        from .verify import verify
        verify(config)
    elif args.stage == "audit_training":
        # Аудит обучения использует train-пары, не читая проверочную разметку.
        from .verify import audit_training
        audit_training(config)
    else:
        # Опечатка в имени этапа должна завершиться ошибкой, а не пустым запуском.
        raise ValueError(f"Неизвестный этап: {args.stage}")


if __name__ == "__main__":
    main()
