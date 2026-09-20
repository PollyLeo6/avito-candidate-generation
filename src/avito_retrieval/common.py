"""Общие правила обработки текста, идентификаторов и метрики."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from functools import lru_cache

import bm25s as bm25s
import numpy as np
import snowballstemmer

# Контекст включает фильтры и локацию: одинаковый текст не всегда означает один поиск.
query_columns = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
# Названия каналов совпадают с каталогами соответствующих BM25-индексов.
text_fields = {
    "title": "item_title_raw",
    "description": "item_description_raw",
    "params": "item_infm_params_text",
}
# Фиксированный seed сохраняет разбиение между повторными запусками.
seed = 20260919
token_pattern = re.compile(r"[^\W_]+", re.UNICODE)
cyrillic_pattern = re.compile(r"^[а-я]+$")
russian_stemmer = snowballstemmer.stemmer("russian")


def normalize(text):
    """Убирает различия в регистре, форме Unicode, букве ё и пробелах."""
    return re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", text or "").lower().replace("ё", "е")
    ).strip()


def group_key(text):
    """Нормализует запись запроса для группировки по последовательности слов."""
    # Пунктуация и лишние пробелы не должны разносить запрос по разным частям.
    return " ".join(token_pattern.findall(normalize(text)))


def stable_hash(value, salt=""):
    """Даёт воспроизводимый хеш; salt разделяет независимые операции отбора."""
    # Явная сериализация не зависит от случайного hash() текущего процесса.
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return hashlib.blake2b(
        (str(seed) + ":" + salt + ":" + raw).encode("utf-8"), digest_size=16
    ).hexdigest()


def query_key(row):
    """Идентифицирует сочетание текста запроса, фильтров, категории и локации."""
    return stable_hash([row[c] for c in query_columns], "context")


def fold_for_group(group):
    """Назначает группе train/dev/holdout с долями примерно 80/10/10."""
    # Решение зависит только от группы, поэтому её контексты не разделяются.
    bucket = int(stable_hash(group, "split")[:8], 16) % 10000
    return "train" if bucket < 8000 else ("dev" if bucket < 9000 else "holdout")


@lru_cache(maxsize=250000)
def stem(token):
    """Сокращает русское слово до основы, сохраняя числа и другие алфавиты."""
    return russian_stemmer.stemWord(token) if cyrillic_pattern.fullmatch(token) else token


def tokenize(text, stemming=True):
    """Выделяет буквенно-числовые токены и при необходимости применяет стемминг."""
    tokens = token_pattern.findall(normalize(text))
    # Сохраняем отрицания, числа и короткие обозначения.
    return [stem(t) for t in tokens] if stemming else tokens


def stable_topk(scores, k):
    """Возвращает индексы лучших оценок; при равенстве берёт меньший индекс."""
    # Ограничиваем k размером массива, в том числе для пустой выдачи.
    scores = np.asarray(scores)
    k = min(k, len(scores))
    if k <= 0:
        return np.empty(0, dtype=np.int32)
    # Сначала находим порог без полной сортировки, затем разбираем равные оценки.
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    greater = np.flatnonzero(scores > threshold)
    equal = np.flatnonzero(scores == threshold)[: k - len(greater)]
    candidates = np.concatenate([greater, equal])
    return candidates[np.lexsort((candidates, -scores[candidates]))].astype(np.int32)


def recall_at(predicted, positives, k=50):
    """Recall одного запроса. Повторы ID не увеличивают число попаданий."""
    # В знаменателе все известные положительные объявления этого запроса.
    gold = set(positives)
    if not gold:
        raise ValueError("Нельзя оценить запрос без positives")
    return len(set(predicted[:k]) & gold) / len(gold)


def rrf_scores(channels, weights, n_docs, constant=60):
    """Объединяет каналы по местам объявлений, независимо от шкалы BM25."""
    # Каждый канал добавляет вес, убывающий с позицией объявления в его выдаче.
    score = np.zeros(n_docs, dtype=np.float32)
    for name, weight in weights.items():
        ids = channels[name]
        score[ids] += weight / (constant + np.arange(1, len(ids) + 1, dtype=np.float32))
    return score


def save_json(path, value):
    """Сохраняет читаемый JSON в UTF-8, создавая каталог при необходимости."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def file_hash(path):
    """Считает SHA-256 порциями, чтобы не загружать большой файл целиком."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def log(*args):
    """Сразу выводит прогресс, в том числе при запуске из ноутбука."""
    print(*args, flush=True)
