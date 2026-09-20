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

query_columns = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
text_fields = {
    "title": "item_title_raw",
    "description": "item_description_raw",
    "params": "item_infm_params_text",
}
seed = 20260919
token_pattern = re.compile(r"[^\W_]+", re.UNICODE)
cyrillic_pattern = re.compile(r"^[а-я]+$")
russian_stemmer = snowballstemmer.stemmer("russian")


def normalize(text):
    return re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", text or "").lower().replace("ё", "е")
    ).strip()


def group_key(text):
    # Пунктуация и лишние пробелы не должны разносить запрос по разным частям.
    return " ".join(token_pattern.findall(normalize(text)))


def stable_hash(value, salt=""):
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return hashlib.blake2b(
        (str(seed) + ":" + salt + ":" + raw).encode("utf-8"), digest_size=16
    ).hexdigest()


def query_key(row):
    return stable_hash([row[c] for c in query_columns], "context")


def fold_for_group(group):
    bucket = int(stable_hash(group, "split")[:8], 16) % 10000
    return "train" if bucket < 8000 else ("dev" if bucket < 9000 else "holdout")


@lru_cache(maxsize=250000)
def stem(token):
    return russian_stemmer.stemWord(token) if cyrillic_pattern.fullmatch(token) else token


def tokenize(text, stemming=True):
    tokens = token_pattern.findall(normalize(text))
    # Сохраняем отрицания, числа и короткие обозначения.
    return [stem(t) for t in tokens] if stemming else tokens


def stable_topk(scores, k):
    """При равных оценках побеждает меньший item_id."""
    scores = np.asarray(scores)
    k = min(k, len(scores))
    if k <= 0:
        return np.empty(0, dtype=np.int32)
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    greater = np.flatnonzero(scores > threshold)
    equal = np.flatnonzero(scores == threshold)[: k - len(greater)]
    candidates = np.concatenate([greater, equal])
    return candidates[np.lexsort((candidates, -scores[candidates]))].astype(np.int32)


def recall_at(predicted, positives, k=50):
    """Recall одного запроса. Повторы ID не увеличивают число попаданий."""
    gold = set(positives)
    if not gold:
        raise ValueError("Нельзя оценить запрос без positives")
    return len(set(predicted[:k]) & gold) / len(gold)


def rrf_scores(channels, weights, n_docs, constant=60):
    score = np.zeros(n_docs, dtype=np.float32)
    for name, weight in weights.items():
        ids = channels[name]
        score[ids] += weight / (constant + np.arange(1, len(ids) + 1, dtype=np.float32))
    return score


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def log(*args):
    print(*args, flush=True)
