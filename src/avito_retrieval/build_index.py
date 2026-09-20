"""Последовательная индексация полей с порционным чтением текстов."""

import json
import time
from array import array
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from .common import bm25s, file_hash, log, save_json, text_fields, tokenize


def build(validation, out, field, stemming=True):
    """Создаёт BM25-индекс одного поля в порядке doc_id из манифеста."""
    # Манифест задаёт одинаковый порядок объявлений для всех индексов.
    validation = Path(validation)
    out = Path(out)
    started = time.time()
    corpus = pl.read_parquet(validation / "corpus_manifest.parquet", columns=["item_id", "doc_id"])
    id_to_row = dict(corpus.iter_rows())
    n = corpus.height
    vocab = {}
    tokens_by_doc = [None] * n
    # Читаем тексты порциями, а в памяти оставляем числовые ID токенов.
    for batch in pq.ParquetFile(validation / "corpus.parquet").iter_batches(
        batch_size=2048, columns=["item_id", text_fields[field]], use_threads=False
    ):
        for row in batch.to_pylist():
            ids = []
            for token in tokenize(row[text_fields[field]], stemming=stemming):
                idx = vocab.get(token)
                if idx is None:
                    idx = len(vocab)
                    vocab[token] = idx
                ids.append(idx)
            tokens_by_doc[id_to_row[row["item_id"]]] = array("I", ids)
    # Не строим индекс, пока не собраны тексты всех ID из манифеста.
    if any(x is None for x in tokens_by_doc):
        raise ValueError("Для части объявлений не найден текст")
    log(
        "Tokenized",
        field,
        "stemming",
        stemming,
        "vocab",
        len(vocab),
        "tokens",
        sum(map(len, tokens_by_doc)),
        "seconds",
        round(time.time() - started, 1),
    )
    # Статистики BM25 считаются по текстам корпуса, без разметки запросов.
    retriever = bm25s.BM25(k1=1.5, b=0.75, method="lucene", idf_method="lucene", backend="numpy")
    retriever.index((tokens_by_doc, vocab), show_progress=False)
    retriever.save(str(out))
    # Параметры и хеши позволяют связать сохранённый индекс с исходным корпусом.
    meta = {
        "field": field,
        "stemming": stemming,
        "documents": n,
        "vocabulary": len(vocab),
        "tokens": sum(map(len, tokens_by_doc)),
        "nnz": len(retriever.scores["data"]),
        "seconds": time.time() - started,
        "k1": 1.5,
        "b": 0.75,
        "method": "lucene",
        "corpus_manifest_sha256": file_hash(validation / "corpus_manifest.parquet"),
        "corpus_sha256": file_hash(validation / "corpus.parquet"),
        "normalization": "NFKC lower ё→е; Unicode alphanumeric tokens; no stopword removal",
    }
    save_json(out / "build_metadata.json", meta)
    log("INDEX DONE", json.dumps(meta, ensure_ascii=False))
