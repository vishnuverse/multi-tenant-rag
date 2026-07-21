"""Shared Chroma and BM25 index contracts and persistence."""

from __future__ import annotations

import pickle
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

COLLECTION_NAME = "chunks"
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True, slots=True)
class BM25Store:
    bm25: BM25Okapi
    chunk_ids: tuple[str, ...]
    chunk_id_to_company: dict[str, str]


def tokenize_for_bm25(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


def build_bm25_store(chunks: Sequence[tuple[str, str, str]]) -> BM25Store:
    """Build a BM25 index from `(chunk_id, company, raw_text)` tuples."""

    chunk_ids = [chunk_id for chunk_id, _company, _text in chunks]
    corpus = [tokenize_for_bm25(text) for _chunk_id, _company, text in chunks]
    chunk_id_to_company = {chunk_id: company for chunk_id, company, _text in chunks}
    return BM25Store(
        bm25=BM25Okapi(corpus),
        chunk_ids=tuple(chunk_ids),
        chunk_id_to_company=chunk_id_to_company,
    )


def persist_bm25_store(path: Path, store: BM25Store) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bm25": store.bm25,
        "chunk_ids": list(store.chunk_ids),
        "chunk_id_to_company": store.chunk_id_to_company,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_bm25_store(path: Path) -> BM25Store:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return BM25Store(
        bm25=payload["bm25"],
        chunk_ids=tuple(payload["chunk_ids"]),
        chunk_id_to_company=payload["chunk_id_to_company"],
    )
