"""Semantic answer cache for near-duplicate document questions."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from multi_tenant_rag.config import Settings


def index_version_path(data_dir: Path | str) -> Path:
    """Return the path of the persisted retrieval index version marker."""

    return Path(data_dir) / "index_version.txt"


def read_index_version(data_dir: Path | str) -> str:
    """Read the current index version, or ``unknown`` when missing."""

    path = index_version_path(data_dir)
    if not path.exists():
        return "unknown"
    value = path.read_text(encoding="utf-8").strip()
    return value or "unknown"


def write_index_version(data_dir: Path | str, version: str) -> Path:
    """Persist the current retrieval index version."""

    path = index_version_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{version.strip()}\n", encoding="utf-8")
    return path


@dataclass(frozen=True, slots=True)
class CacheEntry:
    email: str
    allowed_companies: tuple[str, ...]
    model_id: str
    query_embedding: tuple[float, ...]
    chunk_ids: tuple[str, ...]
    answer: str
    retrieved: tuple[dict[str, str | int | float | bool], ...]
    created_at: float
    index_version: str = "unknown"


@dataclass(slots=True)
class SemanticAnswerCache:
    """In-memory semantic cache keyed by user, ACL scope, and embedding."""

    settings: Settings
    _entries: OrderedDict[str, CacheEntry] = field(
        default_factory=OrderedDict,
        repr=False,
        compare=False,
    )

    def lookup(
        self,
        *,
        email: str,
        allowed_companies: Sequence[str],
        model_id: str,
        query_embedding: Sequence[float],
        chunk_ids: Sequence[str] | None = None,
        index_version: str | None = None,
    ) -> CacheEntry | None:
        now = time.time()
        self._evict_expired(now)
        normalized_email = email.strip().casefold()
        normalized_scope = _normalize_scope(allowed_companies)
        normalized_model = _normalize_model_id(model_id)
        current_version = index_version or read_index_version(self.settings.data_dir)
        query_vector = tuple(float(value) for value in query_embedding)
        best: CacheEntry | None = None
        best_score = -1.0

        for entry in self._entries.values():
            if entry.email != normalized_email:
                continue
            if entry.allowed_companies != normalized_scope:
                continue
            if entry.model_id != normalized_model:
                continue
            if entry.index_version != current_version:
                continue
            if any(
                str(chunk.get("company", "")).casefold() not in normalized_scope
                for chunk in entry.retrieved
                if chunk.get("company")
            ):
                continue
            score = cosine_similarity(query_vector, entry.query_embedding)
            if score < self.settings.semantic_cache_threshold:
                continue
            if chunk_ids is not None and tuple(chunk_ids) != entry.chunk_ids:
                continue
            if score > best_score:
                best = entry
                best_score = score
        if best is not None:
            self._entries.move_to_end(self._entry_key(best))
        return best

    def store(
        self,
        *,
        email: str,
        allowed_companies: Sequence[str],
        model_id: str,
        query_embedding: Sequence[float],
        chunk_ids: Sequence[str],
        answer: str,
        retrieved: Sequence[dict[str, str | int | float | bool]],
        index_version: str | None = None,
    ) -> None:
        now = time.time()
        version = index_version or read_index_version(self.settings.data_dir)
        entry = CacheEntry(
            email=email.strip().casefold(),
            allowed_companies=_normalize_scope(allowed_companies),
            model_id=_normalize_model_id(model_id),
            query_embedding=tuple(float(value) for value in query_embedding),
            chunk_ids=tuple(chunk_ids),
            answer=answer,
            retrieved=tuple(retrieved),
            created_at=now,
            index_version=version,
        )
        key = self._entry_key(entry)
        self._entries[key] = entry
        self._entries.move_to_end(key)
        self._evict_expired(now)
        while len(self._entries) > self.settings.semantic_cache_max_entries:
            self._entries.popitem(last=False)

    def _entry_key(self, entry: CacheEntry) -> str:
        return "|".join(
            [
                entry.email,
                ",".join(entry.allowed_companies),
                entry.model_id,
                entry.index_version,
                str(hash(entry.query_embedding)),
                ",".join(entry.chunk_ids),
            ]
        )

    def _evict_expired(self, now: float) -> None:
        ttl = self.settings.semantic_cache_ttl_seconds
        expired = [
            key for key, entry in self._entries.items() if now - entry.created_at > ttl
        ]
        for key in expired:
            self._entries.pop(key, None)


def _normalize_scope(companies: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        sorted({company.strip().casefold() for company in companies if company.strip()})
    )


def _normalize_model_id(model_id: str) -> str:
    normalized = model_id.strip()
    if not normalized:
        raise ValueError("model_id must not be blank")
    return normalized


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
