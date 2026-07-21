"""Company-filtered dense retrieval over ChromaDB."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import chromadb
from chromadb.api.models.Collection import Collection

from multi_tenant_rag.ai.embeddings import EmbeddingClient
from multi_tenant_rag.config import ConfigurationError, Settings, load_settings
from multi_tenant_rag.rag.index import (
    COLLECTION_NAME,
    BM25Store,
    load_bm25_store,
)

# Chart/table axis labels that dominate dense hits for theme/revenue queries.
_QUARTER_LABEL_RE = re.compile(r"\bQ[1-4]\s*20\d{2}\b", re.IGNORECASE)


def is_low_signal_chunk(text: str) -> bool:
    """True for titles, chart axes, and other chunks that cannot ground an answer."""

    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) < 80:
        return True
    words = cleaned.split()
    if len(words) < 12:
        return True
    quarter_hits = len(_QUARTER_LABEL_RE.findall(cleaned))
    # e.g. "Q3 2023 Q4 2023 Q1 2024 Q2 2024 Q3 2024 Q4 2024"
    if quarter_hits >= 3 and len(words) <= quarter_hits * 3:
        return True
    alpha = sum(character.isalpha() for character in cleaned)
    if quarter_hits >= 2 and alpha / max(len(cleaned), 1) < 0.4:
        return True
    # Cover/title lines: short, no figures, little prose substance.
    if len(cleaned) < 160 and len(words) <= 16:
        has_figure = bool(re.search(r"[$%•]|\d{1,3}(?:,\d{3})+", cleaned))
        if not has_figure:
            return True
    return False


def citation_label(chunk: Mapping[str, Any]) -> str:
    """Build the citation token supplied to answer-generation prompts."""

    company = str(chunk.get("company", "unknown")).upper()
    page = chunk.get("page", "?")
    doc_type = chunk.get("doc_type", "doc")
    metadata = chunk.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    quarter = metadata.get("quarter") or chunk.get("quarter")
    fiscal_year = metadata.get("fiscal_year") or chunk.get("fiscal_year")
    if quarter and fiscal_year:
        return f"{company}-{quarter}-FY{fiscal_year}-{doc_type}-p{page}"
    return f"{company}-{doc_type}-p{page}"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A ranked chunk returned from dense search."""

    chunk_id: str
    text: str
    company: str
    source: str
    page: int
    doc_type: str
    rrf_score: float
    rerank_score: float
    metadata: Mapping[str, str | int | float | bool]
    dense_rank: int | None = None
    sparse_rank: int | None = None


@dataclass(slots=True)
class RetrievalEngine:
    """Run access-controlled dense retrieval over Chroma."""

    settings: Settings
    collection: Collection
    embedder: EmbeddingClient = field(repr=False)
    bm25_store: BM25Store | None = None

    async def search(
        self,
        query: str,
        allowed_companies: Sequence[str],
        *,
        k: int | None = None,
        query_variants: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Embed once and return top-k company-filtered Chroma hits."""

        del query_variants  # Multi-query expansion is handled before retrieval.
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must be non-empty")

        allowed = _normalize_allowed_companies(allowed_companies)
        if not allowed:
            return []

        final_k = k if k is not None else min(3, self.settings.final_top_k)
        # Over-fetch so chart/title junk can be dropped without emptying top-k.
        fetch_k = max(final_k * 5, 15)
        embedding = await self.embedder.embed_query(normalized_query)
        response = await asyncio.to_thread(
            self._dense_query,
            embedding,
            allowed,
            fetch_k,
        )
        ids = list((response.get("ids") or [[]])[0])
        documents = list((response.get("documents") or [[]])[0])
        metadatas = list((response.get("metadatas") or [[]])[0])
        distances = list((response.get("distances") or [[]])[0])

        results: list[RetrievedChunk] = []
        dense_rank = 0
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            if not isinstance(metadata, Mapping):
                continue
            company = str(metadata.get("company", "")).casefold()
            if company not in allowed:
                continue
            chunk_text = str(text or "")
            if is_low_signal_chunk(chunk_text):
                continue
            dense_rank += 1
            # Cosine distance → similarity-like score for callers.
            score = max(0.0, 1.0 - float(distance))
            results.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    text=chunk_text,
                    company=str(metadata["company"]),
                    source=str(metadata.get("source", "")),
                    page=int(metadata.get("page", 0) or 0),
                    doc_type=str(metadata.get("doc_type", "doc")),
                    rrf_score=score,
                    rerank_score=score,
                    metadata=cast(dict[str, str | int | float | bool], dict(metadata)),
                    dense_rank=dense_rank,
                    sparse_rank=None,
                )
            )
            if len(results) >= final_k:
                break
        return results

    def _dense_query(
        self,
        embedding: Sequence[float],
        allowed_companies: frozenset[str],
        n_results: int,
    ) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self.collection.query(
                query_embeddings=cast(Any, [list(embedding)]),
                n_results=n_results,
                where=cast(
                    Any,
                    {"company": {"$in": sorted(allowed_companies)}},
                ),
                include=["documents", "metadatas", "distances"],
            ),
        )

    def warm_reranker(self) -> None:
        """No-op kept for call-site compatibility after hybrid removal."""

        return None


def merge_retrieved_chunks(
    fresh: Sequence[Mapping[str, Any]],
    prior: Sequence[Mapping[str, Any]],
    *,
    fresh_limit: int = 3,
    prior_limit: int = 2,
    total_limit: int = 5,
) -> list[dict[str, Any]]:
    """Prefer fresh hits, then previously cited chunks, deduped by chunk ID."""

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in list(fresh)[:fresh_limit] + list(prior)[:prior_limit]:
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        merged.append(dict(chunk))
        if len(merged) >= total_limit:
            break
    return merged


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    rrf_k: int,
) -> list[tuple[str, float]]:
    """Fuse ranked chunk-id lists with reciprocal rank fusion."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def top_bm25_chunk_ids(
    *,
    scores: Sequence[float],
    chunk_ids: Sequence[str],
    chunk_id_to_company: Mapping[str, str],
    allowed_companies: frozenset[str],
    top_k: int,
) -> list[str]:
    """Return top BM25 chunk ids restricted to allowed companies."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if len(scores) != len(chunk_ids):
        raise ValueError("scores and chunk_ids length mismatch")

    ranked = sorted(
        (
            (score, chunk_id)
            for score, chunk_id in zip(scores, chunk_ids, strict=True)
            if score > 0
            if chunk_id_to_company.get(chunk_id, "").casefold() in allowed_companies
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    return [chunk_id for _score, chunk_id in ranked[:top_k]]


def _normalize_allowed_companies(companies: Sequence[str]) -> frozenset[str]:
    normalized = {
        company.strip().casefold() for company in companies if company.strip()
    }
    return frozenset(normalized)


def build_retrieval_engine(settings: Settings | None = None) -> RetrievalEngine:
    """Construct a retrieval engine from on-disk indexes."""

    resolved = settings or load_settings()
    if not resolved.chroma_dir.is_dir():
        raise ConfigurationError(
            f"Chroma directory not found at {resolved.chroma_dir}. Run ingestion first."
        )

    client = chromadb.PersistentClient(path=str(resolved.chroma_dir))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() == 0:
        raise ConfigurationError(
            "Chroma collection is empty. Run ingestion before searching."
        )

    bm25_store = (
        load_bm25_store(resolved.bm25_index_path)
        if resolved.bm25_index_path.is_file()
        else None
    )
    return RetrievalEngine(
        settings=resolved,
        collection=collection,
        embedder=EmbeddingClient(resolved),
        bm25_store=bm25_store,
    )


_ENGINE: RetrievalEngine | None = None


def get_retrieval_engine(settings: Settings | None = None) -> RetrievalEngine:
    """Return a process-wide retrieval engine singleton."""

    global _ENGINE
    if _ENGINE is None or settings is not None:
        _ENGINE = build_retrieval_engine(settings)
    return _ENGINE
