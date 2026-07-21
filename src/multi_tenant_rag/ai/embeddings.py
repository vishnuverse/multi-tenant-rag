"""Embedding clients for ingestion and query-time retrieval."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

from openai import APIError, AsyncOpenAI
from sentence_transformers import SentenceTransformer

from multi_tenant_rag.config import Settings
from multi_tenant_rag.telemetry import UsageCounters, usage_counters


class EmbeddingAPIError(RuntimeError):
    """A sanitized failure from the configured embedding service."""

    def __init__(self) -> None:
        super().__init__("Embedding service request failed")


class EmbeddingBackend(Protocol):
    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


@dataclass(slots=True)
class EmbeddingClient:
    """Embed text via OpenRouter or a local sentence-transformers model."""

    settings: Settings
    counters: UsageCounters | None = None
    _client: AsyncOpenAI | None = field(default=None, repr=False, compare=False)
    _local_model: SentenceTransformer | None = field(
        default=None, repr=False, compare=False
    )
    _query_cache: OrderedDict[str, list[float]] = field(
        default_factory=OrderedDict, repr=False, compare=False
    )

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.settings.use_local_embedding:
            return await asyncio.to_thread(self._embed_batch_local, list(texts))
        return await self._embed_batch_openrouter(list(texts))

    async def embed_query(self, text: str) -> list[float]:
        normalized = text.strip()
        if not normalized:
            raise ValueError("query text must be non-empty")

        cached = self._query_cache.get(normalized)
        if cached is not None:
            self._query_cache.move_to_end(normalized)
            return cached

        vector = (await self.embed_batch([normalized]))[0]
        self._remember_query(normalized, vector)
        return vector

    def _remember_query(self, text: str, vector: list[float]) -> None:
        self._query_cache[text] = vector
        self._query_cache.move_to_end(text)
        while len(self._query_cache) > 256:
            self._query_cache.popitem(last=False)

    def _embed_batch_local(self, texts: list[str]) -> list[list[float]]:
        model = self._local_model or SentenceTransformer(
            self.settings.local_embed_model
        )
        self._local_model = model
        vectors = model.encode(texts, normalize_embeddings=True)
        return [
            vector.tolist() if hasattr(vector, "tolist") else list(vector)
            for vector in vectors
        ]

    async def _embed_batch_openrouter(self, texts: list[str]) -> list[list[float]]:
        client = self._client or self._build_openrouter_client()
        self._client = client
        batch_size = self.settings.embed_batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                response = await client.embeddings.create(
                    model=self.settings.model_embed,
                    input=batch,
                )
            except APIError as exc:
                raise EmbeddingAPIError from exc
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
            tokens = response.usage.total_tokens if response.usage is not None else 0
            input_cost, _output_cost = self.settings.embedding_cost(
                self.settings.model_embed
            )
            self.usage.record_embedding(
                tokens=tokens,
                cost_usd=(tokens / 1_000_000) * input_cost,
            )
        return vectors

    @property
    def usage(self) -> UsageCounters:
        """Return explicit counters or the current request-local counters."""

        return self.counters or usage_counters()

    def _build_openrouter_client(self) -> AsyncOpenAI:
        headers: dict[str, str] = {}
        if self.settings.openrouter_app_url:
            headers["HTTP-Referer"] = self.settings.openrouter_app_url
        if self.settings.openrouter_app_title:
            headers["X-Title"] = self.settings.openrouter_app_title
        return AsyncOpenAI(
            api_key=self.settings.require_openrouter_api_key(),
            base_url=self.settings.openrouter_base_url,
            default_headers=headers or None,
        )


@lru_cache(maxsize=256)
def cached_local_query_embedding(model_name: str, text: str) -> tuple[float, ...]:
    """Process-local LRU cache for identical local query embeddings."""

    model = SentenceTransformer(model_name)
    vector = model.encode(text, normalize_embeddings=True)
    return tuple(float(value) for value in vector.tolist())
