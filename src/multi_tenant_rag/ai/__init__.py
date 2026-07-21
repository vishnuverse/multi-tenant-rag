"""AI clients with lazy public exports."""

from __future__ import annotations

from typing import Any

__all__ = [
    "EmbeddingBackend",
    "EmbeddingClient",
    "LLMClient",
    "build_openrouter_client",
    "parse_json_object",
]


def __getattr__(name: str) -> Any:
    if name in {"EmbeddingBackend", "EmbeddingClient"}:
        from multi_tenant_rag.ai import embeddings

        return getattr(embeddings, name)
    if name in {"LLMClient", "build_openrouter_client", "parse_json_object"}:
        from multi_tenant_rag.ai import client

        return getattr(client, name)
    raise AttributeError(name)
