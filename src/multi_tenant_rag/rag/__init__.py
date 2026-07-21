"""Retrieval-augmented generation domain with lazy public exports."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BBox",
    "RetrievedChunk",
    "RetrievalEngine",
    "parse_bboxes",
    "serialize_bboxes",
]


def __getattr__(name: str) -> Any:
    if name in {"BBox", "parse_bboxes", "serialize_bboxes"}:
        from multi_tenant_rag.rag import geometry

        return getattr(geometry, name)
    if name in {"RetrievedChunk", "RetrievalEngine"}:
        from multi_tenant_rag.rag import retrieval

        return getattr(retrieval, name)
    raise AttributeError(name)
