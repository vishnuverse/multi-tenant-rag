"""Conversational workflow with lazy public exports."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ChatState",
    "Intent",
    "apply_verification",
    "build_graph",
    "checkpoint_thread_id",
    "get_graph",
    "reset_graph_for_tests",
    "resolve_allowed_companies",
    "retrieved_chunk_to_dict",
]


def __getattr__(name: str) -> Any:
    if name in {"ChatState", "Intent"}:
        from multi_tenant_rag.workflow import state

        return getattr(state, name)
    if name in {
        "apply_verification",
        "build_graph",
        "checkpoint_thread_id",
        "get_graph",
        "reset_graph_for_tests",
        "resolve_allowed_companies",
        "retrieved_chunk_to_dict",
    }:
        from multi_tenant_rag.workflow import graph

        return getattr(graph, name)
    raise AttributeError(name)
