"""Persistent storage adapters with lazy public exports."""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_chainlit_data_layer",
    "chainlit_conninfo",
    "close_persistence",
    "ensure_chainlit_schema",
    "get_async_checkpointer",
    "migrate_chainlit_schema",
    "repair_orphan_message_parents",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from multi_tenant_rag.storage import persistence

        return getattr(persistence, name)
    raise AttributeError(name)
