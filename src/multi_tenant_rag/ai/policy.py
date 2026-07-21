"""Model-selection policy shared by workflow nodes."""

from __future__ import annotations

from multi_tenant_rag.config import Settings


def resolve_chat_model(selected_model: str | None, settings: Settings) -> str:
    """Use a request override when present, otherwise the configured default."""

    override = (selected_model or "").strip()
    return override or settings.default_chat_model
