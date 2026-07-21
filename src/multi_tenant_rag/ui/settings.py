"""Validated Chainlit model settings backed by the OpenRouter catalog."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import chainlit as cl
from chainlit.input_widget import Select

from multi_tenant_rag.ai.catalog import (
    CatalogResult,
    ChatModel,
    catalog_for,
)
from multi_tenant_rag.config import Settings, load_settings

MODEL_SETTING_ID = "model"


@dataclass(frozen=True, slots=True)
class ModelSelection:
    """A validated canonical model ID and optional user-facing notice."""

    model_id: str
    notice: str | None = None
    pending_model_id: str | None = None


def build_model_options(models: Sequence[ChatModel]) -> dict[str, str]:
    """Map display labels to canonical IDs for Chainlit's Select widget."""

    return {model.label: model.id for model in models}


def _model_map(models: Sequence[ChatModel]) -> dict[str, ChatModel]:
    return {model.id: model for model in models}


def resolve_model_selection(
    requested_model: object,
    catalog: CatalogResult,
    *,
    default_model: str,
) -> ModelSelection:
    """Restore an available ID or choose the configured/safest catalog default."""

    models_by_id = _model_map(catalog.models)
    if isinstance(requested_model, str) and requested_model in models_by_id:
        notice = (
            "The live model catalog is unavailable; using the last known catalog."
            if catalog.stale
            else None
        )
        return ModelSelection(requested_model, notice)

    selected = models_by_id.get(default_model)
    if selected is None:
        if not catalog.models:
            raise ValueError("The compatible model catalog is empty")
        selected = catalog.models[0]

    if isinstance(requested_model, str) and requested_model and catalog.outage:
        notice = (
            "The live model catalog is unavailable, so the saved model could not "
            f"be revalidated. Temporarily using {selected.name}."
        )
        return ModelSelection(
            selected.id,
            notice,
            pending_model_id=requested_model,
        )
    if isinstance(requested_model, str) and requested_model:
        notice = f"The saved model is unavailable; using {selected.name} instead."
    elif catalog.used_fallback:
        notice = "The live model catalog is unavailable; using fallback models."
    else:
        notice = None
    return ModelSelection(selected.id, notice)


def resolve_settings_update(
    payload: Mapping[str, object],
    *,
    current_model: object,
    pending_model: object = None,
    catalog: CatalogResult,
    default_model: str,
) -> ModelSelection:
    """Validate a browser settings payload without trusting its model ID."""

    models_by_id = _model_map(catalog.models)
    candidate = payload.get(MODEL_SETTING_ID)
    if isinstance(candidate, str) and candidate in models_by_id:
        return ModelSelection(candidate)

    if isinstance(current_model, str) and current_model in models_by_id:
        safe = models_by_id[current_model]
    else:
        fallback = resolve_model_selection(
            None,
            catalog,
            default_model=default_model,
        )
        safe = models_by_id[fallback.model_id]
    if catalog.outage and isinstance(pending_model, str) and pending_model:
        return ModelSelection(
            safe.id,
            "The live model catalog is unavailable; the requested model could "
            f"not be validated. Continuing temporarily with {safe.name}.",
            pending_model_id=pending_model,
        )
    return ModelSelection(
        safe.id,
        f"That model is unavailable; restored {safe.name}.",
    )


def model_from_thread(
    thread: Mapping[str, Any],
    *,
    session_chat_settings: object = None,
) -> str | None:
    """Read the model persisted by Chainlit, preferring restored session metadata."""

    if isinstance(session_chat_settings, Mapping):
        session_model = session_chat_settings.get(MODEL_SETTING_ID)
        if isinstance(session_model, str):
            return session_model
    metadata = thread.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    chat_settings = metadata.get("chat_settings")
    if not isinstance(chat_settings, Mapping):
        return None
    model = chat_settings.get(MODEL_SETTING_ID)
    return model if isinstance(model, str) else None


def build_chat_settings(
    models: Sequence[ChatModel],
    *,
    selected_model: str,
) -> cl.ChatSettings:
    """Create Chainlit's searchable model dropdown."""

    return cl.ChatSettings(
        [
            Select(
                id=MODEL_SETTING_ID,
                label="Model",
                items=build_model_options(models),
                initial_value=selected_model,
                tooltip="Choose the model used for every LLM call in this chat.",
            )
        ]
    )


async def current_catalog(settings: Settings | None = None) -> CatalogResult:
    """Load the process-cached catalog for UI handlers."""

    resolved_settings = settings or load_settings()
    return await catalog_for(resolved_settings).get_models()


async def synchronize_model_settings(
    selection: ModelSelection,
    catalog: CatalogResult,
) -> None:
    """Synchronize the canonical model across session and Chainlit metadata/UI."""

    cl.user_session.set("selected_model", selection.model_id)
    cl.user_session.set("pending_model_id", selection.pending_model_id)
    chat_settings = build_chat_settings(
        catalog.models,
        selected_model=selection.model_id,
    )
    if selection.pending_model_id is not None:
        await chat_settings.refresh()
        return
    await chat_settings.send()


async def send_model_settings(
    requested_model: object = None,
    *,
    settings: Settings | None = None,
) -> ModelSelection:
    """Validate a selection, send the dropdown, and bind its canonical ID."""

    resolved_settings = settings or load_settings()
    catalog = await current_catalog(resolved_settings)
    selection = resolve_model_selection(
        requested_model,
        catalog,
        default_model=resolved_settings.default_chat_model,
    )
    await synchronize_model_settings(selection, catalog)
    return selection


@cl.on_settings_update
async def on_settings_update(payload: dict[str, Any]) -> None:
    """Reject stale/injected IDs and restore a safe canonical selection."""

    settings = load_settings()
    catalog = await current_catalog(settings)
    selection = resolve_settings_update(
        payload,
        current_model=cl.user_session.get("selected_model"),
        pending_model=cl.user_session.get("pending_model_id"),
        catalog=catalog,
        default_model=settings.default_chat_model,
    )
    await synchronize_model_settings(selection, catalog)
    if selection.notice:
        await cl.Message(content=selection.notice).send()
