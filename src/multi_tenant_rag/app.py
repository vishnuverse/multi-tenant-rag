"""Thin Chainlit entry point; importing chat registers all handlers."""

from multi_tenant_rag.ui.chat import (
    configure_data_layer,
    on_chat_resume,
    on_chat_start,
    on_message,
    password_auth,
)

__all__ = [
    "configure_data_layer",
    "on_chat_resume",
    "on_chat_start",
    "on_message",
    "password_auth",
]
