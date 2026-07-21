"""Compatibility re-exports for the cross-cutting streaming sink."""

from multi_tenant_rag.streaming import (
    TokenSink,
    emit_token,
    reset_token_sink,
    set_token_sink,
)

__all__ = ["TokenSink", "emit_token", "reset_token_sink", "set_token_sink"]
