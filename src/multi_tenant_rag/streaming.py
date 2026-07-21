"""Process-local token streaming shared across workflow, AI, and UI."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token

TokenSink = Callable[[str], Awaitable[None]]

_token_sink: ContextVar[TokenSink | None] = ContextVar("token_sink", default=None)


def set_token_sink(sink: TokenSink | None) -> Token[TokenSink | None]:
    """Bind a sink for the current task; return a reset token."""

    return _token_sink.set(sink)


def reset_token_sink(token: Token[TokenSink | None]) -> None:
    """Restore the previous sink binding."""

    _token_sink.reset(token)


async def emit_token(token: str) -> None:
    """Forward a token to the active sink when one is bound."""

    if not token:
        return
    sink = _token_sink.get()
    if sink is not None:
        await sink(token)
