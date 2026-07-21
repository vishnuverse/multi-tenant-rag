"""Disk persistence for Chainlit threads and LangGraph checkpoints."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from importlib import resources
from pathlib import Path
from typing import Any

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from multi_tenant_rag.config import Settings, load_settings

_CHECKPOINTER: AsyncSqliteSaver | None = None
_CHECKPOINTER_CONN: aiosqlite.Connection | None = None
_CHECKPOINTER_INIT_LOCK: asyncio.Lock | None = None
_SCHEMA_READY: set[str] = set()


def _register_sqlite_adapters() -> None:
    """SQLite cannot bind list/dict; store them as JSON strings."""

    sqlite3.register_adapter(list, json.dumps)
    sqlite3.register_adapter(dict, json.dumps)


# Columns Chainlit 2.11+ may write that older local schemas omit.
_STEPS_COLUMN_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("autoCollapse", "BOOLEAN"),
    ("defaultOpen", "BOOLEAN"),
    ("command", "TEXT"),
    ("modes", "JSON"),
    ("waitForAnswer", "BOOLEAN"),
    ("language", "TEXT"),
    ("indent", "INT"),
    ("tags", "JSON"),
)


def migrate_chainlit_schema(db_path: Path) -> list[str]:
    """Add any missing columns to an existing Chainlit SQLite DB."""

    if not db_path.is_file():
        return []
    added: list[str] = []
    with sqlite3.connect(db_path) as conn:
        existing = {
            row[1] for row in conn.execute('PRAGMA table_info("steps")').fetchall()
        }
        for column, sql_type in _STEPS_COLUMN_MIGRATIONS:
            if column in existing:
                continue
            conn.execute(f'ALTER TABLE steps ADD COLUMN "{column}" {sql_type}')
            added.append(column)
        conn.commit()
    return added


def repair_orphan_message_parents(db_path: Path) -> int:
    """Clear parentId on messages whose parent step is missing.

    Chainlit wraps ``on_message`` in a run step. Assistant replies inherit that
    parentId, but the run step is often absent from SQLite. On chat switch the
    UI nests replies under a missing parent and they disappear — only user
    questions remain visible.
    """

    if not db_path.is_file():
        return 0
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE steps
            SET "parentId" = NULL
            WHERE "type" IN ('assistant_message', 'user_message')
              AND "parentId" IS NOT NULL
              AND "parentId" NOT IN (SELECT "id" FROM steps)
            """
        )
        repaired = int(cursor.rowcount or 0)
        # Fill missing timestamps so resumed threads keep assistant replies ordered.
        stamped = conn.execute(
            """
            UPDATE steps
            SET "createdAt" = COALESCE(
                (
                    SELECT MAX(peer."createdAt")
                    FROM steps AS peer
                    WHERE peer."threadId" = steps."threadId"
                      AND peer."createdAt" IS NOT NULL
                ),
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            )
            WHERE "type" = 'assistant_message'
              AND ("createdAt" IS NULL OR "createdAt" = '')
              AND "output" IS NOT NULL
              AND TRIM("output") != ''
            """
        )
        conn.commit()
        return repaired + int(stamped.rowcount or 0)


def ensure_chainlit_schema(db_path: Path) -> None:
    """Create Chainlit tables if missing (idempotent)."""

    key = str(db_path.resolve())
    already = key in _SCHEMA_READY and db_path.is_file()

    _register_sqlite_adapters()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not already:
        schema_path = resources.files("multi_tenant_rag.storage").joinpath(
            "chainlit_schema.sql"
        )
        schema = schema_path.read_text(encoding="utf-8")
        with sqlite3.connect(db_path) as conn:
            conn.executescript(schema)
            conn.commit()
        _SCHEMA_READY.add(key)

    # Migrate + repair on every boot (idempotent).
    migrate_chainlit_schema(db_path)
    repair_orphan_message_parents(db_path)


def chainlit_conninfo(db_path: Path) -> str:
    ensure_chainlit_schema(db_path)
    # Absolute path so relative CWD changes do not break the connection.
    return f"sqlite+aiosqlite:///{db_path.resolve().as_posix()}"


def build_chainlit_data_layer(settings: Settings | None = None) -> Any:
    """Return a SQLAlchemy data layer backed by local SQLite."""

    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

    resolved = settings or load_settings()
    db_path = resolved.chainlit_db_path
    return SQLAlchemyDataLayer(
        conninfo=chainlit_conninfo(db_path),
        show_logger=False,
    )


async def get_async_checkpointer(
    settings: Settings | None = None,
) -> AsyncSqliteSaver:
    """Process-wide AsyncSqliteSaver for LangGraph conversation state."""

    global _CHECKPOINTER, _CHECKPOINTER_CONN, _CHECKPOINTER_INIT_LOCK
    if _CHECKPOINTER is not None:
        return _CHECKPOINTER

    if _CHECKPOINTER_INIT_LOCK is None:
        _CHECKPOINTER_INIT_LOCK = asyncio.Lock()
    async with _CHECKPOINTER_INIT_LOCK:
        if _CHECKPOINTER is not None:
            return _CHECKPOINTER

        resolved = settings or load_settings()
        db_path = resolved.checkpoint_db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(str(db_path))
        checkpointer = AsyncSqliteSaver(connection)
        try:
            await checkpointer.setup()
        except BaseException:
            await connection.close()
            raise
        _CHECKPOINTER_CONN = connection
        _CHECKPOINTER = checkpointer
        return checkpointer


async def close_persistence() -> None:
    """Close open SQLite connections (tests / shutdown)."""

    global _CHECKPOINTER, _CHECKPOINTER_CONN, _CHECKPOINTER_INIT_LOCK
    if _CHECKPOINTER_CONN is not None:
        await _CHECKPOINTER_CONN.close()
    _CHECKPOINTER = None
    _CHECKPOINTER_CONN = None
    _CHECKPOINTER_INIT_LOCK = None
