"""LangGraph construction and lifecycle for multi-tenant document Q&A."""

from __future__ import annotations

import asyncio
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from multi_tenant_rag.ai.client import LLMClient
from multi_tenant_rag.config import Settings, load_settings
from multi_tenant_rag.rag.retrieval import get_retrieval_engine
from multi_tenant_rag.workflow.nodes import (
    WorkflowNodes,
)
from multi_tenant_rag.workflow.nodes import (
    apply_verification as apply_verification,
)
from multi_tenant_rag.workflow.nodes import (
    resolve_allowed_companies as resolve_allowed_companies,
)
from multi_tenant_rag.workflow.nodes import (
    retrieved_chunk_to_dict as retrieved_chunk_to_dict,
)
from multi_tenant_rag.workflow.routing import route_intent
from multi_tenant_rag.workflow.state import ChatState


def build_graph(
    settings: Settings | None = None,
    *,
    checkpointer: Any | None = None,
) -> Any:
    """Compile the LangGraph application with explicit node dependencies."""

    resolved = settings or load_settings()
    nodes = WorkflowNodes(
        settings=resolved,
        llm=LLMClient(resolved),
        retrieval=get_retrieval_engine(resolved),
    )
    saver = checkpointer if checkpointer is not None else MemorySaver()

    graph = StateGraph(ChatState)
    graph.add_node("classify_intent", nodes.classify_intent)
    graph.add_node("chitchat_reply", nodes.chitchat_reply)
    graph.add_node("meta_reply", nodes.meta_reply)
    graph.add_node("expand_query", nodes.expand_query)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("answer_from_docs", nodes.answer_from_docs)
    graph.add_node("verify_numbers", nodes.verify_numbers)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "chitchat_reply": "chitchat_reply",
            "meta_reply": "meta_reply",
            "expand_query": "expand_query",
        },
    )
    graph.add_edge("chitchat_reply", END)
    graph.add_edge("meta_reply", END)
    graph.add_edge("expand_query", "retrieve")
    graph.add_edge("retrieve", "answer_from_docs")
    graph.add_edge("answer_from_docs", "verify_numbers")
    graph.add_edge("verify_numbers", END)

    return graph.compile(checkpointer=saver)


_GRAPH: Any | None = None
_GRAPH_INIT_LOCK: asyncio.Lock | None = None


def checkpoint_thread_id(email: str, session_id: str) -> str:
    """Namespace checkpoint identity by authenticated user."""

    return f"{email.strip().casefold()}:{session_id}"


async def get_graph(settings: Settings | None = None) -> Any:
    """Return the process-wide graph with a persistent SQLite checkpointer."""

    global _GRAPH, _GRAPH_INIT_LOCK
    if _GRAPH is not None:
        return _GRAPH

    if _GRAPH_INIT_LOCK is None:
        _GRAPH_INIT_LOCK = asyncio.Lock()
    async with _GRAPH_INIT_LOCK:
        if _GRAPH is not None:
            return _GRAPH
        from multi_tenant_rag.storage.persistence import get_async_checkpointer

        resolved = settings or load_settings()
        checkpointer = await get_async_checkpointer(resolved)
        graph = build_graph(resolved, checkpointer=checkpointer)
        _GRAPH = graph
        return graph


def reset_graph_for_tests() -> None:
    """Drop the cached graph (unit tests only)."""

    global _GRAPH, _GRAPH_INIT_LOCK
    _GRAPH = None
    _GRAPH_INIT_LOCK = None
