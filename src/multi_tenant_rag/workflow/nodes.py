"""Reusable node helpers for the conversational workflow."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from multi_tenant_rag.ai.policy import resolve_chat_model
from multi_tenant_rag.ai.prompts import (
    ANSWER_FROM_DOCS_SYSTEM,
    CHITCHAT_REPLY_SYSTEM,
    META_REPLY_SYSTEM,
    UNVERIFIED_REFUSAL,
    canned_meta_reply,
    canned_small_talk,
)
from multi_tenant_rag.config import Settings
from multi_tenant_rag.rag.retrieval import (
    RetrievedChunk,
    citation_label,
    merge_retrieved_chunks,
)
from multi_tenant_rag.rag.verification import drop_unverified_sentences, verify_answer
from multi_tenant_rag.streaming import emit_token
from multi_tenant_rag.workflow.routing import expand_search_query, heuristic_intent
from multi_tenant_rag.workflow.state import ChatState, Intent


def resolve_allowed_companies(
    settings: Settings,
    email: str,
    requested: Sequence[str] | None = None,
) -> list[str]:
    """Return authorized companies for ``email``, ignoring caller widening."""

    del requested
    return list(settings.allowed_companies(email))


def apply_verification(
    answer: str,
    chunks: Sequence[Mapping[str, str | int | float | bool] | str],
) -> tuple[str, dict[str, int | list[str]]]:
    """Verify figures; drop unsupported claims, refuse only if nothing remains."""

    verification = verify_answer(answer, chunks)
    unverified = verification.get("unverified") or []
    if not unverified:
        return answer, verification
    if not isinstance(unverified, list):
        return UNVERIFIED_REFUSAL, verification
    trimmed = drop_unverified_sentences(answer, [str(item) for item in unverified])
    if trimmed:
        return trimmed, verification
    return UNVERIFIED_REFUSAL, verification


def retrieved_chunk_to_dict(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "company": chunk.company,
        "source": chunk.source,
        "page": chunk.page,
        "doc_type": chunk.doc_type,
        "score": chunk.rerank_score,
        "rrf_score": chunk.rrf_score,
        "metadata": dict(chunk.metadata),
    }


def latest_user_message(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def history_text(messages: Sequence[BaseMessage]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in messages[-6:]:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        history.append({"role": role, "content": str(message.content)})
    return history


def message_dicts(messages: Sequence[BaseMessage]) -> list[dict[str, str]]:
    return [
        {
            "role": "assistant" if isinstance(message, AIMessage) else "user",
            "content": str(message.content),
        }
        for message in messages[-6:]
    ]


def format_excerpts(chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        formatted.append(
            {
                "citation": citation_label(chunk),
                "text": chunk.get("text", ""),
                "company": chunk.get("company"),
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "doc_type": chunk.get("doc_type"),
                "speaker": metadata.get("speaker"),
                "quarter": metadata.get("quarter"),
                "fiscal_year": metadata.get("fiscal_year"),
            }
        )
    return formatted


def record_timing(
    state: ChatState,
    node: str,
    started: float,
) -> dict[str, float]:
    timings = dict(state.get("node_timings") or {})
    timings[node] = (time.perf_counter() - started) * 1000
    return timings


@dataclass(slots=True)
class WorkflowNodes:
    """Stateful workflow node set with explicit runtime dependencies."""

    settings: Settings
    llm: Any
    retrieval: Any

    async def classify_intent(self, state: ChatState) -> dict[str, Any]:
        """Heuristic-only routing; document questions never call a model."""

        started = time.perf_counter()
        allowed = resolve_allowed_companies(
            self.settings,
            state["email"],
            state.get("allowed_companies"),
        )
        user_text = latest_user_message(state["messages"])
        intent: Intent = heuristic_intent(user_text) or "DOC"
        return {
            "intent": intent,
            "allowed_companies": allowed,
            "node_timings": record_timing(state, "classify_intent", started),
        }

    async def chitchat_reply(self, state: ChatState) -> dict[str, Any]:
        started = time.perf_counter()
        user_text = latest_user_message(state["messages"])
        if heuristic_intent(user_text) == "SMALL_TALK":
            answer = canned_small_talk(user_text)
            await emit_token(answer)
        else:
            answer = await self.llm.stream_chat_completion(
                cast(
                    Any,
                    [
                        {"role": "system", "content": CHITCHAT_REPLY_SYSTEM},
                        *message_dicts(state["messages"]),
                    ],
                ),
                model=resolve_chat_model(state.get("selected_model"), self.settings),
            )
            answer = answer.strip()
        return {
            "answer": answer,
            "retrieved": [],
            "verification": None,
            "messages": [AIMessage(content=answer)],
            "node_timings": record_timing(state, "chitchat_reply", started),
        }

    async def meta_reply(self, state: ChatState) -> dict[str, Any]:
        started = time.perf_counter()
        allowed = resolve_allowed_companies(
            self.settings,
            state["email"],
            state.get("allowed_companies"),
        )
        user_text = latest_user_message(state["messages"])
        if heuristic_intent(user_text) == "META":
            answer = canned_meta_reply(allowed)
            await emit_token(answer)
        else:
            answer = await self.llm.stream_chat_completion(
                [
                    {"role": "system", "content": META_REPLY_SYSTEM},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "allowed_companies": allowed,
                                "question": user_text,
                            }
                        ),
                    },
                ],
                model=resolve_chat_model(state.get("selected_model"), self.settings),
            )
            answer = answer.strip()
        return {
            "answer": answer,
            "allowed_companies": allowed,
            "retrieved": [],
            "verification": None,
            "messages": [AIMessage(content=answer)],
            "node_timings": record_timing(state, "meta_reply", started),
        }

    async def expand_query(self, state: ChatState) -> dict[str, Any]:
        """Deterministic follow-up expansion for the search embedding only."""

        started = time.perf_counter()
        allowed = resolve_allowed_companies(
            self.settings,
            state["email"],
            state.get("allowed_companies"),
        )
        user_text = latest_user_message(state["messages"])
        search_query = expand_search_query(user_text, state.get("last_user_question"))
        return {
            "standalone_query": search_query,
            "last_user_question": user_text,
            "allowed_companies": allowed,
            "node_timings": record_timing(state, "expand_query", started),
        }

    async def retrieve(self, state: ChatState) -> dict[str, Any]:
        """Fresh filtered search, merged with up to two prior cited chunks."""

        started = time.perf_counter()
        allowed = resolve_allowed_companies(
            self.settings,
            state["email"],
            state.get("allowed_companies"),
        )
        query = state.get("standalone_query") or latest_user_message(state["messages"])
        prior = list(state.get("retrieved") or [])
        chunks = await self.retrieval.search(query, allowed, k=3)
        fresh = [retrieved_chunk_to_dict(chunk) for chunk in chunks]
        merged = merge_retrieved_chunks(fresh, prior)
        return {
            "allowed_companies": allowed,
            "retrieved": merged,
            "node_timings": record_timing(state, "retrieve", started),
        }

    async def answer_from_docs(self, state: ChatState) -> dict[str, Any]:
        started = time.perf_counter()
        allowed = resolve_allowed_companies(
            self.settings,
            state["email"],
            state.get("allowed_companies"),
        )
        question = latest_user_message(state["messages"])
        excerpts = format_excerpts(state.get("retrieved") or [])
        answer = await self.llm.stream_chat_completion(
            [
                {"role": "system", "content": ANSWER_FROM_DOCS_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "allowed_companies": allowed,
                            "question": question,
                            "history": history_text(state["messages"][:-1]),
                            "excerpts": excerpts,
                        }
                    ),
                },
            ],
            model=resolve_chat_model(state.get("selected_model"), self.settings),
        )
        cleaned = answer.strip()
        return {
            "answer": cleaned,
            "allowed_companies": allowed,
            "node_timings": record_timing(state, "answer_from_docs", started),
        }

    async def verify_numbers(self, state: ChatState) -> dict[str, Any]:
        started = time.perf_counter()
        final_answer, verification = apply_verification(
            state.get("answer") or "",
            state.get("retrieved") or [],
        )
        return {
            "answer": final_answer,
            "verification": verification,
            "messages": [AIMessage(content=final_answer)],
            "node_timings": record_timing(state, "verify_numbers", started),
        }
