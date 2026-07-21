"""State schema shared by the conversational workflow."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

Intent = Literal["SMALL_TALK", "META", "DOC"]


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    email: str
    selected_model: str | None
    allowed_companies: list[str]
    intent: Intent | None
    last_user_question: str | None
    standalone_query: str | None
    retrieved: list[dict[str, Any]]
    answer: str | None
    verification: dict[str, Any] | None
    node_timings: dict[str, float]
