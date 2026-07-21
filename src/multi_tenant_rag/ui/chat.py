"""Chainlit UI for the multi-tenant RAG demo."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import chainlit as cl
from chainlit.data import get_data_layer as get_active_data_layer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from multi_tenant_rag.ai.client import ChatModelAPIError
from multi_tenant_rag.ai.embeddings import EmbeddingAPIError
from multi_tenant_rag.config import load_settings
from multi_tenant_rag.storage.persistence import build_chainlit_data_layer
from multi_tenant_rag.streaming import reset_token_sink, set_token_sink
from multi_tenant_rag.telemetry import reset_usage_counters, turn_telemetry
from multi_tenant_rag.ui.citations import PdfCitation, select_pdf_citations
from multi_tenant_rag.ui.pdf import (
    render_highlighted_pdf,
    resolve_highlight_bboxes,
    resolve_pdf_path,
)
from multi_tenant_rag.ui.settings import (
    current_catalog,
    model_from_thread,
    resolve_model_selection,
    send_model_settings,
    synchronize_model_settings,
)
from multi_tenant_rag.workflow import checkpoint_thread_id, get_graph

logger = logging.getLogger(__name__)

load_dotenv()


@cl.data_layer
def configure_data_layer() -> Any:
    """Persist threads/messages in SQLite under DATA_DIR."""

    return build_chainlit_data_layer()


def _bind_user_session(user: cl.User) -> list[str]:
    email = str(user.identifier).strip().lower()
    companies = list(load_settings().allowed_companies(email))
    if not companies:
        companies = list(user.metadata.get("companies", []))
    cl.user_session.set("email", email)
    cl.user_session.set("companies", companies)
    return companies


@cl.password_auth_callback
def password_auth(username: str, password: str) -> cl.User | None:
    del password
    settings = load_settings()
    email = username.strip().lower()
    companies = settings.allowed_companies(email)
    if not companies:
        return None
    return cl.User(
        identifier=email,
        metadata={"companies": list(companies)},
    )


def _toplevel_message(
    content: str,
    *,
    elements: list[cl.Pdf] | None = None,
) -> cl.Message:
    """Build a message that stays visible when switching chats / resuming history.

    Chainlit wraps handlers in a run step; child messages get that parentId. If
    the parent isn't persisted, the UI hides the reply on thread resume.
    """

    message = cl.Message(content=content, elements=elements or [])
    message.parent_id = None
    return message


def build_welcome_content(
    identifier: str,
    companies: list[str],
    *,
    notice: str | None,
) -> str:
    """Build signed-in guidance without coupling its contract to Chainlit DOM."""

    content = (
        f"Signed in as **{identifier}**. "
        f"You can query: {', '.join(companies)}.\n\n"
        "Ask about earnings, margins, revenue, or speakers in your documents.\n\n"
        "Use the settings button beside the message box to choose the OpenRouter "
        "model and see pricing."
    )
    return content + (f"\n\n{notice}" if notice else "")


@cl.on_chat_start
async def on_chat_start() -> None:
    user = cl.user_session.get("user")
    if user is None:
        return
    companies = _bind_user_session(user)
    # Warm graph/retrieval/reranker off the critical path of the first question.
    await get_graph()
    selection = await send_model_settings()
    welcome = _toplevel_message(
        build_welcome_content(
            str(user.identifier),
            companies,
            notice=selection.notice,
        )
    )
    await welcome.send()
    await _persist_toplevel_step(welcome)


@cl.on_chat_resume
async def on_chat_resume(thread: dict[str, Any]) -> None:
    """Rebind ACL session state when opening a saved thread from history."""

    user = cl.user_session.get("user")
    if user is None:
        return
    _bind_user_session(user)
    await get_graph()
    restored_model = model_from_thread(
        thread,
        session_chat_settings=cl.user_session.get("chat_settings"),
    )
    selection = await send_model_settings(restored_model)
    if selection.notice:
        await _toplevel_message(selection.notice).send()
    # Fix any orphan parentIds left by earlier turns before the UI renders.
    from multi_tenant_rag.storage.persistence import repair_orphan_message_parents

    repair_orphan_message_parents(load_settings().chainlit_db_path)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    email = cl.user_session.get("email")
    if not email:
        await _toplevel_message("Please sign in to continue.").send()
        return
    settings = load_settings()
    companies = list(settings.allowed_companies(email))
    if not companies:
        await _toplevel_message("Your account no longer has document access.").send()
        return

    catalog = await current_catalog(settings)
    current_model = cl.user_session.get("selected_model")
    requested_model = cl.user_session.get("pending_model_id") or current_model
    selection = resolve_model_selection(
        requested_model,
        catalog,
        default_model=settings.default_chat_model,
    )
    selected_model = selection.model_id
    if (
        current_model != selected_model
        or selection.pending_model_id != cl.user_session.get("pending_model_id")
    ):
        await synchronize_model_settings(selection, catalog)
    if selection.notice:
        await _toplevel_message(selection.notice).send()

    reset_usage_counters()
    telemetry = turn_telemetry()
    telemetry.start_turn()
    graph = await get_graph()
    thread_id = checkpoint_thread_id(email, cl.context.session.thread_id)
    bypass_cache = message.content.strip().lower().startswith("/regenerate")
    user_text = message.content.removeprefix("/regenerate").strip() or message.content

    input_state = _build_input_state(
        user_text,
        email=email,
        companies=companies,
        selected_model=selected_model,
        bypass_cache=bypass_cache,
    )
    config = {"configurable": {"thread_id": thread_id}}

    # Do NOT send() an empty message first — that persists output="" and can
    # race with the final update, wiping assistant replies from history.
    # parent_id must stay None so chat-switch / history resume still shows replies.
    reply = _toplevel_message("")

    async def _sink(token: str) -> None:
        reply.parent_id = None
        await reply.stream_token(token)

    sink_token = set_token_sink(_sink)
    final_state: dict[str, Any] | None = None
    step_lines: list[str] = []
    api_error: ChatModelAPIError | EmbeddingAPIError | None = None
    try:
        async with cl.Step(name="Running assistant pipeline") as pipeline_step:
            async for event in graph.astream_events(
                input_state,
                config=config,
                version="v2",
            ):
                if event["event"] == "on_chain_end" and event.get("name") in {
                    "classify_intent",
                    "expand_query",
                    "retrieve",
                    "answer_from_docs",
                    "verify_numbers",
                    "chitchat_reply",
                    "meta_reply",
                }:
                    output = event.get("data", {}).get("output") or {}
                    if isinstance(output, dict):
                        line = _step_line(event["name"], output)
                        if line:
                            step_lines.append(line)
                        final_state = _merge_state(final_state, output)
                if (
                    event["event"] == "on_chain_end"
                    and event.get("name") == "LangGraph"
                ):
                    output = event.get("data", {}).get("output")
                    if isinstance(output, dict):
                        final_state = output

            if final_state is None:
                snapshot = await graph.aget_state(config)
                final_state = dict(snapshot.values)

            pipeline_step.output = "\n".join(step_lines) or _summary_line(final_state)
    except ChatModelAPIError as exc:
        api_error = exc
        logger.warning("Selected model %s failed", exc.model_id)
    except EmbeddingAPIError as exc:
        api_error = exc
        logger.warning("Embedding service request failed")
    finally:
        reset_token_sink(sink_token)

    telemetry_error: str | None = None
    retrieved: list[dict[str, Any]] = []
    elements: list[cl.Pdf] = []
    intent: Any = None
    standalone_query: Any = None
    verification: Any = None
    node_timings: dict[str, float] = {}
    if isinstance(api_error, ChatModelAPIError):
        telemetry.counters.record_missing_llm_usage(model_id=api_error.model_id)
        answer = _model_error_message(api_error.model_id)
        telemetry_error = "model_api_error"
    elif isinstance(api_error, EmbeddingAPIError):
        answer = _embedding_error_message()
        telemetry_error = "embedding_api_error"
    else:
        assert final_state is not None
        answer = str(final_state.get("answer") or "I could not generate a response.")
        retrieved = final_state.get("retrieved") or []
        answer, citations = select_pdf_citations(answer, retrieved)
        elements = _build_pdf_elements(
            citations,
            pdf_dir=settings.pdf_dir,
            allowed_companies=companies,
        )
        intent = final_state.get("intent")
        standalone_query = final_state.get("standalone_query")
        verification = final_state.get("verification")
        node_timings = {
            str(node): float(latency)
            for node, latency in (final_state.get("node_timings") or {}).items()
        }

    # Element names must appear verbatim in content so Chainlit linkifies them.
    await _finalize_assistant_reply(reply, answer=answer, elements=elements)
    if elements:
        try:
            await cl.ElementSidebar.set_title("Sources")
            await cl.ElementSidebar.set_elements(elements)
        except Exception:
            logger.exception("Failed to open element sidebar for PDF sources")

    telemetry.write_turn(
        thread_id=thread_id,
        user=email,
        selected_model=selected_model,
        intent=intent,
        standalone_query=standalone_query,
        chunk_ids=[str(item.get("chunk_id", "")) for item in retrieved],
        rerank_scores=[
            float(item.get("score", 0.0))
            for item in retrieved
            if item.get("score") is not None
        ],
        verification=verification if isinstance(verification, dict) else None,
        cache_hit=False,
        node_timings=node_timings,
        error=telemetry_error,
    )


def _build_input_state(
    user_text: str,
    *,
    email: str,
    companies: list[str],
    selected_model: str,
    bypass_cache: bool,
) -> dict[str, Any]:
    """Build turn input without clearing checkpointed memory fields.

    ``last_user_question`` and ``retrieved`` are intentionally omitted so the
    LangGraph checkpointer can retain them across follow-up turns.
    """

    del bypass_cache
    return {
        "messages": [HumanMessage(content=user_text)],
        "email": email,
        "selected_model": selected_model,
        "allowed_companies": companies,
        "intent": None,
        "standalone_query": None,
        "answer": None,
        "verification": None,
        "node_timings": {},
    }


def _model_error_message(selected_model: str) -> str:
    return (
        f"Model `{selected_model}` could not complete this request. "
        "Please try again or choose another model."
    )


def _embedding_error_message() -> str:
    return "The embedding service is temporarily unavailable. Please try again."


async def _persist_toplevel_step(message: cl.Message) -> None:
    """Await a DB upsert with parentId cleared for history/chat-switch visibility."""

    from datetime import UTC, datetime

    message.parent_id = None
    data_layer = get_active_data_layer()
    if data_layer is None:
        return
    step = message.to_dict()
    step["parentId"] = None
    # Chainlit sometimes omits createdAt on streamed assistant updates; without
    # it SQLite stores NULL and resumed threads hide or mis-order replies.
    if not step.get("createdAt"):
        step["createdAt"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        await data_layer.update_step(step)
    except Exception:
        logger.exception("Failed to persist message to chat history")


async def _finalize_assistant_reply(
    reply: cl.Message,
    *,
    answer: str,
    elements: list[cl.Pdf],
) -> None:
    """Publish the final answer and await DB persistence for history resume."""

    reply.parent_id = None
    reply.content = answer
    reply.elements = elements
    if reply.streaming:
        # Tokens already hit the UI via stream_start; commit final text + elements.
        await reply.update()
    else:
        # Cache hits / canned replies never streamed — send once with full content.
        await reply.send()

    # Chainlit's update/send persist via fire-and-forget create_task. Await an
    # explicit upsert so history always stores the final assistant output.
    await _persist_toplevel_step(reply)


def _build_pdf_elements(
    citations: list[PdfCitation],
    *,
    pdf_dir: Path,
    allowed_companies: list[str],
) -> list[cl.Pdf]:
    """Render highlighted temp PDFs (ACL-gated) for side-panel citations."""

    if not citations:
        return []

    output_dir = Path(tempfile.mkdtemp(prefix="mtrag-hl-"))
    elements: list[cl.Pdf] = []
    for citation in citations:
        source = resolve_pdf_path(
            pdf_dir,
            citation.pdf_relpath,
            company=citation.company,
            allowed_companies=allowed_companies,
        )
        if source is None:
            logger.warning(
                "Skipping PDF citation %s (ACL or missing path)",
                citation.label,
            )
            continue
        try:
            boxes = resolve_highlight_bboxes(
                source,
                page=citation.page,
                stored=citation.bboxes,
                search_texts=citation.search_texts,
            )
            if boxes:
                path = render_highlighted_pdf(
                    source,
                    page=citation.page,
                    bboxes=boxes,
                    output_dir=output_dir,
                )
            else:
                logger.warning(
                    "No highlight boxes for %s page %s; opening plain PDF",
                    citation.label,
                    citation.page,
                )
                path = source
        except Exception:
            logger.exception(
                "Highlight failed for %s; using original PDF",
                citation.label,
            )
            path = source
        elements.append(
            cl.Pdf(
                name=citation.label,
                path=str(path),
                page=citation.page,
                display="side",
            )
        )
    return elements


def _step_line(node: str, output: dict[str, Any]) -> str:
    labels = {
        "classify_intent": "Classifying intent",
        "expand_query": "Expanding search query",
        "retrieve": "Retrieving documents",
        "answer_from_docs": "Composing answer",
        "verify_numbers": "Verifying figures",
        "chitchat_reply": "Replying",
        "meta_reply": "Explaining capabilities",
    }
    detail = ""
    if node == "classify_intent":
        detail = str(output.get("intent", ""))
    elif node == "expand_query":
        detail = str(output.get("standalone_query", ""))
    elif node == "retrieve":
        detail = f"{len(output.get('retrieved') or [])} chunks"
    elif node == "verify_numbers":
        verification = output.get("verification") or {}
        detail = (
            f"{verification.get('figures_verified', 0)}/"
            f"{verification.get('figures_found', 0)} figures verified"
        )
    label = labels.get(node, node)
    return f"{label}: {detail}" if detail else label


def _merge_state(
    current: dict[str, Any] | None,
    update: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(current or {})
    merged.update(update)
    return merged


def _summary_line(state: dict[str, Any]) -> str:
    timings = state.get("node_timings") or {}
    total_ms = sum(float(value) for value in timings.values())
    verification = state.get("verification") or {}
    verified = verification.get("figures_verified", 0)
    found = verification.get("figures_found", 0)
    return f"~{total_ms / 1000:.1f}s · {verified}/{found} figures verified"
