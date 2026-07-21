"""Citation labels and PDF side-panel selection for grounded answers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from multi_tenant_rag.rag.geometry import BBox, parse_bboxes
from multi_tenant_rag.rag.retrieval import citation_label as citation_label
from multi_tenant_rag.ui.pdf import search_texts_from_chunk

# Model / legacy cites, e.g. [APPLE-press_release p.1]
_BRACKET_PAGE_CITE_RE = re.compile(
    r"\[([^\[\]]+?)\s+p\.(\d+)\]",
    re.IGNORECASE,
)
# Bracket-free tokens we emit, e.g. APPLE-press_release-p1
_TOKEN_PAGE_CITE_RE = re.compile(
    r"\b([A-Z][A-Z0-9]*(?:-[A-Za-z0-9_]+)+-p)(\d+)\b",
)


@dataclass(frozen=True, slots=True)
class PdfCitation:
    """One side-panel PDF element to attach to an answer."""

    label: str
    company: str
    pdf_relpath: str
    page: int
    bboxes: tuple[BBox, ...]
    score: float
    search_texts: tuple[str, ...] = ()


def _chunk_pdf_relpath(chunk: Mapping[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    if isinstance(metadata, Mapping):
        rel = metadata.get("pdf_relpath")
        if rel:
            return str(rel)
    rel = chunk.get("pdf_relpath")
    if rel:
        return str(rel)
    company = str(chunk.get("company", "")).strip().lower()
    source = str(chunk.get("source", "")).strip()
    if company and source:
        return f"{company}/{source}"
    return ""


def _chunk_bboxes(chunk: Mapping[str, Any]) -> list[BBox]:
    metadata = chunk.get("metadata") or {}
    raw: Any = None
    if isinstance(metadata, Mapping):
        raw = metadata.get("bboxes")
    if raw is None:
        raw = chunk.get("bboxes")
    return parse_bboxes(raw)


def _chunk_score(chunk: Mapping[str, Any]) -> float:
    score = chunk.get("score")
    if score is None:
        return 0.0
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def _page_key(chunk: Mapping[str, Any]) -> tuple[str, int] | None:
    relpath = _chunk_pdf_relpath(chunk)
    try:
        page = int(chunk.get("page", 0))
    except (TypeError, ValueError):
        return None
    if not relpath or page < 1:
        return None
    return (relpath, page)


def _best_chunk_for_page(
    retrieved: Sequence[Mapping[str, Any]],
    page: int,
    *,
    company_hint: str | None = None,
) -> Mapping[str, Any] | None:
    candidates = [chunk for chunk in retrieved if _safe_page(chunk) == page]
    if company_hint:
        narrowed = [
            chunk
            for chunk in candidates
            if str(chunk.get("company", "")).casefold() == company_hint.casefold()
        ]
        if narrowed:
            candidates = narrowed
    if not candidates:
        return None
    return max(candidates, key=_chunk_score)


def _safe_page(chunk: Mapping[str, Any]) -> int | None:
    try:
        page = int(chunk.get("page", 0))
    except (TypeError, ValueError):
        return None
    return page if page >= 1 else None


def _company_hint_from_cite_body(body: str) -> str | None:
    token = body.strip().split("-", 1)[0].strip()
    return token.lower() if token.isalpha() else None


def select_pdf_citations(
    answer: str,
    retrieved: Sequence[Mapping[str, Any]],
    *,
    sources_footer_cap: int = 3,
) -> tuple[str, list[PdfCitation]]:
    """Rewrite cites into Chainlit-safe ``Source-N`` links and attach PDFs.

    Chainlit only linkifies exact element ``name`` strings in the message.
    Using ``Source-1`` avoids bracket/markdown collisions with model cites like
    ``[APPLE-press_release p.1]``.
    """

    if not retrieved:
        return answer, []

    # (pdf_relpath, page) -> best chunk
    page_groups: dict[tuple[str, int], Mapping[str, Any]] = {}
    for chunk in retrieved:
        key = _page_key(chunk)
        if key is None:
            continue
        prior = page_groups.get(key)
        if prior is None or _chunk_score(chunk) > _chunk_score(prior):
            page_groups[key] = chunk

    # Pages explicitly mentioned by the model (legacy or token form).
    mentioned_keys: list[tuple[str, int]] = []
    replacements: list[tuple[str, tuple[str, int]]] = []

    for match in _BRACKET_PAGE_CITE_RE.finditer(answer):
        body, page_s = match.group(1), match.group(2)
        page = int(page_s)
        matched_chunk = _best_chunk_for_page(
            retrieved,
            page,
            company_hint=_company_hint_from_cite_body(body),
        )
        if matched_chunk is None:
            continue
        key = _page_key(matched_chunk)
        if key is None:
            continue
        mentioned_keys.append(key)
        replacements.append((match.group(0), key))

    for match in _TOKEN_PAGE_CITE_RE.finditer(answer):
        page = int(match.group(2))
        body = match.group(1)[:-2]  # drop trailing "-p"
        matched_chunk = _best_chunk_for_page(
            retrieved,
            page,
            company_hint=_company_hint_from_cite_body(body),
        )
        if matched_chunk is None:
            continue
        key = _page_key(matched_chunk)
        if key is None:
            continue
        mentioned_keys.append(key)
        replacements.append((match.group(0), key))

    # Ordered unique page keys to attach.
    ordered_keys: list[tuple[str, int]] = []
    for key in mentioned_keys:
        if key not in ordered_keys and key in page_groups:
            ordered_keys.append(key)

    if not ordered_keys:
        ranked = sorted(
            page_groups.items(),
            key=lambda item: _chunk_score(item[1]),
            reverse=True,
        )[:sources_footer_cap]
        ordered_keys = [key for key, _chunk in ranked]

    key_to_source: dict[tuple[str, int], str] = {
        key: f"Source-{index}" for index, key in enumerate(ordered_keys, start=1)
    }

    final_answer = answer
    # Replace longer cite strings first.
    for cite_text, key in sorted(
        replacements,
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        source_name = key_to_source.get(key)
        if source_name and cite_text in final_answer:
            final_answer = final_answer.replace(cite_text, source_name)

    # Strip any leftover bracket cites that we could not map.
    final_answer = _BRACKET_PAGE_CITE_RE.sub("", final_answer)
    final_answer = re.sub(r"[ \t]+\n", "\n", final_answer)
    final_answer = re.sub(r"  +", " ", final_answer).strip()

    source_line = " · ".join(key_to_source[key] for key in ordered_keys)
    if source_line and source_line not in final_answer:
        final_answer = f"{final_answer}\n\nSources: {source_line}"

    # Prefer answer figures so highlights track the reply, not the whole page.
    answer_search = search_texts_from_chunk(answer)

    citations: list[PdfCitation] = []
    for key in ordered_keys:
        chunk = page_groups[key]
        relpath, page = key
        search_hints: list[str] = list(answer_search)
        for other in retrieved:
            if _page_key(other) != key:
                continue
            for hint in search_texts_from_chunk(str(other.get("text", ""))):
                if hint not in search_hints:
                    search_hints.append(hint)
        # Leave stored page-wide boxes empty; runtime search uses answer tokens.
        # Ambiguous search → plain PDF (no invented highlight).
        citations.append(
            PdfCitation(
                label=key_to_source[key],
                company=str(chunk.get("company", "")),
                pdf_relpath=relpath,
                page=page,
                bboxes=(),
                score=_chunk_score(chunk),
                search_texts=tuple(search_hints[:10]),
            )
        )
    return final_answer, citations
