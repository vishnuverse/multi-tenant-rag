"""Document-aware chunking with citation-preserving metadata."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from multi_tenant_rag.rag.geometry import BBox, serialize_bboxes

DocType = Literal["transcript", "press_release", "sec_8k"]

# Max highlight rectangles stored / drawn per chunk.
MAX_BBOXES = 5


@dataclass(frozen=True, slots=True)
class TextBlock:
    """A layout block with PDF page coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass(frozen=True, slots=True)
class PageText:
    """Text extracted from one source page."""

    page: int
    text: str
    blocks: tuple[TextBlock, ...] = ()


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    """Metadata needed to filter, cite, and contextualize a chunk."""

    company: str
    source: str
    doc_type: DocType
    page: int
    chunk_id: str
    quarter: str | None
    fiscal_year: int | None
    section: str | None
    speaker: str | None
    has_table: bool
    pdf_relpath: str = ""
    bboxes: str = "[]"
    quote: str | None = None


@dataclass(frozen=True, slots=True)
class Chunk:
    """An immutable raw document chunk and its metadata."""

    text: str
    metadata: ChunkMetadata


@dataclass(frozen=True, slots=True)
class _Piece:
    page: int
    text: str
    section: str | None = None
    speaker: str | None = None
    has_table: bool = False


_SEC_FORM_RE = re.compile(r"\b(?:form\s+)?8-k\b", re.IGNORECASE)
_SEC_COMMISSION_RE = re.compile(
    r"\b(?:u\.?s\.?\s+)?securities\s+and\s+exchange\s+commission\b",
    re.IGNORECASE,
)
_TRANSCRIPT_DISCLAIMER_RE = re.compile(
    r"\bthis\s+transcript\s+is\s+provided\b",
    re.IGNORECASE,
)
_CONFERENCE_CALL_RE = re.compile(
    r"\b(?:earnings\s+)?(?:conference|results)\s+call\b",
    re.IGNORECASE,
)
_LISTEN_ONLY_RE = re.compile(r"\blisten-only\s+mode\b", re.IGNORECASE)
_COLON_SPEAKER_RE = re.compile(r"^\s*([^:\n]{1,100}):(?:\s+(.*))?\s*$")
_ROLE_SPEAKER_RE = re.compile(r"^\s*([^—\n]{1,80}?)\s+[—-]\s+([^\n]{1,80})\s*$")
_COMMA_ROLE_SPEAKER_RE = re.compile(
    r"^\s*([A-Z][A-Za-zÀ-ÖØ-öø-ÿ.'’]+"
    r"(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ.'’]+){0,4}),\s*"
    r"([^:\n]*(?:CEO|CFO|Chief|President|Officer|Director|Investor Relations)"
    r"[^:\n]*)\s*$",
    re.IGNORECASE,
)
_INLINE_COLON_SPEAKER_RE = re.compile(
    r"\b(?:(?P<operator>Operator|Analyst|Question|Answer)|"
    r"(?P<name>[A-Z][A-Za-zÀ-ÖØ-öø-ÿ.'’]+"
    r"(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ.'’]+){1,3},\s*"
    r"[^:]{0,80}(?:CEO|CFO|Chief|President|Officer|Director|Investor Relations)"
    r"[^:]{0,30})):\s+"
)
_INLINE_PLAIN_SPEAKER_RE = re.compile(
    r"(?:^(?P<plain_start>[A-Z][A-Za-zÀ-ÖØ-öø-ÿ.'’]+"
    r"(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ.'’]+){0,4})|"
    r"(?<=[.!?])\s+(?P<plain_after>[A-Z][A-Za-zÀ-ÖØ-öø-ÿ.'’]+"
    r"(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ.'’]+){0,4})):\s+"
)
_QA_HEADING_RE = re.compile(
    r"^\s*(?:questions?\s*(?:and|&)\s*answers?|q\s*&\s*a)\s*:?\s*$",
    re.IGNORECASE,
)
_ITEM_RE = re.compile(r"^\s*(Item\s+\d+\.\d{2}\b[^\n]*)$", re.IGNORECASE)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_NUMBER_RE = re.compile(r"[-+($]?\d[\d,.%()$-]*")
_NON_SPEAKER_LABELS = frozenset(
    {
        "adjusted ebitda",
        "agenda",
        "business highlights",
        "cash flow",
        "ebitda",
        "expenses",
        "financial highlights",
        "financial results",
        "free cash flow",
        "guidance",
        "highlights",
        "income",
        "key metrics",
        "margin",
        "margins",
        "metric",
        "metrics",
        "operating income",
        "outlook",
        "overview",
        "performance",
        "results",
        "revenue",
        "revenue growth",
        "sales",
        "summary",
    }
)


def detect_document_type(pages: Sequence[PageText]) -> DocType:
    """Classify a document from extracted page text."""

    all_text = "\n".join(page.text for page in pages)
    if _SEC_FORM_RE.search(all_text) and _SEC_COMMISSION_RE.search(all_text):
        return "sec_8k"

    front_text = "\n".join(page.text for page in pages[:3])
    labelled_lines = sum(
        _speaker_from_line(line) is not None for line in front_text.splitlines()
    )
    normalized_front_text = " ".join(front_text.split())
    inline_labels = len(_inline_speaker_matches(normalized_front_text))
    if max(labelled_lines, inline_labels) >= 3:
        return "transcript"
    if _transcript_keyword_signals(normalized_front_text):
        return "transcript"
    if labelled_lines >= 2 and _CONFERENCE_CALL_RE.search(normalized_front_text):
        return "transcript"

    return "press_release"


def _transcript_keyword_signals(normalized_front_text: str) -> bool:
    if _TRANSCRIPT_DISCLAIMER_RE.search(normalized_front_text):
        return True
    if _LISTEN_ONLY_RE.search(normalized_front_text) and re.search(
        r"\bOperator\s*:", normalized_front_text, re.IGNORECASE
    ):
        return True
    return False


def extract_period(front_matter: str) -> tuple[str | None, int | None]:
    """Extract a normalized fiscal quarter and year from front matter."""

    quarter_words = {
        "first": "Q1",
        "second": "Q2",
        "third": "Q3",
        "fourth": "Q4",
    }

    match = re.search(
        r"\bQ([1-4])\s+(?:FY\s*)?(\d{2,4})\b",
        front_matter,
        re.IGNORECASE,
    )
    if match:
        return f"Q{match.group(1)}", _normalize_year(match.group(2))

    match = re.search(
        r"\b(?:FY\s*)?(\d{2,4})\s+Q([1-4])\b",
        front_matter,
        re.IGNORECASE,
    )
    if match:
        return f"Q{match.group(2)}", _normalize_year(match.group(1))

    match = re.search(
        r"\b(?:FY|Fiscal\s+Year)\s*(\d{2,4})\s+"
        r"(First|Second|Third|Fourth)\s+Quarter\b",
        front_matter,
        re.IGNORECASE,
    )
    if match:
        return quarter_words[match.group(2).lower()], _normalize_year(match.group(1))

    match = re.search(
        r"\b(First|Second|Third|Fourth)\s+Quarter\s+"
        r"(?:(?:FY|Fiscal(?:\s+Year)?)\s*)?(\d{2,4})\b",
        front_matter,
        re.IGNORECASE,
    )
    if match:
        return quarter_words[match.group(1).lower()], _normalize_year(match.group(2))

    return None, None


def chunk_document(
    pages: Sequence[PageText],
    company: str,
    source: str,
    min_chars: int = 200,
    max_chars: int = 1200,
) -> list[Chunk]:
    """Chunk pages according to document structure without crossing pages."""

    if min_chars <= 0:
        raise ValueError("min_chars must be positive")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if min_chars > max_chars:
        raise ValueError("min_chars cannot exceed max_chars")

    doc_type = detect_document_type(pages)
    front_matter = "\n".join(page.text for page in pages[:3])
    quarter, fiscal_year = extract_period(front_matter)

    pieces: list[_Piece] = []
    for page in pages:
        if doc_type == "transcript":
            page_pieces = _transcript_pieces(page)
        else:
            page_pieces = _release_pieces(page, include_items=doc_type == "sec_8k")
        split_pieces = [
            split_piece
            for piece in page_pieces
            for split_piece in _split_piece(piece, max_chars)
        ]
        pieces.extend(_merge_short_pieces(split_pieces, min_chars, max_chars))

    chunks: list[Chunk] = []
    for piece in pieces:
        text = piece.text.strip()
        if not text:
            continue
        chunk_id = _chunk_id(
            company,
            source,
            piece,
            doc_type,
            text,
            ordinal=len(chunks),
        )
        metadata = ChunkMetadata(
            company=company,
            source=source,
            doc_type=doc_type,
            page=piece.page,
            chunk_id=chunk_id,
            quarter=quarter,
            fiscal_year=fiscal_year,
            section=piece.section,
            speaker=piece.speaker,
            has_table=piece.has_table,
            pdf_relpath=f"{company}/{source}",
        )
        chunks.append(Chunk(text=text, metadata=metadata))
    return assign_chunk_bboxes(chunks, pages)


def assign_chunk_bboxes(
    chunks: Sequence[Chunk],
    pages: Sequence[PageText],
) -> list[Chunk]:
    """Attach bbox JSON (and optional quote) using page layout blocks."""

    blocks_by_page = {page.page: page.blocks for page in pages}
    updated: list[Chunk] = []
    for chunk in chunks:
        blocks = blocks_by_page.get(chunk.metadata.page, ())
        bboxes, quote = match_bboxes_for_text(chunk.text, blocks)
        if not bboxes and not quote:
            updated.append(chunk)
            continue

        metadata = ChunkMetadata(
            company=chunk.metadata.company,
            source=chunk.metadata.source,
            doc_type=chunk.metadata.doc_type,
            page=chunk.metadata.page,
            chunk_id=chunk.metadata.chunk_id,
            quarter=chunk.metadata.quarter,
            fiscal_year=chunk.metadata.fiscal_year,
            section=chunk.metadata.section,
            speaker=chunk.metadata.speaker,
            has_table=chunk.metadata.has_table,
            pdf_relpath=chunk.metadata.pdf_relpath
            or f"{chunk.metadata.company}/{chunk.metadata.source}",
            bboxes=serialize_bboxes(bboxes),
            quote=quote,
        )
        updated.append(Chunk(text=chunk.text, metadata=metadata))
    return updated


def match_bboxes_for_text(
    text: str,
    blocks: Sequence[TextBlock],
) -> tuple[list[BBox], str | None]:
    """Return up to MAX_BBOXES rects whose text overlaps the chunk."""

    normalized_chunk = _normalize_match_text(text)
    if not normalized_chunk or not blocks:
        return [], None

    scored: list[tuple[float, BBox, str]] = []
    for block in blocks:
        block_norm = _normalize_match_text(block.text)
        if not block_norm:
            continue
        overlap = _token_overlap_ratio(normalized_chunk, block_norm)
        if overlap < 0.35 and block_norm not in normalized_chunk:
            continue
        if overlap < 0.15:
            continue
        scored.append(
            (
                overlap * ((block.x1 - block.x0) * (block.y1 - block.y0) + 1.0),
                BBox(x0=block.x0, y0=block.y0, x1=block.x1, y1=block.y1),
                block.text.strip(),
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[:MAX_BBOXES]
    if not selected:
        return [], None
    quote = selected[0][2][:240] or None
    return [item[1] for item in selected], quote


def _normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9$.,%]+", left))
    right_tokens = set(re.findall(r"[a-z0-9$.,%]+", right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(right_tokens)


def contextualize_chunk(chunk: Chunk) -> str:
    """Prepend an adaptive retrieval header without altering the raw chunk."""

    metadata = chunk.metadata
    document_labels = {
        "transcript": "Transcript",
        "press_release": "Press Release",
        "sec_8k": "SEC 8-K",
    }
    fields = [
        f"Company: {metadata.company}",
        f"Document: {document_labels[metadata.doc_type]}",
        f"Source: {metadata.source}",
        f"Page: {metadata.page}",
    ]
    if metadata.quarter and metadata.fiscal_year:
        fields.append(f"Period: {metadata.quarter} FY{metadata.fiscal_year}")
    elif metadata.quarter:
        fields.append(f"Period: {metadata.quarter}")
    elif metadata.fiscal_year:
        fields.append(f"Fiscal year: FY{metadata.fiscal_year}")
    if metadata.section:
        section = "Q&A" if metadata.section == "qa" else metadata.section
        fields.append(f"Section: {section}")
    if metadata.speaker:
        fields.append(f"Speaker: {metadata.speaker}")
    if metadata.has_table:
        fields.append("Content: Table")
    return f"[{' | '.join(fields)}]\n{chunk.text}"


def _normalize_year(value: str) -> int:
    year = int(value)
    return 2000 + year if len(value) == 2 else year


def _speaker_from_line(line: str) -> tuple[str, str] | None:
    colon_match = _COLON_SPEAKER_RE.match(line)
    if colon_match:
        speaker = _parse_colon_speaker_label(colon_match.group(1))
        if speaker is not None:
            return speaker, (colon_match.group(2) or "").strip()

    role_match = _ROLE_SPEAKER_RE.match(line)
    if role_match:
        speaker = role_match.group(1).strip()
        role = role_match.group(2).strip()
        if _is_credible_person_name(speaker) and _is_credible_role_phrase(role):
            return speaker, ""

    comma_role_match = _COMMA_ROLE_SPEAKER_RE.match(line)
    if comma_role_match:
        speaker = comma_role_match.group(1).strip()
        role = comma_role_match.group(2).strip()
        if _is_credible_person_name(speaker) and _is_credible_role_phrase(role):
            return speaker, ""
    return None


def _parse_colon_speaker_label(label: str) -> str | None:
    normalized = label.strip()
    generic_roles = {
        "operator": "Operator",
        "analyst": "Analyst",
        "question": "Question",
        "answer": "Answer",
    }
    generic = generic_roles.get(normalized.casefold())
    if generic is not None:
        return generic

    name = normalized
    role: str | None = None
    parenthesized_role = re.fullmatch(r"(.+?)\s*\(([^()]{1,50})\)", normalized)
    if parenthesized_role:
        name = parenthesized_role.group(1).strip()
        role = parenthesized_role.group(2).strip()
    elif "," in normalized:
        name, role = (part.strip() for part in normalized.split(",", maxsplit=1))

    if role is not None and not _is_credible_role_phrase(role):
        return None
    return name if _is_credible_person_name(name) else None


def _is_credible_person_name(value: str) -> bool:
    normalized = " ".join(value.split())
    if normalized.casefold() in _NON_SPEAKER_LABELS:
        return False
    business_terms = {
        "adjusted",
        "business",
        "cash",
        "ebitda",
        "expenses",
        "financial",
        "flow",
        "free",
        "growth",
        "guidance",
        "highlights",
        "income",
        "margin",
        "margins",
        "metric",
        "metrics",
        "operating",
        "outlook",
        "performance",
        "results",
        "revenue",
        "sales",
    }
    parts = normalized.split()
    if not 1 <= len(parts) <= 5:
        return False
    if any(part.casefold().strip(".'’") in business_terms for part in parts):
        return False
    connectors = {"de", "del", "van", "von"}
    name_token = re.compile(
        r"(?:[A-ZÀ-ÖØ-Þ]\.|[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ]*"
        r"(?:[-'’][A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ]*)*)"
    )
    return all(
        part.casefold() in connectors or name_token.fullmatch(part) is not None
        for part in parts
    )


def _is_credible_role_phrase(value: str) -> bool:
    role = " ".join(value.strip().split())
    if len(role) > 60:
        return False
    if re.search(
        r"\b(?:was|is|are|were|remains?|increased|decreased|grew|declined)\b",
        role,
        re.IGNORECASE,
    ):
        return False
    role_patterns = (
        r"(?:CEO|CFO|COO|CTO)",
        r"Chief(?:\s+[A-Za-z&-]+){1,4}\s+Officer",
        r"President(?:\s+and\s+(?:CEO|CFO|COO|CTO))?",
        r"(?:VP|Vice\s+President)(?:,?\s+(?:of\s+)?[A-Za-z& -]+)?",
        r"(?:Senior\s+|Managing\s+)?Director(?:,?\s+[A-Za-z& -]*Relations)",
        r"(?:Founder|Chairman)(?:\s+and\s+(?:CEO|CFO|COO|CTO|"
        r"Chief(?:\s+[A-Za-z&-]+){1,4}\s+Officer))?",
        r"(?:Analyst|Operator)",
    )
    return any(re.fullmatch(pattern, role, re.IGNORECASE) for pattern in role_patterns)


def _transcript_pieces(page: PageText) -> list[_Piece]:
    pieces: list[_Piece] = []
    section = "prepared"
    speaker: str | None = None
    lines: list[str] = []

    def flush() -> None:
        nonlocal lines
        text = "\n".join(lines).strip()
        if text:
            pieces.append(
                _Piece(
                    page=page.page,
                    text=text,
                    section=section,
                    speaker=speaker,
                )
            )
        lines = []

    for raw_line in page.text.splitlines():
        line = raw_line.strip()
        if _QA_HEADING_RE.match(line):
            flush()
            section = "qa"
            speaker = None
            continue

        inline_matches = _inline_speaker_matches(line)
        if inline_matches:
            prefix = line[: inline_matches[0].start()].strip()
            if prefix:
                lines.append(prefix)
            for match_index, match in enumerate(inline_matches):
                flush()
                inline_speaker = _inline_speaker_name(match)
                if inline_speaker is None:
                    raise ValueError("unvalidated inline speaker match")
                speaker = inline_speaker
                text_end = (
                    inline_matches[match_index + 1].start()
                    if match_index + 1 < len(inline_matches)
                    else len(line)
                )
                opening_text = line[match.end() : text_end].strip()
                if opening_text:
                    lines.append(opening_text)
            continue

        speaker_match = _speaker_from_line(line)
        if speaker_match:
            flush()
            speaker, opening_text = speaker_match
            if opening_text:
                lines.append(opening_text)
            continue

        if line:
            lines.append(line)
        elif lines:
            lines.append("")

    flush()
    return pieces


def _release_pieces(page: PageText, *, include_items: bool) -> list[_Piece]:
    lines = page.text.splitlines()
    pieces: list[_Piece] = []
    section: str | None = None
    paragraph: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        nonlocal paragraph
        text = "\n".join(paragraph).strip()
        if text:
            pieces.append(_Piece(page=page.page, text=text, section=section))
        paragraph = []

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            flush_paragraph()
            index += 1
            continue

        table_block = _table_block_at(lines, index)
        if table_block is not None:
            table_lines, next_index = table_block
            flush_paragraph()
            if pieces and _is_scale_or_period_context(pieces[-1].text):
                prior = pieces.pop()
                table_lines = [*prior.text.splitlines(), *table_lines]
            pieces.append(
                _Piece(
                    page=page.page,
                    text="\n".join(table_lines),
                    section=section,
                    has_table=True,
                )
            )
            index = next_index
            continue

        item_match = _ITEM_RE.match(line) if include_items else None
        if item_match:
            flush_paragraph()
            section = item_match.group(1).strip()
            paragraph.append(section)
            index += 1
            continue

        if _is_section_header(line):
            flush_paragraph()
            section = line
            paragraph.append(line)
            index += 1
            continue

        paragraph.append(line)
        index += 1

    flush_paragraph()
    return pieces


def _is_section_header(line: str) -> bool:
    if len(line) > 100 or len(line.split()) > 12:
        return False
    letters = [character for character in line if character.isalpha()]
    if letters and all(character.isupper() for character in letters):
        return True

    words = re.findall(r"[A-Za-z][A-Za-z'’&-]*", line)
    if not 1 <= len(words) <= 6 or line.endswith((".", ":", ";")):
        return False
    lowercase_words = {"and", "of", "the", "for", "in", "to"}
    return all(word.lower() in lowercase_words or word[0].isupper() for word in words)


_SCALE_LINE_RE = re.compile(
    r"\(\s*in\s+(?:millions|thousands|billions)(?:\s*,\s*except[^)]*)?\s*\)",
    re.IGNORECASE,
)
_PERIOD_LINE_RE = re.compile(
    r"\b(?:three|six|nine)\s+months\s+ended\b|\b(?:quarter|year)\s+ended\b",
    re.IGNORECASE,
)


def _is_scale_or_period_context(text: str) -> bool:
    """True when a short preceding piece is a statement scale/period header."""

    stripped = text.strip()
    if not stripped or len(stripped) > 240:
        return False
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines or len(lines) > 4:
        return False
    return all(
        _SCALE_LINE_RE.search(line) is not None
        or _PERIOD_LINE_RE.search(line) is not None
        or (
            len(line) <= 100
            and line.isupper()
            and any(
                token in line for token in ("STATEMENT", "CONSOLIDATED", "OPERATIONS")
            )
        )
        for line in lines
    )


def _is_table_line(line: str) -> bool:
    columns = _table_columns(line)
    if len(columns) < 3 or not any(character.isalpha() for character in columns[0]):
        return False
    return sum(_NUMBER_RE.search(column) is not None for column in columns[1:]) >= 2


def _is_table_data_row(line: str) -> bool:
    columns = _table_columns(line)
    if not columns:
        return False
    label = columns[0]
    return _is_table_line(line) and not label.endswith((".", "!", "?", ";"))


def _is_table_header(line: str) -> bool:
    columns = _table_columns(line)
    if len(columns) < 3:
        return False
    label = columns[0]
    if not any(character.isalpha() for character in label):
        return False
    if label.endswith((".", "!", "?", ";")):
        return False
    period_marker = re.compile(
        r"\b(?:current|prior|quarter|year|FY|Q[1-4]|\d{4})\b",
        re.IGNORECASE,
    )
    return all(period_marker.search(column) is not None for column in columns[1:])


def _table_columns(line: str) -> list[str]:
    separator = r"\s*\|\s*" if "|" in line else r"\s{2,}"
    return [
        column.strip() for column in re.split(separator, line.strip()) if column.strip()
    ]


def _table_block_at(lines: Sequence[str], start: int) -> tuple[list[str], int] | None:
    header = lines[start].strip()
    if not _is_table_header(header):
        return None

    rows: list[str] = []
    index = start + 1
    while index < len(lines):
        row = lines[index].strip()
        if not row or not _is_table_data_row(row):
            break
        rows.append(row)
        index += 1
    if not rows:
        return None
    return [header, *rows], index


def _inline_speaker_name(match: re.Match[str]) -> str | None:
    groups = match.groupdict()
    label = (
        groups.get("operator")
        or groups.get("name")
        or groups.get("plain_start")
        or groups.get("plain_after")
    )
    return _parse_colon_speaker_label(label) if label is not None else None


def _inline_speaker_matches(text: str) -> list[re.Match[str]]:
    strict_matches = [
        match
        for match in _INLINE_COLON_SPEAKER_RE.finditer(text)
        if _inline_speaker_name(match) is not None
    ]
    plain_matches = [
        match
        for match in _INLINE_PLAIN_SPEAKER_RE.finditer(text)
        if _inline_speaker_name(match) is not None
    ]
    if not strict_matches and len(plain_matches) < 3:
        return []

    matches = strict_matches.copy()
    for plain_match in plain_matches:
        overlaps_strict = any(
            plain_match.start() < strict_match.end()
            and strict_match.start() < plain_match.end()
            for strict_match in strict_matches
        )
        if not overlaps_strict:
            matches.append(plain_match)
    return sorted(matches, key=lambda match: match.start())


def _split_piece(piece: _Piece, max_chars: int) -> list[_Piece]:
    text = piece.text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [_replace_piece_text(piece, text)]

    if piece.has_table:
        units = text.splitlines()
        separator = "\n"
    else:
        units = [
            sentence.strip()
            for paragraph in re.split(r"\n\s*\n", text)
            for sentence in _SENTENCE_BOUNDARY_RE.split(paragraph.strip())
            if sentence.strip()
        ]
        separator = " "

    parts: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.extend(_hard_split(unit, max_chars))
            continue
        candidate = f"{current}{separator}{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
        else:
            parts.append(current)
            current = unit
    if current:
        parts.append(current)

    return [_replace_piece_text(piece, part) for part in parts if part.strip()]


def _hard_split(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars + 1)
        if split_at <= 0:
            split_at = max_chars
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _replace_piece_text(piece: _Piece, text: str) -> _Piece:
    return _Piece(
        page=piece.page,
        text=text,
        section=piece.section,
        speaker=piece.speaker,
        has_table=piece.has_table,
    )


def _merge_short_pieces(
    pieces: Sequence[_Piece], min_chars: int, max_chars: int
) -> list[_Piece]:
    merged: list[_Piece] = []
    for piece in pieces:
        if not piece.text.strip():
            continue
        separator = "\n" if piece.has_table else "\n\n"
        if (
            merged
            and len(merged[-1].text) < min_chars
            and _same_citation_context(merged[-1], piece)
            and len(merged[-1].text) + len(separator) + len(piece.text) <= max_chars
        ):
            previous = merged.pop()
            merged.append(
                _replace_piece_text(previous, previous.text + separator + piece.text)
            )
        else:
            merged.append(piece)
    return merged


def _same_citation_context(left: _Piece, right: _Piece) -> bool:
    return (
        left.page == right.page
        and left.section == right.section
        and left.speaker == right.speaker
        and left.has_table == right.has_table
    )


def _chunk_id(
    company: str,
    source: str,
    piece: _Piece,
    doc_type: DocType,
    text: str,
    *,
    ordinal: int,
) -> str:
    payload = "\x1f".join(
        (
            company,
            source,
            str(piece.page),
            doc_type,
            piece.section or "",
            piece.speaker or "",
            "1" if piece.has_table else "0",
            str(ordinal),
            text,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
