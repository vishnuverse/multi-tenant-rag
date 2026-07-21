"""PDF path ACL checks and burn-in highlight rendering for citations."""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from pathlib import Path

import fitz  # PyMuPDF

from multi_tenant_rag.rag.geometry import (
    BBox as BBox,
)
from multi_tenant_rag.rag.geometry import (
    parse_bboxes as parse_bboxes,
)
from multi_tenant_rag.rag.geometry import (
    serialize_bboxes as serialize_bboxes,
)

_AMOUNT_RE = re.compile(
    r"\$\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?|"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|"
    r"\b\d{1,3}(?:\.\d+)?%|"
    r"\b\d{4,7}\b",
)


def resolve_pdf_path(
    pdf_dir: Path,
    pdf_relpath: str,
    *,
    company: str,
    allowed_companies: Sequence[str],
) -> Path | None:
    """Resolve a corpus-relative PDF path, failing closed on ACL or escape."""

    allowed = {item.strip().lower() for item in allowed_companies if item.strip()}
    company_key = company.strip().lower()
    if not company_key or company_key not in allowed:
        return None

    rel = pdf_relpath.strip().replace("\\", "/")
    if not rel or rel.startswith("/") or ".." in Path(rel).parts:
        return None

    parts = Path(rel).parts
    if not parts or parts[0].lower() != company_key:
        return None

    root = pdf_dir.resolve()
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


def search_texts_from_chunk(text: str, *, limit: int = 8) -> list[str]:
    """Derive short strings that PyMuPDF can locate on a page."""

    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []

    candidates: list[str] = []
    for match in _AMOUNT_RE.finditer(cleaned):
        token = match.group(0).replace(" ", "")
        if token not in candidates:
            candidates.append(token)
        # Also try without $ / commas for alternate encodings.
        digits = re.sub(r"[^\d.]", "", token)
        if digits and digits not in candidates:
            candidates.append(digits)

    # Prefer distinctive multi-word phrases from the chunk.
    for phrase in (
        "Total net sales",
        "Net sales",
        "Gross margin",
        "Operating income",
        "Net income",
    ):
        if phrase.casefold() in cleaned.casefold() and phrase not in candidates:
            candidates.insert(0, phrase)

    return candidates[:limit]


def find_bboxes_by_search(
    source: Path,
    *,
    page: int,
    search_texts: Sequence[str],
    max_boxes: int = 5,
) -> list[BBox]:
    """Locate highlight rects by searching page text (runtime fallback)."""

    if page < 1 or not search_texts:
        return []

    doc = fitz.open(source)
    try:
        if page > doc.page_count:
            return []
        pdf_page = doc.load_page(page - 1)
        found: list[BBox] = []
        seen: set[tuple[float, float, float, float]] = set()
        for raw in search_texts:
            text = raw.strip()
            if len(text) < 3:
                continue
            variants = [text]
            if "," in text:
                variants.append(text.replace(",", ""))
            if text.startswith("$"):
                variants.append(text.lstrip("$").strip())
            for variant in variants:
                try:
                    hits = pdf_page.search_for(variant)
                except Exception:
                    continue
                for rect in hits:
                    key = (
                        round(rect.x0, 1),
                        round(rect.y0, 1),
                        round(rect.x1, 1),
                        round(rect.y1, 1),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    # Pad slightly so the highlight is easy to see.
                    pad = 2.0
                    found.append(
                        BBox(
                            x0=float(rect.x0) - pad,
                            y0=float(rect.y0) - pad,
                            x1=float(rect.x1) + pad,
                            y1=float(rect.y1) + pad,
                        )
                    )
                    if len(found) >= max_boxes:
                        return found
        return found
    finally:
        doc.close()


def resolve_highlight_bboxes(
    source: Path,
    *,
    page: int,
    stored: Sequence[BBox],
    search_texts: Sequence[str],
) -> list[BBox]:
    """Prefer stored ingest bboxes; fall back to live page search."""

    if stored:
        return list(stored)[:5]
    # Prefer numeric hits (the cited figure) over broad phrases like "Net sales".
    amounts = [text for text in search_texts if _AMOUNT_RE.search(text)]
    phrases = [text for text in search_texts if text not in amounts]
    boxes = find_bboxes_by_search(source, page=page, search_texts=amounts)
    if boxes:
        return boxes[:3]
    return find_bboxes_by_search(source, page=page, search_texts=phrases)


def render_highlighted_pdf(
    source: Path,
    *,
    page: int,
    bboxes: Sequence[BBox],
    output_dir: Path,
) -> Path:
    """Write a temp copy with yellow rects drawn into page content.

    Content-stream drawing (not annotations) so Chainlit's PDF.js viewer
    shows the highlight.
    """

    if page < 1:
        raise ValueError("page must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"hl-{uuid.uuid4().hex}.pdf"

    doc = fitz.open(source)
    try:
        if page > doc.page_count:
            raise ValueError(f"page {page} out of range for {source}")
        if bboxes:
            pdf_page = doc.load_page(page - 1)
            for box in bboxes:
                rect = fitz.Rect(*box.as_tuple())
                # Drawn into the page so viewers that skip annots still show it.
                pdf_page.draw_rect(
                    rect,
                    color=(0.85, 0.55, 0.0),
                    fill=(1.0, 0.92, 0.35),
                    width=1.2,
                    fill_opacity=0.45,
                    overlay=True,
                )
        doc.save(output_path, garbage=3, deflate=True)
    finally:
        doc.close()
    return output_path
