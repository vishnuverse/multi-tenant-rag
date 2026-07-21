"""Numeric integrity verification for grounded answers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

_FIGURE_RE = re.compile(
    r"""
    (?:
        \(\$\s?\d[\d,]*(?:\.\d+)?\)\s?(?:billion|million|bn|mn|[bm])\b
        |
        \$\s?\(?-?\d[\d,]*(?:\.\d+)?\)?\s?(?:billion|million|bn|mn|[bmk])\b
        |
        \$\s?-?\d[\d,]*(?:\.\d+)?\b
        |
        \b-?\d[\d,]*(?:\.\d+)?\s?(?:billion|million|bn|mn)\b
        |
        \b(?:up|down)\s+\d[\d,]*(?:\.\d+)?(?:%|\s+percent\b)
        |
        \b\d[\d,]*(?:\.\d+)?(?:%|\s+percent\b)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_SCALE = {
    "billion": 1_000_000_000,
    "bn": 1_000_000_000,
    "b": 1_000_000_000,
    "million": 1_000_000,
    "mn": 1_000_000,
    "m": 1_000_000,
    "k": 1_000,
}
_UNIT_SCALE_RE = re.compile(
    r"\(\s*in\s+(billions|millions|thousands)\b|\bin\s+(billions|millions|thousands)\b",
    re.IGNORECASE,
)
_UNIT_SCALE = {
    "billions": 1_000_000_000,
    "millions": 1_000_000,
    "thousands": 1_000,
}
# Financial statements often omit "$" on later columns: "Total net sales (1)  124,300"
_BARE_AMOUNT_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
_METRIC_TERMS = frozenset(
    {
        "revenue",
        "sales",
        "profit",
        "income",
        "earnings",
        "eps",
        "margin",
        "cash",
        "ebitda",
        "operating",
        "gross",
        "net",
        "cost",
        "expense",
        "expenses",
        "loss",
        "losses",
        "guidance",
        "outlook",
    }
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_AMOUNT_RE = re.compile(r"^amount:(-?\d+(?:\.\d+)?)$")


def extract_figures(text: str, *, include_bare_amounts: bool = False) -> list[str]:
    """Extract currency, percentage, and large numeric figures from text."""

    figured_spans = [match.span() for match in _FIGURE_RE.finditer(text)]
    figures = [text[start:end].strip() for start, end in figured_spans]
    if not include_bare_amounts:
        return figures

    for match in _BARE_AMOUNT_RE.finditer(text):
        start, end = match.span()
        overlaps = any(
            start < figured_end and end > figured_start
            for figured_start, figured_end in figured_spans
        )
        if overlaps:
            continue
        figures.append(match.group(0))
    return figures


def detect_unit_scale(text: str) -> int:
    """Return an implied dollar scale from headers like ``(In millions)``."""

    match = _UNIT_SCALE_RE.search(text)
    if match is None:
        return 1
    unit = (match.group(1) or match.group(2) or "").casefold()
    return _UNIT_SCALE.get(unit, 1)


def normalize_figure(value: str, *, implied_scale: int = 1) -> str:
    """Normalize a figure string for comparison across formatting variants."""

    cleaned = value.strip().casefold()
    cleaned = cleaned.replace(",", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\$\s+", "$", cleaned)
    cleaned = cleaned.replace("( ", "(").replace(" )", ")")

    percent_match = re.fullmatch(
        r"(?:up|down)\s+(-?\d+(?:\.\d+)?)(?:%|\s+percent)",
        cleaned,
    )
    if percent_match:
        direction = "up" if cleaned.startswith("up") else "down"
        return f"{direction}:{percent_match.group(1)}"

    percent_only = re.fullmatch(r"(-?\d+(?:\.\d+)?)(?:%|\s+percent)", cleaned)
    if percent_only:
        return f"percent:{percent_only.group(1)}"

    paren_currency = re.fullmatch(
        r"(?:\(\$?(-?\d+(?:\.\d+)?)\)|\$\((-?\d+(?:\.\d+)?)\))"
        r"\s*(billion|million|bn|mn|[bmk])?",
        cleaned,
    )
    if paren_currency:
        number = float(paren_currency.group(1) or paren_currency.group(2))
        scale = paren_currency.group(3)
        if scale:
            number *= _SCALE[scale]
        elif _should_apply_implied_scale(number, implied_scale):
            number *= implied_scale
        if number.is_integer():
            return f"amount:-{int(abs(number))}"
        return f"amount:-{abs(number)}"

    currency_match = re.fullmatch(
        r"\$?\(?(-?\d+(?:\.\d+)?)\)?(?:\s*(billion|million|bn|mn|[bmk]))?",
        cleaned,
    )
    if currency_match is None:
        return cleaned

    number = float(currency_match.group(1))
    scale = currency_match.group(2)
    if scale:
        number *= _SCALE[scale]
    elif _should_apply_implied_scale(number, implied_scale):
        number *= implied_scale
    sign = "-" if cleaned.startswith("(") or cleaned.startswith("$(") else ""
    if number < 0:
        sign = "-"
        number = abs(number)
    if number.is_integer():
        return f"amount:{sign}{int(number)}"
    return f"amount:{sign}{number}"


def verify_answer(
    answer: str,
    chunks: Sequence[Mapping[str, str | int | float | bool] | str],
) -> dict[str, int | list[str]]:
    """Verify that figures in an answer appear in retrieved chunk text."""

    corpus_scale = _corpus_unit_scale(chunks)
    chunk_views = _chunk_views(chunks, corpus_scale=corpus_scale)
    figures = extract_figures(answer)
    unverified: list[str] = []
    verified_count = 0

    for figure in figures:
        normalized = normalize_figure(figure)
        answer_sentence = _sentence_for_figure(answer, figure)
        metrics = _metric_terms(answer_sentence)
        if _figure_supported(normalized, figure, chunk_views, metrics):
            verified_count += 1
        else:
            unverified.append(figure)

    return {
        "figures_found": len(figures),
        "figures_verified": verified_count,
        "unverified": unverified,
    }


def _corpus_unit_scale(
    chunks: Sequence[Mapping[str, str | int | float | bool] | str],
) -> int:
    """Use the strongest unit header across retrieved excerpts."""

    scale = 1
    for chunk in chunks:
        text = chunk if isinstance(chunk, str) else chunk.get("text")
        if isinstance(text, str):
            scale = max(scale, detect_unit_scale(text))
    return scale


def _should_apply_implied_scale(number: float, implied_scale: int) -> bool:
    """Apply statement unit headers to line-item amounts, not EPS-like decimals."""

    if implied_scale <= 1:
        return False
    if not number.is_integer() and abs(number) < 1_000:
        return False
    return abs(number) >= 100


def _chunk_views(
    chunks: Sequence[Mapping[str, str | int | float | bool] | str],
    *,
    corpus_scale: int = 1,
) -> list[tuple[str, str, int]]:
    """Return ``(sentence, full_chunk_text, scale)`` rows for matching."""

    views: list[tuple[str, str, int]] = []
    for chunk in chunks:
        text = chunk if isinstance(chunk, str) else chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        scale = max(detect_unit_scale(text), corpus_scale)
        body = text.strip()
        parts = _SENTENCE_SPLIT_RE.split(body)
        for part in parts:
            if part.strip():
                views.append((part.strip(), body, scale))
    return views


def _sentence_for_figure(text: str, figure: str) -> str:
    for sentence in _SENTENCE_SPLIT_RE.split(text.strip()):
        if figure in sentence:
            return sentence
    return text


def _metric_terms(text: str) -> set[str]:
    tokens = {token.casefold().strip(".,;:()") for token in text.split()}
    return {token for token in tokens if token in _METRIC_TERMS}


def _figure_supported(
    normalized: str,
    raw: str,
    chunk_views: Sequence[tuple[str, str, int]],
    metrics: set[str],
) -> bool:
    for sentence, chunk_text, scale in chunk_views:
        if not _figure_in_text(normalized, raw, sentence, implied_scale=scale):
            continue
        if not metrics:
            return True
        source_metrics = _metric_terms(sentence) | _metric_terms(chunk_text)
        # Transcripts often state "$12 billion" without "revenue"; allow those.
        if not source_metrics:
            return True
        # Metric may sit in a table header outside the local sentence.
        if metrics & source_metrics:
            return True
    return False


def drop_unverified_sentences(answer: str, unverified: Sequence[str]) -> str | None:
    """Remove sentences that carry unverified figures; keep grounded remainder."""

    if not unverified:
        return answer.strip() or None
    kept: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(answer.strip()):
        text = sentence.strip()
        if not text:
            continue
        if any(figure in text for figure in unverified):
            continue
        kept.append(text)
    if not kept:
        return None
    return " ".join(kept)


def _figure_in_text(
    normalized: str,
    raw: str,
    text: str,
    *,
    implied_scale: int = 1,
) -> bool:
    if raw in text:
        return True
    # Always consider bare thousands-separated amounts in statements.
    candidates = extract_figures(text, include_bare_amounts=True)
    for candidate in candidates:
        bare = normalize_figure(candidate)
        if bare == normalized:
            return True
        if (
            implied_scale > 1
            and normalize_figure(candidate, implied_scale=implied_scale) == normalized
        ):
            return True
        # Earnings PDFs often print "$124,300" under an unseen "(In millions)" header
        # while the model answers "$124.3 billion" / "$124,300 million".
        if _statement_scale_equivalent(normalized, bare):
            return True
    return False


def _parse_amount(normalized: str) -> float | None:
    match = _AMOUNT_RE.fullmatch(normalized)
    if match is None:
        return None
    return float(match.group(1))


def _statement_scale_equivalent(answer_norm: str, source_norm: str) -> bool:
    """Match scaled answers to unscaled statement line items (>= 1000)."""

    answer_amount = _parse_amount(answer_norm)
    source_amount = _parse_amount(source_norm)
    if answer_amount is None or source_amount is None:
        return False
    if answer_amount == source_amount:
        return True
    if not float(source_amount).is_integer() or abs(source_amount) < 1_000:
        return False
    for factor in (1_000_000, 1_000_000_000):
        scaled = source_amount * factor
        # Allow mild rounding when models say "$69.1 billion" for $69,138 (millions).
        tolerance = max(100_000_000.0, abs(scaled) * 0.005)
        if abs(answer_amount - scaled) <= tolerance:
            return True
    return False
