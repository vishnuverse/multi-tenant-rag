"""Evaluation metrics for the golden dataset."""

from __future__ import annotations

from collections.abc import Sequence


def faithfulness(answer: str, contexts: Sequence[str]) -> float:
    if not answer.strip():
        return 0.0
    if not contexts:
        return 0.0
    overlap = sum(1 for token in _tokens(answer) if token in _token_set(contexts))
    return overlap / max(len(_tokens(answer)), 1)


def answer_relevance(answer: str, question: str) -> float:
    if not answer.strip() or not question.strip():
        return 0.0
    answer_tokens = set(_tokens(answer))
    question_tokens = set(_tokens(question))
    if not question_tokens:
        return 0.0
    return len(answer_tokens & question_tokens) / len(question_tokens)


def context_precision(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
) -> float:
    if not retrieved_sources:
        return 0.0
    expected = {source.casefold() for source in expected_sources}
    hits = sum(1 for source in retrieved_sources if source.casefold() in expected)
    return hits / len(retrieved_sources)


def context_recall(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
) -> float:
    if not expected_sources:
        return 1.0
    expected = {source.casefold() for source in expected_sources}
    hits = sum(
        1
        for source in expected
        if any(source in retrieved.casefold() for retrieved in retrieved_sources)
    )
    return hits / len(expected)


def numeric_accuracy(
    answer: str,
    expected_figures: Sequence[str],
) -> float:
    if not expected_figures:
        return 1.0
    answer_lower = answer.casefold()
    hits = sum(1 for figure in expected_figures if figure.casefold() in answer_lower)
    return hits / len(expected_figures)


def isolation_check(retrieved_companies: Sequence[str], allowed: Sequence[str]) -> bool:
    allowed_set = {company.casefold() for company in allowed}
    return all(company.casefold() in allowed_set for company in retrieved_companies)


def _tokens(text: str) -> list[str]:
    return [token for token in text.casefold().split() if token]


def _token_set(texts: Sequence[str]) -> set[str]:
    tokens: set[str] = set()
    for text in texts:
        tokens.update(_tokens(text))
    return tokens
