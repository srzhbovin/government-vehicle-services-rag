"""Deterministic metrics for RAG retrieval and generated answers."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence


TOKEN_PATTERN = re.compile(r"[a-zа-яё0-9]+", flags=re.IGNORECASE)
INSUFFICIENT_PATTERNS = (
    "insufficient",
    "not enough information",
    "недостаточно информации",
    "недостаточно данных",
)


def normalize_answer(value: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(value.lower()))


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_scores(prediction: str, reference: str) -> tuple[float, float, float]:
    predicted = TOKEN_PATTERN.findall(prediction.lower())
    expected = TOKEN_PATTERN.findall(reference.lower())
    if not predicted or not expected:
        same = float(predicted == expected)
        return same, same, same
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def citation_numbers(answer: str, source: str | None = None) -> list[int]:
    combined = f"{answer}\n{source or ''}"
    return sorted({int(value) for value in re.findall(r"\[(\d+)\]", combined)})


def citations_are_valid(citations: Sequence[int], source_count: int) -> bool:
    return bool(citations) and all(1 <= value <= source_count for value in citations)


def retrieval_hit(
    retrieved_document_ids: Sequence[str],
    gold_document_ids: Sequence[str],
    k: int,
) -> bool:
    return bool(set(retrieved_document_ids[:k]) & set(gold_document_ids))


def retrieval_recall(
    retrieved_document_ids: Sequence[str],
    gold_document_ids: Sequence[str],
    k: int,
) -> float:
    gold = set(gold_document_ids)
    if not gold:
        return 0.0
    return len(set(retrieved_document_ids[:k]) & gold) / len(gold)


def reciprocal_rank(
    retrieved_document_ids: Sequence[str],
    gold_document_ids: Sequence[str],
) -> float:
    gold = set(gold_document_ids)
    for rank, document_id in enumerate(retrieved_document_ids, start=1):
        if document_id in gold:
            return 1.0 / rank
    return 0.0


def mean(values: Iterable[float | int | None]) -> float:
    usable = [float(value) for value in values if value is not None and math.isfinite(value)]
    return sum(usable) / len(usable) if usable else 0.0


def error_categories(row: dict, semantic_threshold: float = 0.60) -> list[str]:
    categories = []
    if row.get("generation_error"):
        return ["generation_error"]
    if not row.get("retrieval_hit_at_5"):
        categories.append("retrieval_miss")
    if row.get("parse_success") is False:
        categories.append("json_parse_error")
    elif row.get("schema_valid") is False:
        categories.append("schema_validation_error")
    if not row.get("citation_valid"):
        categories.append("citation_error")
    answer = str(row.get("answer") or "").lower()
    if row.get("retrieval_hit_at_5") and any(
        marker in answer for marker in INSUFFICIENT_PATTERNS
    ):
        categories.append("false_insufficient_context")
    if row.get("retrieval_hit_at_5") and (
        row.get("semantic_similarity") or 0.0
    ) < semantic_threshold:
        categories.append("low_answer_similarity")
    if (row.get("reference_token_recall") or 0.0) < 0.35:
        categories.append("low_reference_coverage")
    if row.get("required_terms") and (row.get("required_fact_coverage") or 0.0) < 0.8:
        categories.append("missing_required_facts")
    if (
        row.get("confidence") is not None
        and row["confidence"] >= 0.8
        and "low_answer_similarity" in categories
    ):
        categories.append("overconfident_answer")
    return categories or ["ok"]
