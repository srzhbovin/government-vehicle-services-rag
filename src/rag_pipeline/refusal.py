"""Retrieval-based refusal checks for questions outside the document corpus."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .retriever import RetrievedChunk


TOKEN_PATTERN = re.compile(r"[a-zа-яё0-9]+", flags=re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "can",
    "do",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "with",
    "you",
    "your",
    "а",
    "в",
    "где",
    "для",
    "и",
    "как",
    "мне",
    "на",
    "о",
    "по",
    "с",
    "что",
    "это",
}


@dataclass(frozen=True)
class RefusalFeatures:
    top_score: float | None
    score_gap: float | None
    lexical_overlap: float
    matched_query_terms: int
    query_terms: int


@dataclass(frozen=True)
class RefusalDecision:
    refused: bool
    reason: str | None
    features: RefusalFeatures


def refusal_features(question: str, chunks: Sequence[RetrievedChunk]) -> RefusalFeatures:
    scores = [float(chunk.score) for chunk in chunks]
    top_score = scores[0] if scores else None
    score_gap = scores[0] - scores[1] if len(scores) > 1 else None
    query_terms = meaningful_terms(question)
    context_terms = set()
    for chunk in chunks[:3]:
        context_terms.update(meaningful_terms(f"{chunk.title or ''} {chunk.text}"))
    matched = query_terms & context_terms
    return RefusalFeatures(
        top_score=top_score,
        score_gap=score_gap,
        lexical_overlap=(len(matched) / len(query_terms) if query_terms else 0.0),
        matched_query_terms=len(matched),
        query_terms=len(query_terms),
    )


def should_refuse(
    question: str,
    chunks: Sequence[RetrievedChunk],
    *,
    min_top_score: float,
    min_lexical_overlap: float,
) -> RefusalDecision:
    features = refusal_features(question, chunks)
    reasons = []
    if features.top_score is None:
        reasons.append("retrieval returned no chunks")
    elif features.top_score < min_top_score:
        reasons.append(
            f"top retrieval score {features.top_score:.3f} is below threshold {min_top_score:.3f}"
        )
    if features.lexical_overlap < min_lexical_overlap:
        reasons.append(
            f"lexical overlap {features.lexical_overlap:.3f} is below threshold {min_lexical_overlap:.3f}"
        )
    return RefusalDecision(
        refused=bool(reasons),
        reason="; ".join(reasons) if reasons else None,
        features=features,
    )


def refusal_answer(question: str, language: str) -> str:
    selected = language
    if selected == "auto":
        selected = "ru" if re.search(r"[а-яё]", question, flags=re.IGNORECASE) else "en"
    if selected == "ru":
        return (
            "В найденных DMV-документах недостаточно информации, чтобы надёжно "
            "ответить на этот вопрос."
        )
    return (
        "I do not have enough information in the retrieved DMV documents to answer "
        "this question reliably."
    )


def meaningful_terms(text: str) -> set[str]:
    terms = {
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if len(token) > 2 and token.lower() not in STOPWORDS
    }
    return terms
