#!/usr/bin/env python3
"""BM25-inspired sentence selection for compact RAG documents.

The goal is not abstractive summarization. The script keeps original sentences
that contain important domain terms, so the resulting compact document remains
traceable to the source text.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "prepared" / "dmv_documents.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "prepared" / "dmv_documents_compact_bm25.jsonl"
TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z'-]{2,}|\d+[a-zA-Z0-9-]*")
SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")

STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "also",
    "and",
    "any",
    "are",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "can",
    "cannot",
    "could",
    "did",
    "does",
    "doing",
    "don",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "into",
    "its",
    "itself",
    "more",
    "most",
    "must",
    "not",
    "off",
    "once",
    "only",
    "other",
    "our",
    "ours",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "too",
    "under",
    "until",
    "very",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
}


class CompressionError(ValueError):
    """Raised when compact document generation cannot continue."""


@dataclass(frozen=True)
class Segment:
    text: str
    start: int
    end: int
    tokens: list[str]

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def normalize_inline_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")
    return " ".join(normalized.split())


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token.lower() not in STOPWORDS
    ]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"path does not exist: {path}")
    records = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise CompressionError(
                    f"invalid JSON at {path}:{line_number}: {error.msg}"
                ) from error
    if not records:
        raise CompressionError(f"no documents found in: {path}")
    return records


def split_long_segment(text: str, offset: int, max_words: int = 70) -> list[Segment]:
    words = list(re.finditer(r"\S+", text))
    if len(words) <= max_words:
        return [Segment(text.strip(), offset, offset + len(text), tokenize(text))]

    segments = []
    for start_word in range(0, len(words), max_words):
        end_word = min(start_word + max_words, len(words))
        start = words[start_word].start()
        end = words[end_word - 1].end()
        segment_text = text[start:end].strip()
        if segment_text:
            segments.append(
                Segment(
                    segment_text,
                    offset + start,
                    offset + end,
                    tokenize(segment_text),
                )
            )
    return segments


def split_segments(text: str) -> list[Segment]:
    segments = []
    for match in SENTENCE_PATTERN.finditer(text):
        raw = match.group(0).strip()
        if not raw:
            continue

        if len(raw.split()) > 90:
            for separator in ("; ", ": ", ", "):
                if separator in raw:
                    local_start = match.start()
                    position = 0
                    for piece in raw.split(separator):
                        piece = piece.strip()
                        if not piece:
                            continue
                        found = text.find(piece, local_start + position)
                        found = found if found != -1 else match.start()
                        segments.extend(split_long_segment(piece, found))
                        position = found + len(piece) - local_start
                    break
            else:
                segments.extend(split_long_segment(raw, match.start()))
        else:
            segments.append(Segment(raw, match.start(), match.end(), tokenize(raw)))

    if not segments:
        return [Segment(text.strip(), 0, len(text), tokenize(text))]
    return segments


def document_frequency(documents: Iterable[dict[str, Any]]) -> dict[str, int]:
    frequencies: dict[str, int] = collections.Counter()
    for document in documents:
        frequencies.update(set(tokenize(document.get("text", ""))))
    return dict(frequencies)


def idf_scores(documents: list[dict[str, Any]]) -> dict[str, float]:
    df = document_frequency(documents)
    document_count = len(documents)
    return {
        token: math.log(1 + (document_count - count + 0.5) / (count + 0.5))
        for token, count in df.items()
    }


def important_terms(
    document: dict[str, Any],
    idf: dict[str, float],
    top_terms: int,
) -> list[tuple[str, float]]:
    term_counts = collections.Counter(tokenize(document.get("text", "")))
    scored_terms = [
        (term, count * idf.get(term, 0.0))
        for term, count in term_counts.items()
        if len(term) >= 3
    ]
    scored_terms.sort(key=lambda item: item[1], reverse=True)
    return scored_terms[:top_terms]


def bm25_score(
    segment: Segment,
    query_terms: list[tuple[str, float]],
    idf: dict[str, float],
    avg_segment_length: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not segment.tokens:
        return 0.0

    term_counts = collections.Counter(segment.tokens)
    score = 0.0
    segment_length = len(segment.tokens)
    normalizer = k1 * (1 - b + b * segment_length / max(avg_segment_length, 1.0))
    for term, query_weight in query_terms:
        frequency = term_counts.get(term, 0)
        if frequency == 0:
            continue
        term_idf = idf.get(term, 0.0)
        bm25_component = (frequency * (k1 + 1)) / (frequency + normalizer)
        score += term_idf * bm25_component * (1 + math.log1p(query_weight))
    return score


def prune_spans_for_text(
    metadata: dict[str, Any],
    compact_text: str,
) -> dict[str, Any]:
    pruned = dict(metadata)
    spans = metadata.get("spans")
    if not isinstance(spans, dict):
        return pruned

    compact_normalized = normalize_inline_text(compact_text)
    kept_spans = {}
    for span_id, span in spans.items():
        if not isinstance(span, dict) or not isinstance(span.get("text_sp"), str):
            continue
        span_text = normalize_inline_text(span["text_sp"])
        if span_text and span_text in compact_normalized:
            kept_spans[span_id] = span
    pruned["spans"] = kept_spans
    return pruned


def compress_document(
    document: dict[str, Any],
    idf: dict[str, float],
    target_ratio: float,
    top_terms: int,
    min_segments: int,
    max_segments: int,
    keep_first_segments: int,
) -> dict[str, Any]:
    text = document.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise CompressionError(f"document has empty text: {document.get('document_id')}")

    segments = split_segments(text)
    if len(segments) <= min_segments:
        compact_text = text
        selected_indices = list(range(len(segments)))
        query_terms = important_terms(document, idf, top_terms)
    else:
        query_terms = important_terms(document, idf, top_terms)
        avg_segment_length = statistics.mean(max(len(segment.tokens), 1) for segment in segments)
        scored_segments = [
            (
                index,
                bm25_score(segment, query_terms, idf, avg_segment_length),
                len(segment.text),
            )
            for index, segment in enumerate(segments)
        ]
        scored_segments.sort(key=lambda item: (item[1], item[2]), reverse=True)

        target_characters = max(1, int(len(text) * target_ratio))
        selected = set(range(min(keep_first_segments, len(segments))))
        selected_characters = sum(len(segments[index].text) for index in selected)

        for index, _, _ in scored_segments:
            if len(selected) >= max_segments:
                break
            if index in selected:
                continue
            selected.add(index)
            selected_characters += len(segments[index].text)
            if selected_characters >= target_characters and len(selected) >= min_segments:
                break

        while len(selected) < min_segments and len(selected) < len(segments):
            for index, _, _ in scored_segments:
                if index not in selected:
                    selected.add(index)
                    break

        selected_indices = sorted(selected)
        compact_text = " ".join(segments[index].text.strip() for index in selected_indices)

    metadata = prune_spans_for_text(document.get("metadata", {}), compact_text)
    metadata["compression"] = {
        "method": "bm25_sentence_selection",
        "target_ratio": target_ratio,
        "top_terms": top_terms,
        "original_characters": len(text),
        "compact_characters": len(compact_text),
        "original_words": len(text.split()),
        "compact_words": len(compact_text.split()),
        "total_segments": len(segments),
        "selected_segments": len(selected_indices),
        "selected_segment_indices": selected_indices,
        "important_terms": [
            {"term": term, "score": round(score, 6)}
            for term, score in query_terms[: min(20, len(query_terms))]
        ],
    }

    compact_document = dict(document)
    compact_document["text"] = compact_text
    compact_document["metadata"] = metadata
    compact_document["stats"] = {
        "characters": len(compact_text),
        "words": len(compact_text.split()),
    }
    return compact_document


def compress_documents(
    documents: list[dict[str, Any]],
    target_ratio: float = 0.45,
    top_terms: int = 40,
    min_segments: int = 3,
    max_segments: int = 30,
    keep_first_segments: int = 1,
) -> list[dict[str, Any]]:
    if not 0 < target_ratio <= 1:
        raise CompressionError("--target-ratio must be in the interval (0, 1]")
    if top_terms <= 0:
        raise CompressionError("--top-terms must be positive")
    if min_segments <= 0:
        raise CompressionError("--min-segments must be positive")
    if max_segments < min_segments:
        raise CompressionError("--max-segments must be >= --min-segments")

    idf = idf_scores(documents)
    return [
        compress_document(
            document=document,
            idf=idf,
            target_ratio=target_ratio,
            top_terms=top_terms,
            min_segments=min_segments,
            max_segments=max_segments,
            keep_first_segments=keep_first_segments,
        )
        for document in documents
    ]


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def print_summary(original: list[dict[str, Any]], compact: list[dict[str, Any]]) -> None:
    original_chars = sum(len(document["text"]) for document in original)
    compact_chars = sum(len(document["text"]) for document in compact)
    original_words = sum(len(document["text"].split()) for document in original)
    compact_words = sum(len(document["text"].split()) for document in compact)
    ratio = compact_chars / max(original_chars, 1)

    print(f"Documents:           {len(compact)}")
    print(f"Original characters: {original_chars}")
    print(f"Compact characters:  {compact_chars}")
    print(f"Character ratio:     {ratio:.3f}")
    print(f"Original words:      {original_words}")
    print(f"Compact words:       {compact_words}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact RAG documents with BM25-style sentence selection."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to prepared document JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination for compact document JSONL.",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.45,
        help="Target compact/original character ratio per document.",
    )
    parser.add_argument(
        "--top-terms",
        type=int,
        default=40,
        help="Number of important document terms used as the BM25 query.",
    )
    parser.add_argument(
        "--min-segments",
        type=int,
        default=3,
        help="Minimum number of sentence-like segments to keep per document.",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=30,
        help="Maximum number of sentence-like segments to keep per document.",
    )
    parser.add_argument(
        "--keep-first-segments",
        type=int,
        default=1,
        help="Always keep this many leading segments for context.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        documents = read_jsonl(args.path)
        compact = compress_documents(
            documents,
            target_ratio=args.target_ratio,
            top_terms=args.top_terms,
            min_segments=args.min_segments,
            max_segments=args.max_segments,
            keep_first_segments=args.keep_first_segments,
        )
        write_jsonl(compact, args.output)
    except (CompressionError, FileNotFoundError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print_summary(documents, compact)
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
