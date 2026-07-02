#!/usr/bin/env python3
"""Compare retrieval quality for several chunking strategies.

The script intentionally avoids heavyweight dependencies. It implements
LangChain-like baselines locally:

- CharacterTextSplitter: fixed character windows;
- RecursiveCharacterTextSplitter: recursive separator-aware splitting;
- TokenTextSplitter: fixed token/word windows;
- Semantic-lite splitter: sentence grouping with TF-IDF cosine boundary hints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from RAG._archive_nested_pipeline.scripts.compress_documents_bm25 import split_segments, tokenize
from RAG._archive_nested_pipeline.scripts.evaluate_bm25_retrieval import evaluate, read_jsonl
from RAG._archive_nested_pipeline.scripts.recursive_chunk_documents import chunk_document as recursive_chunk_document


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS = ROOT / "data" / "prepared" / "dmv_documents.jsonl"
DEFAULT_QUESTIONS = ROOT / "data" / "prepared" / "dmv_questions_validation.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "experiments" / "chunking"
DEFAULT_REPORT = ROOT / "reports" / "chunking_comparison.md"


@dataclass(frozen=True)
class ChunkConfig:
    splitter: str
    chunk_size: int
    chunk_overlap: int
    unit: str
    semantic_threshold: float | None = None

    @property
    def label(self) -> str:
        pieces = [
            self.splitter,
            f"size={self.chunk_size}",
            f"overlap={self.chunk_overlap}",
            self.unit,
        ]
        if self.semantic_threshold is not None:
            pieces.append(f"threshold={self.semantic_threshold}")
        return " | ".join(pieces)


@dataclass(frozen=True)
class TextBoundary:
    start: int
    end: int


WORD_PATTERN = r"\S+"


def validate_size_overlap(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def trim_boundary(text: str, start: int, end: int) -> TextBoundary | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return TextBoundary(start, end)


def make_chunk_records(
    document: dict[str, Any],
    boundaries: Iterable[TextBoundary],
    config: ChunkConfig,
) -> list[dict[str, Any]]:
    text = document["text"]
    records = []
    seen = set()
    for boundary in boundaries:
        trimmed = trim_boundary(text, boundary.start, boundary.end)
        if trimmed is None:
            continue
        if (trimmed.start, trimmed.end) in seen:
            continue
        seen.add((trimmed.start, trimmed.end))
        chunk_text = text[trimmed.start : trimmed.end]
        records.append(
            {
                "chunk_id": f"{document['document_id']}::{config.splitter}::{len(records):04d}",
                "document_id": document["document_id"],
                "title": document.get("title"),
                "domain": document.get("domain"),
                "chunk_index": len(records),
                "text": chunk_text,
                "splitter": config.splitter,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "unit": config.unit,
                "semantic_threshold": config.semantic_threshold,
                "boundaries": {
                    "char_start": trimmed.start,
                    "char_end": trimmed.end,
                },
                "stats": {
                    "characters": len(chunk_text),
                    "words": len(chunk_text.split()),
                },
            }
        )
    return records


def character_boundaries(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextBoundary]:
    validate_size_overlap(chunk_size, chunk_overlap)
    step = chunk_size - chunk_overlap
    boundaries = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        boundaries.append(TextBoundary(start, end))
        if end >= len(text):
            break
        start += step
    return boundaries


def character_chunks(document: dict[str, Any], config: ChunkConfig) -> list[dict[str, Any]]:
    return make_chunk_records(
        document,
        character_boundaries(document["text"], config.chunk_size, config.chunk_overlap),
        config,
    )


def token_boundaries(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextBoundary]:
    validate_size_overlap(chunk_size, chunk_overlap)
    import re

    tokens = list(re.finditer(WORD_PATTERN, text))
    if not tokens:
        return []

    step = chunk_size - chunk_overlap
    boundaries = []
    token_start = 0
    while token_start < len(tokens):
        token_end = min(token_start + chunk_size, len(tokens))
        boundaries.append(
            TextBoundary(tokens[token_start].start(), tokens[token_end - 1].end())
        )
        if token_end >= len(tokens):
            break
        token_start += step
    return boundaries


def token_chunks(document: dict[str, Any], config: ChunkConfig) -> list[dict[str, Any]]:
    return make_chunk_records(
        document,
        token_boundaries(document["text"], config.chunk_size, config.chunk_overlap),
        config,
    )


def recursive_chunks(document: dict[str, Any], config: ChunkConfig) -> list[dict[str, Any]]:
    chunks = recursive_chunk_document(
        document,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    records = []
    for chunk in chunks:
        record = chunk.to_record()
        record["splitter"] = config.splitter
        record["chunk_size"] = config.chunk_size
        record["chunk_overlap"] = config.chunk_overlap
        record["unit"] = config.unit
        record["semantic_threshold"] = config.semantic_threshold
        records.append(record)
    return records


def segment_idf(segments: list[Any]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for segment in segments:
        document_frequency.update(set(segment.tokens))

    segment_count = len(segments)
    return {
        token: math.log(1 + (segment_count - frequency + 0.5) / (frequency + 0.5))
        for token, frequency in document_frequency.items()
    }


def tfidf_cosine(left_tokens: list[str], right_tokens: list[str], idf: dict[str, float]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0

    left = Counter(left_tokens)
    right = Counter(right_tokens)
    shared_terms = set(left).intersection(right)
    dot = sum(left[term] * right[term] * idf.get(term, 0.0) ** 2 for term in shared_terms)
    left_norm = math.sqrt(
        sum((frequency * idf.get(term, 0.0)) ** 2 for term, frequency in left.items())
    )
    right_norm = math.sqrt(
        sum((frequency * idf.get(term, 0.0)) ** 2 for term, frequency in right.items())
    )
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def trailing_overlap_segments(segments: list[Any], chunk_overlap: int) -> list[Any]:
    selected = []
    total = 0
    for segment in reversed(segments):
        length = segment.end - segment.start
        if selected and total + length > chunk_overlap:
            break
        selected.insert(0, segment)
        total += length
        if total >= chunk_overlap:
            break
    return selected


def semantic_boundaries(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    threshold: float,
) -> list[TextBoundary]:
    validate_size_overlap(chunk_size, chunk_overlap)
    segments = split_segments(text)
    if len(segments) <= 1:
        return character_boundaries(text, chunk_size, chunk_overlap)

    idf = segment_idf(segments)
    min_chunk_size = int(chunk_size * 0.55)
    boundaries = []
    current = []
    segment_index = 0

    while segment_index < len(segments):
        if not current:
            current.append(segments[segment_index])
            segment_index += 1

        added_new_segments = 0
        while segment_index < len(segments):
            next_segment = segments[segment_index]
            projected_length = next_segment.end - current[0].start
            current_length = current[-1].end - current[0].start

            if projected_length > chunk_size:
                if added_new_segments == 0 and len(current) > 1:
                    current.pop(0)
                    continue
                break

            if current_length >= min_chunk_size and added_new_segments > 0:
                similarity = tfidf_cosine(current[-1].tokens, next_segment.tokens, idf)
                if similarity <= threshold:
                    break

            current.append(next_segment)
            segment_index += 1
            added_new_segments += 1

        boundaries.append(TextBoundary(current[0].start, current[-1].end))
        if segment_index >= len(segments):
            break

        overlap = trailing_overlap_segments(current, chunk_overlap)
        while overlap and segments[segment_index].end - overlap[0].start > chunk_size:
            overlap.pop(0)
        current = overlap

    return boundaries


def semantic_chunks(document: dict[str, Any], config: ChunkConfig) -> list[dict[str, Any]]:
    threshold = config.semantic_threshold
    if threshold is None:
        threshold = 0.08
    return make_chunk_records(
        document,
        semantic_boundaries(
            document["text"],
            config.chunk_size,
            config.chunk_overlap,
            threshold,
        ),
        config,
    )


def build_chunks(documents: list[dict[str, Any]], config: ChunkConfig) -> list[dict[str, Any]]:
    splitter_map = {
        "character": character_chunks,
        "recursive": recursive_chunks,
        "token": token_chunks,
        "semantic_lite": semantic_chunks,
    }
    splitter = splitter_map[config.splitter]
    chunks = []
    for document in documents:
        chunks.extend(splitter(document, config))
    return chunks


def default_configs() -> list[ChunkConfig]:
    configs: list[ChunkConfig] = []

    for splitter in ("character", "recursive"):
        for size in (400, 500, 800):
            for overlap in (0, 100, 200):
                if overlap < size:
                    configs.append(ChunkConfig(splitter, size, overlap, "characters"))

    for size in (80, 120, 160):
        for overlap in (0, 30, 60):
            if overlap < size:
                configs.append(ChunkConfig("token", size, overlap, "tokens"))

    for size in (500, 800):
        for overlap in (100, 200):
            for threshold in (0.05, 0.12):
                configs.append(
                    ChunkConfig(
                        "semantic_lite",
                        size,
                        overlap,
                        "characters",
                        semantic_threshold=threshold,
                    )
                )

    return configs


def summarize_chunks(chunks: list[dict[str, Any]]) -> dict[str, float | int]:
    char_counts = [chunk["stats"]["characters"] for chunk in chunks]
    word_counts = [chunk["stats"]["words"] for chunk in chunks]
    return {
        "chunks": len(chunks),
        "avg_chars": statistics.mean(char_counts),
        "min_chars": min(char_counts),
        "max_chars": max(char_counts),
        "avg_words": statistics.mean(word_counts),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
    }


def run_experiment(
    documents: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    config: ChunkConfig,
) -> dict[str, Any]:
    chunks = build_chunks(documents, config)
    chunk_stats = summarize_chunks(chunks)
    metrics = evaluate(chunks, questions)
    return {
        **asdict(config),
        "label": config.label,
        **chunk_stats,
        **metrics,
    }


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "splitter",
        "chunk_size",
        "chunk_overlap",
        "unit",
        "semantic_threshold",
        "chunks",
        "avg_chars",
        "avg_words",
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "mrr@10",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fieldnames})


def format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(records: list[dict[str, Any]]) -> str:
    headers = [
        "Rank",
        "Splitter",
        "Size",
        "Overlap",
        "Unit",
        "Threshold",
        "Chunks",
        "Avg chars",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "MRR@10",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for rank, record in enumerate(records, start=1):
        values = [
            rank,
            record["splitter"],
            record["chunk_size"],
            record["chunk_overlap"],
            record["unit"],
            record.get("semantic_threshold") or "",
            record["chunks"],
            f"{record['avg_chars']:.1f}",
            format_metric(record["recall@1"]),
            format_metric(record["recall@5"]),
            format_metric(record["recall@10"]),
            format_metric(record["mrr@10"]),
        ]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def build_report(records: list[dict[str, Any]], documents_path: Path, questions_path: Path) -> str:
    sorted_records = sorted(
        records,
        key=lambda record: (record["recall@10"], record["recall@5"], record["mrr@10"]),
        reverse=True,
    )
    best = sorted_records[0]
    company_matches = [
        record
        for record in records
        if record["splitter"] == "recursive"
        and record["chunk_size"] == 500
        and record["chunk_overlap"] == 200
    ]
    company = company_matches[0] if company_matches else None
    company_line = (
        f"- Baseline `recursive 500/200`: "
        f"Recall@10={company['recall@10']:.4f}, "
        f"Recall@5={company['recall@5']:.4f}, чанков={company['chunks']}."
        if company
        else "- Baseline `recursive 500/200` не попал в этот укороченный прогон."
    )

    return f"""# Chunking comparison

Сравнение способов разбиения DMV-документов для RAG retrieval.

Документы: `{documents_path}`

Вопросы для проверки: `{questions_path}`

Метрика: document-level Recall@k. Вопрос считается найденным, если среди top-k
чанков есть хотя бы один чанк из правильного `gold_document_ids`.

## Главное

- Лучший вариант по Recall@10: `{best['splitter']}`, size={best['chunk_size']}, overlap={best['chunk_overlap']}, Recall@10={best['recall@10']:.4f}.
{company_line}
- TokenTextSplitter в этом датасете оказался сильным конкурентом, потому что BM25 retrieval тоже работает по словам/термам.
- Semantic-lite реализован без внешних embedding-моделей, через sentence grouping и TF-IDF similarity; это полезный эксперимент, но не полноценный embedding semantic splitter.

## Top results

{markdown_table(sorted_records[:15])}

## All results

{markdown_table(sorted_records)}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare chunking strategies for RAG retrieval.")
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="If positive, run only the first N configs for a quick smoke test.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    documents = read_jsonl(args.documents)
    questions = read_jsonl(args.questions)
    configs = default_configs()
    if args.top > 0:
        configs = configs[: args.top]

    results = []
    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {config.label}")
        result = run_experiment(documents, questions, config)
        results.append(result)
        print(
            f"  chunks={result['chunks']} "
            f"r@5={result['recall@5']:.4f} "
            f"r@10={result['recall@10']:.4f}"
        )

    sorted_results = sorted(
        results,
        key=lambda record: (record["recall@10"], record["recall@5"], record["mrr@10"]),
        reverse=True,
    )

    jsonl_path = args.output_dir / "chunking_comparison_results.jsonl"
    csv_path = args.output_dir / "chunking_comparison_results.csv"
    write_jsonl(sorted_results, jsonl_path)
    write_csv(sorted_results, csv_path)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        build_report(sorted_results, args.documents, args.questions),
        encoding="utf-8",
    )

    best = sorted_results[0]
    print()
    print(
        "Best: "
        f"{best['splitter']} size={best['chunk_size']} "
        f"overlap={best['chunk_overlap']} "
        f"recall@10={best['recall@10']:.4f}"
    )
    print(f"JSONL:  {jsonl_path}")
    print(f"CSV:    {csv_path}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
