#!/usr/bin/env python3
"""Evaluate simple BM25 retrieval over RAG chunks with document-level Recall@k."""

from __future__ import annotations

import argparse
import collections
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compress_documents_bm25 import tokenize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data" / "prepared" / "dmv_chunks_recursive.jsonl"
DEFAULT_QUESTIONS = ROOT / "data" / "prepared" / "dmv_questions_validation.jsonl"


@dataclass(frozen=True)
class BM25Index:
    chunks: list[dict[str, Any]]
    inverted_index: dict[str, list[tuple[int, int]]]
    idf: dict[str, float]
    lengths: list[int]
    avg_length: float


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_index(chunks: list[dict[str, Any]]) -> BM25Index:
    inverted_index: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    document_frequency: dict[str, int] = collections.Counter()
    lengths = []

    for chunk_index, chunk in enumerate(chunks):
        tokens = tokenize(chunk.get("text", ""))
        counts = collections.Counter(tokens)
        lengths.append(len(tokens))
        document_frequency.update(counts.keys())
        for token, frequency in counts.items():
            inverted_index[token].append((chunk_index, frequency))

    chunk_count = len(chunks)
    idf = {
        token: math.log(1 + (chunk_count - frequency + 0.5) / (frequency + 0.5))
        for token, frequency in document_frequency.items()
    }
    avg_length = sum(lengths) / max(len(lengths), 1)
    return BM25Index(
        chunks=chunks,
        inverted_index=dict(inverted_index),
        idf=idf,
        lengths=lengths,
        avg_length=avg_length,
    )


def search(
    index: BM25Index,
    query: str,
    top_k: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict[str, Any]]:
    query_terms = collections.Counter(tokenize(query))
    scores: dict[int, float] = collections.Counter()

    for term, query_frequency in query_terms.items():
        postings = index.inverted_index.get(term, [])
        term_idf = index.idf.get(term, 0.0)
        for chunk_index, term_frequency in postings:
            length = index.lengths[chunk_index]
            normalizer = k1 * (1 - b + b * length / max(index.avg_length, 1.0))
            bm25 = (term_frequency * (k1 + 1)) / (term_frequency + normalizer)
            scores[chunk_index] += term_idf * bm25 * query_frequency

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [index.chunks[chunk_index] for chunk_index, _ in ranked[:top_k]]


def evaluate(
    chunks: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    cutoffs: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float | int]:
    index = build_index(chunks)
    max_k = max(cutoffs)
    usable_questions = [
        question for question in questions if question.get("gold_document_ids")
    ]
    hits = {cutoff: 0 for cutoff in cutoffs}
    reciprocal_ranks = []

    for question in usable_questions:
        gold_document_ids = set(question["gold_document_ids"])
        retrieved_chunks = search(index, question.get("question", ""), max_k)
        retrieved_document_ids = [
            chunk.get("document_id") for chunk in retrieved_chunks
        ]

        first_hit_rank = None
        for rank, document_id in enumerate(retrieved_document_ids, start=1):
            if document_id in gold_document_ids:
                first_hit_rank = rank
                break

        if first_hit_rank is not None:
            reciprocal_ranks.append(1 / first_hit_rank)
        else:
            reciprocal_ranks.append(0)

        for cutoff in cutoffs:
            if any(
                document_id in gold_document_ids
                for document_id in retrieved_document_ids[:cutoff]
            ):
                hits[cutoff] += 1

    total = len(usable_questions)
    result: dict[str, float | int] = {"questions": total}
    for cutoff in cutoffs:
        result[f"recall@{cutoff}"] = hits[cutoff] / max(total, 1)
    result["mrr@10"] = sum(reciprocal_ranks) / max(total, 1)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BM25 retrieval over chunks.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks = read_jsonl(args.chunks)
    questions = read_jsonl(args.questions)
    result = evaluate(chunks, questions)

    print(f"Chunks:    {len(chunks)}")
    print(f"Questions: {result['questions']}")
    for metric in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10"):
        print(f"{metric}: {result[metric]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
