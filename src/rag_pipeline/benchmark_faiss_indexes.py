#!/usr/bin/env python3
"""Benchmark FAISS indexes for RAG retrieval.

The benchmark uses dense lexical TF-IDF vectors. This keeps the experiment
fully local and reproducible while still exercising real FAISS index types:
Flat, IVF and HNSW.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from compress_documents_bm25 import tokenize
from evaluate_bm25_retrieval import read_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS = ROOT / "data" / "current" / "prepared" / "dmv_chunks_token_120_0.jsonl"
DEFAULT_QUESTIONS = ROOT / "data" / "current" / "prepared" / "dmv_questions_validation.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "experiments" / "faiss"
DEFAULT_INDEX_DIR = ROOT / "data" / "current" / "indexes" / "faiss"
DEFAULT_REPORT = ROOT / "reports" / "faiss_comparison.md"


@dataclass(frozen=True)
class FaissIndexConfig:
    name: str
    kind: str
    nlist: int | None = None
    nprobe: int | None = None
    hnsw_m: int | None = None
    ef_construction: int | None = None
    ef_search: int | None = None


def default_index_configs() -> list[FaissIndexConfig]:
    return [
        FaissIndexConfig(name="flat_ip", kind="flat"),
        FaissIndexConfig(name="ivf32_nprobe4", kind="ivf", nlist=32, nprobe=4),
        FaissIndexConfig(name="ivf32_nprobe16", kind="ivf", nlist=32, nprobe=16),
        FaissIndexConfig(
            name="hnsw_m16_ef32",
            kind="hnsw",
            hnsw_m=16,
            ef_construction=80,
            ef_search=32,
        ),
        FaissIndexConfig(
            name="hnsw_m32_ef64",
            kind="hnsw",
            hnsw_m=32,
            ef_construction=120,
            ef_search=64,
        ),
    ]


def build_vocabulary(
    chunks: list[dict[str, Any]],
    dimension: int,
) -> tuple[dict[str, int], dict[str, float]]:
    document_frequency: collections.Counter[str] = collections.Counter()
    term_frequency: collections.Counter[str] = collections.Counter()

    for chunk in chunks:
        tokens = tokenize(chunk.get("text", ""))
        term_frequency.update(tokens)
        document_frequency.update(set(tokens))

    selected_terms = [
        term
        for term, _ in term_frequency.most_common(dimension)
    ]
    vocabulary = {term: index for index, term in enumerate(selected_terms)}
    chunk_count = len(chunks)
    idf = {
        term: math.log(
            1 + (chunk_count - document_frequency[term] + 0.5)
            / (document_frequency[term] + 0.5)
        )
        for term in selected_terms
    }
    return vocabulary, idf


def vectorize_texts(
    texts: list[str],
    vocabulary: dict[str, int],
    idf: dict[str, float],
):
    import numpy as np
    import faiss

    matrix = np.zeros((len(texts), len(vocabulary)), dtype="float32")
    for row_index, text in enumerate(texts):
        counts = collections.Counter(tokenize(text))
        for term, count in counts.items():
            column_index = vocabulary.get(term)
            if column_index is None:
                continue
            matrix[row_index, column_index] = (1 + math.log(count)) * idf[term]

    faiss.normalize_L2(matrix)
    return matrix


def build_index(config: FaissIndexConfig, vectors):
    import faiss

    dimension = vectors.shape[1]
    if config.kind == "flat":
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        return index

    if config.kind == "ivf":
        quantizer = faiss.IndexFlatIP(dimension)
        nlist = min(config.nlist or 32, max(1, vectors.shape[0]))
        index = faiss.IndexIVFFlat(
            quantizer,
            dimension,
            nlist,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.train(vectors)
        index.add(vectors)
        index.nprobe = min(config.nprobe or 4, nlist)
        return index

    if config.kind == "hnsw":
        try:
            index = faiss.IndexHNSWFlat(
                dimension,
                config.hnsw_m or 16,
                faiss.METRIC_INNER_PRODUCT,
            )
        except TypeError:
            index = faiss.IndexHNSWFlat(dimension, config.hnsw_m or 16)
            index.metric_type = faiss.METRIC_INNER_PRODUCT

        index.hnsw.efConstruction = config.ef_construction or 80
        index.add(vectors)
        index.hnsw.efSearch = config.ef_search or 32
        return index

    raise ValueError(f"unsupported FAISS index kind: {config.kind}")


def index_size_bytes(index) -> int:
    import faiss

    return len(faiss.serialize_index(index))


def save_index(index, path: Path) -> None:
    import faiss

    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def evaluate_search_results(
    chunks: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    indices,
    cutoffs: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float | int]:
    usable_questions = [
        question for question in questions if question.get("gold_document_ids")
    ]
    hits = {cutoff: 0 for cutoff in cutoffs}
    reciprocal_ranks = []

    for question, row in zip(usable_questions, indices):
        gold_document_ids = set(question["gold_document_ids"])
        retrieved_document_ids = []
        for chunk_index in row:
            if chunk_index < 0:
                continue
            retrieved_document_ids.append(chunks[int(chunk_index)]["document_id"])

        first_hit_rank = None
        for rank, document_id in enumerate(retrieved_document_ids, start=1):
            if document_id in gold_document_ids:
                first_hit_rank = rank
                break

        reciprocal_ranks.append(0 if first_hit_rank is None else 1 / first_hit_rank)

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


def benchmark_index(
    config: FaissIndexConfig,
    chunks: list[dict[str, Any]],
    chunk_vectors,
    query_vectors,
    questions: list[dict[str, Any]],
    repeats: int,
    top_k: int,
    index_dir: Path,
) -> dict[str, Any]:
    start = time.perf_counter()
    index = build_index(config, chunk_vectors)
    build_seconds = time.perf_counter() - start

    index.search(query_vectors[: min(5, len(query_vectors))], top_k)
    search_start = time.perf_counter()
    distances = None
    indices = None
    for _ in range(repeats):
        distances, indices = index.search(query_vectors, top_k)
    search_seconds = (time.perf_counter() - search_start) / max(repeats, 1)

    assert distances is not None
    assert indices is not None

    metrics = evaluate_search_results(chunks, questions, indices)
    size_bytes = index_size_bytes(index)
    index_path = index_dir / f"{config.name}.index"
    save_index(index, index_path)

    return {
        **asdict(config),
        "chunks": len(chunks),
        "dimension": int(chunk_vectors.shape[1]),
        "vectors_memory_bytes": int(chunk_vectors.nbytes),
        "index_size_bytes": size_bytes,
        "index_path": str(index_path.relative_to(ROOT)),
        "build_ms": build_seconds * 1000,
        "search_ms_total": search_seconds * 1000,
        "search_ms_per_query": search_seconds * 1000 / len(query_vectors),
        **metrics,
    }


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "name",
        "kind",
        "nlist",
        "nprobe",
        "hnsw_m",
        "ef_construction",
        "ef_search",
        "chunks",
        "dimension",
        "index_size_bytes",
        "build_ms",
        "search_ms_total",
        "search_ms_per_query",
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "mrr@10",
        "index_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fieldnames})


def markdown_table(records: list[dict[str, Any]]) -> str:
    headers = [
        "Index",
        "Build ms",
        "Search ms/query",
        "Index size",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "MRR@10",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    record["name"],
                    f"{record['build_ms']:.2f}",
                    f"{record['search_ms_per_query']:.4f}",
                    f"{record['index_size_bytes'] / 1024:.1f} KB",
                    f"{record['recall@1']:.4f}",
                    f"{record['recall@5']:.4f}",
                    f"{record['recall@10']:.4f}",
                    f"{record['mrr@10']:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_report(
    records: list[dict[str, Any]],
    chunks_path: Path,
    questions_path: Path,
    dimension: int,
) -> str:
    by_recall = sorted(
        records,
        key=lambda record: (record["recall@10"], record["recall@5"], record["mrr@10"]),
        reverse=True,
    )
    by_speed = sorted(records, key=lambda record: record["search_ms_per_query"])
    best_quality = by_recall[0]
    fastest = by_speed[0]

    return f"""# FAISS index comparison

Исследование FAISS-индексов для retrieval поверх DMV chunks.

Корпус чанков: `{chunks_path.relative_to(ROOT)}`

Validation-вопросы: `{questions_path.relative_to(ROOT)}`

Векторы: dense TF-IDF, L2-normalized, dimension={dimension}. Поиск идёт через
inner product, то есть эквивалент cosine similarity для нормализованных векторов.

## Главное

- Лучшее качество: `{best_quality['name']}`, Recall@10={best_quality['recall@10']:.4f}.
- Самый быстрый поиск: `{fastest['name']}`, {fastest['search_ms_per_query']:.4f} ms/query.
- На текущем небольшом корпусе Flat практически оптимален: он точный, простой и быстрый.
- IVF и HNSW полезнее становятся на больших корпусах, где Flat уже слишком дорогой по времени.

## Results

{markdown_table(records)}

## Рекомендации

- До десятков/сотен тысяч чанков: начинать с `IndexFlatIP`.
- Когда корпус растёт и latency становится проблемой: пробовать IVF, подбирать `nlist` и `nprobe`.
- Для больших динамичных коллекций и быстрых approximate-запросов: пробовать HNSW, подбирать `M` и `efSearch`.
- Качество индекса нельзя оценивать отдельно от embeddings: после перехода с TF-IDF vectors на neural embeddings benchmark надо повторить.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FAISS Flat/IVF/HNSW indexes.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    import faiss

    args = parse_args()
    faiss.omp_set_num_threads(args.threads)

    chunks = read_jsonl(args.chunks)
    questions = read_jsonl(args.questions)
    usable_questions = [
        question for question in questions if question.get("gold_document_ids")
    ]

    vocabulary, idf = build_vocabulary(chunks, args.dimension)
    chunk_vectors = vectorize_texts(
        [chunk.get("text", "") for chunk in chunks],
        vocabulary,
        idf,
    )
    query_vectors = vectorize_texts(
        [question.get("question", "") for question in usable_questions],
        vocabulary,
        idf,
    )

    results = []
    for config in default_index_configs():
        print(f"Benchmarking {config.name}")
        result = benchmark_index(
            config=config,
            chunks=chunks,
            chunk_vectors=chunk_vectors,
            query_vectors=query_vectors,
            questions=usable_questions,
            repeats=args.repeats,
            top_k=args.top_k,
            index_dir=args.index_dir,
        )
        results.append(result)
        print(
            f"  build={result['build_ms']:.2f}ms "
            f"search={result['search_ms_per_query']:.4f}ms/query "
            f"r@10={result['recall@10']:.4f}"
        )

    results = sorted(
        results,
        key=lambda record: (record["recall@10"], -record["search_ms_per_query"]),
        reverse=True,
    )

    jsonl_path = args.output_dir / "faiss_index_results.jsonl"
    csv_path = args.output_dir / "faiss_index_results.csv"
    write_jsonl(results, jsonl_path)
    write_csv(results, csv_path)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        build_report(results, args.chunks, args.questions, len(vocabulary)),
        encoding="utf-8",
    )

    print()
    print(f"JSONL:  {jsonl_path}")
    print(f"CSV:    {csv_path}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
