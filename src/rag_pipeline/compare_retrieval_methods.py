#!/usr/bin/env python3
"""Compare retrieval methods on the MultiDoc2Dial DMV corpus.

The experiment uses one corpus, one set of questions and one evaluation
function for every method.  It compares:

* exhaustive cosine similarity;
* BM25;
* exact FAISS IndexFlatIP search;
* BM25 + FAISS fusion with Reciprocal Rank Fusion (RRF);
* optional cross-encoder reranking of the hybrid candidate list.

Validation questions are used to choose the retrieval method.  Test questions
are evaluated only to report the final, unbiased quality estimate.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS = ROOT / "data" / "current" / "prepared" / "dmv_chunks_token_120_0.jsonl"
DEFAULT_VALIDATION = ROOT / "data" / "current" / "prepared" / "dmv_questions_validation.jsonl"
DEFAULT_TEST = ROOT / "data" / "current" / "prepared" / "dmv_questions_test.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "experiments" / "retrieval"
DEFAULT_INDEX_DIR = ROOT / "data" / "current" / "indexes" / "retrieval"
DEFAULT_REPORT = ROOT / "reports" / "retrieval_comparison.md"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L2-v2"

TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)?", flags=re.UNICODE)
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "cannot",
    "could", "did", "do", "does", "doing", "don", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "me", "more", "most",
    "must", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "would", "you", "your", "yours",
    "yourself", "yourselves",
}


class RetrievalExperimentError(RuntimeError):
    """Raised when experiment input or configuration is invalid."""


@dataclass(frozen=True)
class RankedItem:
    chunk_index: int
    score: float


def lexical_tokenize(text: str) -> list[str]:
    """Tokenize English text for BM25 and the local TF-IDF fallback."""

    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text or "")
        if token.lower() not in STOPWORDS
    ]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RetrievalExperimentError(f"Файл не найден: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RetrievalExperimentError(
                    f"Некорректный JSON в {path}, строка {line_number}: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise RetrievalExperimentError(
                    f"Ожидался JSON-объект в {path}, строка {line_number}"
                )
            records.append(value)

    if not records:
        raise RetrievalExperimentError(f"Файл не содержит записей: {path}")
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


class BM25Retriever:
    """Small dependency-free Okapi BM25 implementation with an inverted index."""

    def __init__(
        self,
        texts: Sequence[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not texts:
            raise RetrievalExperimentError("Нельзя построить BM25 по пустому корпусу")
        if k1 <= 0 or not 0 <= b <= 1:
            raise RetrievalExperimentError("Для BM25 нужны k1 > 0 и 0 <= b <= 1")

        self.k1 = k1
        self.b = b
        self.document_count = len(texts)
        self.lengths = np.zeros(self.document_count, dtype=np.float32)
        self.postings: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        document_frequency: collections.Counter[str] = collections.Counter()

        for document_index, text in enumerate(texts):
            counts = collections.Counter(lexical_tokenize(text))
            self.lengths[document_index] = sum(counts.values())
            document_frequency.update(counts.keys())
            for term, frequency in counts.items():
                self.postings[term].append((document_index, frequency))

        self.average_length = float(np.mean(self.lengths)) or 1.0
        self.idf = {
            term: math.log(1 + (self.document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        result = np.zeros(self.document_count, dtype=np.float32)
        query_terms = collections.Counter(lexical_tokenize(query))
        for term, query_frequency in query_terms.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for document_index, term_frequency in self.postings.get(term, ()):
                length_normalizer = self.k1 * (
                    1 - self.b + self.b * self.lengths[document_index] / self.average_length
                )
                term_score = idf * (
                    term_frequency * (self.k1 + 1) / (term_frequency + length_normalizer)
                )
                result[document_index] += float(query_frequency) * term_score
        return result

    def search(self, query: str, top_k: int) -> list[RankedItem]:
        return rank_scores(self.scores(query), top_k)


class TfidfEncoder:
    """Local deterministic encoder used for tests and offline smoke runs."""

    name = "tfidf"

    def __init__(self, max_features: int = 4096) -> None:
        if max_features < 1:
            raise RetrievalExperimentError("max_features должен быть положительным")
        self.max_features = max_features
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}

    def fit(self, texts: Sequence[str]) -> None:
        document_frequency: collections.Counter[str] = collections.Counter()
        term_frequency: collections.Counter[str] = collections.Counter()
        for text in texts:
            tokens = lexical_tokenize(text)
            term_frequency.update(tokens)
            document_frequency.update(set(tokens))

        selected_terms = sorted(
            document_frequency,
            key=lambda term: (-term_frequency[term], term),
        )[: self.max_features]
        self.vocabulary = {term: index for index, term in enumerate(selected_terms)}
        document_count = len(texts)
        self.idf = {
            term: math.log((1 + document_count) / (1 + document_frequency[term])) + 1
            for term in selected_terms
        }
        if not self.vocabulary:
            raise RetrievalExperimentError("Не удалось построить TF-IDF словарь")

    @property
    def dimension(self) -> int:
        return len(self.vocabulary)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self.vocabulary:
            raise RetrievalExperimentError("Сначала вызовите TfidfEncoder.fit")
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for term, frequency in collections.Counter(lexical_tokenize(text)).items():
                column = self.vocabulary.get(term)
                if column is not None:
                    matrix[row, column] = (1 + math.log(frequency)) * self.idf[term]
        return normalize_rows(matrix)


class SentenceTransformerEncoder:
    """Neural sentence embedding adapter loaded only when requested."""

    name = "sentence-transformers"

    def __init__(self, model_name: str, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RetrievalExperimentError(
                "Не установлен sentence-transformers. Выполните "
                "`python -m pip install -r requirements.txt` или используйте "
                "`--embedding-backend tfidf` для локальной проверки."
            ) from error

        kwargs: dict[str, Any] = {}
        if device:
            kwargs["device"] = device
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, **kwargs)
        get_dimension = getattr(
            self.model,
            "get_embedding_dimension",
            self.model.get_sentence_embedding_dimension,
        )
        self._dimension = int(get_dimension())

    def fit(self, texts: Sequence[str]) -> None:
        del texts

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=64,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)


class FaissFlatRetriever:
    """Exact FAISS inner-product search over normalized vectors."""

    def __init__(self, document_vectors: np.ndarray) -> None:
        try:
            import faiss
        except ImportError as error:
            raise RetrievalExperimentError(
                "Не установлен faiss-cpu. Выполните `python -m pip install -r requirements.txt`."
            ) from error
        if document_vectors.ndim != 2 or len(document_vectors) == 0:
            raise RetrievalExperimentError("FAISS нужны непустые двумерные векторы")
        self.faiss = faiss
        self.index = faiss.IndexFlatIP(int(document_vectors.shape[1]))
        self.index.add(np.ascontiguousarray(document_vectors, dtype=np.float32))

    def search(self, query_vectors: np.ndarray, top_k: int) -> list[list[RankedItem]]:
        top_k = min(max(1, top_k), self.index.ntotal)
        scores, indices = self.index.search(
            np.ascontiguousarray(query_vectors, dtype=np.float32),
            top_k,
        )
        rankings: list[list[RankedItem]] = []
        for row_scores, row_indices in zip(scores, indices):
            rankings.append(
                [
                    RankedItem(int(index), float(score))
                    for score, index in zip(row_scores, row_indices)
                    if index >= 0
                ]
            )
        return rankings

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(self.index, str(path))

    def serialized_size(self) -> int:
        return len(self.faiss.serialize_index(self.index))


class CrossEncoderReranker:
    """Optional learned reranker for the hybrid candidate pool."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RetrievalExperimentError(
                "Для reranker нужен sentence-transformers из requirements.txt."
            ) from error
        kwargs: dict[str, Any] = {}
        if device:
            kwargs["device"] = device
        self.model_name = model_name
        self.model = CrossEncoder(model_name, **kwargs)

    def rerank_many(
        self,
        queries: Sequence[str],
        candidates: Sequence[Sequence[RankedItem]],
        chunk_texts: Sequence[str],
        *,
        top_k: int,
        batch_size: int,
    ) -> list[list[RankedItem]]:
        pairs: list[tuple[str, str]] = []
        offsets = [0]
        for query, query_candidates in zip(queries, candidates):
            pairs.extend((query, chunk_texts[item.chunk_index]) for item in query_candidates)
            offsets.append(len(pairs))

        raw_scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=len(pairs) > 100,
        )
        scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
        output: list[list[RankedItem]] = []
        for row, query_candidates in enumerate(candidates):
            start, end = offsets[row], offsets[row + 1]
            scored = [
                RankedItem(item.chunk_index, float(score))
                for item, score in zip(query_candidates, scores[start:end])
            ]
            scored.sort(key=lambda item: (-item.score, item.chunk_index))
            output.append(scored[:top_k])
        return output


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, norms, out=matrix, where=norms > 0)
    return matrix


def rank_scores(scores: np.ndarray, top_k: int) -> list[RankedItem]:
    """Return deterministic descending ranking (lower index breaks ties)."""

    if scores.ndim != 1:
        raise RetrievalExperimentError("Ожидался одномерный массив оценок")
    top_k = min(max(1, top_k), len(scores))
    order = np.lexsort((np.arange(len(scores)), -scores))[:top_k]
    return [RankedItem(int(index), float(scores[index])) for index in order]


def cosine_search(document_vectors: np.ndarray, query_vectors: np.ndarray, top_k: int) -> list[list[RankedItem]]:
    rankings: list[list[RankedItem]] = []
    for query_vector in query_vectors:
        rankings.append(rank_scores(document_vectors @ query_vector, top_k))
    return rankings


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RankedItem]],
    *,
    top_k: int,
    rrf_k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[RankedItem]:
    if not rankings:
        return []
    if rrf_k < 1:
        raise RetrievalExperimentError("rrf_k должен быть положительным")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings) or any(weight < 0 for weight in weights):
        raise RetrievalExperimentError("Некорректные веса RRF")

    scores: collections.defaultdict[int, float] = collections.defaultdict(float)
    for weight, ranking in zip(weights, rankings):
        for rank, item in enumerate(ranking, start=1):
            scores[item.chunk_index] += weight / (rrf_k + rank)
    fused = [RankedItem(index, score) for index, score in scores.items()]
    fused.sort(key=lambda item: (-item.score, item.chunk_index))
    return fused[:top_k]


def validate_inputs(chunks: Sequence[dict[str, Any]], questions: Sequence[dict[str, Any]]) -> None:
    chunk_ids: set[str] = set()
    document_ids: set[str] = set()
    for row, chunk in enumerate(chunks, start=1):
        for field in ("chunk_id", "document_id", "text"):
            if not chunk.get(field):
                raise RetrievalExperimentError(f"У чанка {row} отсутствует поле {field!r}")
        if chunk["chunk_id"] in chunk_ids:
            raise RetrievalExperimentError(f"Повторяющийся chunk_id: {chunk['chunk_id']}")
        chunk_ids.add(chunk["chunk_id"])
        document_ids.add(chunk["document_id"])

    usable = 0
    for question in questions:
        if not question.get("question") or not question.get("gold_document_ids"):
            continue
        usable += 1
        missing = set(question["gold_document_ids"]) - document_ids
        if missing:
            raise RetrievalExperimentError(
                f"Для вопроса {question.get('question_id')} нет gold-документа в корпусе: "
                + ", ".join(sorted(missing))
            )
    if not usable:
        raise RetrievalExperimentError("Нет вопросов с question и gold_document_ids")


def usable_questions(questions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        question
        for question in questions
        if question.get("question") and question.get("gold_document_ids")
    ]


def evaluate_rankings(
    chunks: Sequence[dict[str, Any]],
    questions: Sequence[dict[str, Any]],
    rankings: Sequence[Sequence[RankedItem]],
    *,
    cutoffs: Sequence[int] = (1, 3, 5, 10),
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    questions = usable_questions(questions)
    if len(questions) != len(rankings):
        raise RetrievalExperimentError("Число rankings не совпадает с числом вопросов")

    hits = {cutoff: 0 for cutoff in cutoffs}
    reciprocal_ranks: list[float] = []
    details: list[dict[str, Any]] = []
    for question, ranking in zip(questions, rankings):
        gold = set(question["gold_document_ids"])
        retrieved_document_ids = [chunks[item.chunk_index]["document_id"] for item in ranking]
        first_hit_rank = next(
            (
                rank
                for rank, document_id in enumerate(retrieved_document_ids, start=1)
                if document_id in gold
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if first_hit_rank is None else 1.0 / first_hit_rank)
        for cutoff in cutoffs:
            if gold.intersection(retrieved_document_ids[:cutoff]):
                hits[cutoff] += 1

        details.append(
            {
                "question_id": question.get("question_id"),
                "question": question["question"],
                "gold_document_ids": sorted(gold),
                "first_hit_rank": first_hit_rank,
                "retrieved_chunk_ids": [chunks[item.chunk_index]["chunk_id"] for item in ranking],
                "retrieved_document_ids": retrieved_document_ids,
                "scores": [item.score for item in ranking],
            }
        )

    total = len(questions)
    metrics: dict[str, float | int] = {"questions": total}
    for cutoff in cutoffs:
        metrics[f"recall@{cutoff}"] = hits[cutoff] / total
    metrics["mrr@10"] = float(np.mean(reciprocal_ranks))
    return metrics, details


def corpus_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_encoder(args: argparse.Namespace):
    if args.embedding_backend == "tfidf":
        return TfidfEncoder(max_features=args.tfidf_features)
    return SentenceTransformerEncoder(args.embedding_model, device=args.device)


def evaluate_split(
    *,
    split: str,
    questions: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    chunk_texts: Sequence[str],
    bm25: BM25Retriever,
    encoder: Any,
    document_vectors: np.ndarray,
    faiss_retriever: FaissFlatRetriever,
    reranker: CrossEncoderReranker | None,
    args: argparse.Namespace,
    build_times: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared_questions = usable_questions(questions)
    queries = [question["question"] for question in prepared_questions]
    candidate_k = min(args.candidate_k, len(chunks))
    top_k = min(args.top_k, candidate_k)

    encode_started = time.perf_counter()
    query_vectors = encoder.encode_queries(queries)
    query_encode_ms = (time.perf_counter() - encode_started) * 1000 / len(queries)

    bm25_started = time.perf_counter()
    bm25_rankings = [bm25.search(query, candidate_k) for query in queries]
    bm25_search_ms = (time.perf_counter() - bm25_started) * 1000 / len(queries)

    cosine_started = time.perf_counter()
    cosine_candidates = cosine_search(document_vectors, query_vectors, candidate_k)
    cosine_search_ms = (time.perf_counter() - cosine_started) * 1000 / len(queries) + query_encode_ms

    faiss_started = time.perf_counter()
    faiss_candidates = faiss_retriever.search(query_vectors, candidate_k)
    faiss_search_ms = (time.perf_counter() - faiss_started) * 1000 / len(queries) + query_encode_ms

    fusion_started = time.perf_counter()
    hybrid_candidates = [
        reciprocal_rank_fusion(
            [bm25_ranking, faiss_ranking],
            top_k=candidate_k,
            rrf_k=args.rrf_k,
            weights=[args.bm25_weight, 1.0 - args.bm25_weight],
        )
        for bm25_ranking, faiss_ranking in zip(bm25_rankings, faiss_candidates)
    ]
    fusion_ms = (time.perf_counter() - fusion_started) * 1000 / len(queries)
    hybrid_search_ms = bm25_search_ms + faiss_search_ms + fusion_ms

    method_rankings: dict[str, list[list[RankedItem]]] = {
        "cosine": [ranking[:top_k] for ranking in cosine_candidates],
        "bm25": [ranking[:top_k] for ranking in bm25_rankings],
        "faiss_flat": [ranking[:top_k] for ranking in faiss_candidates],
        "hybrid_rrf": [ranking[:top_k] for ranking in hybrid_candidates],
    }
    method_latency = {
        "cosine": cosine_search_ms,
        "bm25": bm25_search_ms,
        "faiss_flat": faiss_search_ms,
        "hybrid_rrf": hybrid_search_ms,
    }
    method_build = {
        "cosine": build_times["encoder_and_vectors"],
        "bm25": build_times["bm25"],
        "faiss_flat": build_times["encoder_and_vectors"] + build_times["faiss"],
        "hybrid_rrf": build_times["bm25"] + build_times["encoder_and_vectors"] + build_times["faiss"],
    }

    if reranker is not None:
        rerank_started = time.perf_counter()
        reranked = reranker.rerank_many(
            queries,
            [ranking[: args.reranker_candidate_k] for ranking in hybrid_candidates],
            chunk_texts,
            top_k=top_k,
            batch_size=args.reranker_batch_size,
        )
        rerank_ms = (time.perf_counter() - rerank_started) * 1000 / len(queries)
        method_rankings["hybrid_reranked"] = reranked
        method_latency["hybrid_reranked"] = hybrid_search_ms + rerank_ms
        method_build["hybrid_reranked"] = method_build["hybrid_rrf"] + build_times["reranker"]

    records: list[dict[str, Any]] = []
    per_query_records: list[dict[str, Any]] = []
    for method, rankings in method_rankings.items():
        metrics, details = evaluate_rankings(chunks, prepared_questions, rankings)
        records.append(
            {
                "split": split,
                "method": method,
                "embedding_backend": args.embedding_backend,
                "embedding_model": args.embedding_model if args.embedding_backend == "sentence-transformers" else "tfidf",
                "reranker_model": args.reranker_model if method == "hybrid_reranked" else None,
                "chunks": len(chunks),
                "dimension": int(document_vectors.shape[1]),
                "candidate_k": candidate_k,
                "top_k": top_k,
                "build_ms": method_build[method],
                "search_ms_per_query": method_latency[method],
                **metrics,
            }
        )
        for detail in details:
            per_query_records.append({"split": split, "method": method, **detail})
    return records, per_query_records


def selection_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
    """Choose for a top-5 RAG context: Recall@5, then MRR and Recall@10."""

    return (
        float(record["recall@5"]),
        float(record["mrr@10"]),
        float(record["recall@10"]),
        -float(record["search_ms_per_query"]),
    )


def write_results_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    fields = [
        "split", "method", "embedding_backend", "embedding_model", "reranker_model",
        "chunks", "dimension", "candidate_k", "top_k", "questions", "build_ms",
        "search_ms_per_query", "recall@1", "recall@3", "recall@5", "recall@10", "mrr@10",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})


def markdown_table(records: Sequence[dict[str, Any]]) -> str:
    headers = ["Метод", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "мс/запрос"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for record in sorted(records, key=selection_key, reverse=True):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record["method"]),
                    f"{record['recall@1']:.4f}",
                    f"{record['recall@3']:.4f}",
                    f"{record['recall@5']:.4f}",
                    f"{record['recall@10']:.4f}",
                    f"{record['mrr@10']:.4f}",
                    f"{record['search_ms_per_query']:.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_report(
    records: Sequence[dict[str, Any]],
    *,
    args: argparse.Namespace,
    chunks_path: Path,
    validation_path: Path,
    test_path: Path,
) -> str:
    validation = [record for record in records if record["split"] == "validation"]
    test = [record for record in records if record["split"] == "test"]
    best = max(validation, key=selection_key)
    test_best = next((record for record in test if record["method"] == best["method"]), None)
    base_hybrid = next(
        (record for record in validation if record["method"] == "hybrid_rrf"),
        None,
    )
    backend = (
        f"SentenceTransformers `{args.embedding_model}`"
        if args.embedding_backend == "sentence-transformers"
        else f"локальный TF-IDF ({args.tfidf_features} признаков)"
    )
    reranker_note = (
        f"Дополнительно проверен cross-encoder `{args.reranker_model}`."
        if args.use_reranker
        else "Reranker не включался в этот прогон; его можно проверить флагом `--use-reranker`."
    )
    test_summary = (
        f"На отложенном test-наборе выбранный метод получил Recall@5={test_best['recall@5']:.4f}, "
        f"Recall@10={test_best['recall@10']:.4f}, MRR@10={test_best['mrr@10']:.4f}."
        if test_best
        else "Test-набор в этом прогоне не оценивался."
    )
    latency_summary = (
        f"Reranker повысил Recall@5 относительно `hybrid_rrf` с "
        f"{base_hybrid['recall@5']:.4f} до {best['recall@5']:.4f}, но среднее время выросло с "
        f"{base_hybrid['search_ms_per_query']:.3f} до {best['search_ms_per_query']:.3f} мс/запрос. "
        "Для максимального качества следует использовать `hybrid_reranked`; для минимальной "
        "задержки — `hybrid_rrf` без reranker."
        if best["method"] == "hybrid_reranked" and base_hybrid
        else ""
    )

    return f"""# Сравнение методов retrieval

Эксперимент проведён на чанках MultiDoc2Dial DMV. Все методы получают один и тот же корпус и
одни и те же вопросы. Метод для итогового RAG выбирается **только по validation-набору**;
test-набор используется для окончательной проверки после выбора.

## Конфигурация

- чанки: `{chunks_path.relative_to(ROOT)}`;
- validation: `{validation_path.relative_to(ROOT)}`;
- test: `{test_path.relative_to(ROOT)}`;
- векторизация: {backend};
- размер списка кандидатов: {args.candidate_k};
- итоговый top-k: {args.top_k};
- гибрид: weighted RRF, BM25 weight={args.bm25_weight:.2f}, FAISS weight={1 - args.bm25_weight:.2f}, k={args.rrf_k};
- reranker получает первые {args.reranker_candidate_k} кандидатов гибридного поиска;
- {reranker_note}

Здесь Recall@k — доля вопросов, для которых среди первых k чанков найден хотя бы один чанк
из правильного документа. MRR@10 дополнительно учитывает, насколько высоко появился первый
правильный документ.

## Validation

{markdown_table(validation)}

## Test

{markdown_table(test) if test else "Test-набор не оценивался."}

## Выбор для итогового RAG

Основная метрика выбора — Recall@5: генератору обычно передаётся небольшой контекст, поэтому
важно найти правильный документ именно в первой пятёрке. При равенстве используются MRR@10,
Recall@10 и затем скорость.

По этому правилу выбран **`{best['method']}`**: validation Recall@5={best['recall@5']:.4f},
Recall@10={best['recall@10']:.4f}, MRR@10={best['mrr@10']:.4f}.
{test_summary}

{latency_summary}

Прямой cosine и `FAISS IndexFlatIP` используют одинаковые нормализованные векторы. Их качество
должно быть одинаковым или отличаться только на ничьих; различие между ними — способ выполнения
поиска. Для текущего небольшого корпуса Flat-индекс предпочтительнее приближённых IVF/HNSW.

## Воспроизведение

```powershell
python src/rag_pipeline/compare_retrieval_methods.py
```

Быстрая полностью локальная проверка без загрузки нейросетевой модели:

```powershell
python src/rag_pipeline/compare_retrieval_methods.py --embedding-backend tfidf
```

Артефакты эксперимента находятся в `data/experiments/retrieval`, а готовый Flat-индекс —
в `data/current/indexes/retrieval`.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сравнить cosine, BM25, FAISS и гибридный retrieval на DMV-корпусе."
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--validation-questions", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test-questions", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--embedding-backend",
        choices=("sentence-transformers", "tfidf"),
        default="sentence-transformers",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--tfidf-features", type=int, default=4096)
    parser.add_argument("--device", help="Например, cpu, cuda или mps")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=60)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--bm25-weight", type=float, default=0.5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-candidate-k", type=int, default=30)
    parser.add_argument("--reranker-batch-size", type=int, default=64)
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args(argv)

    if args.top_k < 1 or args.candidate_k < args.top_k:
        parser.error("Нужно 1 <= top-k <= candidate-k")
    if not 0 <= args.bm25_weight <= 1:
        parser.error("bm25-weight должен лежать в диапазоне [0, 1]")
    if args.reranker_batch_size < 1:
        parser.error("reranker-batch-size должен быть положительным")
    if not args.top_k <= args.reranker_candidate_k <= args.candidate_k:
        parser.error("Нужно top-k <= reranker-candidate-k <= candidate-k")
    return args


def run_experiment(args: argparse.Namespace) -> list[dict[str, Any]]:
    chunks = read_jsonl(args.chunks)
    validation_questions = read_jsonl(args.validation_questions)
    test_questions = [] if args.skip_test else read_jsonl(args.test_questions)
    validate_inputs(chunks, validation_questions)
    if test_questions:
        validate_inputs(chunks, test_questions)

    chunk_texts = [str(chunk["text"]) for chunk in chunks]
    print(f"Корпус: {len(chunks)} чанков")

    bm25_started = time.perf_counter()
    bm25 = BM25Retriever(chunk_texts, k1=args.bm25_k1, b=args.bm25_b)
    bm25_build_ms = (time.perf_counter() - bm25_started) * 1000

    encoder_started = time.perf_counter()
    encoder = make_encoder(args)
    encoder.fit(chunk_texts)
    document_vectors = encoder.encode_documents(chunk_texts)
    encoder_build_ms = (time.perf_counter() - encoder_started) * 1000

    faiss_started = time.perf_counter()
    faiss_retriever = FaissFlatRetriever(document_vectors)
    faiss_build_ms = (time.perf_counter() - faiss_started) * 1000

    reranker = None
    reranker_build_ms = 0.0
    if args.use_reranker:
        reranker_started = time.perf_counter()
        reranker = CrossEncoderReranker(args.reranker_model, device=args.device)
        reranker_build_ms = (time.perf_counter() - reranker_started) * 1000

    build_times = {
        "bm25": bm25_build_ms,
        "encoder_and_vectors": encoder_build_ms,
        "faiss": faiss_build_ms,
        "reranker": reranker_build_ms,
    }
    print(
        f"Векторы: backend={args.embedding_backend}, dimension={document_vectors.shape[1]}, "
        f"build={encoder_build_ms:.1f} ms"
    )

    records: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    split_inputs = [("validation", validation_questions)]
    if test_questions:
        split_inputs.append(("test", test_questions))

    for split, questions in split_inputs:
        print(f"Оценка: {split} ({len(usable_questions(questions))} вопросов)")
        split_records, split_details = evaluate_split(
            split=split,
            questions=questions,
            chunks=chunks,
            chunk_texts=chunk_texts,
            bm25=bm25,
            encoder=encoder,
            document_vectors=document_vectors,
            faiss_retriever=faiss_retriever,
            reranker=reranker,
            args=args,
            build_times=build_times,
        )
        records.extend(split_records)
        per_query.extend(split_details)
        for record in sorted(split_records, key=selection_key, reverse=True):
            print(
                f"  {record['method']:<18} "
                f"R@5={record['recall@5']:.4f} R@10={record['recall@10']:.4f} "
                f"MRR={record['mrr@10']:.4f} {record['search_ms_per_query']:.3f} ms/q"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_results_csv(args.output_dir / "retrieval_comparison.csv", records)
    write_jsonl(args.output_dir / "retrieval_comparison.jsonl", records)
    write_jsonl(args.output_dir / "retrieval_per_query.jsonl", per_query)

    index_path = args.index_dir / "faiss_flat.index"
    embeddings_path = args.index_dir / "chunk_embeddings.npy"
    faiss_retriever.save(index_path)
    args.index_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, document_vectors, allow_pickle=False)
    validation_records = [record for record in records if record["split"] == "validation"]
    best = max(validation_records, key=selection_key)
    write_json(
        args.index_dir / "index_metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "chunks_path": str(args.chunks.relative_to(ROOT) if args.chunks.is_relative_to(ROOT) else args.chunks),
            "chunks_sha256": corpus_sha256(args.chunks),
            "chunks": len(chunks),
            "embedding_backend": args.embedding_backend,
            "embedding_model": args.embedding_model if args.embedding_backend == "sentence-transformers" else "tfidf",
            "dimension": int(document_vectors.shape[1]),
            "faiss_index": str(index_path.relative_to(ROOT) if index_path.is_relative_to(ROOT) else index_path),
            "faiss_index_size_bytes": faiss_retriever.serialized_size(),
            "selected_method": best["method"],
            "selection_rule": "validation recall@5, then mrr@10, recall@10, latency",
            "bm25": {"k1": args.bm25_k1, "b": args.bm25_b},
            "hybrid": {"rrf_k": args.rrf_k, "bm25_weight": args.bm25_weight},
            "reranker_model": args.reranker_model if args.use_reranker else None,
            "reranker_candidate_k": args.reranker_candidate_k if args.use_reranker else None,
        },
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        build_report(
            records,
            args=args,
            chunks_path=args.chunks,
            validation_path=args.validation_questions,
            test_path=args.test_questions,
        ),
        encoding="utf-8",
    )
    print(f"Отчёт: {args.report}")
    return records


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_experiment(parse_args(argv))
    except (RetrievalExperimentError, OSError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
