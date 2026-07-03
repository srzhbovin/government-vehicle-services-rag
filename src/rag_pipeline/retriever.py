"""Runtime hybrid retriever backed by BM25, FAISS and an optional reranker."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .compare_retrieval_methods import (
    BM25Retriever,
    CrossEncoderReranker,
    RankedItem,
    read_jsonl,
    reciprocal_rank_fusion,
)
from .settings import Settings


class RetrieverError(RuntimeError):
    """Raised when retrieval assets cannot be loaded or queried."""


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    chunk_index: int
    chunk_id: str
    document_id: str
    title: str | None
    text: str
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    method: str
    elapsed_ms: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class HybridRetriever:
    """Hybrid BM25 + exact FAISS retrieval with optional cross-encoder reranking."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chunks: list[dict[str, Any]] = []
        self.bm25: BM25Retriever | None = None
        self.encoder: Any = None
        self.faiss_index: Any = None
        self.reranker: CrossEncoderReranker | None = None
        self._ready = False
        self._inference_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def method_name(self) -> str:
        return "hybrid_reranked" if self.settings.use_reranker else "hybrid_rrf"

    def initialize(self) -> None:
        if self._ready:
            return
        for path in (
            self.settings.chunks_path,
            self.settings.index_path,
            self.settings.embeddings_path,
            self.settings.index_metadata_path,
        ):
            if not path.is_file():
                raise RetrieverError(
                    f"Retrieval asset is missing: {path}. Run `python -m rag_pipeline.build_rag_index`."
                )

        metadata = json.loads(
            self.settings.index_metadata_path.read_text(encoding="utf-8")
        )
        expected_hash = metadata.get("chunks_sha256")
        actual_hash = file_sha256(self.settings.chunks_path)
        if expected_hash and expected_hash != actual_hash:
            raise RetrieverError(
                "The FAISS index was built for another chunks file. Rebuild the RAG index."
            )
        indexed_model = metadata.get("embedding_model")
        if indexed_model and indexed_model != self.settings.embedding_model:
            raise RetrieverError(
                "RAG_EMBEDDING_MODEL does not match index metadata. Rebuild the index or restore the model setting."
            )

        self.chunks = read_jsonl(self.settings.chunks_path)
        chunk_texts = [str(chunk["text"]) for chunk in self.chunks]
        self.bm25 = BM25Retriever(chunk_texts)

        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RetrieverError(
                "Retrieval dependencies are not installed. Run `python -m pip install -r requirements.txt`."
            ) from error

        self.faiss_index = faiss.read_index(str(self.settings.index_path))
        if int(self.faiss_index.ntotal) != len(self.chunks):
            raise RetrieverError("FAISS index size does not match the chunks file")

        encoder_kwargs: dict[str, Any] = {}
        if self.settings.device:
            encoder_kwargs["device"] = self.settings.device
        self.encoder = SentenceTransformer(
            self.settings.embedding_model,
            **encoder_kwargs,
        )
        if self.settings.use_reranker:
            self.reranker = CrossEncoderReranker(
                self.settings.reranker_model,
                device=self.settings.device,
            )
        self._ready = True

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        if not self._ready:
            self.initialize()
        assert self.bm25 is not None

        requested_k = top_k or self.settings.top_k
        if not 1 <= requested_k <= 10:
            raise RetrieverError("top_k must be between 1 and 10")
        candidate_k = min(self.settings.candidate_k, len(self.chunks))

        started = time.perf_counter()
        with self._inference_lock:
            query_vector = self.encoder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
            faiss_scores, faiss_indices = self.faiss_index.search(
                query_vector,
                candidate_k,
            )
            faiss_ranking = [
                RankedItem(int(index), float(score))
                for score, index in zip(faiss_scores[0], faiss_indices[0])
                if index >= 0
            ]
            bm25_ranking = self.bm25.search(query, candidate_k)
            fused = reciprocal_rank_fusion(
                [bm25_ranking, faiss_ranking],
                top_k=candidate_k,
                rrf_k=self.settings.rrf_k,
                weights=[
                    self.settings.bm25_weight,
                    1.0 - self.settings.bm25_weight,
                ],
            )

            ranking = fused
            if self.reranker is not None:
                ranking = self.reranker.rerank_many(
                    [query],
                    [fused[: self.settings.reranker_candidate_k]],
                    [str(chunk["text"]) for chunk in self.chunks],
                    top_k=max(requested_k, self.settings.top_k),
                    batch_size=self.settings.reranker_candidate_k,
                )[0]

        elapsed_ms = (time.perf_counter() - started) * 1000
        output = []
        for rank, item in enumerate(ranking[:requested_k], start=1):
            chunk = self.chunks[item.chunk_index]
            output.append(
                RetrievedChunk(
                    rank=rank,
                    chunk_index=item.chunk_index,
                    chunk_id=str(chunk["chunk_id"]),
                    document_id=str(chunk["document_id"]),
                    title=chunk.get("title"),
                    text=str(chunk["text"]),
                    score=float(item.score),
                )
            )
        return RetrievalResult(
            chunks=output,
            method=self.method_name,
            elapsed_ms=elapsed_ms,
        )

