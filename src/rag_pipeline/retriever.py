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
    source_url: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    method: str
    elapsed_ms: float


def chunk_search_text(chunk: dict[str, Any]) -> str:
    """Text used for embeddings and lexical search."""

    title = str(chunk.get("title") or chunk.get("document_id") or "").strip()
    text = str(chunk.get("text") or "").strip()
    if not title:
        return text
    return f"{title}\n{title}\n{text}"


def chunk_rerank_text(chunk: dict[str, Any]) -> str:
    title = str(chunk.get("title") or chunk.get("document_id") or "").strip()
    text = str(chunk.get("text") or "").strip()
    return f"Title: {title}\nText: {text}" if title else text


def expand_query(query: str) -> str:
    """Small DMV-specific query expansion for common ambiguous requests."""

    normalized = query.lower()
    additions: list[str] = []
    lost_terms = ("lost", "stolen", "destroyed", "missing")

    if any(term in normalized for term in lost_terms):
        if "license" in normalized or "permit" in normalized:
            additions.extend(
                [
                    "replace license or permit",
                    "duplicate driver license",
                    "replace online",
                    "replace by mail",
                    "DMV office",
                    "$17.50 fee",
                ]
            )
        if "plate" in normalized or "plates" in normalized:
            additions.extend(
                [
                    "lost stolen destroyed plates",
                    "surrender registration",
                    "police report",
                    "MV-78B",
                ]
            )

    if "change" in normalized and "address" in normalized:
        additions.extend(["change address", "within 10 days", "license permit registration"])

    if not additions:
        return query
    return query + "\n" + " ".join(dict.fromkeys(additions))


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
        chunk_texts = [chunk_search_text(chunk) for chunk in self.chunks]
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
        self.encoder = self._load_sentence_transformer(
            SentenceTransformer,
            self.settings.embedding_model,
            encoder_kwargs,
        )
        if self.settings.use_reranker:
            self.reranker = CrossEncoderReranker(
                self.settings.reranker_model,
                device=self.settings.device,
            )
        self._ready = True

    def _load_sentence_transformer(
        self,
        sentence_transformer_cls: Any,
        model_name: str,
        kwargs: dict[str, Any],
    ) -> Any:
        try:
            return sentence_transformer_cls(
                model_name,
                **kwargs,
                local_files_only=True,
            )
        except TypeError:
            return sentence_transformer_cls(model_name, **kwargs)
        except Exception as local_error:
            try:
                return sentence_transformer_cls(model_name, **kwargs)
            except Exception as error:
                raise RetrieverError(
                    f"Cannot load embedding model `{model_name}`. "
                    "Check the local model cache or internet connection."
                ) from error

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        context_window: int | None = None,
        intro_chunks: int | None = None,
        max_context_chunks: int | None = None,
    ) -> RetrievalResult:
        if not self._ready:
            self.initialize()
        assert self.bm25 is not None

        requested_k = top_k or self.settings.top_k
        if not 1 <= requested_k <= 10:
            raise RetrieverError("top_k must be between 1 and 10")
        selected_window = (
            self.settings.context_window
            if context_window is None
            else context_window
        )
        selected_intro = (
            self.settings.context_intro_chunks
            if intro_chunks is None
            else intro_chunks
        )
        selected_max_context = (
            self.settings.max_context_chunks
            if max_context_chunks is None
            else max_context_chunks
        )
        if selected_window < 0:
            raise RetrieverError("context_window cannot be negative")
        if selected_intro < 0:
            raise RetrieverError("intro_chunks cannot be negative")
        if selected_max_context < requested_k:
            raise RetrieverError("max_context_chunks must be greater than or equal to top_k")
        candidate_k = min(self.settings.candidate_k, len(self.chunks))

        started = time.perf_counter()
        with self._inference_lock:
            search_query = expand_query(query) if self.settings.use_query_expansion else query
            query_vector = self.encoder.encode(
                [search_query],
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
            bm25_ranking = self.bm25.search(search_query, candidate_k)
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
                    [search_query],
                    [fused[: self.settings.reranker_candidate_k]],
                    [chunk_rerank_text(chunk) for chunk in self.chunks],
                    top_k=max(requested_k, self.settings.top_k),
                    batch_size=self.settings.reranker_candidate_k,
                )[0]
            ranking = self._expand_context(
                ranking,
                requested_k,
                context_window=selected_window,
                intro_chunks=selected_intro,
                max_context_chunks=selected_max_context,
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        output = []
        for rank, item in enumerate(ranking, start=1):
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
                    source_url=(
                        str(
                            chunk.get("source_url")
                            or chunk.get("url")
                            or chunk.get("link")
                            or ""
                        ).strip()
                        or None
                    ),
                )
            )
        return RetrievalResult(
            chunks=output,
            method=self.method_name,
            elapsed_ms=elapsed_ms,
        )

    def _expand_context(
        self,
        ranking: list[RankedItem],
        requested_k: int,
        *,
        context_window: int,
        intro_chunks: int,
        max_context_chunks: int,
    ) -> list[RankedItem]:
        if not ranking:
            return []
        if context_window == 0 and intro_chunks == 0:
            return ranking[:requested_k]
        limit = max(requested_k, max_context_chunks)

        by_document: dict[str, list[int]] = {}
        for global_index, chunk in enumerate(self.chunks):
            document_id = str(chunk.get("document_id"))
            by_document.setdefault(document_id, []).append(global_index)
        for indices in by_document.values():
            indices.sort(key=lambda index: int(self.chunks[index].get("chunk_index") or 0))

        selected: list[RankedItem] = []
        seen: set[int] = set()

        def add(chunk_index: int, score: float) -> None:
            if chunk_index in seen or len(selected) >= limit:
                return
            seen.add(chunk_index)
            selected.append(RankedItem(chunk_index, score))

        for item in ranking[:requested_k]:
            chunk = self.chunks[item.chunk_index]
            document_id = str(chunk.get("document_id"))
            local_index = int(chunk.get("chunk_index") or 0)
            document_indices = by_document.get(document_id, [])

            intro_indices = document_indices[:intro_chunks]
            neighbor_indices = [
                index
                for index in document_indices
                if abs(int(self.chunks[index].get("chunk_index") or 0) - local_index)
                <= context_window
            ]
            context_indices = sorted(
                {*intro_indices, *neighbor_indices, item.chunk_index},
                key=lambda index: int(self.chunks[index].get("chunk_index") or 0),
            )
            for index in context_indices:
                add(index, item.score)
            if len(selected) >= limit:
                break

        for item in ranking:
            add(item.chunk_index, item.score)
            if len(selected) >= limit:
                break
        return selected
