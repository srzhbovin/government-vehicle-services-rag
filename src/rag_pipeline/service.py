"""Application service combining retrieval and grounded answer generation."""

from __future__ import annotations

import time
from typing import Protocol

from .generator import GenerationResult, build_generator
from .retriever import HybridRetriever, RetrievalResult, RetrievedChunk
from .schemas import (
    AskResponse,
    RetrievalResponse,
    SourceItem,
    TimingInfo,
)
from .settings import Settings


class RetrieverProtocol(Protocol):
    ready: bool
    method_name: str

    def initialize(self) -> None: ...

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult: ...


class GeneratorProtocol(Protocol):
    configured: bool

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        language: str = "auto",
    ) -> GenerationResult: ...


class RAGService:
    def __init__(
        self,
        settings: Settings,
        retriever: RetrieverProtocol | None = None,
        generator: GeneratorProtocol | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever or HybridRetriever(settings)
        self.generator = generator or build_generator(settings)

    @property
    def ready(self) -> bool:
        return self.retriever.ready

    @property
    def generator_configured(self) -> bool:
        return self.generator.configured

    @property
    def retrieval_method(self) -> str:
        return self.retriever.method_name

    def initialize(self) -> None:
        self.retriever.initialize()

    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResponse:
        result = self.retriever.retrieve(question, top_k=top_k)
        return RetrievalResponse(
            question=question,
            retrieval_method=result.method,
            sources=_source_items(result),
            retrieval_ms=round(result.elapsed_ms, 3),
        )

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        language: str = "auto",
    ) -> AskResponse:
        total_started = time.perf_counter()
        retrieval = self.retriever.retrieve(question, top_k=top_k)
        generation = self.generator.generate(question, retrieval.chunks, language)
        total_ms = (time.perf_counter() - total_started) * 1000
        return AskResponse(
            question=question,
            answer=generation.answer,
            retrieval_method=retrieval.method,
            generation_model=generation.model,
            answer_language=language,
            sources=_source_items(retrieval),
            usage=generation.usage,
            timings=TimingInfo(
                retrieval_ms=round(retrieval.elapsed_ms, 3),
                generation_ms=round(generation.elapsed_ms, 3),
                total_ms=round(total_ms, 3),
            ),
        )


def _source_items(result: RetrievalResult) -> list[SourceItem]:
    return [
        SourceItem(
            rank=chunk.rank,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            title=chunk.title,
            score=chunk.score,
            text=chunk.text,
        )
        for chunk in result.chunks
    ]
