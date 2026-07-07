"""Application service combining retrieval and grounded answer generation."""

from __future__ import annotations

import time
from typing import Protocol

from .generator import GenerationError, GenerationResult, GuardrailResult, build_generator
from .retriever import HybridRetriever, RetrievalResult, RetrievedChunk
from .schemas import (
    AskResponse,
    GuardrailInfo,
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
        strategy: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
    ) -> GenerationResult: ...

    def judge_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        answer: str,
        language: str = "auto",
    ) -> GuardrailResult: ...


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
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        use_judge: bool | None = None,
    ) -> AskResponse:
        total_started = time.perf_counter()
        requested_top_k = top_k or self.settings.top_k
        selected_temperature = (
            self.settings.generation_temperature
            if temperature is None
            else temperature
        )
        selected_top_p = self.settings.generation_top_p if top_p is None else top_p
        selected_max_tokens = (
            self.settings.generation_max_tokens
            if max_output_tokens is None
            else max_output_tokens
        )
        should_judge = self.settings.enable_judge if use_judge is None else use_judge

        retrieval = self.retriever.retrieve(question, top_k=requested_top_k)
        generation = self.generator.generate(
            question,
            retrieval.chunks,
            language,
            temperature=selected_temperature,
            top_p=selected_top_p,
            max_output_tokens=selected_max_tokens,
        )
        final_answer = generation.answer
        original_answer = None
        source = generation.source
        judge_ms: float | None = None
        guardrail = GuardrailInfo(enabled=should_judge)

        if should_judge:
            try:
                judgment = self.generator.judge_answer(
                    question,
                    retrieval.chunks,
                    generation.answer,
                    language,
                )
            except GenerationError as error:
                guardrail = GuardrailInfo(
                    enabled=True,
                    checked=False,
                    accepted=None,
                    verdict="judge_error",
                    reason=str(error),
                )
            else:
                judge_ms = (
                    round(judgment.elapsed_ms, 3)
                    if judgment.elapsed_ms is not None
                    else None
                )
                corrected = bool(
                    not judgment.accepted and judgment.corrected_answer
                )
                if corrected and judgment.corrected_answer:
                    original_answer = generation.answer
                    final_answer = judgment.corrected_answer
                    source = judgment.source or source
                guardrail = GuardrailInfo(
                    enabled=True,
                    checked=judgment.checked,
                    accepted=judgment.accepted,
                    verdict=judgment.verdict,
                    score=judgment.score,
                    reason=judgment.reason,
                    corrected=corrected,
                )

        total_ms = (time.perf_counter() - total_started) * 1000
        return AskResponse(
            question=question,
            answer=final_answer,
            original_answer=original_answer,
            retrieval_method=retrieval.method,
            generation_model=generation.model,
            prompt_strategy=generation.prompt_strategy,
            answer_language=language,
            generation_temperature=selected_temperature,
            generation_top_p=selected_top_p,
            top_k=requested_top_k,
            confidence=generation.confidence,
            source=source,
            structured_parse_success=generation.parse_success,
            schema_valid=generation.schema_valid,
            guardrail=guardrail,
            sources=_source_items(retrieval),
            usage=generation.usage,
            timings=TimingInfo(
                retrieval_ms=round(retrieval.elapsed_ms, 3),
                generation_ms=round(generation.elapsed_ms, 3),
                judge_ms=judge_ms,
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
