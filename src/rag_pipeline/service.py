"""Application service combining retrieval and grounded answer generation."""

from __future__ import annotations

import time
from typing import Protocol

from .generator import GenerationError, GenerationResult, GuardrailResult, build_generator
from .retriever import HybridRetriever, RetrievalResult, RetrievedChunk
from .schemas import (
    AdaptiveContextInfo,
    AskResponse,
    ContextParameters,
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

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        context_window: int | None = None,
        intro_chunks: int | None = None,
        max_context_chunks: int | None = None,
    ) -> RetrievalResult: ...


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

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        context_window: int | None = None,
        intro_chunks: int | None = None,
        max_context_chunks: int | None = None,
    ) -> RetrievalResponse:
        requested_top_k = top_k or self.settings.top_k
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
        result = self.retriever.retrieve(
            question,
            top_k=requested_top_k,
            context_window=selected_window,
            intro_chunks=selected_intro,
            max_context_chunks=selected_max_context,
        )
        return RetrievalResponse(
            question=question,
            retrieval_method=result.method,
            top_k=requested_top_k,
            context_window=selected_window,
            intro_chunks=selected_intro,
            max_context_chunks=selected_max_context,
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
        context_window: int | None = None,
        intro_chunks: int | None = None,
        max_context_chunks: int | None = None,
        use_adaptive_context: bool | None = None,
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
        should_judge = self.settings.enable_judge if use_judge is None else use_judge
        should_adapt_context = (
            self.settings.enable_adaptive_context
            if use_adaptive_context is None
            else use_adaptive_context
        )
        initial_context = ContextParameters(
            top_k=requested_top_k,
            context_window=selected_window,
            intro_chunks=selected_intro,
            max_context_chunks=selected_max_context,
        )
        active_context = initial_context
        adaptive_context = AdaptiveContextInfo(
            enabled=bool(should_judge and should_adapt_context),
            initial=initial_context,
        )

        retrieval = self.retriever.retrieve(
            question,
            top_k=requested_top_k,
            context_window=selected_window,
            intro_chunks=selected_intro,
            max_context_chunks=selected_max_context,
        )
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
        generation_ms = generation.elapsed_ms
        retrieval_ms = retrieval.elapsed_ms

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
                if (
                    should_adapt_context
                    and judgment.checked
                    and judgment.accepted is False
                ):
                    retry_context = self._adaptive_context_parameters(initial_context)
                    adaptive_context = AdaptiveContextInfo(
                        enabled=True,
                        triggered=retry_context != initial_context,
                        reason=_adaptive_reason(judgment),
                        initial=initial_context,
                        retry=retry_context,
                    )
                    if retry_context != initial_context:
                        try:
                            retry_retrieval = self.retriever.retrieve(
                                question,
                                top_k=retry_context.top_k,
                                context_window=retry_context.context_window,
                                intro_chunks=retry_context.intro_chunks,
                                max_context_chunks=retry_context.max_context_chunks,
                            )
                            retry_generation = self.generator.generate(
                                question,
                                retry_retrieval.chunks,
                                language,
                                temperature=selected_temperature,
                                top_p=selected_top_p,
                                max_output_tokens=selected_max_tokens,
                            )
                            retrieval_ms += retry_retrieval.elapsed_ms
                            generation_ms += retry_generation.elapsed_ms
                            retry_judgment = self.generator.judge_answer(
                                question,
                                retry_retrieval.chunks,
                                retry_generation.answer,
                                language,
                            )
                        except GenerationError as error:
                            adaptive_context.reason = (
                                f"{adaptive_context.reason}; adaptive retry failed: {error}"
                            )
                        else:
                            if retry_judgment.elapsed_ms is not None:
                                judge_ms = (judge_ms or 0.0) + round(
                                    retry_judgment.elapsed_ms,
                                    3,
                                )
                            retry_corrected = bool(
                                not retry_judgment.accepted
                                and retry_judgment.corrected_answer
                            )
                            if retry_judgment.accepted or retry_corrected:
                                retrieval = retry_retrieval
                                generation = retry_generation
                                active_context = retry_context
                                adaptive_context.used_retry_answer = True
                                final_answer = retry_generation.answer
                                original_answer = None
                                source = retry_generation.source
                                if retry_corrected and retry_judgment.corrected_answer:
                                    original_answer = retry_generation.answer
                                    final_answer = retry_judgment.corrected_answer
                                    source = retry_judgment.source or source
                                guardrail = GuardrailInfo(
                                    enabled=True,
                                    checked=retry_judgment.checked,
                                    accepted=retry_judgment.accepted,
                                    verdict=retry_judgment.verdict,
                                    score=retry_judgment.score,
                                    reason=retry_judgment.reason,
                                    corrected=retry_corrected,
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
            top_k=active_context.top_k,
            context_window=active_context.context_window,
            intro_chunks=active_context.intro_chunks,
            max_context_chunks=active_context.max_context_chunks,
            confidence=generation.confidence,
            source=source,
            structured_parse_success=generation.parse_success,
            schema_valid=generation.schema_valid,
            guardrail=guardrail,
            adaptive_context=adaptive_context,
            sources=_source_items(retrieval),
            usage=generation.usage,
            timings=TimingInfo(
                retrieval_ms=round(retrieval_ms, 3),
                generation_ms=round(generation_ms, 3),
                judge_ms=judge_ms,
                total_ms=round(total_ms, 3),
            ),
        )

    def _adaptive_context_parameters(
        self,
        current: ContextParameters,
    ) -> ContextParameters:
        return ContextParameters(
            top_k=max(current.top_k, self.settings.adaptive_top_k),
            context_window=max(
                current.context_window,
                self.settings.adaptive_context_window,
            ),
            intro_chunks=max(
                current.intro_chunks,
                self.settings.adaptive_context_intro_chunks,
            ),
            max_context_chunks=max(
                current.max_context_chunks,
                self.settings.adaptive_max_context_chunks,
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


def _adaptive_reason(judgment: GuardrailResult) -> str:
    verdict = judgment.verdict or "unknown"
    score = "n/a" if judgment.score is None else f"{judgment.score:.2f}"
    reason = judgment.reason or "judge rejected the first answer"
    return f"first judge verdict={verdict}, score={score}; {reason}"
