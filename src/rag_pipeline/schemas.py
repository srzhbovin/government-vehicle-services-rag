"""Pydantic contracts shared by the RAG core, API and user interface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)
    language: Literal["auto", "ru", "en"] = "auto"
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_output_tokens: int | None = Field(default=None, ge=64, le=2000)
    use_judge: bool | None = None


class SourceItem(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    title: str | None = None
    score: float
    text: str


class RetrievalResponse(BaseModel):
    question: str
    retrieval_method: str
    sources: list[SourceItem]
    retrieval_ms: float


class GenerationUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class TimingInfo(BaseModel):
    retrieval_ms: float
    generation_ms: float
    judge_ms: float | None = None
    total_ms: float


class GuardrailInfo(BaseModel):
    enabled: bool
    checked: bool = False
    accepted: bool | None = None
    verdict: str | None = None
    score: float | None = None
    reason: str | None = None
    corrected: bool = False


class AskResponse(BaseModel):
    question: str
    answer: str
    original_answer: str | None = None
    retrieval_method: str
    generation_model: str
    prompt_strategy: str
    answer_language: str
    generation_temperature: float
    generation_top_p: float
    top_k: int
    confidence: float | None = None
    source: str | None = None
    structured_parse_success: bool | None = None
    schema_valid: bool | None = None
    guardrail: GuardrailInfo
    sources: list[SourceItem]
    usage: GenerationUsage
    timings: TimingInfo


class HealthResponse(BaseModel):
    status: str
    retrieval_ready: bool
    generator_configured: bool
    retrieval_method: str
    embedding_model: str
    llm_provider: str
    generation_model: str
    prompt_strategy: str
    generation_temperature: float
    generation_top_p: float
    top_k: int
    context_window: int
    context_intro_chunks: int
    max_context_chunks: int
    judge_enabled: bool
    judge_min_score: float
