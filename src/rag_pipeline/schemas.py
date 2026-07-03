"""Pydantic contracts shared by the RAG core, API and user interface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)
    language: Literal["auto", "ru", "en"] = "auto"


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
    total_ms: float


class AskResponse(BaseModel):
    question: str
    answer: str
    retrieval_method: str
    generation_model: str
    answer_language: str
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
