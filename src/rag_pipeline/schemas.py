"""Pydantic contracts shared by the RAG core, API and user interface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)
    context_window: int | None = Field(default=None, ge=0, le=5)
    intro_chunks: int | None = Field(default=None, ge=0, le=10)
    max_context_chunks: int | None = Field(default=None, ge=1, le=30)
    language: Literal["auto", "ru", "en"] = "auto"
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_output_tokens: int | None = Field(default=None, ge=64, le=2000)
    use_judge: bool | None = None
    use_adaptive_context: bool | None = None

    @model_validator(mode="after")
    def validate_context_size(self) -> "AskRequest":
        if (
            self.top_k is not None
            and self.max_context_chunks is not None
            and self.max_context_chunks < self.top_k
        ):
            raise ValueError("max_context_chunks must be greater than or equal to top_k")
        return self


class ChatRequest(AskRequest):
    question: str = Field(default="chat", min_length=1, max_length=2000)
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_messages(self) -> "ChatRequest":
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("chat must contain at least one user message")
        if self.messages[-1].role != "user":
            raise ValueError("last chat message must be from the user")
        self.question = self.messages[-1].content
        return self


class SourceItem(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    title: str | None = None
    score: float
    text: str
    source_url: str | None = None


class RetrievalResponse(BaseModel):
    question: str
    retrieval_method: str
    top_k: int
    context_window: int
    intro_chunks: int
    max_context_chunks: int
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


class ContextParameters(BaseModel):
    top_k: int
    context_window: int
    intro_chunks: int
    max_context_chunks: int


class AdaptiveContextInfo(BaseModel):
    enabled: bool
    triggered: bool = False
    used_retry_answer: bool = False
    reason: str | None = None
    initial: ContextParameters | None = None
    retry: ContextParameters | None = None


class RefusalInfo(BaseModel):
    enabled: bool
    refused: bool = False
    strategy: str | None = None
    reason: str | None = None
    top_score: float | None = None
    min_top_score: float | None = None
    lexical_overlap: float | None = None
    min_lexical_overlap: float | None = None


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
    context_window: int
    intro_chunks: int
    max_context_chunks: int
    confidence: float | None = None
    source: str | None = None
    structured_parse_success: bool | None = None
    schema_valid: bool | None = None
    guardrail: GuardrailInfo
    adaptive_context: AdaptiveContextInfo
    refusal: RefusalInfo
    sources: list[SourceItem]
    usage: GenerationUsage
    timings: TimingInfo


class ChatResponse(AskResponse):
    standalone_question: str
    history_messages: int


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
    adaptive_context_enabled: bool
    adaptive_top_k: int
    adaptive_context_window: int
    adaptive_context_intro_chunks: int
    adaptive_max_context_chunks: int
    refusal_gate_enabled: bool
    refusal_min_top_score: float
    refusal_min_lexical_overlap: float
    judge_enabled: bool
    judge_min_score: float
