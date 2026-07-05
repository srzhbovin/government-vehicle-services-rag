"""Prompt strategies and structured RAG answer parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class PromptStrategy(StrEnum):
    PLAIN = "plain"
    JSON = "json"
    PYDANTIC = "pydantic"
    STRUCTURED_OUTPUT = "structured_output"


class StructuredRAGAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        min_length=1,
        description="Concise grounded answer with fragment citations such as [1].",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence that the supplied fragments fully support the answer. "
            "Use 1 only for direct and complete support; lower it for missing or conflicting facts."
        ),
    )
    source: str = Field(
        min_length=1,
        description="Comma-separated supporting fragment markers, for example [1], [2].",
    )


@dataclass(frozen=True)
class PromptSpec:
    strategy: PromptStrategy
    instructions: str
    response_format: dict[str, Any] | None = None


@dataclass(frozen=True)
class ParsedModelAnswer:
    answer: str
    confidence: float | None
    source: str | None
    parse_success: bool | None
    schema_valid: bool | None
    parse_error: str | None = None


GROUNDING_INSTRUCTIONS = """You are an assistant for New York State DMV services.
Answer only from the supplied document fragments. Do not use outside knowledge and do not invent
requirements, dates, fees, addresses, or links. If the fragments are insufficient, state that
clearly. Present the general rule before exceptions. Never turn a conditional statement into a
universal rule. Answer in the requested language, keep the answer concise and practical, and cite
supporting fragments with markers such as [1] and [2]. Only the markers immediately before each
fragment Title are valid citations. Ignore bracketed
numbers inside fragment text because they are links from the original document, not fragment IDs."""


def prompt_spec(strategy: PromptStrategy | str) -> PromptSpec:
    selected = PromptStrategy(strategy)
    if selected is PromptStrategy.PLAIN:
        return PromptSpec(
            strategy=selected,
            instructions=(
                GROUNDING_INSTRUCTIONS
                + "\nReturn a direct plain-text answer with citations."
            ),
        )

    if selected is PromptStrategy.JSON:
        return PromptSpec(
            strategy=selected,
            instructions=(
                GROUNDING_INSTRUCTIONS
                + "\nReturn only one valid JSON object with exactly three fields: "
                "answer as a string with citations, confidence as a number from 0 to 1, and "
                "source as a string containing supporting fragment markers. Use confidence 1 "
                "only when the context directly and completely supports the answer. Do not use "
                "Markdown fences."
            ),
        )

    schema = StructuredRAGAnswer.model_json_schema()
    if selected is PromptStrategy.PYDANTIC:
        return PromptSpec(
            strategy=selected,
            instructions=(
                GROUNDING_INSTRUCTIONS
                + "\nReturn only JSON that validates against this Pydantic JSON Schema. "
                "Do not use Markdown fences.\n"
                + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            ),
        )

    return PromptSpec(
        strategy=selected,
        instructions=GROUNDING_INSTRUCTIONS,
        response_format={
            "type": "json_schema",
            "name": "rag_answer",
            "description": "Grounded DMV answer, confidence and supporting fragment markers.",
            "schema": schema,
            "strict": True,
        },
    )


def parse_model_answer(
    raw_output: str,
    strategy: PromptStrategy | str,
) -> ParsedModelAnswer:
    selected = PromptStrategy(strategy)
    raw = raw_output.strip()
    if selected is PromptStrategy.PLAIN:
        citations = sorted({int(value) for value in re.findall(r"\[(\d+)\]", raw)})
        source = ", ".join(f"[{value}]" for value in citations) or None
        return ParsedModelAnswer(
            answer=raw,
            confidence=None,
            source=source,
            parse_success=None,
            schema_valid=None,
        )

    try:
        payload = json.loads(_strip_markdown_fence(raw))
    except (json.JSONDecodeError, TypeError) as error:
        return ParsedModelAnswer(
            answer=raw,
            confidence=None,
            source=None,
            parse_success=False,
            schema_valid=False,
            parse_error=f"invalid_json: {error}",
        )

    if not isinstance(payload, dict):
        return ParsedModelAnswer(
            answer=raw,
            confidence=None,
            source=None,
            parse_success=True,
            schema_valid=False,
            parse_error="top_level_json_is_not_an_object",
        )

    try:
        validated = StructuredRAGAnswer.model_validate(payload)
    except ValidationError as error:
        answer = payload.get("answer")
        return ParsedModelAnswer(
            answer=str(answer).strip() if answer else raw,
            confidence=_optional_float(payload.get("confidence")),
            source=_optional_string(payload.get("source")),
            parse_success=True,
            schema_valid=False,
            parse_error=f"schema_validation: {error.errors(include_url=False)}",
        )

    return ParsedModelAnswer(
        answer=validated.answer.strip(),
        confidence=validated.confidence,
        source=validated.source.strip(),
        parse_success=True,
        schema_valid=True,
    )


def _strip_markdown_fence(value: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL)
    return match.group(1).strip() if match else value


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
