"""Yandex AI Studio and local LLM clients with the baseline grounded prompt."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

from .prompting import (
    JUDGE_INSTRUCTIONS,
    PromptStrategy,
    judge_response_format,
    parse_model_answer,
    parse_judge_answer,
    prompt_spec,
)
from .retriever import RetrievedChunk
from .schemas import GenerationUsage
from .settings import Settings, SettingsError


BASELINE_INSTRUCTIONS = prompt_spec(PromptStrategy.PLAIN).instructions


class GenerationError(RuntimeError):
    """Raised when the external model cannot return a usable answer."""


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    model: str
    usage: GenerationUsage
    elapsed_ms: float
    prompt_strategy: str = PromptStrategy.PLAIN.value
    confidence: float | None = None
    source: str | None = None
    parse_success: bool | None = None
    schema_valid: bool | None = None
    parse_error: str | None = None
    raw_output: str | None = None


@dataclass(frozen=True)
class GuardrailResult:
    checked: bool
    accepted: bool | None
    verdict: str | None
    score: float | None
    reason: str | None
    corrected_answer: str | None = None
    source: str | None = None
    elapsed_ms: float | None = None
    parse_success: bool | None = None
    schema_valid: bool | None = None
    parse_error: str | None = None


def build_context_prompt(
    question: str,
    chunks: Sequence[RetrievedChunk],
    language: str = "auto",
) -> str:
    blocks = []
    for chunk in chunks:
        title = chunk.title or chunk.document_id
        blocks.append(
            f"[{chunk.rank}] Title: {title}\n"
            f"Document ID: {chunk.document_id}\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Text: {chunk.text}"
        )
    context = "\n\n".join(blocks)
    language_instruction = {
        "auto": "Answer in the same language as the user's question.",
        "ru": "Answer in Russian.",
        "en": "Answer in English.",
    }.get(language, "Answer in the same language as the user's question.")
    return (
        "Use the following retrieved document fragments to answer the question.\n\n"
        f"DOCUMENT FRAGMENTS\n{context}\n\n"
        f"QUESTION\n{question}\n\n"
        f"LANGUAGE\n{language_instruction}\n\n"
        "Return a direct answer with citations to the fragment numbers."
    )


def build_judge_prompt(
    question: str,
    chunks: Sequence[RetrievedChunk],
    answer: str,
    language: str = "auto",
) -> str:
    return (
        build_context_prompt(question, chunks, language)
        + "\n\nASSISTANT ANSWER TO CHECK\n"
        + answer
        + "\n\nReturn the groundedness judgment."
    )


class YandexGenerator:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings
        self._client = http_client

    @property
    def configured(self) -> bool:
        return self.settings.generator_configured

    def generate(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        language: str = "auto",
        strategy: PromptStrategy | str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
    ) -> GenerationResult:
        try:
            self.settings.require_generator_credentials()
        except SettingsError as error:
            raise GenerationError(str(error)) from error

        selected_strategy = PromptStrategy(strategy or self.settings.prompt_strategy)
        spec = prompt_spec(selected_strategy)
        payload: dict[str, Any] = {
            "model": self.settings.yandex_model_uri,
            "instructions": spec.instructions,
            "input": build_context_prompt(question, chunks, language),
            "temperature": (
                self.settings.generation_temperature
                if temperature is None
                else temperature
            ),
            "top_p": self.settings.generation_top_p if top_p is None else top_p,
            "max_output_tokens": (
                self.settings.generation_max_tokens
                if max_output_tokens is None
                else max_output_tokens
            ),
        }
        if spec.response_format is not None:
            payload["text"] = {"format": spec.response_format}
        headers = {
            "Authorization": f"Api-Key {self.settings.yandex_api_key}",
            "Content-Type": "application/json",
        }

        started = time.perf_counter()
        client = self._client or httpx.Client(
            timeout=self.settings.generation_timeout_seconds
        )
        should_close = self._client is None
        data: dict[str, Any] | None = None
        models = list(
            dict.fromkeys(
                [self.settings.yandex_model, *self.settings.yandex_fallback_models]
            )
        )
        try:
            last_error: httpx.HTTPStatusError | None = None
            for model_index, model in enumerate(models):
                payload["model"] = self.settings.model_uri(model)
                response = client.post(
                    self.settings.yandex_api_url,
                    headers=headers,
                    json=payload,
                )
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as error:
                    last_error = error
                    has_fallback = model_index < len(models) - 1
                    if has_fallback and response.status_code in {400, 404}:
                        continue
                    status = response.status_code
                    message = _safe_error_message(response)
                    raise GenerationError(
                        f"Yandex AI Studio returned HTTP {status}: {message}"
                    ) from error
                data = response.json()
                break
            if data is None and last_error is not None:
                status = last_error.response.status_code
                message = _safe_error_message(last_error.response)
                raise GenerationError(
                    f"No configured Yandex model is available (HTTP {status}): {message}"
                ) from last_error
        except (httpx.HTTPError, ValueError) as error:
            raise GenerationError(f"Yandex AI Studio request failed: {error}") from error
        finally:
            if should_close:
                client.close()

        assert data is not None
        _raise_yandex_error_if_present(data, "Yandex AI Studio")
        raw_output = _extract_output_text(data)
        parsed = parse_model_answer(raw_output, selected_strategy)
        usage_data = data.get("usage") or {}
        elapsed_ms = (time.perf_counter() - started) * 1000
        return GenerationResult(
            answer=parsed.answer,
            model=str(data.get("model") or self.settings.yandex_model_uri),
            usage=GenerationUsage(
                input_tokens=_optional_int(usage_data.get("input_tokens")),
                output_tokens=_optional_int(usage_data.get("output_tokens")),
                total_tokens=_optional_int(usage_data.get("total_tokens")),
            ),
            elapsed_ms=elapsed_ms,
            prompt_strategy=selected_strategy.value,
            confidence=parsed.confidence,
            source=parsed.source,
            parse_success=parsed.parse_success,
            schema_valid=parsed.schema_valid,
            parse_error=parsed.parse_error,
            raw_output=raw_output,
        )

    def judge_answer(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        answer: str,
        language: str = "auto",
    ) -> GuardrailResult:
        try:
            self.settings.require_generator_credentials()
        except SettingsError as error:
            raise GenerationError(str(error)) from error

        payload: dict[str, Any] = {
            "model": self.settings.yandex_model_uri,
            "instructions": JUDGE_INSTRUCTIONS,
            "input": build_judge_prompt(question, chunks, answer, language),
            "temperature": self.settings.judge_temperature,
            "top_p": 1.0,
            "max_output_tokens": self.settings.judge_max_tokens,
            "text": {"format": judge_response_format()},
        }
        headers = {
            "Authorization": f"Api-Key {self.settings.yandex_api_key}",
            "Content-Type": "application/json",
        }

        started = time.perf_counter()
        client = self._client or httpx.Client(
            timeout=self.settings.generation_timeout_seconds
        )
        should_close = self._client is None
        data: dict[str, Any] | None = None
        models = list(
            dict.fromkeys(
                [self.settings.yandex_model, *self.settings.yandex_fallback_models]
            )
        )
        try:
            last_error: httpx.HTTPStatusError | None = None
            for model_index, model in enumerate(models):
                payload["model"] = self.settings.model_uri(model)
                response = client.post(
                    self.settings.yandex_api_url,
                    headers=headers,
                    json=payload,
                )
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as error:
                    last_error = error
                    has_fallback = model_index < len(models) - 1
                    if has_fallback and response.status_code in {400, 404}:
                        continue
                    status = response.status_code
                    message = _safe_error_message(response)
                    raise GenerationError(
                        f"Yandex AI Studio judge returned HTTP {status}: {message}"
                    ) from error
                data = response.json()
                break
            if data is None and last_error is not None:
                status = last_error.response.status_code
                message = _safe_error_message(last_error.response)
                raise GenerationError(
                    f"No configured Yandex judge model is available (HTTP {status}): {message}"
                ) from last_error
        except (httpx.HTTPError, ValueError) as error:
            raise GenerationError(f"Yandex AI Studio judge request failed: {error}") from error
        finally:
            if should_close:
                client.close()

        assert data is not None
        _raise_yandex_error_if_present(data, "Yandex AI Studio judge")
        raw_output = _extract_output_text(data)
        parsed = parse_judge_answer(raw_output)
        return GuardrailResult(
            checked=True,
            accepted=(
                parsed.schema_valid
                and parsed.verdict == "grounded"
                and parsed.score >= self.settings.judge_min_score
            ),
            verdict=parsed.verdict,
            score=parsed.score,
            reason=parsed.reason,
            corrected_answer=parsed.corrected_answer,
            source=parsed.source,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            parse_success=parsed.parse_success,
            schema_valid=parsed.schema_valid,
            parse_error=parsed.parse_error,
        )


class LocalOpenAICompatibleGenerator:
    """Generator for LM Studio and other OpenAI-compatible local servers."""

    def __init__(
        self,
        settings: Settings,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings
        self._client = http_client
        self._resolved_model: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.settings.local_llm_base_url)

    def generate(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        language: str = "auto",
        strategy: PromptStrategy | str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
    ) -> GenerationResult:
        selected_strategy = PromptStrategy(strategy or self.settings.prompt_strategy)
        if selected_strategy is PromptStrategy.STRUCTURED_OUTPUT:
            selected_strategy = PromptStrategy.PYDANTIC
        spec = prompt_spec(selected_strategy)
        client = self._client or httpx.Client(
            timeout=self.settings.generation_timeout_seconds
        )
        should_close = self._client is None
        started = time.perf_counter()
        try:
            model = self._resolved_model or self._resolve_model(client)
            self._resolved_model = model
            response = client.post(
                f"{self.settings.local_llm_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.local_llm_api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": spec.instructions},
                        {
                            "role": "user",
                            "content": build_context_prompt(question, chunks, language),
                        },
                    ],
                    "temperature": (
                        self.settings.generation_temperature
                        if temperature is None
                        else temperature
                    ),
                    "top_p": self.settings.generation_top_p if top_p is None else top_p,
                    "max_tokens": (
                        self.settings.generation_max_tokens
                        if max_output_tokens is None
                        else max_output_tokens
                    ),
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
        except httpx.ConnectError as error:
            raise GenerationError(
                "Local LLM server is unavailable. Start the LM Studio server on port 1234."
            ) from error
        except httpx.HTTPStatusError as error:
            raise GenerationError(
                f"Local LLM server returned HTTP {error.response.status_code}: "
                f"{_safe_error_message(error.response)}"
            ) from error
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as error:
            raise GenerationError(f"Local LLM request failed: {error}") from error
        finally:
            if should_close:
                client.close()

        try:
            raw_output = str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as error:
            raise GenerationError("Local LLM server returned no text") from error
        if not raw_output:
            raise GenerationError("Local LLM server returned an empty answer")
        parsed = parse_model_answer(raw_output, selected_strategy)
        usage_data = data.get("usage") or {}
        return GenerationResult(
            answer=parsed.answer,
            model=str(data.get("model") or model),
            usage=GenerationUsage(
                input_tokens=_optional_int(usage_data.get("prompt_tokens")),
                output_tokens=_optional_int(usage_data.get("completion_tokens")),
                total_tokens=_optional_int(usage_data.get("total_tokens")),
            ),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            prompt_strategy=selected_strategy.value,
            confidence=parsed.confidence,
            source=parsed.source,
            parse_success=parsed.parse_success,
            schema_valid=parsed.schema_valid,
            parse_error=parsed.parse_error,
            raw_output=raw_output,
        )

    def judge_answer(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        answer: str,
        language: str = "auto",
    ) -> GuardrailResult:
        client = self._client or httpx.Client(
            timeout=self.settings.generation_timeout_seconds
        )
        should_close = self._client is None
        started = time.perf_counter()
        schema = judge_response_format()["schema"]
        try:
            model = self._resolved_model or self._resolve_model(client)
            self._resolved_model = model
            response = client.post(
                f"{self.settings.local_llm_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.local_llm_api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                JUDGE_INSTRUCTIONS
                                + "\nReturn only JSON matching this schema:\n"
                                + str(schema)
                            ),
                        },
                        {
                            "role": "user",
                            "content": build_judge_prompt(
                                question,
                                chunks,
                                answer,
                                language,
                            ),
                        },
                    ],
                    "temperature": self.settings.judge_temperature,
                    "top_p": 1.0,
                    "max_tokens": self.settings.judge_max_tokens,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
        except httpx.ConnectError as error:
            raise GenerationError(
                "Local LLM server is unavailable. Start the LM Studio server on port 1234."
            ) from error
        except httpx.HTTPStatusError as error:
            raise GenerationError(
                f"Local LLM judge returned HTTP {error.response.status_code}: "
                f"{_safe_error_message(error.response)}"
            ) from error
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as error:
            raise GenerationError(f"Local LLM judge request failed: {error}") from error
        finally:
            if should_close:
                client.close()

        try:
            raw_output = str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as error:
            raise GenerationError("Local LLM judge returned no text") from error
        parsed = parse_judge_answer(raw_output)
        return GuardrailResult(
            checked=True,
            accepted=(
                parsed.schema_valid
                and parsed.verdict == "grounded"
                and parsed.score >= self.settings.judge_min_score
            ),
            verdict=parsed.verdict,
            score=parsed.score,
            reason=parsed.reason,
            corrected_answer=parsed.corrected_answer,
            source=parsed.source,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            parse_success=parsed.parse_success,
            schema_valid=parsed.schema_valid,
            parse_error=parsed.parse_error,
        )

    def _resolve_model(self, client: httpx.Client) -> str:
        try:
            response = client.get(f"{self.settings.local_llm_base_url}/models")
            response.raise_for_status()
            model_ids = [
                str(item["id"])
                for item in response.json().get("data", [])
                if item.get("id") and "embed" not in str(item["id"]).lower()
            ]
        except (httpx.HTTPError, ValueError, KeyError) as error:
            raise GenerationError(
                "Could not list models from the local LLM server"
            ) from error
        if self.settings.local_llm_model in model_ids:
            return self.settings.local_llm_model
        if model_ids:
            return model_ids[0]
        raise GenerationError(
            "No text generation model is available in LM Studio. Download or load a model first."
        )


def build_generator(settings: Settings):
    if settings.llm_provider == "lmstudio":
        return LocalOpenAICompatibleGenerator(settings)
    return YandexGenerator(settings)


def _extract_output_text(data: dict[str, Any]) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    texts = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                texts.append(content["text"].strip())
    answer = "\n".join(text for text in texts if text)
    if not answer:
        raise GenerationError("Yandex AI Studio returned no text")
    return answer


def _safe_error_message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return "request rejected"
    message = data.get("error") or data.get("message") or "request rejected"
    if isinstance(message, dict):
        message = message.get("message") or message.get("code") or "request rejected"
    text = str(message)
    if "api key" in text.lower():
        return "API key was rejected"
    return text[:300]


def _raise_yandex_error_if_present(data: dict[str, Any], label: str) -> None:
    if data.get("status") != "failed" and not data.get("error"):
        return
    error = data.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("code") or "request rejected"
    else:
        message = error or "request rejected"
    text = str(message)
    if "api key" in text.lower():
        text = "API key was rejected"
    raise GenerationError(f"{label} returned an error: {text[:300]}")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
