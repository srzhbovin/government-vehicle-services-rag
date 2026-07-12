"""Application settings loaded from environment variables and a local .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SettingsError(ValueError):
    """Raised when application settings are inconsistent."""


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SettingsError(f"Expected a boolean value, got: {value!r}")


def _as_int(value: str | None, default: int, name: str) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise SettingsError(f"{name} must be an integer") from error


def _as_float(value: str | None, default: float, name: str) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as error:
        raise SettingsError(f"{name} must be a number") from error


@dataclass(frozen=True)
class Settings:
    project_root: Path
    chunks_path: Path
    index_path: Path
    embeddings_path: Path
    index_metadata_path: Path
    embedding_model: str
    reranker_model: str
    use_reranker: bool
    device: str | None
    candidate_k: int
    reranker_candidate_k: int
    top_k: int
    context_window: int
    context_intro_chunks: int
    max_context_chunks: int
    enable_adaptive_context: bool
    adaptive_top_k: int
    adaptive_context_window: int
    adaptive_context_intro_chunks: int
    adaptive_max_context_chunks: int
    enable_refusal_gate: bool
    refusal_min_top_score: float
    refusal_min_lexical_overlap: float
    use_query_expansion: bool
    rrf_k: int
    bm25_weight: float
    llm_provider: str
    prompt_strategy: str
    local_llm_base_url: str
    local_llm_model: str
    local_llm_api_key: str
    yandex_api_key: str | None
    yandex_folder_id: str | None
    yandex_model: str
    yandex_fallback_models: tuple[str, ...]
    yandex_api_url: str
    generation_temperature: float
    generation_top_p: float
    generation_max_tokens: int
    generation_timeout_seconds: float
    enable_judge: bool
    judge_min_score: float
    judge_temperature: float
    judge_max_tokens: int
    eager_load: bool
    backend_url: str

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Settings":
        load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

        project_root = PROJECT_ROOT
        llm_provider = os.getenv("RAG_LLM_PROVIDER", "yandex").strip().lower()
        chunks_path = Path(
            os.getenv(
                "RAG_CHUNKS_PATH",
                project_root / "data" / "current" / "prepared" / "dmv_chunks_token_120_0.jsonl",
            )
        )
        index_dir = Path(
            os.getenv(
                "RAG_INDEX_DIR",
                project_root / "data" / "current" / "indexes" / "retrieval",
            )
        )

        settings = cls(
            project_root=project_root,
            chunks_path=chunks_path,
            index_path=index_dir / "faiss_flat.index",
            embeddings_path=index_dir / "chunk_embeddings.npy",
            index_metadata_path=index_dir / "index_metadata.json",
            embedding_model=os.getenv(
                "RAG_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            reranker_model=os.getenv(
                "RAG_RERANKER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L2-v2",
            ),
            use_reranker=_as_bool(os.getenv("RAG_USE_RERANKER"), True),
            device=os.getenv("RAG_DEVICE") or None,
            candidate_k=_as_int(os.getenv("RAG_CANDIDATE_K"), 60, "RAG_CANDIDATE_K"),
            reranker_candidate_k=_as_int(
                os.getenv("RAG_RERANKER_CANDIDATE_K"),
                20,
                "RAG_RERANKER_CANDIDATE_K",
            ),
            top_k=_as_int(os.getenv("RAG_TOP_K"), 3, "RAG_TOP_K"),
            context_window=_as_int(
                os.getenv("RAG_CONTEXT_WINDOW"),
                0,
                "RAG_CONTEXT_WINDOW",
            ),
            context_intro_chunks=_as_int(
                os.getenv("RAG_CONTEXT_INTRO_CHUNKS"),
                0,
                "RAG_CONTEXT_INTRO_CHUNKS",
            ),
            max_context_chunks=_as_int(
                os.getenv("RAG_MAX_CONTEXT_CHUNKS"),
                6,
                "RAG_MAX_CONTEXT_CHUNKS",
            ),
            enable_adaptive_context=_as_bool(
                os.getenv("RAG_ENABLE_ADAPTIVE_CONTEXT"),
                True,
            ),
            adaptive_top_k=_as_int(
                os.getenv("RAG_ADAPTIVE_TOP_K"),
                5,
                "RAG_ADAPTIVE_TOP_K",
            ),
            adaptive_context_window=_as_int(
                os.getenv("RAG_ADAPTIVE_CONTEXT_WINDOW"),
                1,
                "RAG_ADAPTIVE_CONTEXT_WINDOW",
            ),
            adaptive_context_intro_chunks=_as_int(
                os.getenv("RAG_ADAPTIVE_CONTEXT_INTRO_CHUNKS"),
                1,
                "RAG_ADAPTIVE_CONTEXT_INTRO_CHUNKS",
            ),
            adaptive_max_context_chunks=_as_int(
                os.getenv("RAG_ADAPTIVE_MAX_CONTEXT_CHUNKS"),
                8,
                "RAG_ADAPTIVE_MAX_CONTEXT_CHUNKS",
            ),
            enable_refusal_gate=_as_bool(os.getenv("RAG_ENABLE_REFUSAL_GATE"), True),
            refusal_min_top_score=_as_float(
                os.getenv("RAG_REFUSAL_MIN_TOP_SCORE"),
                -2.0,
                "RAG_REFUSAL_MIN_TOP_SCORE",
            ),
            refusal_min_lexical_overlap=_as_float(
                os.getenv("RAG_REFUSAL_MIN_LEXICAL_OVERLAP"),
                0.0,
                "RAG_REFUSAL_MIN_LEXICAL_OVERLAP",
            ),
            use_query_expansion=_as_bool(os.getenv("RAG_USE_QUERY_EXPANSION"), True),
            rrf_k=_as_int(os.getenv("RAG_RRF_K"), 60, "RAG_RRF_K"),
            bm25_weight=_as_float(
                os.getenv("RAG_BM25_WEIGHT"),
                0.5,
                "RAG_BM25_WEIGHT",
            ),
            llm_provider=llm_provider,
            prompt_strategy=os.getenv(
                "RAG_PROMPT_STRATEGY",
                "structured_output",
            ).strip().lower(),
            local_llm_base_url=os.getenv(
                "LOCAL_LLM_BASE_URL",
                (
                    "http://127.0.0.1:18000/v1"
                    if llm_provider == "lmdeploy"
                    else "http://127.0.0.1:1234/v1"
                ),
            ).rstrip("/"),
            local_llm_model=os.getenv(
                "LOCAL_LLM_MODEL",
                (
                    "qwen2.5-3b-instruct"
                    if llm_provider == "lmdeploy"
                    else "google/gemma-3-4b"
                ),
            ),
            local_llm_api_key=os.getenv("LOCAL_LLM_API_KEY", "local"),
            yandex_api_key=os.getenv("YANDEX_CLOUD_API_KEY") or None,
            yandex_folder_id=(
                os.getenv("YANDEX_CLOUD_FOLDER_ID")
                or os.getenv("YANDEX_CLOUD_FOLDER")
                or None
            ),
            yandex_model=os.getenv("YANDEX_GPT_MODEL", "qwen3.6-35b-a3b"),
            yandex_fallback_models=tuple(
                model.strip()
                for model in os.getenv(
                    "YANDEX_GPT_FALLBACK_MODELS",
                    "yandexgpt-5-lite,yandexgpt-5.1,yandexgpt-5-pro",
                ).split(",")
                if model.strip()
            ),
            yandex_api_url=os.getenv(
                "YANDEX_RESPONSES_URL",
                "https://ai.api.cloud.yandex.net/v1/responses",
            ),
            generation_temperature=_as_float(
                os.getenv("YANDEX_TEMPERATURE"),
                0.1,
                "YANDEX_TEMPERATURE",
            ),
            generation_top_p=_as_float(
                os.getenv("YANDEX_TOP_P"),
                0.9,
                "YANDEX_TOP_P",
            ),
            generation_max_tokens=_as_int(
                os.getenv("YANDEX_MAX_OUTPUT_TOKENS"),
                700,
                "YANDEX_MAX_OUTPUT_TOKENS",
            ),
            generation_timeout_seconds=_as_float(
                os.getenv("YANDEX_TIMEOUT_SECONDS"),
                90.0,
                "YANDEX_TIMEOUT_SECONDS",
            ),
            enable_judge=_as_bool(os.getenv("RAG_ENABLE_JUDGE"), True),
            judge_min_score=_as_float(
                os.getenv("RAG_JUDGE_MIN_SCORE"),
                0.72,
                "RAG_JUDGE_MIN_SCORE",
            ),
            judge_temperature=_as_float(
                os.getenv("RAG_JUDGE_TEMPERATURE"),
                0.0,
                "RAG_JUDGE_TEMPERATURE",
            ),
            judge_max_tokens=_as_int(
                os.getenv("RAG_JUDGE_MAX_OUTPUT_TOKENS"),
                500,
                "RAG_JUDGE_MAX_OUTPUT_TOKENS",
            ),
            eager_load=_as_bool(os.getenv("RAG_EAGER_LOAD"), True),
            backend_url=os.getenv("RAG_BACKEND_URL", "http://127.0.0.1:8000"),
        )
        settings.validate()
        return settings

    @property
    def generator_configured(self) -> bool:
        if self.llm_provider in {"lmstudio", "lmdeploy"}:
            return bool(self.local_llm_base_url)
        return bool(self.yandex_api_key and self.yandex_folder_id)

    @property
    def generation_model_name(self) -> str:
        if self.llm_provider in {"lmstudio", "lmdeploy"}:
            return self.local_llm_model
        return self.yandex_model

    @property
    def yandex_model_uri(self) -> str:
        return self.model_uri(self.yandex_model)

    def model_uri(self, model: str) -> str:
        if not self.yandex_folder_id:
            raise SettingsError("YANDEX_CLOUD_FOLDER_ID is not configured")
        return f"gpt://{self.yandex_folder_id}/{model}"

    def require_generator_credentials(self) -> None:
        missing = []
        if not self.yandex_api_key:
            missing.append("YANDEX_CLOUD_API_KEY")
        if not self.yandex_folder_id:
            missing.append("YANDEX_CLOUD_FOLDER_ID")
        if missing:
            raise SettingsError("Missing Yandex Cloud settings: " + ", ".join(missing))

    def validate(self) -> None:
        if self.llm_provider not in {"lmstudio", "lmdeploy", "yandex"}:
            raise SettingsError(
                "RAG_LLM_PROVIDER must be 'lmstudio', 'lmdeploy' or 'yandex'"
            )
        if self.prompt_strategy not in {
            "plain",
            "json",
            "pydantic",
            "structured_output",
        }:
            raise SettingsError(
                "RAG_PROMPT_STRATEGY must be plain, json, pydantic or structured_output"
            )
        if not 1 <= self.top_k <= self.reranker_candidate_k <= self.candidate_k:
            raise SettingsError(
                "Expected RAG_TOP_K <= RAG_RERANKER_CANDIDATE_K <= RAG_CANDIDATE_K"
            )
        if self.rrf_k < 1:
            raise SettingsError("RAG_RRF_K must be positive")
        if self.context_window < 0:
            raise SettingsError("RAG_CONTEXT_WINDOW cannot be negative")
        if self.context_intro_chunks < 0:
            raise SettingsError("RAG_CONTEXT_INTRO_CHUNKS cannot be negative")
        if self.max_context_chunks < self.top_k:
            raise SettingsError("RAG_MAX_CONTEXT_CHUNKS must be greater than or equal to RAG_TOP_K")
        if not 1 <= self.adaptive_top_k <= self.reranker_candidate_k:
            raise SettingsError("Expected RAG_ADAPTIVE_TOP_K <= RAG_RERANKER_CANDIDATE_K")
        if self.adaptive_context_window < 0:
            raise SettingsError("RAG_ADAPTIVE_CONTEXT_WINDOW cannot be negative")
        if self.adaptive_context_intro_chunks < 0:
            raise SettingsError("RAG_ADAPTIVE_CONTEXT_INTRO_CHUNKS cannot be negative")
        if self.adaptive_max_context_chunks < self.adaptive_top_k:
            raise SettingsError(
                "RAG_ADAPTIVE_MAX_CONTEXT_CHUNKS must be greater than or equal to RAG_ADAPTIVE_TOP_K"
            )
        if not 0 <= self.refusal_min_lexical_overlap <= 1:
            raise SettingsError("RAG_REFUSAL_MIN_LEXICAL_OVERLAP must be between 0 and 1")
        if not 0 <= self.bm25_weight <= 1:
            raise SettingsError("RAG_BM25_WEIGHT must be between 0 and 1")
        if not 0 <= self.generation_temperature <= 1:
            raise SettingsError("YANDEX_TEMPERATURE must be between 0 and 1")
        if not 0 <= self.generation_top_p <= 1:
            raise SettingsError("YANDEX_TOP_P must be between 0 and 1")
        if self.generation_max_tokens < 1:
            raise SettingsError("YANDEX_MAX_OUTPUT_TOKENS must be positive")
        if not 0 <= self.judge_min_score <= 1:
            raise SettingsError("RAG_JUDGE_MIN_SCORE must be between 0 and 1")
        if not 0 <= self.judge_temperature <= 1:
            raise SettingsError("RAG_JUDGE_TEMPERATURE must be between 0 and 1")
        if self.judge_max_tokens < 1:
            raise SettingsError("RAG_JUDGE_MAX_OUTPUT_TOKENS must be positive")
