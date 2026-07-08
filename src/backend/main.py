"""FastAPI application exposing retrieval and answer generation endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from rag_pipeline.generator import GenerationError
from rag_pipeline.retriever import RetrieverError
from rag_pipeline.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    RetrievalResponse,
)
from rag_pipeline.service import RAGService
from rag_pipeline.settings import Settings, SettingsError


def create_app(
    service: RAGService | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    application_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rag_service = service or RAGService(application_settings)
        if application_settings.eager_load and service is None:
            await run_in_threadpool(app.state.rag_service.initialize)
        yield

    app = FastAPI(
        title="DMV RAG API",
        version="0.1.0",
        description="Baseline RAG over MultiDoc2Dial DMV documents",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:8501",
            "http://localhost:8501",
            "http://127.0.0.1:7860",
            "http://localhost:7860",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/", tags=["system"])
    async def root() -> dict[str, str]:
        return {
            "name": "DMV RAG API",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health(request: Request) -> HealthResponse:
        rag = _service(request)
        return HealthResponse(
            status=("ok" if rag.ready and rag.generator_configured else "degraded"),
            retrieval_ready=rag.ready,
            generator_configured=rag.generator_configured,
            retrieval_method=rag.retrieval_method,
            embedding_model=application_settings.embedding_model,
            llm_provider=application_settings.llm_provider,
            generation_model=application_settings.generation_model_name,
            prompt_strategy=application_settings.prompt_strategy,
            generation_temperature=application_settings.generation_temperature,
            generation_top_p=application_settings.generation_top_p,
            top_k=application_settings.top_k,
            context_window=application_settings.context_window,
            context_intro_chunks=application_settings.context_intro_chunks,
            max_context_chunks=application_settings.max_context_chunks,
            adaptive_context_enabled=application_settings.enable_adaptive_context,
            adaptive_top_k=application_settings.adaptive_top_k,
            adaptive_context_window=application_settings.adaptive_context_window,
            adaptive_context_intro_chunks=application_settings.adaptive_context_intro_chunks,
            adaptive_max_context_chunks=application_settings.adaptive_max_context_chunks,
            judge_enabled=application_settings.enable_judge,
            judge_min_score=application_settings.judge_min_score,
        )

    @app.post(
        "/api/v1/retrieve",
        response_model=RetrievalResponse,
        tags=["rag"],
    )
    async def retrieve(payload: AskRequest, request: Request) -> RetrievalResponse:
        try:
            return await run_in_threadpool(
                _service(request).retrieve,
                payload.question,
                payload.top_k,
                payload.context_window,
                payload.intro_chunks,
                payload.max_context_chunks,
            )
        except RetrieverError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post("/api/v1/ask", response_model=AskResponse, tags=["rag"])
    async def ask(payload: AskRequest, request: Request) -> AskResponse:
        try:
            return await run_in_threadpool(
                _service(request).answer,
                payload.question,
                payload.top_k,
                payload.language,
                payload.temperature,
                payload.top_p,
                payload.max_output_tokens,
                payload.use_judge,
                payload.context_window,
                payload.intro_chunks,
                payload.max_context_chunks,
                payload.use_adaptive_context,
            )
        except RetrieverError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except GenerationError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except SettingsError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    return app


def _service(request: Request) -> RAGService:
    return request.app.state.rag_service


app = create_app()
