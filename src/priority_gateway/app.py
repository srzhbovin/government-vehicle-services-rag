"""FastAPI reverse proxy that schedules requests before LiteLLM."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import AuthenticationError, ConfigManager, GatewayConfig
from .scheduler import PriorityScheduler, QueueFullError, QueueTimeoutError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def create_app(
    config_path: str | Path | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    policy_path = Path(
        config_path
        or os.getenv(
            "PRIORITY_GATEWAY_CONFIG",
            PROJECT_ROOT / "deploy" / "priority_gateway" / "priorities.yaml",
        )
    )
    manager = ConfigManager(policy_path)
    initial = manager.get()
    scheduler = _build_scheduler(initial)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config_manager = manager
        app.state.scheduler = scheduler
        app.state.http_client = httpx.AsyncClient(
            timeout=initial.upstream_timeout_seconds,
            transport=transport,
        )
        yield
        await app.state.http_client.aclose()

    app = FastAPI(
        title="LiteLLM Priority Gateway",
        version="0.1.0",
        description="OpenAI-compatible priority queue in front of LiteLLM",
        lifespan=lifespan,
    )

    @app.get("/health", tags=["system"])
    async def health(request: Request) -> dict[str, object]:
        config = request.app.state.config_manager.get()
        return {
            "status": "ok",
            "upstream": config.upstream_url,
            "policy_file": str(policy_path),
            "policy_reload_error": request.app.state.config_manager.last_reload_error,
            "scheduler": await request.app.state.scheduler.snapshot(),
        }

    @app.get("/metrics", tags=["system"])
    async def metrics(request: Request) -> dict[str, object]:
        return await request.app.state.scheduler.snapshot()

    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        tags=["proxy"],
    )
    async def proxy(path: str, request: Request) -> Response:
        config: GatewayConfig = request.app.state.config_manager.get()
        try:
            group = config.classify(request.headers)
        except AuthenticationError as error:
            raise HTTPException(status_code=401, detail=str(error)) from error
        scheduler: PriorityScheduler = request.app.state.scheduler
        try:
            lease = await scheduler.acquire(group.name, group.priority)
        except QueueFullError as error:
            raise HTTPException(status_code=429, detail=str(error)) from error
        except QueueTimeoutError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

        upstream_response: httpx.Response | None = None
        try:
            body = await request.body()
            headers = _upstream_headers(request, config)
            target = f"{config.upstream_url}/v1/{path}"
            upstream_request = request.app.state.http_client.build_request(
                request.method,
                target,
                params=request.query_params,
                headers=headers,
                content=body,
            )
            stream = _requests_stream(body)
            upstream_response = await request.app.state.http_client.send(
                upstream_request,
                stream=stream,
            )
            response_headers = _response_headers(upstream_response)
            response_headers["x-priority-group"] = group.name
            response_headers["x-queue-time-ms"] = f"{lease.queue_ms:.3f}"
            if stream:
                return StreamingResponse(
                    _stream_and_release(upstream_response, lease),
                    status_code=upstream_response.status_code,
                    headers=response_headers,
                    media_type=upstream_response.headers.get("content-type"),
                )
            status_code = upstream_response.status_code
            content = await upstream_response.aread()
            await upstream_response.aclose()
            upstream_response = None
            await lease.release()
            return Response(
                content=content,
                status_code=status_code,
                headers=response_headers,
            )
        except httpx.TimeoutException as error:
            if upstream_response is not None:
                await upstream_response.aclose()
            await lease.release()
            return JSONResponse(
                status_code=504, content={"error": {"message": str(error)}}
            )
        except httpx.HTTPError as error:
            if upstream_response is not None:
                await upstream_response.aclose()
            await lease.release()
            return JSONResponse(
                status_code=502, content={"error": {"message": str(error)}}
            )
        except Exception:
            if upstream_response is not None:
                await upstream_response.aclose()
            await lease.release()
            raise

    return app


def _build_scheduler(config: GatewayConfig) -> PriorityScheduler:
    scheduler = config.scheduler
    return PriorityScheduler(
        strategy=scheduler.strategy,
        max_concurrency=scheduler.max_concurrency,
        max_queue_size=scheduler.max_queue_size,
        queue_timeout_seconds=scheduler.queue_timeout_seconds,
        aging_interval_seconds=scheduler.aging_interval_seconds,
        starvation_timeout_seconds=scheduler.starvation_timeout_seconds,
    )


def _upstream_headers(request: Request, config: GatewayConfig) -> dict[str, str]:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _HOP_BY_HOP | {"host", "content-length", "x-user-group"}
    }
    if config.upstream_api_key:
        headers["authorization"] = f"Bearer {config.upstream_api_key}"
    return headers


def _response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in _HOP_BY_HOP | {"content-length", "content-encoding"}
    }


def _requests_stream(body: bytes) -> bool:
    if not body:
        return False
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("stream") is True


async def _stream_and_release(response: httpx.Response, lease):
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    finally:
        await response.aclose()
        await lease.release()


app = create_app()
