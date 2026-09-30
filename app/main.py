"""
SafeSight AI — FastAPI Application Factory
===========================================
Creates, configures, and returns the FastAPI application instance.

Lifecycle:
  startup  → initialise all ML models, connect to Redis
  shutdown → teardown models, release GPU memory

Middleware:
  - CORS (configurable origins)
  - Request ID injection
  - Structured access logging
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.logging_config import configure_logging, get_logger
from app.models.registry import initialise_models, teardown_models
from app.routes.moderation import router
from app.ui.splash import display_splash_screen

# Bootstrap logging before anything else
configure_logging()
logger = get_logger("main")
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan context manager (FastAPI 0.93+ style)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and graceful shutdown.

    Startup:
      - Load all ML models (CLIP, VLM, DeBERTa, Whisper) concurrently
      - Attempt Redis connection (non-fatal if unavailable)
      - Attach registry and redis client to app.state

    Shutdown:
      - Release model GPU memory
      - Close Redis connection
    """
    logger.info("SafeSight AI starting up…", env=settings.app_env)

    # ── Load models ────────────────────────────────────────────────────────────
    registry = await initialise_models()
    app.state.registry = registry

    # ── Connect Redis (optional) ───────────────────────────────────────────────
    redis_client = None
    try:
        import aioredis  # type: ignore

        redis_client = await aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        await redis_client.ping()
        logger.info("Redis connected", url=settings.redis_url)
    except Exception as exc:
        logger.warning(
            "Redis unavailable — running without cache",
            error=str(exc),
        )

    app.state.redis = redis_client

    logger.info(
        "SafeSight AI ready",
        host=settings.app_host,
        port=settings.app_port,
        models=registry.health_map(),
    )

    # ── Display sleek CLI splash screen ────────────────────────────────────────
    display_splash_screen(
        clear=True,
        local_server=f"http://{settings.app_host}:{settings.app_port}",
        api_docs=f"http://{settings.app_host}:{settings.app_port}/docs",
    )

    yield  # ← Application runs here

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("SafeSight AI shutting down…")

    await teardown_models()

    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")

    logger.info("Shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
# Application factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    FastAPI application factory.
    Returns a fully configured app instance ready to be served by uvicorn.
    """
    app = FastAPI(
        title="SafeSight AI",
        summary="Real-Time Multimodal Content Moderation API",
        description=(
            "Production-grade two-tier AI content moderation system. "
            "Processes text, images, video frames, and audio files in real time, "
            "detecting deepfakes, AI-generated media, and policy violations with "
            "structured, human-readable explanations."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    @app.on_event("startup")
    async def on_startup() -> None:
        display_splash_screen(
            clear=True,
            local_server=f"http://{settings.app_host}:{settings.app_port}",
            api_docs=f"http://{settings.app_host}:{settings.app_port}/docs",
        )

    # ── CORS middleware ────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request ID + Access Logging middleware ─────────────────────────────────
    @app.middleware("http")
    async def request_logging_middleware(
        request: Request, call_next
    ) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        t0 = time.perf_counter()

        logger.info(
            "→ Incoming request",
            rid=request_id,
            method=request.method,
            path=request.url.path,
        )

        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "← Request complete",
            rid=request_id,
            status=response.status_code,
            latency_ms=f"{elapsed_ms:.1f}",
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = f"{elapsed_ms:.1f}"
        return response

    # ── Mount routers ──────────────────────────────────────────────────────────
    app.include_router(router)

    # ── Root Convenience Aliases (e.g. /moderate, /moderate/text, /health) ────
    from app.routes.moderation import (
        moderate_multimodal,
        moderate_text,
        moderate_image,
        health_check,
    )
    from app.schemas import ModerationResult, HealthResponse

    app.add_api_route("/moderate", moderate_multimodal, methods=["POST"], response_model=ModerationResult, include_in_schema=False)
    app.add_api_route("/moderate/multimodal", moderate_multimodal, methods=["POST"], response_model=ModerationResult, include_in_schema=False)
    app.add_api_route("/moderate/text", moderate_text, methods=["POST"], response_model=ModerationResult, include_in_schema=False)
    app.add_api_route("/moderate/image", moderate_image, methods=["POST"], response_model=ModerationResult, include_in_schema=False)
    app.add_api_route("/health", health_check, methods=["GET"], response_model=HealthResponse, include_in_schema=False)

    from fastapi.responses import RedirectResponse

    @app.get("/moderate", include_in_schema=False)
    @app.get("/moderate/multimodal", include_in_schema=False)
    @app.get("/moderate/text", include_in_schema=False)
    @app.get("/moderate/image", include_in_schema=False)
    async def redirect_moderate_to_docs():
        """Redirect direct browser navigation to interactive Swagger docs."""
        return RedirectResponse(url="/docs")

    # ── Root redirect ──────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "SafeSight AI",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


# ── Entry point for `python -m app.main` or direct uvicorn invocation ─────────
app = create_app()
