"""
SafeSight AI — API Route Handlers
====================================
FastAPI router definitions for all moderation endpoints.

Endpoints:
  POST /api/v1/moderate/multimodal  — Full multimodal (image + text + audio + video)
  POST /api/v1/moderate/text        — Text-only fast moderation
  POST /api/v1/moderate/image       — Image + optional text moderation
  GET  /api/v1/health               — Model health check
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from app.config import get_settings
from app.logging_config import get_logger
from app.models.registry import ModelRegistry, get_registry
from app.pipeline.orchestrator import run_moderation_pipeline
from app.schemas import (
    HealthResponse,
    ModerationResult,
    TextModerationRequest,
)

logger = get_logger("routes")
settings = get_settings()

router = APIRouter(prefix="/api/v1", tags=["Moderation"])


# ─────────────────────────────────────────────────────────────────────────────
# Dependency: extract model registry from app state
# ─────────────────────────────────────────────────────────────────────────────

def get_model_registry(request: Request) -> ModelRegistry:
    """FastAPI dependency: returns the pre-loaded ModelRegistry from app state."""
    registry: ModelRegistry | None = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not yet initialised. Please retry in a few moments.",
        )
    return registry


def get_redis(request: Request):
    """FastAPI dependency: returns the Redis client from app state (may be None)."""
    return getattr(request.app.state, "redis", None)


RegistryDep = Annotated[ModelRegistry, Depends(get_model_registry)]
RedisDep = Annotated[object, Depends(get_redis)]


# ─────────────────────────────────────────────────────────────────────────────
# POST /moderate/multimodal
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/moderate/multimodal",
    response_model=ModerationResult,
    summary="Full Multimodal Content Moderation",
    description=(
        "Submit content containing any combination of text, image, audio, and/or video "
        "for comprehensive real-time moderation through the two-tier pipeline."
    ),
    status_code=status.HTTP_200_OK,
)
async def moderate_multimodal(
    registry: RegistryDep,
    redis: RedisDep,
    text: Annotated[str | None, Form(description="Text caption or body")] = None,
    image: Annotated[
        UploadFile | None,
        File(description="Image file (JPEG, PNG, WEBP…)"),
    ] = None,
    audio: Annotated[
        UploadFile | None,
        File(description="Audio file (WAV, MP3, FLAC…)"),
    ] = None,
    video: Annotated[
        UploadFile | None,
        File(description="Video file (MP4, WebM…)"),
    ] = None,
) -> ModerationResult:
    """Handle multimodal moderation requests with any combination of media types."""
    t0 = time.perf_counter()

    if not any([text, image, audio, video]):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one of text, image, audio, or video must be provided.",
        )

    # Read file bytes concurrently using asyncio
    import asyncio

    async def _read(upload: UploadFile | None) -> bytes | None:
        if upload is None:
            return None
        return await upload.read()

    image_bytes, audio_bytes, video_bytes = await asyncio.gather(
        _read(image),
        _read(audio),
        _read(video),
    )

    result = await run_moderation_pipeline(
        registry=registry,
        text=text,
        image_bytes=image_bytes,
        audio_bytes=audio_bytes,
        video_bytes=video_bytes,
        redis_client=redis,
    )

    logger.info(
        "Multimodal request complete",
        decision=result.decision,
        latency_ms=f"{(time.perf_counter() - t0) * 1000:.1f}",
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# POST /moderate/text
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/moderate/text",
    response_model=ModerationResult,
    summary="Text-Only Content Moderation",
    description="Fast text moderation endpoint. Accepts JSON body. Optimized for high-throughput text-only workloads.",
    status_code=status.HTTP_200_OK,
)
async def moderate_text(
    request_body: TextModerationRequest,
    registry: RegistryDep,
    redis: RedisDep,
) -> ModerationResult:
    """Moderate a text-only submission (JSON body)."""
    text_input = (
        f"{request_body.context} | {request_body.text}"
        if request_body.context
        else request_body.text
    )

    result = await run_moderation_pipeline(
        registry=registry,
        text=text_input,
        redis_client=redis,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# POST /moderate/image
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/moderate/image",
    response_model=ModerationResult,
    summary="Image + Text Moderation",
    description="Upload an image and optional caption for bimodal content moderation.",
    status_code=status.HTTP_200_OK,
)
async def moderate_image(
    registry: RegistryDep,
    redis: RedisDep,
    image: Annotated[UploadFile, File(description="Image file to moderate")],
    caption: Annotated[
        str | None,
        Form(description="Optional text caption accompanying the image"),
    ] = None,
) -> ModerationResult:
    """Moderate an image with optional text caption."""
    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Image file is empty.",
        )

    result = await run_moderation_pipeline(
        registry=registry,
        text=caption,
        image_bytes=image_bytes,
        redis_client=redis,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health Check",
    description="Returns the loaded status of all ML models and the active device.",
    status_code=status.HTTP_200_OK,
)
async def health_check(registry: RegistryDep) -> HealthResponse:
    """Check that all models are loaded and the system is healthy."""
    return HealthResponse(
        status="healthy" if registry.is_healthy() else "degraded",
        models_loaded=registry.health_map(),
        device=settings.device,
    )
