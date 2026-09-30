"""
SafeSight AI — Master Pipeline Orchestrator
============================================
The central async orchestrator that:

  1. Receives raw media inputs (text, image bytes, video bytes, audio bytes)
  2. Dispatches concurrent media pre-processing jobs
  3. Executes the cascaded Tier-1 → Tier-2 moderation pipeline
  4. Assembles the final ModerationResult
  5. Optionally caches results in Redis

This is the single entry point for all moderation logic. Routes, background
workers, and test harnesses call ONLY this module — never the tier modules
directly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

from PIL import Image

from app.config import get_settings
from app.logging_config import get_logger
from app.models.registry import ModelRegistry
from app.pipeline.tier1_gate import run_tier1_gate
from app.pipeline.tier2_vlm import run_tier2_vlm
from app.processors.audio_processor import transcribe_audio
from app.processors.image_processor import bytes_to_pil, extract_video_frames
from app.processors.text_processor import preprocess_text
from app.utils.preprocessor import preprocess_adversarial_text, preprocess_image
from app.utils.forensics_detector import run_full_forensics, AuthenticityVerdict as ForensicVerdict
from app.schemas import (
    AuthenticityReportSchema,
    DetectedViolation,
    MediaType,
    ModerationDecision,
    ModerationResult,
    PipelineTier,
    ReasoningBlock,
    SeverityLevel,
    ViolationCategory,
)

logger = get_logger("orchestrator")
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_cache_key(
    text: str | None,
    image_bytes: bytes | None,
    audio_bytes: bytes | None,
    video_bytes: bytes | None,
) -> str:
    """Deterministic SHA-256 cache key from all input modalities."""
    h = hashlib.sha256()
    if text:
        h.update(text.encode("utf-8"))
    if image_bytes:
        h.update(image_bytes)
    if audio_bytes:
        h.update(audio_bytes)
    if video_bytes:
        h.update(video_bytes[:4096])  # Hash only first 4KB of video for speed
    return f"safesight:v1:{h.hexdigest()}"


async def _cache_get(key: str, redis_client: Any) -> ModerationResult | None:
    """Attempt to retrieve a cached result from Redis."""
    if redis_client is None:
        return None
    try:
        raw = await redis_client.get(key)
        if raw:
            return ModerationResult.model_validate_json(raw)
    except Exception as exc:
        logger.warning("Redis GET failed", key=key, error=str(exc))
    return None


async def _cache_set(
    key: str, result: ModerationResult, redis_client: Any
) -> None:
    """Store a moderation result in Redis with TTL."""
    if redis_client is None:
        return
    try:
        await redis_client.setex(
            key,
            settings.cache_ttl_seconds,
            result.model_dump_json(),
        )
    except Exception as exc:
        logger.warning("Redis SET failed", key=key, error=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Tier-1 Short-circuit Decision Builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_tier1_result(safe_score: float) -> tuple[
    ModerationDecision,
    float,
    list[DetectedViolation],
    ReasoningBlock,
]:
    """
    When Tier-1 short-circuits (no Tier-2 needed), build a complete
    decision tuple from the safe_score alone.
    """
    if safe_score >= settings.tier1_confidence_threshold:
        return (
            ModerationDecision.APPROVED,
            round(safe_score, 4),
            [],
            ReasoningBlock(
                text_explanation="Content passed Tier-1 fast-path screening with high safety confidence.",
                visual_explanation="Visual content scored above the safe threshold in CLIP embedding space.",
                bimodal_synergy="No harmful bimodal synergy detected in fast-path screening.",
            ),
        )
    else:
        # Low score — high-confidence harm, auto-flag
        confidence = round(1.0 - safe_score, 4)
        violations = [
            DetectedViolation(
                category=ViolationCategory.HATE_SPEECH_AND_HARASSMENT,
                policy_code="MLC_H1.1",
                severity=SeverityLevel.HIGH if confidence > 0.80 else SeverityLevel.MEDIUM,
                confidence=confidence,
            )
        ]
        return (
            ModerationDecision.FLAGGED,
            confidence,
            violations,
            ReasoningBlock(
                text_explanation="Content triggered hard-block patterns or scored below the suspicious threshold.",
                visual_explanation="Visual embedding similarity strongly correlates with known unsafe content categories.",
                bimodal_synergy="Combined text and visual signals indicate high-confidence policy violation.",
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_moderation_pipeline(
    registry: ModelRegistry,
    text: str | None = None,
    image_bytes: bytes | None = None,
    audio_bytes: bytes | None = None,
    video_bytes: bytes | None = None,
    redis_client: Any = None,
) -> ModerationResult:
    """
    Main async pipeline orchestrator.

    Execution flow:
      ┌─────────────────────────────────────────────────────────┐
      │ 1. Cache lookup (Redis)                                  │
      │ 2. Concurrent media pre-processing (asyncio.gather)      │
      │    ├─ image_bytes  → PIL Image                           │
      │    ├─ video_bytes  → key frames → PIL Image              │
      │    ├─ audio_bytes  → Whisper transcript → text           │
      │    └─ text         → normalize / clean                   │
      │ 3. Tier-1 fast-path gate (<50ms budget)                  │
      │    ├─ short-circuit APPROVED/FLAGGED                     │
      │    └─ borderline → Tier-2                                │
      │ 4. Tier-2 VLM reasoning (<700ms budget)                  │
      │ 5. Assemble ModerationResult                             │
      │ 6. Cache result                                          │
      └─────────────────────────────────────────────────────────┘

    Args:
        registry:     Loaded model registry.
        text:         Raw text content.
        image_bytes:  Raw bytes of an image file.
        audio_bytes:  Raw bytes of an audio file (WAV/MP3/FLAC).
        video_bytes:  Raw bytes of a video file (MP4/WebM).
        redis_client: Optional async Redis client for result caching.

    Returns:
        ModerationResult — the canonical API response object.
    """
    pipeline_start = time.perf_counter()
    loop = asyncio.get_event_loop()

    # ── Step 0a: Adversarial text preprocessing ────────────────────────────────
    # Defeats leet-speak, Unicode homoglyphs, spaced-text evasion BEFORE cache
    # lookup so obfuscated variants map to the same clean cache key.
    if text:
        text = await loop.run_in_executor(None, preprocess_adversarial_text, text)
        logger.debug("Adversarial text preprocessing applied", cleaned_len=len(text))


    # ── Step 0b: Banned image hash pre-screening ───────────────────────────────
    # Fingerprint the image and check against the banned-hash database.
    # Returns an instant FLAGGED result without invoking CLIP or the VLM.
    if image_bytes:
        img_precheck = await loop.run_in_executor(None, preprocess_image, image_bytes)
        if img_precheck.is_banned:
            elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
            logger.warning(
                "Banned image hash matched — instant block",
                dhash=img_precheck.dhash,
                matched_hash=img_precheck.matched_banned_hash,
            )
            return ModerationResult(
                decision=ModerationDecision.FLAGGED,
                confidence_score=1.0,
                detected_violations=[
                    DetectedViolation(
                        category=ViolationCategory.EXPLICIT_CONTENT,
                        policy_code="MLC_E3.1",
                        severity=SeverityLevel.HIGH,
                        confidence=1.0,
                    )
                ],
                reasoning=ReasoningBlock(
                    text_explanation="No text violations detected.",
                    visual_explanation=(
                        f"Image perceptual hash ({img_precheck.dhash}) matched a "
                        "known-banned image in the hash database. Content blocked "
                        "without model inference."
                    ),
                    bimodal_synergy=(
                        "Banned image hash triggered an instant block. "
                        "Text content was not evaluated."
                    ),
                ),
                authenticity_report=AuthenticityReportSchema(
                    verdict="MALICIOUS_TAMPERING",
                    ai_involvement_percentage=100.0,
                    tamper_annotations=[
                        f"Image perceptual dHash ({img_precheck.dhash}) matched known-banned database."
                    ],
                ),
                pipeline_tier_used=PipelineTier.TIER1,
                total_latency_ms=round(elapsed_ms, 2),
                media_types_processed=[MediaType.IMAGE],
            )

    # ── Cache lookup ───────────────────────────────────────────────────────────

    cache_key = _compute_cache_key(text, image_bytes, audio_bytes, video_bytes)
    cached = await _cache_get(cache_key, redis_client)
    if cached:
        logger.info("Cache hit", key=cache_key[:16])
        return cached

    # ── Concurrent Media Forensics Analysis ────────────────────────────────────
    forensics_task = asyncio.create_task(
        run_full_forensics(
            text=text,
            image_bytes=image_bytes,
            audio_bytes=audio_bytes,
            video_bytes=video_bytes,
        )
    )

    # ── Determine present media types ──────────────────────────────────────────

    media_types: list[MediaType] = []
    if text:
        media_types.append(MediaType.TEXT)
    if image_bytes:
        media_types.append(MediaType.IMAGE)
    if audio_bytes:
        media_types.append(MediaType.AUDIO)
    if video_bytes:
        media_types.append(MediaType.VIDEO)

    # ── Concurrent media pre-processing ───────────────────────────────────────

    async def _process_image() -> Image.Image | None:
        if image_bytes:
            return await loop.run_in_executor(None, bytes_to_pil, image_bytes)
        return None

    async def _process_video() -> Image.Image | None:
        if video_bytes:
            frames = await loop.run_in_executor(
                None, extract_video_frames, video_bytes, 3
            )
            return frames[0] if frames else None
        return None

    async def _process_audio() -> str | None:
        if audio_bytes and registry.whisper_model is not None:
            return await loop.run_in_executor(
                None, transcribe_audio, audio_bytes, registry.whisper_model
            )
        return None

    async def _process_text() -> str | None:
        if text:
            return await loop.run_in_executor(None, preprocess_text, text)
        return None

    # Run all pre-processing concurrently
    image, video_frame, audio_transcript, clean_text = await asyncio.gather(
        _process_image(),
        _process_video(),
        _process_audio(),
        _process_text(),
    )

    # Compose final inputs — merge image and video frame (prefer video frame if available)
    primary_image = video_frame or image

    # Merge text sources — transcribed audio appended to original text
    combined_text_parts = [p for p in [clean_text, audio_transcript] if p]
    combined_text = " | ".join(combined_text_parts) if combined_text_parts else None

    if audio_transcript:
        logger.info("Audio transcribed", transcript_preview=audio_transcript[:100])

    # ── Tier-1 Gate ────────────────────────────────────────────────────────────
    tier1_result = await run_tier1_gate(
        text=combined_text,
        image=primary_image,
        registry=registry,
        media_types=media_types,
    )

    # ── Routing decision ───────────────────────────────────────────────────────
    tier_used = PipelineTier.TIER1

    if tier1_result.routed_to_tier2:
        # ── Tier-2 VLM Analysis ────────────────────────────────────────────────
        tier_used = PipelineTier.TIER2
        decision, confidence, violations, reasoning, _tier2_meta = await run_tier2_vlm(
            text=combined_text,
            image=primary_image,
            registry=registry,
        )
    else:
        # Short-circuit: build result from Tier-1 score alone
        decision, confidence, violations, reasoning = _build_tier1_result(
            tier1_result.safe_score
        )

    # ── Await concurrent forensics analysis ────────────────────────────────────
    forensics_report = await forensics_task
    authenticity_report = forensics_report.to_schema()

    # If malicious tampering detected (e.g. face-swap, ELA anomaly), escalate if safe
    if str(authenticity_report.verdict) == "MALICIOUS_TAMPERING":
        if decision == ModerationDecision.APPROVED:
            decision = ModerationDecision.FLAGGED
            confidence = max(confidence, 0.88)
        if not any(v.category == ViolationCategory.DEEPFAKE for v in violations):
            violations.append(
                DetectedViolation(
                    category=ViolationCategory.DEEPFAKE,
                    policy_code="MLC_D4.1",
                    severity=SeverityLevel.HIGH,
                    confidence=0.90,
                )
            )

    # ── Assemble final result ──────────────────────────────────────────────────
    total_latency_ms = (time.perf_counter() - pipeline_start) * 1000

    result = ModerationResult(
        decision=decision,
        confidence_score=confidence,
        detected_violations=violations,
        reasoning=reasoning,
        authenticity_report=authenticity_report,
        pipeline_tier_used=tier_used,
        total_latency_ms=round(total_latency_ms, 2),
        media_types_processed=media_types,
    )

    logger.info(
        "Pipeline complete",
        decision=decision.value if hasattr(decision, "value") else decision,
        confidence=f"{confidence:.3f}",
        tier=tier_used.value if hasattr(tier_used, "value") else tier_used,
        total_ms=f"{total_latency_ms:.1f}",
        violations=len(violations),
    )

    # ── Cache result ───────────────────────────────────────────────────────────
    await _cache_set(cache_key, result, redis_client)

    return result
