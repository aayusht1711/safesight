"""
SafeSight AI — Mock API Server (No GPU / No ML Required)
=========================================================
A fully functional FastAPI server that uses the deterministic mock pipeline
instead of real ML models. Starts in under 2 seconds.

Usage:
    python run_mock_server.py

Endpoints:
    POST /moderate/text          → moderate text only
    POST /moderate/multimodal    → text + image (image hash checked)
    GET  /health                 → health check
    GET  /docs                   → Swagger UI
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ── Force UTF-8 output on Windows ────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.utils.preprocessor import preprocess_adversarial_text, preprocess_image
from app.utils.forensics_detector import run_full_forensics
from tests.mock_pipeline import mock_moderate
from app.schemas import ModerationResult, AuthenticityReportSchema, AuthenticityVerdict

# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SafeSight AI — Mock Server",
    description=(
        "Real-Time Multimodal Content Moderation API\n\n"
        "**Mock mode** — uses deterministic rule-based pipeline. "
        "No GPU or ML models required.\n\n"
        "The adversarial preprocessor IS active: leet-speak, Unicode homoglyphs, "
        "and spaced-text evasion are all defeated before evaluation."
    ),
    version="1.0.0-mock",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Server health check — always returns OK in mock mode."""
    return {
        "status": "ok",
        "mode": "mock",
        "message": "SafeSight AI mock server is running. No ML models loaded.",
        "pipeline": {
            "tier1": "keyword-regex (mock)",
            "tier2": "skipped (mock mode)",
            "preprocessor": "active",
            "cache": "InMemoryCache",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /moderate/text
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/moderate/text",
    response_model=ModerationResult,
    tags=["Moderation"],
    summary="Moderate text content",
    description=(
        "Submit a text string for content moderation.\n\n"
        "**Adversarial preprocessing is applied automatically:**\n"
        "- Leet-speak: `p0is0n` → `poison`\n"
        "- Spaced text: `h a t e` → `hate`\n"
        "- Unicode homoglyphs: Cyrillic chars normalized\n"
        "- Zero-width characters stripped\n"
    ),
)
async def moderate_text(payload: dict):
    """
    Moderate a text string.

    Request body:
        {"text": "your text here"}
    """
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="'text' field is required and cannot be empty.")

    # ── Adversarial preprocessing ──────────────────────────────────────────────
    original_text = text
    cleaned_text = preprocess_adversarial_text(text)

    # ── Concurrent Forensics & Mock pipeline ──────────────────────────────────
    forensics_task = run_full_forensics(text=cleaned_text)
    result = mock_moderate(text=cleaned_text, simulate_latency=True)
    report = await forensics_task
    result.authenticity_report = report.to_schema()

    # Attach preprocessing info to reasoning if text was modified
    if cleaned_text != original_text:
        result.reasoning.text_explanation = (
            f"[Adversarial preprocessing applied: '{original_text[:60]}' → '{cleaned_text[:60]}']\n"
            + result.reasoning.text_explanation
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# POST /moderate/multimodal
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/moderate/multimodal",
    response_model=ModerationResult,
    tags=["Moderation"],
    summary="Moderate text + image",
    description=(
        "Submit text and/or an image for multimodal content moderation.\n\n"
        "Image is fingerprinted with dHash. If the hash matches a known-banned "
        "image, it is instantly blocked without any model inference.\n\n"
        "**Form fields:** `text` (optional), `image` (optional file upload)"
    ),
)
async def moderate_multimodal(
    text: str | None = Form(default=None),
    image: UploadFile | None = File(default=None),
):
    """
    Moderate text + image together.
    At least one of text or image must be provided.
    """
    if not text and not image:
        raise HTTPException(
            status_code=422,
            detail="At least one of 'text' or 'image' must be provided."
        )

    image_bytes: bytes | None = None
    img_dhash: str | None = None
    image_ban_note: str | None = None

    # ── Image preprocessing ────────────────────────────────────────────────────
    if image:
        image_bytes = await image.read()
        img_result = preprocess_image(image_bytes)
        img_dhash = img_result.dhash

        if img_result.is_banned:
            from app.schemas import (
                ModerationDecision, DetectedViolation, ViolationCategory,
                SeverityLevel, ReasoningBlock, PipelineTier, MediaType,
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
                    text_explanation="Image was blocked before text analysis.",
                    visual_explanation=(
                        f"Image hash ({img_dhash}) matched a known-banned "
                        f"hash ({img_result.matched_banned_hash}) in the database. "
                        "Content blocked without model inference."
                    ),
                    bimodal_synergy="Banned image triggered an instant block.",
                ),
                authenticity_report=AuthenticityReportSchema(
                    verdict=AuthenticityVerdict.MALICIOUS_TAMPERING,
                    ai_involvement_percentage=100.0,
                    tamper_annotations=[f"Image hash {img_dhash} matches banned database."],
                ),
                pipeline_tier_used=PipelineTier.TIER1,
                total_latency_ms=1.0,
                media_types_processed=[MediaType.IMAGE],
            )

        image_ban_note = (
            f"Image hash: {img_dhash} — not in banned database, proceeding."
        )

    # ── Text preprocessing ─────────────────────────────────────────────────────
    cleaned_text: str | None = None
    if text:
        cleaned_text = preprocess_adversarial_text(text.strip())

    # ── Concurrent Forensics & Mock pipeline ──────────────────────────────────
    forensics_task = run_full_forensics(text=cleaned_text, image_bytes=image_bytes)
    result = mock_moderate(
        text=cleaned_text,
        image_path=image.filename if image else None,
        simulate_latency=True,
    )
    report = await forensics_task
    result.authenticity_report = report.to_schema()

    if image_ban_note:
        result.reasoning.visual_explanation = (
            image_ban_note + "\n" + result.reasoning.visual_explanation
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# POST /moderate/batch  (bonus endpoint)
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/moderate/batch",
    tags=["Moderation"],
    summary="Moderate a batch of text items",
    description="Submit up to 20 text strings at once and get a decision for each.",
)
async def moderate_batch(payload: dict):
    """
    Request body:
        {"items": ["text 1", "text 2", ...]}
    Returns a list of ModerationResult objects.
    """
    items: list[str] = payload.get("items", [])
    if not items:
        raise HTTPException(status_code=422, detail="'items' list cannot be empty.")
    if len(items) > 20:
        raise HTTPException(status_code=422, detail="Maximum 20 items per batch request.")

    results = []
    for item in items:
        cleaned = preprocess_adversarial_text(item.strip())
        result = mock_moderate(text=cleaned, simulate_latency=False)
        results.append(result.model_dump())

    return {"count": len(results), "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  SafeSight AI — Mock API Server")
    print("  No GPU or ML models required")
    print("═" * 60)
    print("  API:    http://127.0.0.1:8000")
    print("  Docs:   http://127.0.0.1:8000/docs")
    print("  Health: http://127.0.0.1:8000/health")
    print("═" * 60 + "\n")

    uvicorn.run(
        "run_mock_server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
