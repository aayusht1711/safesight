"""
SafeSight AI — Tier-1 Fast-Path Screening Gate
================================================
Implements the sub-50ms lightweight screening layer that determines
whether a post requires deep VLM analysis or can be immediately resolved.

Strategy (cascaded):
  1. Keyword/regex hard-block list — instant FLAGGED for known offensive patterns
  2. CLIP visual embedding similarity — compare image against safe/unsafe anchors
  3. DeBERTa toxicity quick score — fast text classification
  4. Combined score gating — route to Tier-2 or short-circuit

Output: Tier1Result with safe_score ∈ [0, 1]
  - safe_score → TIER1_CONFIDENCE_THRESHOLD : auto-APPROVED
  - safe_score < TIER1_SUSPICIOUS_THRESHOLD : auto-FLAGGED  (high-confidence harm)
  - between thresholds                       : routed to Tier-2
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

from app.config import get_settings
from app.logging_config import get_logger
from app.models.registry import ModelRegistry
from app.processors.keyword_patterns import keyword_hard_block
from app.schemas import MediaType, Tier1Result

logger = get_logger("tier1_gate")
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Safe-anchor text prompts for CLIP zero-shot similarity
# ─────────────────────────────────────────────────────────────────────────────

_SAFE_ANCHORS = [
    "a safe, family-friendly image",
    "normal everyday content",
    "a photo of a person smiling",
    "beautiful landscape photography",
]

_UNSAFE_ANCHORS = [
    "graphic violence or gore",
    "explicit sexual content",
    "hate symbols or extremist imagery",
    "a deepfake or manipulated face",
    "weapons being used violently",
]


# ─────────────────────────────────────────────────────────────────────────────
# Keyword Gate — delegates to keyword_patterns module (no ML deps)
# ─────────────────────────────────────────────────────────────────────────────

# keyword_hard_block is imported directly from app.processors.keyword_patterns


# ─────────────────────────────────────────────────────────────────────────────
# CLIP Visual Safety Score
# ─────────────────────────────────────────────────────────────────────────────

def _compute_clip_safety_score(
    image: Image.Image,
    registry: ModelRegistry,
) -> float:
    """
    Embed the image and compare cosine similarity against safe vs unsafe anchors.

    Returns a float ∈ [0, 1] where 1.0 = maximally safe.
    """
    try:
        import torch  # lazy import — only needed at inference time
    except ImportError:
        # PyTorch not installed — degrade gracefully with neutral score
        return 0.65

    if registry.clip_model is None:
        # CLIP not loaded — return neutral score
        return 0.65

    device = next(registry.clip_model.parameters()).device

    # Image embedding
    img_tensor = registry.clip_preprocess(image).unsqueeze(0).to(device)

    # Text embeddings for anchors
    safe_tokens = registry.clip_tokenizer(_SAFE_ANCHORS).to(device)
    unsafe_tokens = registry.clip_tokenizer(_UNSAFE_ANCHORS).to(device)

    with torch.no_grad():
        img_features = registry.clip_model.encode_image(img_tensor)
        safe_features = registry.clip_model.encode_text(safe_tokens)
        unsafe_features = registry.clip_model.encode_text(unsafe_tokens)

        # Normalise for cosine similarity
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        safe_features = safe_features / safe_features.norm(dim=-1, keepdim=True)
        unsafe_features = unsafe_features / unsafe_features.norm(dim=-1, keepdim=True)

        safe_sim = (img_features @ safe_features.T).mean().item()
        unsafe_sim = (img_features @ unsafe_features.T).mean().item()

    # Normalise to [0, 1] using softmax-like ratio
    safe_score = (safe_sim + 1.0) / 2.0
    unsafe_score = (unsafe_sim + 1.0) / 2.0
    total = safe_score + unsafe_score + 1e-8
    return float(safe_score / total)


# ─────────────────────────────────────────────────────────────────────────────
# DeBERTa Text Safety Score
# ─────────────────────────────────────────────────────────────────────────────

def _compute_text_safety_score(
    text: str,
    registry: ModelRegistry,
) -> float:
    """
    Run DeBERTa text-classification pipeline.
    Returns a float ∈ [0, 1] where 1.0 = safe.
    """
    if not text.strip() or registry.deberta_pipeline is None:
        return 0.70  # neutral fallback

    result = registry.deberta_pipeline(text[:512], truncation=True)[0]
    label: str = result["label"].upper()
    score: float = result["score"]

    # Model labels differ per checkpoint; normalise to safe_score
    if "NON_TOXIC" in label or "SAFE" in label or label == "LABEL_0":
        return score
    else:
        # TOXIC / OFFENSIVE / LABEL_1 → invert to get safe score
        return 1.0 - score


# ─────────────────────────────────────────────────────────────────────────────
# Public Tier-1 Entry Point
# ─────────────────────────────────────────────────────────────────────────────

async def run_tier1_gate(
    text: str | None,
    image: Image.Image | None,
    registry: ModelRegistry,
    media_types: list[MediaType],
) -> Tier1Result:
    """
    Asynchronously execute the Tier-1 fast-path screening gate.

    Runs keyword check (sync, fast) then concurrently runs CLIP and DeBERTa
    in thread-pool executors to avoid blocking the event loop.

    Args:
        text:        Extracted or provided text content.
        image:       PIL Image (may be None for audio/text-only posts).
        registry:    Loaded model registry.
        media_types: Which modalities are present in this request.

    Returns:
        Tier1Result containing safe_score and routing decision.
    """
    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()

    # ── Step 1: Hard-block keyword check (sync, <1ms) ─────────────────────────
    if text:
        blocked, trigger = keyword_hard_block(text)
        if blocked:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.warning("Tier-1 hard-block triggered", trigger=trigger)
            return Tier1Result(
                safe_score=0.0,
                routed_to_tier2=False,
                gate_latency_ms=elapsed_ms,
                reason=f"Hard-block keyword match: '{trigger}'",
            )

    # ── Step 2: Concurrent CLIP + DeBERTa scoring ─────────────────────────────
    async def _neutral() -> float:
        return 0.70

    tasks = []

    if image is not None and MediaType.IMAGE in media_types:
        tasks.append(
            loop.run_in_executor(
                None, _compute_clip_safety_score, image, registry
            )
        )
    else:
        tasks.append(_neutral())  # neutral visual score

    if text:
        tasks.append(
            loop.run_in_executor(
                None, _compute_text_safety_score, text, registry
            )
        )
    else:
        tasks.append(_neutral())  # neutral text score

    visual_score, text_score = await asyncio.gather(*tasks)

    # ── Step 3: Weighted fusion ────────────────────────────────────────────────
    if image is not None and text:
        # Both modalities — give slight edge to visual (harder to fake)
        combined_score = 0.55 * visual_score + 0.45 * text_score
    elif image is not None:
        combined_score = visual_score
    else:
        combined_score = text_score

    # ── Step 4: Routing decision ───────────────────────────────────────────────
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if combined_score >= settings.tier1_confidence_threshold:
        routed = False
        reason = (
            f"Auto-approved: combined safety score {combined_score:.3f} "
            f"≥ threshold {settings.tier1_confidence_threshold}"
        )
    elif combined_score < settings.tier1_suspicious_threshold:
        routed = False  # auto-flag, no need for VLM
        reason = (
            f"Auto-flagged: combined safety score {combined_score:.3f} "
            f"< suspicious threshold {settings.tier1_suspicious_threshold}"
        )
    else:
        routed = True
        reason = (
            f"Borderline score {combined_score:.3f} — routing to Tier-2 VLM analysis"
        )

    logger.info(
        "Tier-1 gate complete",
        visual_score=f"{visual_score:.3f}",
        text_score=f"{text_score:.3f}",
        combined=f"{combined_score:.3f}",
        routed_to_tier2=routed,
        latency_ms=f"{elapsed_ms:.1f}",
    )

    return Tier1Result(
        safe_score=round(combined_score, 4),
        routed_to_tier2=routed,
        gate_latency_ms=round(elapsed_ms, 2),
        reason=reason,
    )
