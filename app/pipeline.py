"""
SafeSight AI — Hierarchical Contextual Forensics Pipeline
==========================================================
Advanced multi-stage pipeline designed for adversarial computer vision,
document forensics, and compression-aware provenance verification.

Key Capabilities:
  1. Provenance Fingerprinting (`get_image_fingerprint`): Analyzes EXIF metadata
     and JPEG Quantization Tables (DQT) to fingerprint source engines (e.g. WhatsApp,
     Facebook, Original Camera).
  2. Differential Error Level Analysis (`perform_error_level_analysis`): Employs
     dual-pass (95% vs 75%) ELA to distinguish benign social-media compression artifacts
     from localized digital splices and generative AI insertions.
  3. Contextual Prompt Generation (`_build_stage2_system_prompt`, `_build_stage2_user_prompt`):
     Dynamically tunes the VLM system prompt based on the image's digital provenance.
  4. Localized Tampering Grounding: Identifies and highlights tampered regions with normalized
     bounding boxes and visual annotation badges.
"""

from __future__ import annotations

from app.pipeline.hierarchical import (
    STAGE1_SYSTEM_PROMPT,
    STAGE1_USER_PROMPT,
    STAGE2_SYSTEM_PROMPT,
    Stage1TextResult,
    Stage2ContextualResult,
    HierarchicalForensicsOutcome,
    _extract_json_payload,
    _build_stage2_system_prompt,
    _build_stage2_user_prompt,
    run_stage1_text_analysis,
    run_stage2_contextual_evaluation,
    run_hierarchical_contextual_forensics,
)

from app.schemas import DetectedTamperedRegion
from app.utils.compression_fingerprint import (
    CompressionFingerprint,
    get_image_fingerprint,
)
from app.utils.ela_detector import (
    ELAResult,
    perform_error_level_analysis,
    perform_ela_async,
)
from app.utils.visualizer import render_annotated_tampering_image


async def run_moderation_pipeline(*args, **kwargs):
    """Forwarding delegate to app.pipeline.orchestrator.run_moderation_pipeline."""
    from app.pipeline.orchestrator import run_moderation_pipeline as _orch_run
    return await _orch_run(*args, **kwargs)


__all__ = [
    "STAGE1_SYSTEM_PROMPT",
    "STAGE1_USER_PROMPT",
    "STAGE2_SYSTEM_PROMPT",
    "Stage1TextResult",
    "Stage2ContextualResult",
    "HierarchicalForensicsOutcome",
    "DetectedTamperedRegion",
    "CompressionFingerprint",
    "ELAResult",
    "get_image_fingerprint",
    "perform_error_level_analysis",
    "perform_ela_async",
    "render_annotated_tampering_image",
    "_extract_json_payload",
    "_build_stage2_system_prompt",
    "_build_stage2_user_prompt",
    "run_stage1_text_analysis",
    "run_stage2_contextual_evaluation",
    "run_hierarchical_contextual_forensics",
    "run_moderation_pipeline",
]
