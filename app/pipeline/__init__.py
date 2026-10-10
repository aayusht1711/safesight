"""
SafeSight AI — Pipeline Package
================================
Provides the master moderation orchestrator, Tier-1 gate, Tier-2 VLM,
and advanced hierarchical contextual forensics.
"""

from __future__ import annotations

from app.pipeline.orchestrator import run_moderation_pipeline
from app.pipeline.tier1_gate import run_tier1_gate
from app.pipeline.tier2_vlm import run_tier2_vlm
from app.pipeline.hierarchical import (
    STAGE1_SYSTEM_PROMPT,
    STAGE1_USER_PROMPT,
    STAGE2_SYSTEM_PROMPT,
    Stage1TextResult,
    Stage2ContextualResult,
    HierarchicalForensicsOutcome,
    _build_stage2_user_prompt,
    _build_stage2_system_prompt,
    _extract_json_payload,
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

__all__ = [
    "run_moderation_pipeline",
    "run_tier1_gate",
    "run_tier2_vlm",
    "run_hierarchical_contextual_forensics",
    "run_stage1_text_analysis",
    "run_stage2_contextual_evaluation",
    "HierarchicalForensicsOutcome",
    "Stage1TextResult",
    "Stage2ContextualResult",
    "DetectedTamperedRegion",
    "CompressionFingerprint",
    "ELAResult",
    "get_image_fingerprint",
    "perform_error_level_analysis",
    "perform_ela_async",
    "render_annotated_tampering_image",
    "STAGE1_SYSTEM_PROMPT",
    "STAGE1_USER_PROMPT",
    "STAGE2_SYSTEM_PROMPT",
    "_build_stage2_user_prompt",
    "_build_stage2_system_prompt",
    "_extract_json_payload",
]
