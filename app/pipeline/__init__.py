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
    _extract_json_payload,
    run_stage1_text_analysis,
    run_stage2_contextual_evaluation,
    run_hierarchical_contextual_forensics,
)

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
    "STAGE1_SYSTEM_PROMPT",
    "STAGE1_USER_PROMPT",
    "STAGE2_SYSTEM_PROMPT",
    "_build_stage2_user_prompt",
    "_extract_json_payload",
]
