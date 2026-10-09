"""
SafeSight AI — Hierarchical Contextual Forensics Pipeline
==========================================================
Advanced multi-stage pipeline designed for adversarial computer vision
and document forensics.

Problem Solved:
  AI-generated scenes mimicking real-world photographs of documents (e.g.
  synthetic handwriting inside a realistic notebook on a wooden desk).
  Naïve single-pass models score AI involvement at ~25% because the background
  paper and desk look photographic.

Solution (Bayesian Contextual Reasoning):
  1. Stage 1 (Textual Element Analysis): Isolates text / handwriting micro-features
     and yields a calibrated `synthetic_text_confidence` (0.0 – 1.0).
  2. ELA Forensics: Performs Error Level Analysis (95% JPEG re-save) to extract
     compression variance, boundary discrepancy, and synthetic glow artifacts.
  3. Stage 2 (Contextual Re-evaluation): Couples Stage 1 findings and ELA visual
     evidence into a specialized prompt that scrutinizes scene physics and
     enforces contextual proportionality (if the document text is synthetic,
     the probability that the entire photograph is a 3D/diffusion render escalates).
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
    _build_stage2_user_prompt,
    run_stage1_text_analysis,
    run_stage2_contextual_evaluation,
    run_hierarchical_contextual_forensics,
)

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
    "_extract_json_payload",
    "_build_stage2_user_prompt",
    "run_stage1_text_analysis",
    "run_stage2_contextual_evaluation",
    "run_hierarchical_contextual_forensics",
    "run_moderation_pipeline",
]
