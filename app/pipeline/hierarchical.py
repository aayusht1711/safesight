"""
SafeSight AI — Hierarchical Contextual Forensics Engine
========================================================
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

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from app.config import get_settings
from app.logging_config import get_logger
from app.models.registry import ModelRegistry
from app.schemas import (
    AuthenticityReportSchema,
    AuthenticityVerdict,
    DetectedViolation,
    MediaType,
    ModerationDecision,
    ModerationResult,
    PipelineTier,
    ReasoningBlock,
    SeverityLevel,
    ViolationCategory,
)
from app.utils.ela_detector import ELAResult, perform_ela_async
from app.utils.ollama_client import get_ollama_client

logger = get_logger("hierarchical_pipeline")
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Specialized Prompts: Stage 1 & Stage 2
# ─────────────────────────────────────────────────────────────────────────────

STAGE1_SYSTEM_PROMPT = """You are an expert Forensic Document Examiner and Computational Typographer.
Your job is to analyze ONLY the textual, typographical, and handwritten elements in the provided image.
Ignore the background, lighting, and scenery. Focus exclusively on the text characters, strokes, and ink.

### FORENSIC EVALUATION CRITERIA:
1. Authentic Handwriting:
   - Natural human motor micro-tremors, uneven pressure gradients (darker where pen pressed harder).
   - Natural ink bleeding into paper fibers, inconsistent letter sizing and baseline drift.
   - Never has identical duplicate letter glyphs (e.g., every 'e' or 't' has natural variations).

2. Synthetic / AI / Digitally Overlaid Text:
   - Unnaturally smooth vector strokes, identical repeated letter glyphs, perfect baseline alignment.
   - Letters that fail to interact with paper texture or creases (floating text).
   - Diffusion artifacts: warped or melted letterforms, nonsensical ligatures, or font inconsistencies.

### OUTPUT FORMAT:
You MUST respond with ONLY a strictly valid JSON object:
{
  "synthetic_text_confidence": <float between 0.0 and 1.0>,
  "text_classification": "AI_SYNTHETIC_TEXT" | "DIGITALLY_OVERLAID" | "AUTHENTIC_HANDWRITING" | "PRINTED_DOCUMENT" | "NO_TEXT",
  "handwriting_variations_detected": <true if authentic motor variation is present, false otherwise>,
  "observations": "Detailed forensic notes on stroke geometry, glyph repetition, and ink-paper interaction."
}
"""

STAGE1_USER_PROMPT = """Analyze only the textual and handwritten elements in this image.
Is the text computer-generated, digitally overlaid, or does it show the natural variations of real handwriting?
Provide a 'synthetic_text_confidence' score from 0.0 to 1.0 along with your specific visual observations.
Output JSON only.
"""

STAGE2_SYSTEM_PROMPT = """You are an expert digital forensics analyst. You understand that AI models can create entire scenes that look real at first glance.
Pay close attention to unnatural uniformity. Real-world objects have imperfections. Look for handwriting that is too consistent, lines that are too straight, and lighting that is too perfect.
If the text is synthetic, there is a high probability the entire scene is a render. Scrutinize the background for supporting evidence of this.

### CORE CONTEXTUAL FORENSIC PRINCIPLES:
1. BAYESIAN CONTEXTUAL COUPLING (DOCUMENT HOST PRINCIPLE):
   - In real-world forensic photography, authentic paper documents do NOT naturally host synthetic AI handwriting or digital overlays unless the document was fabricated.
   - If Stage 1 determined that the text is synthetic ('synthetic_text_confidence' >= 0.65), you must re-evaluate the entire surrounding environment (notebook lines, paper texture, table surface, shadows, and lighting).
   - Do NOT treat the background as an authentic photo simply because it looks plausible in isolation. If the core subject (the written document) is synthetic, the entire scene is likely an end-to-end diffusion synthesis or composite render.

2. EVIDENCE INTEGRATION (ERROR LEVEL ANALYSIS - ELA):
   - You are provided with Error Level Analysis (ELA) quantitative metrics and visual map.
   - In authentic photos, compression error across paper and background is consistent and low.
   - In AI-generated images or digital compositions, high-error glows or sharp discrepancies between text and paper fibers betray artificial synthesis.

3. PROPORTIONAL AI INVOLVEMENT:
   - When a notebook, document, or memo contains synthetic text and was generated as a whole scene, 'ai_involvement_percentage' must NOT be artificially diluted (e.g., 20-30%).
   - Rate 'ai_involvement_percentage' between 85.0% and 100.0% if the focal scene is an AI generation.

### OUTPUT FORMAT:
Output ONLY a strictly valid JSON object:
{
  "overall_authenticity_verdict": "SYNTHETIC_GENERATED" | "MALICIOUS_TAMPERING" | "HYBRID_HUMAN_AI_EDITED" | "AUTHENTIC",
  "ai_involvement_percentage": <float 0.0 to 100.0>,
  "decision": "APPROVED" | "FLAGGED" | "HUMAN_AUDIT_REQUIRED",
  "confidence_score": <float 0.0 to 1.0>,
  "scene_physics_assessment": "Detailed evaluation of paper texture, lighting, shadow angles, and rendering artifacts.",
  "explanation": "Final comprehensive forensic synthesis connecting Stage 1 text analysis, background physics, and ELA signals.",
  "detected_violations": [
    {
      "category": "DEEPFAKE" | "SPAM_AND_MANIPULATION" | "MISINFORMATION",
      "policy_code": "MLC_D4.1",
      "severity": "HIGH" | "MEDIUM" | "LOW",
      "confidence": <float 0.0 to 1.0>
    }
  ],
  "reasoning": {
    "text_explanation": "Stage 1 textual authenticity summary and policy assessment.",
    "visual_explanation": "Stage 2 scene physics and background texture evaluation.",
    "bimodal_synergy": "Contextual fusion explaining how synthetic text combined with background ELA confirms the overall verdict."
  }
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# 2. Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Stage1TextResult:
    """Outcome of Stage 1 textual element analysis."""
    synthetic_text_confidence: float
    text_classification: str
    handwriting_variations_detected: bool
    observations: str
    latency_ms: float = 0.0


@dataclass
class Stage2ContextualResult:
    """Outcome of Stage 2 contextual re-evaluation."""
    overall_authenticity_verdict: AuthenticityVerdict
    ai_involvement_percentage: float
    decision: ModerationDecision
    confidence_score: float
    scene_physics_assessment: str
    explanation: str
    detected_violations: list[DetectedViolation]
    reasoning: ReasoningBlock
    latency_ms: float = 0.0


@dataclass
class HierarchicalForensicsOutcome:
    """Complete multi-stage forensic analysis report."""
    stage1: Stage1TextResult
    stage2: Stage2ContextualResult
    ela_result: ELAResult
    moderation_result: ModerationResult
    total_latency_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. JSON Parser Helper
# ─────────────────────────────────────────────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_payload(raw_text: str) -> dict[str, Any] | None:
    """Robustly parse JSON object from VLM raw string response."""
    if not raw_text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stage 1: Textual Element Analysis
# ─────────────────────────────────────────────────────────────────────────────

async def run_stage1_text_analysis(
    image: Image.Image,
    registry: ModelRegistry | None = None,
) -> Stage1TextResult:
    """
    Stage 1: Focuses exclusively on the rendered text / handwriting.
    Determines whether text is computer-generated, digitally overlaid,
    or possesses genuine human handwriting variation.
    """
    t0 = time.perf_counter()
    ollama_client = get_ollama_client()

    parsed_json, raw_resp, success = await ollama_client.generate_multimodal(
        prompt=STAGE1_USER_PROMPT,
        system_prompt=STAGE1_SYSTEM_PROMPT,
        image=image,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if success and parsed_json:
        synth_conf = float(parsed_json.get("synthetic_text_confidence", 0.5))
        synth_conf = max(0.0, min(1.0, synth_conf))
        classification = str(parsed_json.get("text_classification", "UNKNOWN"))
        has_variation = bool(parsed_json.get("handwriting_variations_detected", False))
        obs = str(parsed_json.get("observations", "No detailed observations provided."))
        return Stage1TextResult(
            synthetic_text_confidence=synth_conf,
            text_classification=classification,
            handwriting_variations_detected=has_variation,
            observations=obs,
            latency_ms=elapsed_ms,
        )

    # Fallback if VLM unavailable or unparseable
    logger.warning("Stage 1 VLM parsing failed, utilizing default fallback heuristics")
    return Stage1TextResult(
        synthetic_text_confidence=0.50,
        text_classification="AMBIGUOUS",
        handwriting_variations_detected=False,
        observations="Automated Stage 1 typography analysis unavailable; routed to contextual gate.",
        latency_ms=elapsed_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Stage 2: Contextual Re-evaluation & ELA Fusion
# ─────────────────────────────────────────────────────────────────────────────

def _build_stage2_user_prompt(
    stage1_result: Stage1TextResult,
    ela_result: ELAResult,
    accompanying_text: str | None = None,
) -> str:
    """Construct the Stage 2 user prompt containing Stage 1 findings & ELA evidence."""
    return f"""Given that the text in this image has a 'synthetic_text_confidence' of {stage1_result.synthetic_text_confidence:.2f} (Classification: {stage1_result.text_classification}; Notes: {stage1_result.observations}), and Error Level Analysis (ELA) exhibits:
- ELA Variance: {ela_result.variance:.1f}
- Text-to-Background Compression Discrepancy: {ela_result.text_vs_background_discrepancy:.2f}
- ELA Forensic Summary: {ela_result.summary_description}
- Accompanying Caption: {accompanying_text or '(none provided)'}

Re-evaluate the entire scene. Do the background elements like paper texture, lighting, and object physics appear consistent with a real photograph, or do they show the subtle uniformity and perfection of a 3D render?
Output your final 'overall_authenticity_verdict', 'ai_involvement_percentage', and 'explanation' as JSON.
"""


async def run_stage2_contextual_evaluation(
    image: Image.Image,
    stage1_result: Stage1TextResult,
    ela_result: ELAResult,
    accompanying_text: str | None = None,
    registry: ModelRegistry | None = None,
) -> Stage2ContextualResult:
    """
    Stage 2: Re-evaluates entire scene given Stage 1 typography assessment
    and ELA compression metrics.
    """
    t0 = time.perf_counter()
    ollama_client = get_ollama_client()

    stage2_prompt = _build_stage2_user_prompt(
        stage1_result=stage1_result,
        ela_result=ela_result,
        accompanying_text=accompanying_text,
    )

    # Pass both the original image and the visual ELA map as evidence
    images_payload = [image, ela_result.ela_image]

    parsed_json, raw_resp, success = await ollama_client.generate_multimodal(
        prompt=stage2_prompt,
        system_prompt=STAGE2_SYSTEM_PROMPT,
        image=images_payload,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if success and parsed_json:
        raw_verdict = str(parsed_json.get("overall_authenticity_verdict", "SYNTHETIC_GENERATED")).upper()
        try:
            verdict = AuthenticityVerdict(raw_verdict)
        except ValueError:
            verdict = AuthenticityVerdict.SYNTHETIC_GENERATED if stage1_result.synthetic_text_confidence > 0.6 else AuthenticityVerdict.AUTHENTIC

        ai_pct = float(parsed_json.get("ai_involvement_percentage", 0.0))

        # Contextual Enforcement: If text is synthetic (>=0.70), elevate AI involvement
        if stage1_result.synthetic_text_confidence >= 0.70 and ai_pct < 75.0:
            logger.info("Contextual escalation applied: synthetic text anchors full scene involvement")
            ai_pct = max(ai_pct, 85.0 + (stage1_result.synthetic_text_confidence - 0.70) * 40.0)
            ai_pct = min(100.0, ai_pct)
            if verdict == AuthenticityVerdict.AUTHENTIC:
                verdict = AuthenticityVerdict.SYNTHETIC_GENERATED

        raw_decision = str(parsed_json.get("decision", "APPROVED")).upper()
        try:
            decision = ModerationDecision(raw_decision)
        except ValueError:
            decision = ModerationDecision.HUMAN_AUDIT_REQUIRED

        conf = float(parsed_json.get("confidence_score", 0.85))
        physics_notes = str(parsed_json.get("scene_physics_assessment", "Scene physics evaluated."))
        explanation = str(parsed_json.get("explanation", "Contextual forensics evaluation complete."))

        violations: list[DetectedViolation] = []
        for v in parsed_json.get("detected_violations", []):
            try:
                violations.append(
                    DetectedViolation(
                        category=ViolationCategory(v["category"]),
                        policy_code=v.get("policy_code", "MLC_D4.1"),
                        severity=SeverityLevel(v.get("severity", "HIGH")),
                        confidence=float(v.get("confidence", 0.9)),
                    )
                )
            except Exception:
                pass

        reasoning_dict = parsed_json.get("reasoning", {})
        reasoning = ReasoningBlock(
            text_explanation=reasoning_dict.get("text_explanation", stage1_result.observations),
            visual_explanation=reasoning_dict.get("visual_explanation", physics_notes),
            bimodal_synergy=reasoning_dict.get("bimodal_synergy", explanation),
        )

        return Stage2ContextualResult(
            overall_authenticity_verdict=verdict,
            ai_involvement_percentage=round(ai_pct, 1),
            decision=decision,
            confidence_score=round(conf, 3),
            scene_physics_assessment=physics_notes,
            explanation=explanation,
            detected_violations=violations,
            reasoning=reasoning,
            latency_ms=elapsed_ms,
        )

    # Deterministic rule-based fallback if VLM call fails
    is_synthetic = (
        stage1_result.synthetic_text_confidence >= 0.65
        or ela_result.is_synthetic_suspicious
        or ela_result.text_vs_background_discrepancy > 1.2
    )

    if is_synthetic:
        fallback_verdict = AuthenticityVerdict.SYNTHETIC_GENERATED
        fallback_ai_pct = max(85.0, stage1_result.synthetic_text_confidence * 100.0)
        fallback_decision = ModerationDecision.HUMAN_AUDIT_REQUIRED
    else:
        fallback_verdict = AuthenticityVerdict.AUTHENTIC
        fallback_ai_pct = 0.0
        fallback_decision = ModerationDecision.APPROVED

    return Stage2ContextualResult(
        overall_authenticity_verdict=fallback_verdict,
        ai_involvement_percentage=round(fallback_ai_pct, 1),
        decision=fallback_decision,
        confidence_score=0.80,
        scene_physics_assessment="Rule-based contextual estimation based on Stage 1 and ELA.",
        explanation=(
            f"Contextual analysis determined {fallback_verdict.value} based on "
            f"synthetic text score {stage1_result.synthetic_text_confidence:.2f} "
            f"and ELA variance {ela_result.variance:.1f}."
        ),
        detected_violations=[
            DetectedViolation(
                category=ViolationCategory.DEEPFAKE,
                policy_code="MLC_D4.1",
                severity=SeverityLevel.HIGH,
                confidence=0.85,
            )
        ] if is_synthetic else [],
        reasoning=ReasoningBlock(
            text_explanation=stage1_result.observations,
            visual_explanation=ela_result.summary_description,
            bimodal_synergy="Contextual coupling applied: synthetic typography escalates scene likelihood.",
        ),
        latency_ms=elapsed_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Master Hierarchical Pipeline Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

async def run_hierarchical_contextual_forensics(
    image: Image.Image,
    text: str | None = None,
    registry: ModelRegistry | None = None,
) -> HierarchicalForensicsOutcome:
    """
    Execute the full two-stage hierarchical prompt chain with ELA evidence coupling.

    Stage 1: Textual Element Analysis
    Parallel: ELA Error Level Analysis computation
    Stage 2: Contextual Re-evaluation & Scene Physics
    """
    pipeline_t0 = time.perf_counter()

    # Step 1: Run Stage 1 text analysis and ELA computation concurrently
    stage1_task = run_stage1_text_analysis(image, registry)
    ela_task = perform_ela_async(image, quality=95)

    stage1_res, ela_res = await asyncio.gather(stage1_task, ela_task)

    # Step 2: Feed Stage 1 output + ELA map into Stage 2 Contextual Re-evaluation
    stage2_res = await run_stage2_contextual_evaluation(
        image=image,
        stage1_result=stage1_res,
        ela_result=ela_res,
        accompanying_text=text,
        registry=registry,
    )

    total_latency = (time.perf_counter() - pipeline_t0) * 1000

    # Build standard ModerationResult
    annotations = [
        f"[Stage 1] Typography analysis: {stage1_res.observations}",
        f"[ELA] Compression profile: {ela_res.summary_description}",
        f"[Stage 2] Scene physics verdict: {stage2_res.explanation}",
    ]

    auth_report = AuthenticityReportSchema(
        verdict=stage2_res.overall_authenticity_verdict,
        ai_involvement_percentage=stage2_res.ai_involvement_percentage,
        tamper_annotations=annotations,
    )

    mod_result = ModerationResult(
        decision=stage2_res.decision,
        confidence_score=stage2_res.confidence_score,
        detected_violations=stage2_res.detected_violations,
        reasoning=stage2_res.reasoning,
        authenticity_report=auth_report,
        pipeline_tier_used=PipelineTier.TIER2,
        total_latency_ms=round(total_latency, 2),
        media_types_processed=[MediaType.IMAGE] + ([MediaType.TEXT] if text else []),
    )

    return HierarchicalForensicsOutcome(
        stage1=stage1_res,
        stage2=stage2_res,
        ela_result=ela_res,
        moderation_result=mod_result,
        total_latency_ms=total_latency,
    )
