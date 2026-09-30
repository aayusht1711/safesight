"""
SafeSight AI — Tier-2 VLM Explainable Reasoning Engine
========================================================
Routes borderline content to the quantized Qwen2.5-VL-3B-Instruct model
for deep bimodal analysis with structured JSON output.

Responsibilities:
  - Build a precise, chain-of-thought prompt embedding both text and image
  - Run the VLM with constrained output format
  - Parse and validate the structured JSON response
  - Fall back gracefully if the VLM output is malformed
  - Track per-request latency for SLO monitoring
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

try:
    import torch
except ImportError:
    torch = None
from PIL import Image

from app.config import get_settings
from app.logging_config import get_logger
from app.models.registry import ModelRegistry
from app.schemas import (
    DetectedViolation,
    ModerationDecision,
    ReasoningBlock,
    SeverityLevel,
    Tier2Result,
    ViolationCategory,
)

logger = get_logger("tier2_vlm")
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# VLM Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert Trust and Safety Auditor specialized in real-time, multimodal content moderation.
Your task is to analyze the provided image/video frame alongside the accompanying text.
You must output a strictly structured JSON response following the requested Pydantic schema.

---
### EVALUATION TAXONOMY:
- APPROVED: Content has no violation, or contains obvious satire, parody, or harmless venting.
- HUMAN_AUDIT_REQUIRED: Content has borderline language, sarcasm directed at non-protected individuals, or complex context that requires human intuition.
- FLAGGED: Content shows direct, explicit violation of platform rules (direct threats, slurs, explicit harm, severe hate speech).

---
### MODERATION CATEGORIES:
- HATE_SPEECH_AND_HARASSMENT: Hate speech, slurs, content targeting protected groups based on race, religion, gender, nationality, etc.
- SELF_HARM: Promotion of suicide, self-injury, or eating disorders — not clinical discussion.
- EXPLICIT_CONTENT: Sexual or nudity content not appropriate for general audiences.
- DEEPFAKE: AI-generated faces, voice cloning, synthetic media used to deceive.
- CYBERBULLYING: Targeted, systematic personal attacks, doxxing, or direct harassment of an individual.
- VIOLENT_THREATS: Explicit credible threats of violence or weapons glorification.
- MISINFORMATION: Dangerous false claims (medical, electoral, safety-critical).
- SPAM_AND_MANIPULATION: Coordinated inauthentic behavior or scam content.

---
### SPECIAL COMPLIANCE RULES:

1. THE SATIRE & HYPERBOLE RULE:
   - Do NOT flag content simply because it contains negative or aggressive-sounding words
     (e.g., "kill", "die", "poison", "trash", "destroy", "murder").
   - Analyze the VISUAL CONTEXT first: If a violent-sounding phrase is paired with a cartoon,
     a common household item, a funny face, food, animals, or a standard meme template,
     check for metaphorical or humorous intent.
   - If the intent is satirical, self-deprecating, hyperbolic, or clearly humorous,
     evaluate as "APPROVED".
   - If you are uncertain but the content leans harmless, classify as "APPROVED" or
     "HUMAN_AUDIT_REQUIRED" — NEVER "FLAGGED" for ambiguous hyperbole.
   - Examples of APPROVED hyperbole: "I could kill for a pizza right now", "This meeting
     is destroying my soul", "I'm dead 💀" (reaction to something funny).

2. THE DEFENSIVE CYBERBULLYING RULE:
   - Cyberbullying requires a TARGETED, SYSTEMATIC, or HIGHLY DIRECT attack against
     a specific, identifiable individual or protected group.
   - If the text/image represents: a generic online argument, an insult without a clear
     identifiable victim, passive-aggressive venting, or a single heated comment —
     do NOT auto-flag as CYBERBULLYING.
   - Classify these ambiguous, context-dependent cases as "HUMAN_AUDIT_REQUIRED" so a
     human moderator can make the final platform decision with full context.
   - Only classify as FLAGGED if the post contains: doxxing (real name/address/contact),
     a sustained harassment campaign, or explicit calls for others to target a person.

3. THE BIMODAL SYNERGY PRINCIPLE:
   - Always consider how text and image INTERACT — they can amplify OR diffuse harm.
   - A dangerous text + innocent image = possible ironic/satirical intent → lower severity.
   - An innocent text + dangerous image = visual content is the primary risk signal.
   - Dangerous text + dangerous image = strong signal → FLAGGED with HIGH confidence.

---
### MANDATORY JSON FORMAT:
Your response must contain ONLY a valid JSON object.
Do not include markdown code blocks, preamble, or any explanation outside the JSON keys.

{
  "decision": "APPROVED" | "FLAGGED" | "HUMAN_AUDIT_REQUIRED",
  "confidence_score": <float 0.0-1.0>,
  "detected_violations": [
    {
      "category": "<CATEGORY_FROM_LIST_ABOVE>",
      "policy_code": "<MLC_CODE>",
      "severity": "HIGH" | "MEDIUM" | "LOW",
      "confidence": <float 0.0-1.0>
    }
  ],
  "reasoning": {
    "text_explanation": "Explain your text analysis here — cite specific phrases.",
    "visual_explanation": "Explain your visual analysis here — describe specific visual elements.",
    "bimodal_synergy": "Explain how text and image combine (or conflict) to change the meaning and harm level."
  }
}
"""


def _build_user_prompt(text: str | None, has_image: bool) -> str:
    """Construct the user-turn prompt for the VLM."""
    parts = ["Analyze the following content for policy violations:"]

    if has_image:
        parts.append("\n[IMAGE]: See attached image above.")

    if text:
        parts.append(f"\n[TEXT CONTENT]: {text}")
    else:
        parts.append("\n[TEXT CONTENT]: (none provided)")

    parts.append(
        "\nProvide your moderation analysis as a single JSON object. "
        "Be precise about specific elements that triggered your assessment."
    )
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# JSON Response Parser
# ─────────────────────────────────────────────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict[str, Any] | None:
    """Extract and parse a JSON object from the VLM's raw text output."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first {...} block
    match = _JSON_BLOCK_RE.search(cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _parse_vlm_output(raw: str) -> tuple[
    ModerationDecision,
    float,
    list[DetectedViolation],
    ReasoningBlock,
    bool,
]:
    """
    Parse the VLM JSON output into validated Pydantic objects.

    Returns:
        (decision, confidence, violations, reasoning, parsed_successfully)
    """
    data = _extract_json(raw)

    if data is None:
        logger.warning("VLM output could not be parsed as JSON", raw_output=raw[:500])
        return (
            ModerationDecision.HUMAN_AUDIT_REQUIRED,
            0.50,
            [],
            ReasoningBlock(
                text_explanation="VLM output parsing failed.",
                visual_explanation="VLM output parsing failed.",
                bimodal_synergy="Manual review required due to parsing error.",
            ),
            False,
        )

    try:
        decision = ModerationDecision(data.get("decision", "HUMAN_AUDIT_REQUIRED"))
        confidence = float(data.get("confidence_score", 0.50))
        confidence = max(0.0, min(1.0, confidence))

        violations: list[DetectedViolation] = []
        for v in data.get("detected_violations", []):
            try:
                violations.append(
                    DetectedViolation(
                        category=ViolationCategory(v["category"]),
                        policy_code=v.get("policy_code", ""),
                        severity=SeverityLevel(v.get("severity", "MEDIUM")),
                        confidence=float(v.get("confidence", 1.0)),
                    )
                )
            except (ValueError, KeyError) as e:
                logger.warning("Skipping malformed violation entry", error=str(e), entry=v)

        reasoning_data = data.get("reasoning", {})
        reasoning = ReasoningBlock(
            text_explanation=reasoning_data.get("text_explanation", "No text analysis provided."),
            visual_explanation=reasoning_data.get("visual_explanation", "No visual analysis provided."),
            bimodal_synergy=reasoning_data.get("bimodal_synergy", "No bimodal analysis provided."),
        )

        return decision, confidence, violations, reasoning, True

    except Exception as exc:
        logger.error("Error constructing Pydantic objects from VLM output", error=str(exc))
        return (
            ModerationDecision.HUMAN_AUDIT_REQUIRED,
            0.50,
            [],
            ReasoningBlock(
                text_explanation=str(exc),
                visual_explanation="Parse error",
                bimodal_synergy="Human review required",
            ),
            False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# VLM Inference (blocking, runs in thread pool)
# ─────────────────────────────────────────────────────────────────────────────

def _run_vlm_inference(
    text: str | None,
    image: Image.Image | None,
    registry: ModelRegistry,
) -> str:
    """
    Synchronous VLM forward pass.
    Must be called via run_in_executor to avoid blocking the event loop.
    """
    if registry.vlm_model is None or registry.vlm_processor is None:
        raise RuntimeError("VLM model is not loaded.")

    processor = registry.vlm_processor
    model = registry.vlm_model

    # Build messages in Qwen2.5-VL chat format
    user_content: list[dict] = []

    if image is not None:
        user_content.append({"type": "image", "image": image})

    user_content.append({
        "type": "text",
        "text": _build_user_prompt(text, image is not None),
    })

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Apply chat template
    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Prepare inputs
    inputs = processor(
        text=[text_input],
        images=[image] if image is not None else None,
        padding=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    # Generate with constrained settings
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,         # Low temp for deterministic safety decisions
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (strip input prompt)
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[:, input_len:]
    raw_output = processor.batch_decode(generated, skip_special_tokens=True)[0]

    return raw_output


# ─────────────────────────────────────────────────────────────────────────────
# Public Tier-2 Entry Point
# ─────────────────────────────────────────────────────────────────────────────

async def run_tier2_vlm(
    text: str | None,
    image: Image.Image | None,
    registry: ModelRegistry,
) -> tuple[
    ModerationDecision,
    float,
    list[DetectedViolation],
    ReasoningBlock,
    Tier2Result,
]:
    """
    Asynchronously execute Tier-2 VLM deep analysis.

    Offloads the blocking GPU inference to a thread-pool executor so the
    event loop remains responsive to other concurrent requests.

    Returns:
        (decision, confidence, violations, reasoning, tier2_meta)
    """
    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()

    # Determine whether to use Ollama client or in-process HuggingFace VLM
    use_ollama = bool(settings.ollama_base_url) or (registry.vlm_model is None)

    if use_ollama:
        from app.utils.ollama_client import get_ollama_client
        ollama_client = get_ollama_client()
        user_prompt = _build_user_prompt(text, image is not None)
        _parsed, raw_output, _ = await ollama_client.generate_multimodal(
            prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT,
            image=image,
        )
    else:
        try:
            raw_output = await asyncio.wait_for(
                loop.run_in_executor(None, _run_vlm_inference, text, image, registry),
                timeout=settings.tier2_timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Tier-2 VLM timed out",
                timeout_ms=settings.tier2_timeout_ms,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return (
                ModerationDecision.HUMAN_AUDIT_REQUIRED,
                0.50,
                [],
                ReasoningBlock(
                    text_explanation="VLM analysis timed out.",
                    visual_explanation="VLM analysis timed out.",
                    bimodal_synergy="Escalated to human review due to timeout.",
                ),
                Tier2Result(
                    raw_vlm_output="TIMEOUT",
                    vlm_latency_ms=elapsed_ms,
                    parsed_successfully=False,
                ),
            )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    decision, confidence, violations, reasoning, parsed_ok = _parse_vlm_output(raw_output)

    logger.info(
        "Tier-2 VLM analysis complete",
        decision=decision.value,
        confidence=f"{confidence:.3f}",
        violations=len(violations),
        parsed=parsed_ok,
        latency_ms=f"{elapsed_ms:.1f}",
    )

    tier2_meta = Tier2Result(
        raw_vlm_output=raw_output,
        vlm_latency_ms=round(elapsed_ms, 2),
        parsed_successfully=parsed_ok,
    )

    return decision, confidence, violations, reasoning, tier2_meta
