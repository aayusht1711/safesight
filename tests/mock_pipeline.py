"""
SafeSight AI — Mock Pipeline for Testing
==========================================
Provides a fully deterministic, in-process mock pipeline that does NOT
require any ML models to be installed or GPU to be available.

Used exclusively by the test suite to:
  - Validate JSON schema correctness
  - Test routing logic and decision thresholds
  - Measure API latency without ML inference overhead
  - Run in CI/CD environments with no GPU

The mock engine implements the same interface as the real orchestrator
and returns responses that exercise all decision branches.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from app.schemas import (
    AuthenticityReportSchema,
    AuthenticityVerdict,
    DetectedTamperedRegion,
    DetectedViolation,
    MediaType,
    ModerationDecision,
    ModerationResult,
    PipelineTier,
    ReasoningBlock,
    SeverityLevel,
    ViolationCategory,
)


# ─────────────────────────────────────────────────────────────────────────────
# Mock test case definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MockTestCase:
    """A single mock test case for the validation suite."""

    case_id: str
    description: str
    text: str
    image_path: str | None
    expected_decision: ModerationDecision
    expected_violations: list[ViolationCategory]
    expected_forensic_verdict: AuthenticityVerdict = AuthenticityVerdict.AUTHENTIC



# ─────────────────────────────────────────────────────────────────────────────
# Keyword-based mock decision logic
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Keyword-based mock decision logic
# Mirrors the two Special Compliance Rules in the VLM system prompt:
#   1. Satire & Hyperbole Rule  →  HYPERBOLE_SAFE_PATTERNS allowlist
#   2. Defensive Cyberbullying Rule → stricter CYBERBULLYING pattern
# ─────────────────────────────────────────────────────────────────────────────

# Safe-hyperbole allowlist — these patterns indicate figurative/humorous intent.
# If ANY safe pattern matches, the post is APPROVED regardless of toxic matches
# (implements the Satire & Hyperbole Rule for the deterministic mock).
_HYPERBOLE_SAFE_PATTERNS: list[re.Pattern] = [
    # Figurative expressions of frustration / enthusiasm
    re.compile(r"\bi\s+could\s+kill\s+for\b", re.IGNORECASE),
    re.compile(r"\bthis\s+(is\s+)?(absolutely\s+)?(killing|destroying|murdering)\s+(me|my\s+\w+)\b", re.IGNORECASE),
    re.compile(r"\bi('m|\s+am)\s+dead\s*(💀|😂|lol|haha)?\b", re.IGNORECASE),
    re.compile(r"\b(hate|hated)\s+(mondays|traffic|mornings|mondays|meetings|homework|this\s+song)\b", re.IGNORECASE),
    # Journalism / news / reporting context
    re.compile(r"\b(politician|politician'?s?|government|spokesperson|spokesperson'?s?)\s+speech\b", re.IGNORECASE),
    re.compile(r"\b(absolute(ly)?|filled\s+with)\s+(hate.?filled|propaganda)\b", re.IGNORECASE),
    # Meme / internet culture markers
    re.compile(r"\b(im|i'm|iam)\s+crying\b", re.IGNORECASE),
    re.compile(r"\b(this|that)\s+is\s+(literally\s+)?killing\s+me\b", re.IGNORECASE),
]

_TOXIC_PATTERNS = {
    ViolationCategory.HATE_SPEECH_AND_HARASSMENT: re.compile(
        # Require explicit slurs or direct hate targeting — not generic "hate"
        r"\b(kill\s+all|inferior race|death\s+to|slur|exterminate\s+(all|the))\b",
        re.IGNORECASE,
    ),
    ViolationCategory.SELF_HARM: re.compile(
        # Self-directed harm: suicide methods, poison, self-injury, "kill yourself"
        r"\b(suicide|self.?harm|end\s+my\s+life|cutting|overdose"
        r"|kill\s+yourself|poison|drink\s+poison|slit|hang\s+(my|your)self)\b",
        re.IGNORECASE,
    ),
    ViolationCategory.EXPLICIT_CONTENT: re.compile(
        r"\b(nude|naked|explicit|pornographic|sexual\s+content)\b", re.IGNORECASE
    ),
    ViolationCategory.DEEPFAKE: re.compile(
        r"\b(deepfake|ai.?generated|synthetic\s+face|voice\s+clone|face\s+swap)\b",
        re.IGNORECASE,
    ),
    ViolationCategory.VIOLENT_THREATS: re.compile(
        # Direct 1st-person threat, explicit weapon/bomb threat, location-based threat
        r"\b(i\s+will\s+kill|going\s+to\s+hurt|bomb\s+threat|shoot\s+up"
        r"|i\s+know\s+where\s+you\s+live)\b",
        re.IGNORECASE,
    ),
    # Defensive Cyberbullying Rule: targeted/systematic interpersonal attacks
    # "kill yourself" is SELF_HARM (self-directed), not here
    ViolationCategory.CYBERBULLYING: re.compile(
        r"\b(nobody\s+likes\s+you|you\s+are\s+worthless\s+and\s+should"
        r"|i\s+will\s+ruin\s+your\s+life|everyone\s+hates\s+you)\b",
        re.IGNORECASE,
    ),
    ViolationCategory.MISINFORMATION: re.compile(
        r"\b(vaccines\s+cause|5g\s+causes|covid\s+is\s+fake|election\s+stolen)\b",
        re.IGNORECASE,
    ),
}


_POLICY_CODES = {
    ViolationCategory.HATE_SPEECH_AND_HARASSMENT: "MLC_H1.1",
    ViolationCategory.SELF_HARM: "MLC_S2.1",
    ViolationCategory.EXPLICIT_CONTENT: "MLC_E3.1",
    ViolationCategory.DEEPFAKE: "MLC_D4.1",
    ViolationCategory.VIOLENT_THREATS: "MLC_V5.1",
    ViolationCategory.CYBERBULLYING: "MLC_H1.2",
    ViolationCategory.MISINFORMATION: "MLC_M6.1",
}



def mock_moderate(
    text: str | None = None,
    image_path: str | None = None,
    simulate_latency: bool = True,
) -> ModerationResult:
    """
    Deterministic mock moderation function.

    Decision logic (rule-based, no ML) — mirrors the VLM compliance rules:
      0. SATIRE & HYPERBOLE RULE: If text matches any safe-hyperbole pattern,
         short-circuit to APPROVED immediately (no further scanning).
      1. Scan text for violation pattern matches.
      2. If matches → FLAGGED (single) or HUMAN_AUDIT_REQUIRED (multi-violation).
      3. If no matches → APPROVED.

    Args:
        text:             Text content to moderate.
        image_path:       Path to image (only used to set media_types in response).
        simulate_latency: If True, adds small sleep to simulate real latency.

    Returns:
        ModerationResult matching the production schema exactly.
    """
    t0 = time.perf_counter()

    if simulate_latency:
        import random
        time.sleep(random.uniform(0.005, 0.025))

    violations: list[DetectedViolation] = []
    media_types: list[MediaType] = []

    if text:
        media_types.append(MediaType.TEXT)
        cleaned = text.lower()

        # ── Step 0: Satire & Hyperbole Rule — allowlist short-circuit ─────────
        for safe_pattern in _HYPERBOLE_SAFE_PATTERNS:
            if safe_pattern.search(text):
                total_ms = (time.perf_counter() - t0) * 1000
                return ModerationResult(
                    decision=ModerationDecision.APPROVED,
                    confidence_score=0.88,
                    detected_violations=[],
                    reasoning=ReasoningBlock(
                        text_explanation=(
                            "Content contains aggressive-sounding language, but the "
                            "Satire & Hyperbole Rule applies — the phrasing is figurative "
                            "or humorous, not a genuine threat or hate expression."
                        ),
                        visual_explanation=(
                            "Visual context (if any) is consistent with satirical or "
                            "everyday content."
                        ),
                        bimodal_synergy=(
                            "Combined signals indicate hyperbolic expression. "
                            "No policy violation detected."
                        ),
                    ),
                    pipeline_tier_used=PipelineTier.TIER1,
                    total_latency_ms=round(total_ms, 2),
                    media_types_processed=media_types,
                    authenticity_report=AuthenticityReportSchema(
                        verdict=AuthenticityVerdict.AUTHENTIC,
                        ai_involvement_percentage=0.0,
                        tamper_annotations=["Satire & Hyperbole Rule: authentic expression."],
                    ),
                )

        # ── Step 1: Toxic violation pattern scan ──────────────────────────────
        for category, pattern in _TOXIC_PATTERNS.items():
            match = pattern.search(cleaned)
            if match:
                violations.append(
                    DetectedViolation(
                        category=category,
                        policy_code=_POLICY_CODES.get(category, "MLC_UNKNOWN"),
                        severity=SeverityLevel.HIGH,
                        confidence=0.93,
                    )
                )


    if image_path:
        suf = Path(image_path).suffix.lower()
        if suf in (".mp3", ".wav", ".flac", ".ogg", ".aac"):
            media_types.append(MediaType.AUDIO)
        elif suf in (".mp4", ".webm", ".avi", ".mov", ".mkv"):
            media_types.append(MediaType.VIDEO)
        else:
            media_types.append(MediaType.IMAGE)

    # Build reasoning
    if violations:
        text_expl = (
            f"Detected {len(violations)} violation(s): "
            + ", ".join(v.category for v in violations)
            + ". Specific trigger patterns matched in submitted text."
        )
        visual_expl = (
            "Image analysis flagged visual content consistent with the detected categories."
            if image_path
            else "No image provided for visual analysis."
        )
        synergy = (
            "Combined text and visual signals amplify confidence in the violation detection."
            if image_path and text
            else "Single-modality assessment; no cross-modal amplification."
        )

        # Multiple violations → escalate to HUMAN_AUDIT if ambiguous
        if len(violations) >= 2:
            decision = ModerationDecision.HUMAN_AUDIT_REQUIRED
            confidence = 0.72
        else:
            decision = ModerationDecision.FLAGGED
            confidence = 0.94

    else:
        decision = ModerationDecision.APPROVED
        confidence = 0.97
        text_expl = "No harmful textual patterns detected. Content appears safe."
        visual_expl = (
            "Image embedding similarity to safe anchors is high."
            if image_path
            else "No image provided."
        )
        synergy = "Content is safe across all analyzed modalities."

    total_ms = (time.perf_counter() - t0) * 1000

    # ── Authenticity / Forensics Report ────────────────────────────────────────
    is_deepfake = any(v.category == ViolationCategory.DEEPFAKE for v in violations)
    is_ai_text = bool(re.search(r"\b(furthermore|moreover|consequently|cutting-edge)\b", cleaned, re.I) and "paradigm" in cleaned) if text else False
    tampered_regions: list[DetectedTamperedRegion] = []
    if is_deepfake:
        auth_verdict = AuthenticityVerdict.MALICIOUS_TAMPERING
        ai_pct = 95.0
        annotations = [
            "Deepfake disclosure or synthetic facial features identified in content.",
            "High inter-frame landmark variance and compression boundary artifacts detected.",
        ]
        tampered_regions = [
            DetectedTamperedRegion(
                region_label="deepfake_face",
                confidence=0.95,
                box_2d=[120, 240, 680, 760],
                forensic_reasoning="Synthesized facial geometry and localized compression boundary artifacts detected.",
            )
        ]
    elif is_ai_text:
        auth_verdict = AuthenticityVerdict.SYNTHETIC_GENERATED
        ai_pct = 92.0
        annotations = [
            "Text exhibits uniform low perplexity and regular sentence length distribution.",
            "Pattern matches LLM output (uniform predictability).",
        ]
        tampered_regions = [
            DetectedTamperedRegion(
                region_label="synthetic_text",
                confidence=0.92,
                box_2d=[60, 40, 940, 960],
                forensic_reasoning="Stage 1 typography analysis detected computer-generated / synthetic text.",
            )
        ]
    else:
        auth_verdict = AuthenticityVerdict.AUTHENTIC
        ai_pct = 0.0
        annotations = [
            "No digital tampering or synthetic generation detected. Content appears authentic."
        ]

    auth_report = AuthenticityReportSchema(
        verdict=auth_verdict,
        ai_involvement_percentage=ai_pct,
        tamper_annotations=annotations,
        tampered_regions=tampered_regions,
    )

    return ModerationResult(
        decision=decision,
        confidence_score=confidence,
        detected_violations=violations,
        reasoning=ReasoningBlock(
            text_explanation=text_expl,
            visual_explanation=visual_expl,
            bimodal_synergy=synergy,
        ),
        authenticity_report=auth_report,
        pipeline_tier_used=PipelineTier.TIER1 if not violations else PipelineTier.TIER2,
        total_latency_ms=round(total_ms, 2),
        media_types_processed=media_types,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standard 10-case test battery
# ─────────────────────────────────────────────────────────────────────────────

MOCK_TEST_CASES: list[MockTestCase] = [
    MockTestCase(
        case_id="TC-001",
        description="Clean post — nature photo with wholesome caption",
        text="Beautiful sunrise over the mountains this morning! 🌄 #nature #photography",
        image_path=None,
        expected_decision=ModerationDecision.APPROVED,
        expected_violations=[],
        expected_forensic_verdict=AuthenticityVerdict.AUTHENTIC,
    ),
    MockTestCase(
        case_id="TC-002",
        description="Hate speech — explicit racial slur",
        text="I hate all [inferior race] people, they should be eliminated.",
        image_path=None,
        expected_decision=ModerationDecision.FLAGGED,
        expected_violations=[ViolationCategory.HATE_SPEECH_AND_HARASSMENT],
        expected_forensic_verdict=AuthenticityVerdict.AUTHENTIC,
    ),
    MockTestCase(
        case_id="TC-003",
        description="Self-harm promotion — direct encouragement",
        text="Feeling hopeless. I keep thinking about self-harm as an escape.",
        image_path=None,
        expected_decision=ModerationDecision.FLAGGED,
        expected_violations=[ViolationCategory.SELF_HARM],
        expected_forensic_verdict=AuthenticityVerdict.AUTHENTIC,
    ),
    MockTestCase(
        case_id="TC-004",
        description="Deepfake disclosure — synthetic media",
        text="This is a deepfake video of the president saying things he never said.",
        image_path=None,
        expected_decision=ModerationDecision.FLAGGED,
        expected_violations=[ViolationCategory.DEEPFAKE],
        expected_forensic_verdict=AuthenticityVerdict.MALICIOUS_TAMPERING,
    ),
    MockTestCase(
        case_id="TC-005",
        description="Violent threat — direct intimidation",
        text="I will kill you if you post that again. I know where you live.",
        image_path=None,
        expected_decision=ModerationDecision.FLAGGED,
        expected_violations=[ViolationCategory.VIOLENT_THREATS],
        expected_forensic_verdict=AuthenticityVerdict.AUTHENTIC,
    ),
    MockTestCase(
        case_id="TC-006",
        description="Cyberbullying — targeted harassment",
        text="Nobody likes you, you are worthless and should kill yourself.",
        image_path=None,
        expected_decision=ModerationDecision.HUMAN_AUDIT_REQUIRED,
        expected_violations=[ViolationCategory.CYBERBULLYING, ViolationCategory.SELF_HARM],
        expected_forensic_verdict=AuthenticityVerdict.AUTHENTIC,
    ),
    MockTestCase(
        case_id="TC-007",
        description="Medical misinformation",
        text="Vaccines cause autism! The government is hiding the truth from you.",
        image_path=None,
        expected_decision=ModerationDecision.FLAGGED,
        expected_violations=[ViolationCategory.MISINFORMATION],
        expected_forensic_verdict=AuthenticityVerdict.AUTHENTIC,
    ),
    MockTestCase(
        case_id="TC-008",
        description="Satire / news context — borderline",
        text="The politician's speech on immigration was absolutely hate-filled propaganda.",
        image_path=None,
        expected_decision=ModerationDecision.APPROVED,
        expected_violations=[],
        expected_forensic_verdict=AuthenticityVerdict.AUTHENTIC,
    ),
    MockTestCase(
        case_id="TC-009",
        description="Explicit content flagging",
        text="Sharing nude explicit content without consent is harassment.",
        image_path=None,
        expected_decision=ModerationDecision.FLAGGED,
        expected_violations=[ViolationCategory.EXPLICIT_CONTENT],
        expected_forensic_verdict=AuthenticityVerdict.AUTHENTIC,
    ),
    MockTestCase(
        case_id="TC-010",
        description="Multi-violation — compound hate + threats",
        text="Death to all [group]! I will kill them all with a bomb threat.",
        image_path=None,
        expected_decision=ModerationDecision.HUMAN_AUDIT_REQUIRED,
        expected_violations=[
            ViolationCategory.HATE_SPEECH_AND_HARASSMENT,
            ViolationCategory.VIOLENT_THREATS,
        ],
        expected_forensic_verdict=AuthenticityVerdict.AUTHENTIC,
    ),
    MockTestCase(
        case_id="TC-011",
        description="Image Inpainting — Localized ELA Tampering",
        text="Photo containing synthetic face inpainting and localized boundary artifacts.",
        image_path=None,
        expected_decision=ModerationDecision.FLAGGED,
        expected_violations=[ViolationCategory.DEEPFAKE],
        expected_forensic_verdict=AuthenticityVerdict.MALICIOUS_TAMPERING,
    ),
    MockTestCase(
        case_id="TC-012",
        description="Audio Splice — Speech Phase Discontinuity",
        text="Audio confession recording containing synthetic voice clone patches.",
        image_path=None,
        expected_decision=ModerationDecision.FLAGGED,
        expected_violations=[ViolationCategory.DEEPFAKE],
        expected_forensic_verdict=AuthenticityVerdict.MALICIOUS_TAMPERING,
    ),
    MockTestCase(
        case_id="TC-013",
        description="Video Face-Swap — Facial Landmark Jitter",
        text="Deepfake face swap footage with anomalous landmark variance.",
        image_path=None,
        expected_decision=ModerationDecision.FLAGGED,
        expected_violations=[ViolationCategory.DEEPFAKE],
        expected_forensic_verdict=AuthenticityVerdict.MALICIOUS_TAMPERING,
    ),
    MockTestCase(
        case_id="TC-014",
        description="AI Authorship — Uniform Perplexity Collapse",
        text="Furthermore, this paradigm facilitates automated multimodal processing across distributed architectures.",
        image_path=None,
        expected_decision=ModerationDecision.APPROVED,
        expected_violations=[],
        expected_forensic_verdict=AuthenticityVerdict.SYNTHETIC_GENERATED,
    ),
]
