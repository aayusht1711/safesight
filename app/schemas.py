"""
SafeSight AI — Pydantic Schemas
================================
Defines the canonical request/response data models used across the API
and internal pipeline stages. All schemas are versioned under this module.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class ModerationDecision(str, Enum):
    APPROVED = "APPROVED"
    FLAGGED = "FLAGGED"
    HUMAN_AUDIT_REQUIRED = "HUMAN_AUDIT_REQUIRED"


class ViolationCategory(str, Enum):
    HATE_SPEECH_AND_HARASSMENT = "HATE_SPEECH_AND_HARASSMENT"
    SELF_HARM = "SELF_HARM"
    EXPLICIT_CONTENT = "EXPLICIT_CONTENT"
    DEEPFAKE = "DEEPFAKE"
    CYBERBULLYING = "CYBERBULLYING"
    VIOLENT_THREATS = "VIOLENT_THREATS"
    MISINFORMATION = "MISINFORMATION"
    SPAM_AND_MANIPULATION = "SPAM_AND_MANIPULATION"


class SeverityLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class MediaType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class PipelineTier(str, Enum):
    TIER1 = "TIER1"
    TIER2 = "TIER2"


# ─────────────────────────────────────────────────────────────────────────────
# Policy Codes — MLC (Moderation Label Codes)
# ─────────────────────────────────────────────────────────────────────────────

POLICY_CODE_MAP: dict[ViolationCategory, str] = {
    ViolationCategory.HATE_SPEECH_AND_HARASSMENT: "MLC_H1.1",
    ViolationCategory.SELF_HARM: "MLC_S2.1",
    ViolationCategory.EXPLICIT_CONTENT: "MLC_E3.1",
    ViolationCategory.DEEPFAKE: "MLC_D4.1",
    ViolationCategory.CYBERBULLYING: "MLC_H1.2",
    ViolationCategory.VIOLENT_THREATS: "MLC_V5.1",
    ViolationCategory.MISINFORMATION: "MLC_M6.1",
    ViolationCategory.SPAM_AND_MANIPULATION: "MLC_SP7.1",
}


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────────────────────────────────────

class DetectedViolation(BaseModel):
    """A single identified policy violation."""

    category: ViolationCategory
    policy_code: str = Field(
        description="MLC code referencing the internal policy rule that was violated."
    )
    severity: SeverityLevel
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Per-violation confidence score.",
    )

    @model_validator(mode="before")
    @classmethod
    def auto_assign_policy_code(cls, data: Any) -> Any:
        """Auto-populate policy_code from the category if not provided."""
        if isinstance(data, dict) and not data.get("policy_code"):
            category = data.get("category")
            if category:
                cat_enum = ViolationCategory(category)
                data["policy_code"] = POLICY_CODE_MAP.get(cat_enum, "MLC_UNKNOWN")
        return data


class ReasoningBlock(BaseModel):
    """Human-readable reasoning explaining the moderation decision."""

    text_explanation: str = Field(
        description="Textual triggers that contributed to the decision."
    )
    visual_explanation: str = Field(
        description="Visual objects or patterns that raised flags."
    )
    bimodal_synergy: str = Field(
        description=(
            "How the combination of text + image creates, amplifies, "
            "or diffuses potential harm."
        )
    )


class Tier1Result(BaseModel):
    """Internal result produced by the Tier-1 fast-path screening gate."""

    safe_score: float = Field(ge=0.0, le=1.0)
    routed_to_tier2: bool
    gate_latency_ms: float
    reason: str


class Tier2Result(BaseModel):
    """Internal result produced by the Tier-2 VLM reasoning stage."""

    raw_vlm_output: str
    vlm_latency_ms: float
    parsed_successfully: bool
    authenticity_report: AuthenticityReportSchema | None = None


class AuthenticityVerdict(str, Enum):
    AUTHENTIC = "AUTHENTIC"
    BENIGN_AI_ASSISTED = "BENIGN_AI_ASSISTED"
    HYBRID_HUMAN_AI_EDITED = "HYBRID_HUMAN_AI_EDITED"
    MALICIOUS_TAMPERING = "MALICIOUS_TAMPERING"
    SYNTHETIC_GENERATED = "SYNTHETIC_GENERATED"


class TamperedRegionCategory(str, Enum):
    """Categorical classification of detected media tampering."""
    INPAINTING = "INPAINTING"
    GENERATIVE_OBJECT = "GENERATIVE_OBJECT"
    SYNTHETIC_TEXT = "SYNTHETIC_TEXT"
    FACE_SWAP = "FACE_SWAP"
    COMPRESSION_ARTIFACT = "COMPRESSION_ARTIFACT"


class DetectedTamperedRegion(BaseModel):
    """
    Normalized 2D bounding box representing localized media tampering,
    inpainting, synthetic text, deepfake faces, or generative artifacts,
    with human-interpretable semantic labeling and forensic explanation.
    """
    region_label: str = Field(
        ...,
        description="A concise, plain-English label describing the exact object (e.g., 'AI Inpainted Birds', 'Synthetic Text Block', 'Cloned Object').",
    )
    category: TamperedRegionCategory = Field(
        default=TamperedRegionCategory.INPAINTING,
        description="Category of detected tampering: INPAINTING | GENERATIVE_OBJECT | SYNTHETIC_TEXT | FACE_SWAP | COMPRESSION_ARTIFACT",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this specific localized region (0.0 to 1.0).",
    )
    box_2d: list[int] = Field(
        ...,
        description="Normalized coordinates [ymin, xmin, ymax, xmax] scaled from 0 to 1000.",
    )
    forensic_explanation: str = Field(
        default="",
        description="A concise 1-sentence explanation of why this region was flagged.",
    )
    forensic_reasoning: str = Field(
        default="",
        description="Legacy alias for forensic_explanation.",
    )

    @model_validator(mode="before")
    @classmethod
    def sync_reasoning_and_explanation(cls, data: Any) -> Any:
        if isinstance(data, dict):
            reasoning = data.get("forensic_reasoning")
            explanation = data.get("forensic_explanation")
            if reasoning and not explanation:
                data["forensic_explanation"] = reasoning
            elif explanation and not reasoning:
                data["forensic_reasoning"] = explanation
            elif not reasoning and not explanation:
                data["forensic_explanation"] = "Localized anomaly detected."
                data["forensic_reasoning"] = "Localized anomaly detected."

            cat_val = data.get("category")
            if not cat_val:
                lbl = str(data.get("region_label", "")).lower()
                if "text" in lbl or "font" in lbl or "handwriting" in lbl:
                    data["category"] = TamperedRegionCategory.SYNTHETIC_TEXT
                elif "face" in lbl or "deepfake" in lbl or "swap" in lbl:
                    data["category"] = TamperedRegionCategory.FACE_SWAP
                elif "bird" in lbl or "object" in lbl or "clone" in lbl or "generative" in lbl:
                    data["category"] = TamperedRegionCategory.GENERATIVE_OBJECT
                elif "compress" in lbl or "dct" in lbl or "block" in lbl:
                    data["category"] = TamperedRegionCategory.COMPRESSION_ARTIFACT
                else:
                    data["category"] = TamperedRegionCategory.INPAINTING
            elif isinstance(cat_val, str) and not isinstance(cat_val, TamperedRegionCategory):
                cat_upper = cat_val.upper().replace(" ", "_")
                if hasattr(TamperedRegionCategory, cat_upper):
                    data["category"] = TamperedRegionCategory[cat_upper]
        return data


class AuthenticityReportSchema(BaseModel):
    """Forensic media authenticity and localized tampering report."""

    verdict: AuthenticityVerdict = Field(
        default=AuthenticityVerdict.AUTHENTIC,
        description="Forensic verdict: AUTHENTIC | BENIGN_AI_ASSISTED | MALICIOUS_TAMPERING | SYNTHETIC_GENERATED",
    )
    ai_involvement_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Estimated AI contribution percentage (0-100%).",
    )
    tamper_annotations: list[str] = Field(
        default_factory=list,
        description="Detailed forensic annotations identifying specific localized or global tampering.",
    )
    tampered_regions: list[DetectedTamperedRegion] = Field(
        default_factory=list,
        description="Normalized bounding boxes highlighting exact localized regions of synthetic manipulation.",
    )
    annotated_image_path: str | None = Field(
        default=None,
        description="Filesystem path or URI to the rendered visual inspection map with bounding box overlays.",
    )
    compression_fingerprint: str | None = Field(
        default=None,
        description="Quantization and metadata compression fingerprint (e.g. WhatsApp_Android_v2.23, Original_Camera_HighQ).",
    )


# Schema aliases for developer convenience
AuthenticityReport = AuthenticityReportSchema


# ─────────────────────────────────────────────────────────────────────────────
# Primary API Response
# ─────────────────────────────────────────────────────────────────────────────

class ModerationResult(BaseModel):
    """
    Canonical moderation response returned by the pipeline.
    This is the authoritative output schema documented in the API contract.
    """

    decision: ModerationDecision
    confidence_score: float = Field(ge=0.0, le=1.0)
    detected_violations: list[DetectedViolation] = Field(default_factory=list)
    reasoning: ReasoningBlock

    # ── Media Authenticity & Deepfake Forensics ──────────────────────────────
    authenticity_report: AuthenticityReportSchema = Field(
        default_factory=AuthenticityReportSchema,
        description="Granular media authenticity and localized tampering analysis.",
    )

    # ── Pipeline telemetry (internal, surfaced for observability) ─────────────
    pipeline_tier_used: PipelineTier
    total_latency_ms: float
    media_types_processed: list[MediaType] = Field(default_factory=list)

    model_config = {"use_enum_values": True}


ModerationResponse = ModerationResult



# ─────────────────────────────────────────────────────────────────────────────
# API Request Models
# ─────────────────────────────────────────────────────────────────────────────

class TextModerationRequest(BaseModel):
    """Request body for text-only moderation endpoint."""

    text: str = Field(..., min_length=1, max_length=10_000)
    context: str | None = Field(
        default=None,
        description="Optional surrounding context for improved accuracy.",
    )


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    models_loaded: dict[str, bool]
    device: str
    version: str = "1.0.0"
