"""
SafeSight AI — Unit Tests for Hierarchical Contextual Forensics & ELA Detector
==============================================================================
Validates:
  1. ELA Detector (app/utils/ela_detector.py)
     - 95% JPEG re-save error calculation
     - Error map scaling and luminance difference
     - Text vs background discrepancy calculation
     - Synthetic glow uniformity heuristic
     - Non-blocking async execution
  2. Hierarchical Prompt Chaining (app/pipeline.py)
     - Stage 1 typography analysis and parsing
     - Stage 2 prompt synthesis with ELA metrics & visual map
     - Bayesian contextual coupling (synthetic handwriting escalating scene involvement to 85-100%)
     - Master pipeline orchestration (gather Stage 1 & ELA -> Stage 2)
"""

from __future__ import annotations

import asyncio
import io
import json
from unittest.mock import AsyncMock, patch
from PIL import Image, ImageDraw
import pytest

from app.schemas import AuthenticityVerdict, ModerationDecision
from app.utils.ela_detector import (
    ELAResult,
    perform_error_level_analysis,
    perform_ela_async,
)
from app.pipeline import (
    Stage1TextResult,
    Stage2ContextualResult,
    HierarchicalForensicsOutcome,
    STAGE1_SYSTEM_PROMPT,
    STAGE1_USER_PROMPT,
    STAGE2_SYSTEM_PROMPT,
    _build_stage2_user_prompt,
    _extract_json_payload,
    run_stage1_text_analysis,
    run_stage2_contextual_evaluation,
    run_hierarchical_contextual_forensics,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_synthetic_document_image() -> Image.Image:
    """Create a synthetic document-like image with crisp vector text on uniform paper."""
    img = Image.new("RGB", (300, 300), color=(245, 245, 240))  # paper background
    draw = ImageDraw.Draw(img)
    # Draw lined paper ruled lines
    for y in range(40, 280, 30):
        draw.line([(20, y), (280, y)], fill=(200, 220, 240), width=1)
    # Draw sharp computer-rendered pseudo-handwriting
    draw.text((40, 70), "Meeting Notes: Project SafeSight", fill=(20, 20, 30))
    draw.text((40, 100), "Review biometric forensics pipeline", fill=(20, 20, 30))
    draw.text((40, 130), "Verify ELA JPEG compression variance", fill=(20, 20, 30))
    return img


@pytest.fixture
def sample_natural_photo_image() -> Image.Image:
    """Create a gradient image resembling photographic noise."""
    img = Image.new("RGB", (100, 100))
    for x in range(100):
        for y in range(100):
            noise = (x * 2 + y * 2) % 256
            img.putpixel((x, y), (noise, (noise + 30) % 256, (noise + 60) % 256))
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 1. ELA Detector Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestELADetector:
    """Tests for app/utils/ela_detector.py."""

    def test_perform_ela_returns_valid_structure(self, sample_synthetic_document_image):
        result = perform_error_level_analysis(sample_synthetic_document_image, quality=95)
        assert isinstance(result, ELAResult)
        assert result.ela_image is not None
        assert result.ela_bytes is not None
        assert isinstance(result.ela_base64, str)
        assert len(result.ela_base64) > 100
        assert result.mean_error >= 0.0
        assert result.max_error >= 0.0
        assert result.variance >= 0.0
        assert 0.0 <= result.high_error_ratio <= 1.0
        assert 0.0 <= result.glow_uniformity_score <= 1.0
        assert "center" in result.quadrant_variances
        assert isinstance(result.text_vs_background_discrepancy, float)
        assert isinstance(result.summary_description, str)

    def test_perform_ela_from_raw_bytes(self, sample_synthetic_document_image):
        buf = io.BytesIO()
        sample_synthetic_document_image.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        result = perform_error_level_analysis(raw_bytes, quality=95)
        assert isinstance(result, ELAResult)
        assert result.mean_error >= 0.0

    def test_ela_to_dict_serialization(self, sample_synthetic_document_image):
        result = perform_error_level_analysis(sample_synthetic_document_image)
        d = result.to_dict()
        assert "mean_error" in d
        assert "variance" in d
        assert "text_vs_background_discrepancy" in d
        assert "glow_uniformity_score" in d
        assert "is_synthetic_suspicious" in d

    @pytest.mark.anyio
    async def test_perform_ela_async(self, sample_synthetic_document_image):
        result = await perform_ela_async(sample_synthetic_document_image, quality=95)
        assert isinstance(result, ELAResult)
        assert result.mean_error >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hierarchical Contextual Pipeline Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHierarchicalPipeline:
    """Tests for app/pipeline.py."""

    def test_json_payload_extractor(self):
        # Case 1: Plain JSON string
        plain = '{"synthetic_text_confidence": 0.95, "verdict": "SYNTHETIC_GENERATED"}'
        assert _extract_json_payload(plain) == {
            "synthetic_text_confidence": 0.95,
            "verdict": "SYNTHETIC_GENERATED",
        }

        # Case 2: Markdown fenced block
        fenced = """Here is your analysis:
```json
{
  "synthetic_text_confidence": 0.88,
  "verdict": "SYNTHETIC_GENERATED"
}
```
Thank you."""
        parsed = _extract_json_payload(fenced)
        assert parsed is not None
        assert parsed["synthetic_text_confidence"] == 0.88

        # Case 3: Invalid / empty string
        assert _extract_json_payload("") is None
        assert _extract_json_payload("No json here") is None

    def test_build_stage2_user_prompt(self, sample_synthetic_document_image):
        stage1 = Stage1TextResult(
            synthetic_text_confidence=0.92,
            text_classification="AI_SYNTHETIC_TEXT",
            handwriting_variations_detected=False,
            observations="Perfect repetitive glyph strokes and lack of pen pressure.",
        )
        ela = perform_error_level_analysis(sample_synthetic_document_image)

        prompt = _build_stage2_user_prompt(
            stage1_result=stage1,
            ela_result=ela,
            accompanying_text="Sample note",
        )

        assert "synthetic_text_confidence" in prompt
        assert "0.92" in prompt
        assert "AI_SYNTHETIC_TEXT" in prompt
        assert "Error Level Analysis" in prompt
        assert "Sample note" in prompt
        assert "overall_authenticity_verdict" in prompt

    @pytest.mark.anyio
    async def test_run_stage1_text_analysis_mock(self, sample_synthetic_document_image):
        mock_response = {
            "synthetic_text_confidence": 0.94,
            "text_classification": "AI_SYNTHETIC_TEXT",
            "handwriting_variations_detected": False,
            "observations": "Glyphs show exact geometric duplication indicative of AI font synthesis.",
        }

        with patch("app.pipeline.hierarchical.get_ollama_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.generate_multimodal.return_value = (
                mock_response,
                json.dumps(mock_response),
                True,
            )
            mock_get_client.return_value = mock_client

            result = await run_stage1_text_analysis(sample_synthetic_document_image)

            assert isinstance(result, Stage1TextResult)
            assert result.synthetic_text_confidence == 0.94
            assert result.text_classification == "AI_SYNTHETIC_TEXT"
            assert result.handwriting_variations_detected is False

    @pytest.mark.anyio
    async def test_bayesian_contextual_escalation(self, sample_synthetic_document_image):
        """
        Critical Test:
        When Stage 1 detects synthetic text (confidence >= 0.70), Stage 2 must NOT
        allow the overall AI involvement to be diluted to ~25%. It must escalate to >= 85%.
        """
        stage1 = Stage1TextResult(
            synthetic_text_confidence=0.85,
            text_classification="AI_SYNTHETIC_TEXT",
            handwriting_variations_detected=False,
            observations="Synthetic handwriting detected in document.",
        )
        ela = perform_error_level_analysis(sample_synthetic_document_image)

        # Simulate a naive VLM that incorrectly guessed 25% AI involvement and AUTHENTIC background
        naive_stage2_response = {
            "overall_authenticity_verdict": "AUTHENTIC",
            "ai_involvement_percentage": 25.0,
            "decision": "APPROVED",
            "confidence_score": 0.80,
            "scene_physics_assessment": "Paper and desk look authentic.",
            "explanation": "Text is synthetic but background paper looks real.",
        }

        with patch("app.pipeline.hierarchical.get_ollama_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.generate_multimodal.return_value = (
                naive_stage2_response,
                json.dumps(naive_stage2_response),
                True,
            )
            mock_get_client.return_value = mock_client

            result = await run_stage2_contextual_evaluation(
                image=sample_synthetic_document_image,
                stage1_result=stage1,
                ela_result=ela,
            )

            # Enforced Bayesian coupling should have escalated AI involvement to >= 85%
            assert result.ai_involvement_percentage >= 85.0
            # And overridden verdict from AUTHENTIC to SYNTHETIC_GENERATED
            assert result.overall_authenticity_verdict == AuthenticityVerdict.SYNTHETIC_GENERATED

    @pytest.mark.anyio
    async def test_end_to_end_hierarchical_forensics(self, sample_synthetic_document_image):
        """
        Verify the master hierarchical pipeline gather flow:
        Stage 1 + ELA run in parallel -> Stage 2 -> HierarchicalForensicsOutcome.
        """
        s1_resp = {
            "synthetic_text_confidence": 0.91,
            "text_classification": "AI_SYNTHETIC_TEXT",
            "handwriting_variations_detected": False,
            "observations": "Unnatural glyph smoothness on paper fibers.",
        }
        s2_resp = {
            "overall_authenticity_verdict": "SYNTHETIC_GENERATED",
            "ai_involvement_percentage": 95.0,
            "decision": "FLAGGED",
            "confidence_score": 0.92,
            "scene_physics_assessment": "Rendered paper texture and synthetic shadows.",
            "explanation": "Bayesian coupling confirms entire document scene is a diffusion render.",
            "detected_violations": [
                {
                    "category": "DEEPFAKE",
                    "policy_code": "MLC_D4.1",
                    "severity": "HIGH",
                    "confidence": 0.95,
                }
            ],
        }

        with patch("app.pipeline.hierarchical.get_ollama_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.generate_multimodal.side_effect = [
                (s1_resp, json.dumps(s1_resp), True),   # Stage 1 call
                (s2_resp, json.dumps(s2_resp), True),   # Stage 2 call
            ]
            mock_get_client.return_value = mock_client

            outcome = await run_hierarchical_contextual_forensics(
                image=sample_synthetic_document_image,
                text="Document memo",
            )

            assert isinstance(outcome, HierarchicalForensicsOutcome)
            assert outcome.stage1.synthetic_text_confidence == 0.91
            assert outcome.stage2.overall_authenticity_verdict == AuthenticityVerdict.SYNTHETIC_GENERATED
            assert outcome.stage2.ai_involvement_percentage == 95.0
            assert outcome.moderation_result.authenticity_report.verdict == AuthenticityVerdict.SYNTHETIC_GENERATED
            assert outcome.moderation_result.authenticity_report.ai_involvement_percentage == 95.0
            assert len(outcome.moderation_result.detected_violations) >= 1
            assert len(outcome.moderation_result.authenticity_report.tampered_regions) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. Localized Grounding & Visualizer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalizedTamperingAndVisualizer:
    """Tests for DetectedTamperedRegion and render_annotated_tampering_image."""

    def test_detected_tampered_region_schema_instantiation(self):
        from app.schemas import DetectedTamperedRegion, AuthenticityReportSchema

        region = DetectedTamperedRegion(
            region_label="synthetic_text",
            confidence=0.96,
            box_2d=[100, 150, 400, 850],
            forensic_reasoning="Synthesized letterforms detected.",
        )
        assert region.region_label == "synthetic_text"
        assert region.confidence == 0.96
        assert region.box_2d == [100, 150, 400, 850]
        assert "Synthesized" in region.forensic_reasoning

        report = AuthenticityReportSchema(
            verdict=AuthenticityVerdict.SYNTHETIC_GENERATED,
            ai_involvement_percentage=96.0,
            tampered_regions=[region],
        )
        assert len(report.tampered_regions) == 1
        assert report.tampered_regions[0].region_label == "synthetic_text"

    def test_render_annotated_tampering_image(self, tmp_path, sample_synthetic_document_image):
        from app.schemas import DetectedTamperedRegion
        from app.utils.visualizer import render_annotated_tampering_image

        regions = [
            DetectedTamperedRegion(
                region_label="synthetic_text",
                confidence=0.95,
                box_2d=[60, 40, 500, 960],
                forensic_reasoning="Computer-generated font glyphs detected.",
            ),
            DetectedTamperedRegion(
                region_label="inpainting",
                confidence=0.82,
                box_2d=[520, 100, 900, 900],
                forensic_reasoning="Compression variance in bottom region.",
            ),
        ]

        target_out = tmp_path / "annotated_test_doc.png"
        saved_path = render_annotated_tampering_image(
            image_input=sample_synthetic_document_image,
            tampered_regions=regions,
            output_path=target_out,
        )

        assert saved_path.exists()
        assert saved_path == target_out.resolve()

        # Check annotated image can be opened and dimensions match
        annotated_img = Image.open(saved_path)
        assert annotated_img.size == sample_synthetic_document_image.size

    @pytest.mark.anyio
    async def test_stage2_parses_explicit_tampered_regions(self, sample_synthetic_document_image):
        stage1 = Stage1TextResult(
            synthetic_text_confidence=0.88,
            text_classification="AI_SYNTHETIC_TEXT",
            handwriting_variations_detected=False,
            observations="Micro-tremors absent.",
        )
        ela = perform_error_level_analysis(sample_synthetic_document_image)

        s2_json = {
            "overall_authenticity_verdict": "SYNTHETIC_GENERATED",
            "ai_involvement_percentage": 94.0,
            "decision": "FLAGGED",
            "confidence_score": 0.93,
            "scene_physics_assessment": "Unnatural shadow angles on desk.",
            "explanation": "Document and desk are fully rendered.",
            "tampered_regions": [
                {
                    "region_label": "synthetic_text",
                    "confidence": 0.95,
                    "box_2d": [100, 80, 400, 920],
                    "forensic_reasoning": "Unnatural glyph smoothness on paper fibers.",
                },
                {
                    "region_label": "inpainting",
                    "confidence": 0.88,
                    "box_2d": [500, 100, 800, 800],
                    "forensic_reasoning": "Spliced border region.",
                },
            ],
            "detected_violations": [],
        }

        with patch("app.pipeline.hierarchical.get_ollama_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.generate_multimodal.return_value = (s2_json, json.dumps(s2_json), True)
            mock_get_client.return_value = mock_client

            res = await run_stage2_contextual_evaluation(
                image=sample_synthetic_document_image,
                stage1_result=stage1,
                ela_result=ela,
            )

            assert len(res.tampered_regions) == 2
            assert res.tampered_regions[0].region_label == "synthetic_text"
            assert res.tampered_regions[0].box_2d == [100, 80, 400, 920]
            assert res.tampered_regions[1].region_label == "inpainting"

