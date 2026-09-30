"""
SafeSight AI — pytest Unit Tests
==================================
Fast, dependency-free unit tests for schemas, text processor,
and mock pipeline logic. Runs without any ML models.

Run:
    pytest tests/test_unit.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemas:
    """Validate Pydantic schema construction and validation."""

    def test_moderation_result_construction(self):
        from app.schemas import (
            DetectedViolation,
            ModerationDecision,
            ModerationResult,
            PipelineTier,
            ReasoningBlock,
            SeverityLevel,
            ViolationCategory,
        )

        result = ModerationResult(
            decision=ModerationDecision.FLAGGED,
            confidence_score=0.95,
            detected_violations=[
                DetectedViolation(
                    category=ViolationCategory.HATE_SPEECH_AND_HARASSMENT,
                    policy_code="MLC_H1.1",
                    severity=SeverityLevel.HIGH,
                    confidence=0.95,
                )
            ],
            reasoning=ReasoningBlock(
                text_explanation="Test text",
                visual_explanation="Test visual",
                bimodal_synergy="Test synergy",
            ),
            pipeline_tier_used=PipelineTier.TIER2,
            total_latency_ms=123.4,
        )

        assert result.decision == "FLAGGED"
        assert result.confidence_score == 0.95
        assert len(result.detected_violations) == 1
        assert result.detected_violations[0].policy_code == "MLC_H1.1"

    def test_auto_policy_code_assignment(self):
        from app.schemas import DetectedViolation, SeverityLevel, ViolationCategory

        v = DetectedViolation(
            category=ViolationCategory.DEEPFAKE,
            severity=SeverityLevel.HIGH,
        )
        assert v.policy_code == "MLC_D4.1"

    def test_confidence_score_bounds(self):
        from pydantic import ValidationError
        from app.schemas import (
            ModerationDecision,
            ModerationResult,
            PipelineTier,
            ReasoningBlock,
        )

        with pytest.raises(ValidationError):
            ModerationResult(
                decision=ModerationDecision.APPROVED,
                confidence_score=1.5,  # > 1.0 — should fail
                reasoning=ReasoningBlock(
                    text_explanation="x",
                    visual_explanation="x",
                    bimodal_synergy="x",
                ),
                pipeline_tier_used=PipelineTier.TIER1,
                total_latency_ms=10.0,
            )

    def test_moderation_result_json_roundtrip(self):
        from app.schemas import (
            ModerationDecision,
            ModerationResult,
            PipelineTier,
            ReasoningBlock,
        )

        result = ModerationResult(
            decision=ModerationDecision.APPROVED,
            confidence_score=0.99,
            detected_violations=[],
            reasoning=ReasoningBlock(
                text_explanation="Clean",
                visual_explanation="Safe",
                bimodal_synergy="No issues",
            ),
            pipeline_tier_used=PipelineTier.TIER1,
            total_latency_ms=22.1,
        )

        json_str = result.model_dump_json()
        restored = ModerationResult.model_validate_json(json_str)
        assert restored.decision == result.decision
        assert restored.confidence_score == result.confidence_score


# ─────────────────────────────────────────────────────────────────────────────
# Text Processor Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTextProcessor:
    """Validate text preprocessing functions."""

    def test_url_removal(self):
        from app.processors.text_processor import preprocess_text

        result = preprocess_text("Check out https://evil.com for free stuff")
        assert "https" not in result
        assert "Check out" in result

    def test_url_preservation(self):
        from app.processors.text_processor import preprocess_text

        result = preprocess_text(
            "Visit https://safe.org for info", preserve_urls=False
        )
        assert "safe.org" not in result

    def test_mention_normalization(self):
        from app.processors.text_processor import preprocess_text

        result = preprocess_text("Hey @john you're great!")
        assert "@john" not in result
        assert "[USER]" in result

    def test_hashtag_expansion(self):
        from app.processors.text_processor import preprocess_text

        result = preprocess_text("Love this! #BeautifulDay")
        assert "#BeautifulDay" not in result
        assert "BeautifulDay" in result

    def test_unicode_normalization(self):
        from app.processors.text_processor import preprocess_text

        # Cyrillic 'а' (U+0430) looks like Latin 'a' — homoglyph attack
        text_with_homoglyphs = "Hаte speech"  # 'а' is Cyrillic
        result = preprocess_text(text_with_homoglyphs)
        # After NFKC normalization the string should be valid text
        assert len(result) > 0

    def test_max_length_truncation(self):
        from app.processors.text_processor import preprocess_text

        long_text = "a" * 5000
        result = preprocess_text(long_text, max_length=100)
        assert len(result) <= 100

    def test_empty_text(self):
        from app.processors.text_processor import preprocess_text

        assert preprocess_text("") == ""
        assert preprocess_text("   ") == ""

    def test_language_hint_detection(self):
        from app.processors.text_processor import extract_language_hints

        hints = extract_language_hints("Hello, how are you today?")
        assert "detected_script" in hints
        assert "estimated_language" in hints


# ─────────────────────────────────────────────────────────────────────────────
# Mock Pipeline Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMockPipeline:
    """Validate mock pipeline decision logic for all test cases."""

    def test_clean_text_approved(self):
        from tests.mock_pipeline import mock_moderate
        from app.schemas import ModerationDecision

        result = mock_moderate(text="Beautiful sunset today! #nature")
        assert result.decision == ModerationDecision.APPROVED
        assert len(result.detected_violations) == 0
        assert result.confidence_score > 0.5

    def test_hate_speech_flagged(self):
        from tests.mock_pipeline import mock_moderate
        from app.schemas import ModerationDecision, ViolationCategory

        result = mock_moderate(
            text="I hate all [inferior race] people, death to them."
        )
        assert result.decision in [
            ModerationDecision.FLAGGED,
            ModerationDecision.HUMAN_AUDIT_REQUIRED,
        ]
        categories = [v.category for v in result.detected_violations]
        assert ViolationCategory.HATE_SPEECH_AND_HARASSMENT in categories

    def test_deepfake_flagged(self):
        from tests.mock_pipeline import mock_moderate
        from app.schemas import ModerationDecision, ViolationCategory

        result = mock_moderate(
            text="This deepfake video shows the president saying he will nuke everyone."
        )
        categories = [v.category for v in result.detected_violations]
        assert ViolationCategory.DEEPFAKE in categories

    def test_self_harm_flagged(self):
        from tests.mock_pipeline import mock_moderate
        from app.schemas import ViolationCategory

        result = mock_moderate(text="I've been thinking about self-harm lately.")
        categories = [v.category for v in result.detected_violations]
        assert ViolationCategory.SELF_HARM in categories

    def test_schema_always_valid(self):
        """Every mock result must pass ModerationResult schema validation."""
        from tests.mock_pipeline import MOCK_TEST_CASES, mock_moderate
        from app.schemas import ModerationResult

        for case in MOCK_TEST_CASES:
            result = mock_moderate(
                text=case.text,
                image_path=case.image_path,
                simulate_latency=False,
            )
            # Re-validate through Pydantic
            json_data = result.model_dump_json()
            restored = ModerationResult.model_validate_json(json_data)
            assert restored.decision is not None
            assert 0.0 <= restored.confidence_score <= 1.0

    def test_all_10_cases_have_valid_decisions(self):
        from tests.mock_pipeline import MOCK_TEST_CASES, mock_moderate
        from app.schemas import ModerationDecision

        valid_decisions = {d.value for d in ModerationDecision}
        for case in MOCK_TEST_CASES:
            result = mock_moderate(text=case.text, simulate_latency=False)
            decision_val = result.decision.value if hasattr(result.decision, "value") else result.decision
            assert decision_val in valid_decisions, f"{case.case_id}: invalid decision {decision_val}"

    def test_no_simulate_latency_is_fast(self):
        """Without latency simulation, each case should complete in < 50ms."""
        import time
        from tests.mock_pipeline import mock_moderate

        t0 = time.perf_counter()
        mock_moderate(text="hello world", simulate_latency=False)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 50, f"Mock pipeline too slow: {elapsed:.1f}ms"


# ─────────────────────────────────────────────────────────────────────────────
# Tier-1 Keyword Gate Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKeywordGate:
    """Test hard-block keyword patterns in the Tier-1 gate."""

    def test_kill_yourself_blocked(self):
        from app.processors.keyword_patterns import keyword_hard_block

        blocked, trigger = keyword_hard_block("you should kill yourself loser")
        assert blocked is True
        assert trigger.lower().replace(" ", "") in ["killyourself", "kys"]

    def test_safe_text_not_blocked(self):
        from app.processors.keyword_patterns import keyword_hard_block

        blocked, _ = keyword_hard_block("I love spending time with my family")
        assert blocked is False

    def test_case_insensitive_matching(self):
        from app.processors.keyword_patterns import keyword_hard_block

        blocked, _ = keyword_hard_block("KiLl YouRsElF now")
        assert blocked is True

    def test_empty_text_not_blocked(self):
        from app.processors.keyword_patterns import keyword_hard_block

        blocked, _ = keyword_hard_block("")
        assert blocked is False
