"""
SafeSight AI — Unit Tests for Ollama Multimodal Adapter
========================================================
Validates image base64 conversion, async Ollama client retry logic,
fail-safe warming-up fallback behavior, and Tier-2 VLM integration.
"""

from __future__ import annotations

import asyncio
import io
import json
import pytest
from unittest.mock import AsyncMock, patch
from PIL import Image

from app.utils.ollama_client import OllamaClient, image_to_base64


class TestOllamaClientUnit:
    """Unit tests for Ollama client utility methods and graceful fallbacks."""

    def test_image_to_base64_pil(self):
        img = Image.new("RGB", (64, 64), color="blue")
        b64 = image_to_base64(img)
        assert b64 is not None
        assert isinstance(b64, str)
        assert len(b64) > 10

    def test_image_to_base64_rgba_converted_to_rgb(self):
        img = Image.new("RGBA", (32, 32), color=(255, 0, 0, 128))
        b64 = image_to_base64(img)
        assert b64 is not None
        assert isinstance(b64, str)

    def test_image_to_base64_raw_bytes(self):
        raw = b"\xFF\xD8\xFF\xE0\x00\x10JFIF"
        b64 = image_to_base64(raw)
        assert b64 is not None
        assert isinstance(b64, str)

    def test_image_to_base64_none(self):
        assert image_to_base64(None) is None

    def test_client_init_defaults(self):
        client = OllamaClient()
        assert "11434" in client.base_url
        assert client.model_name in ("qwen2.5vl:7b", "qwen2.5vl:3b", "qwen2.5-vl:3b")
        assert client.max_retries >= 1

    def test_ollama_client_unreachable_fallback(self):
        """
        Verify that when Ollama is offline or warming up, generate_multimodal
        returns a structured fallback rather than crashing.
        """
        async def _run():
            client = OllamaClient(
                base_url="http://127.0.0.1:59999",  # unused port
                timeout_seconds=0.5,
                max_retries=1,
            )

            fallback_dict, raw_str, ok = await client.generate_multimodal(
                prompt="Analyze this content",
                system_prompt="Safety system prompt",
                image=None,
            )

            assert ok is False
            assert fallback_dict is not None
            assert fallback_dict.get("decision") == "HUMAN_AUDIT_REQUIRED"
            assert fallback_dict.get("confidence_score") == 0.50
            assert "reasoning" in fallback_dict

            # Verify raw_str is a valid JSON string parsable by downstream Pydantic models
            parsed = json.loads(raw_str)
            assert parsed["decision"] == "HUMAN_AUDIT_REQUIRED"

        asyncio.run(_run())

    def test_ollama_client_successful_mock_response(self):
        """Verify successful Ollama 200 JSON response handling."""
        from unittest.mock import MagicMock

        async def _run():
            mock_vlm_output = {
                "decision": "APPROVED",
                "confidence_score": 0.95,
                "detected_violations": [],
                "reasoning": {
                    "text_explanation": "Content is harmless.",
                    "visual_explanation": "Image shows nature scenery.",
                    "bimodal_synergy": "Bimodal elements are fully safe.",
                },
            }

            client = OllamaClient(base_url="http://localhost:11434", max_retries=1)

            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "model": "qwen2.5-vl:3b",
                    "response": json.dumps(mock_vlm_output),
                    "done": True,
                }
                mock_post.return_value = mock_resp

                parsed_dict, raw_str, ok = await client.generate_multimodal(
                    prompt="Analyze content",
                    system_prompt="System instructions",
                )

                assert ok is True
                assert parsed_dict is not None
                assert parsed_dict["decision"] == "APPROVED"
                assert parsed_dict["confidence_score"] == 0.95

        asyncio.run(_run())
