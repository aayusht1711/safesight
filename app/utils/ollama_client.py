"""
SafeSight AI — Ollama Multimodal Client Adapter
=================================================
Asynchronous client adapter communicating with a local or remote
Ollama instance hosting vision-language models (e.g., qwen2.5-vl:3b).

Features:
  - Native asynchronous HTTP requests via httpx.AsyncClient
  - Resilient retry strategy with exponential backoff for warming-up states
  - In-memory Base64 conversion for PIL Images and raw image bytes
  - Constrained JSON generation via Ollama's format="json" API
  - Non-crashing graceful fallbacks when the Ollama daemon is starting up
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from typing import Any

import httpx
from PIL import Image

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger("ollama_client")
settings = get_settings()


def image_to_base64(image_input: Image.Image | bytes | None) -> str | None:
    """
    Convert a PIL Image or raw bytes into a base64-encoded ASCII string
    compatible with Ollama's multimodal API.
    """
    if image_input is None:
        return None

    try:
        if isinstance(image_input, bytes):
            return base64.b64encode(image_input).decode("utf-8")

        if isinstance(image_input, Image.Image):
            buffered = io.BytesIO()
            # If image has alpha channel or palette, convert to RGB for JPEG
            if image_input.mode in ("RGBA", "P", "LA"):
                img_rgb = image_input.convert("RGB")
            else:
                img_rgb = image_input

            img_rgb.save(buffered, format="JPEG", quality=90)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        return None
    except Exception as exc:
        logger.warning("Failed to encode image to base64 for Ollama", error=str(exc))
        return None


class OllamaClient:
    """
    Async client managing communication, retries, and fallbacks with Ollama.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ):
        self.base_url = (
            base_url
            or settings.ollama_base_url
            or "http://localhost:11434"
        ).rstrip("/")
        self.model_name = model_name or settings.vlm_model_name
        self.timeout_seconds = timeout_seconds or settings.ollama_timeout_seconds
        self.max_retries = max_retries or settings.ollama_max_retries

    async def is_healthy(self) -> bool:
        """Ping the Ollama daemon to check if it is responsive."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/")
                return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """Fetch list of models currently loaded or pulled in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    return [m.get("name", "") for m in data.get("models", [])]
        except Exception as exc:
            logger.debug("Failed to list Ollama models", error=str(exc))
        return []

    async def _resolve_active_model(self, client: httpx.AsyncClient) -> str:
        """Resolve the best matching model available in local Ollama instance."""
        target = self.model_name
        try:
            resp = await client.get(f"{self.base_url}/api/tags")
            if resp.status_code == 200:
                available = [m.get("name", "") for m in resp.json().get("models", [])]
                if target in available:
                    return target
                # Check normalized aliases
                candidates = [
                    target.replace("qwen2.5-vl", "qwen2.5vl"),
                    target.replace("qwen2.5vl", "qwen2.5-vl"),
                    "qwen2.5vl:3b",
                    "qwen2.5vl:7b",
                    "qwen2.5vl:latest",
                    "llava:7b",
                ]
                for c in candidates:
                    for a in available:
                        if a == c or a.startswith(c):
                            logger.info("Auto-mapped Ollama model", requested=target, using=a)
                            return a
        except Exception:
            pass
        if target == "qwen2.5-vl:3b":
            return "qwen2.5vl:3b"
        return target

    async def generate_multimodal(
        self,
        prompt: str,
        system_prompt: str | None = None,
        image: Image.Image | bytes | list[Image.Image | bytes] | None = None,
    ) -> tuple[dict[str, Any] | None, str, bool]:
        """
        Send multimodal prompt to Ollama with retry backoff and non-crashing fallback.

        Supports a single image or multiple images (e.g. original image + ELA map).

        Returns:
            (parsed_json_dict_or_fallback, raw_response_string, is_success)
        """
        endpoint = f"{self.base_url}/api/generate"
        last_error = ""

        # Retry loop with exponential backoff
        for attempt in range(1, self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    resolved_model = await self._resolve_active_model(client)

                    payload: dict[str, Any] = {
                        "model": resolved_model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": 0.1,
                        },
                    }

                    if system_prompt:
                        payload["system"] = system_prompt

                    if image is not None:
                        if isinstance(image, list):
                            encoded_list = [image_to_base64(img) for img in image if img is not None]
                            payload["images"] = [b for b in encoded_list if b is not None]
                        else:
                            b64_image = image_to_base64(image)
                            if b64_image:
                                payload["images"] = [b64_image]

                    has_images = bool(payload.get("images"))
                    logger.info(
                        "Dispatching multimodal inference to Ollama",
                        endpoint=endpoint,
                        model=resolved_model,
                        attempt=attempt,
                        has_image=has_images,
                        num_images=len(payload.get("images", [])),
                    )

                    resp = await client.post(endpoint, json=payload)

                    if resp.status_code == 200:
                        data = resp.json()
                        raw_output = data.get("response", "")
                        elapsed_ms = (time.perf_counter() - t0) * 1000

                        logger.info(
                            "Ollama inference response received",
                            status=resp.status_code,
                            latency_ms=f"{elapsed_ms:.1f}",
                        )

                        # Parse JSON from response
                        try:
                            parsed_json = json.loads(raw_output)
                            return parsed_json, raw_output, True
                        except json.JSONDecodeError:
                            logger.warning(
                                "Ollama response was not valid JSON, returning raw text",
                                raw_preview=raw_output[:200],
                            )
                            return None, raw_output, False

                    else:
                        last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                        logger.warning(
                            "Ollama returned non-200 status",
                            status=resp.status_code,
                            attempt=attempt,
                        )

            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_error = f"Connection failed (Ollama daemon may still be warming up): {exc}"
                logger.warning(
                    "Ollama connection error during warm-up",
                    attempt=attempt,
                    error=str(exc),
                )
            except httpx.ReadTimeout as exc:
                last_error = f"Read timeout ({self.timeout_seconds}s): {exc}"
                logger.warning(
                    "Ollama read timeout",
                    attempt=attempt,
                    error=str(exc),
                )
            except Exception as exc:
                last_error = f"Unexpected error communicating with Ollama: {exc}"
                logger.error(
                    "Unexpected error during Ollama inference",
                    attempt=attempt,
                    error=str(exc),
                )

            # Wait before retry if not on the last attempt
            if attempt < self.max_retries:
                backoff = 0.5 * (2 ** (attempt - 1))
                logger.info("Retrying Ollama request after backoff", backoff_sec=backoff)
                await asyncio.sleep(backoff)

        # ── Fallback Graceful Response (No Crashes) ───────────────────────────
        logger.error(
            "All Ollama retries exhausted. Activating non-crashing fail-safe fallback.",
            last_error=last_error,
        )

        fallback_dict = {
            "decision": "HUMAN_AUDIT_REQUIRED",
            "confidence_score": 0.50,
            "detected_violations": [],
            "reasoning": {
                "text_explanation": (
                    f"Tier-2 VLM inference automated review temporarily unavailable "
                    f"({last_error}). Escalated to human audit."
                ),
                "visual_explanation": (
                    "Visual inspection deferred to human moderator queue due to "
                    "background model initialization."
                ),
                "bimodal_synergy": (
                    "Fail-safe posture activated: ambiguous or unverified media "
                    "is held for human audit rather than rejected or allowed blindly."
                ),
            },
        }

        fallback_json = json.dumps(fallback_dict)
        return fallback_dict, fallback_json, False


# Singleton helper instance
_default_client: OllamaClient | None = None


def get_ollama_client() -> OllamaClient:
    """Return singleton Ollama client instance."""
    global _default_client
    if _default_client is None:
        _default_client = OllamaClient()
    return _default_client
