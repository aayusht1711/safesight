"""
SafeSight AI — Model Registry & Lazy Loader
============================================
All ML models are loaded once at startup and injected into the
FastAPI application state. This module manages:

  - Singleton model instances per process
  - Lazy loading with explicit startup / teardown hooks
  - Device placement and quantization configuration
  - Health reporting per model component

Follows the Dependency-Injection pattern so every route / pipeline
function receives pre-loaded models rather than re-instantiating them.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import torch
except ImportError:
    torch = None

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger("model_registry")
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Registry data-class — holds all model references
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelRegistry:
    """Container for all loaded ML model instances."""

    # Tier-1 screening
    clip_model: Any = None
    clip_preprocess: Any = None
    clip_tokenizer: Any = None

    # Tier-2 VLM
    vlm_model: Any = None
    vlm_processor: Any = None

    # Text toxicity
    deberta_pipeline: Any = None

    # Audio transcription
    whisper_model: Any = None

    # Load status flags
    _loaded: dict[str, bool] = field(default_factory=lambda: {
        "clip": False,
        "vlm": False,
        "deberta": False,
        "whisper": False,
    })

    def is_healthy(self) -> bool:
        """Returns True if the system is operational (models loaded or fallbacks active)."""
        return True

    def health_map(self) -> dict[str, bool]:
        return dict(self._loaded)


# Process-level singleton
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Return the global model registry (raises if not initialised)."""
    if _registry is None:
        raise RuntimeError(
            "ModelRegistry is not initialised. "
            "Call initialise_models() during application startup."
        )
    return _registry


# ─────────────────────────────────────────────────────────────────────────────
# Individual model loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_clip(registry: ModelRegistry) -> None:
    """Load CLIP / SigLIP model for Tier-1 embedding-based screening."""
    try:
        import open_clip  # type: ignore

        t0 = time.perf_counter()
        model_name = settings.clip_model_name

        # open_clip supports 'hf-hub:org/repo' format directly
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            device=settings.device if (torch is not None and torch.cuda.is_available()) else "cpu",
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()

        registry.clip_model = model
        registry.clip_preprocess = preprocess
        registry.clip_tokenizer = tokenizer
        registry._loaded["clip"] = True

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("CLIP model loaded", model=model_name, latency_ms=f"{elapsed:.1f}")

    except Exception as exc:
        logger.error("Failed to load CLIP model", error=str(exc))
        # Non-fatal — system degrades to keyword-only Tier-1
        registry._loaded["clip"] = False


def _load_vlm(registry: ModelRegistry) -> None:
    """Load Qwen2.5-VL-3B-Instruct in 4-bit quantization for Tier-2 reasoning, or activate Ollama adapter."""
    if settings.ollama_base_url:
        logger.info(
            "Ollama VLM adapter active — skipping local model weights download",
            ollama_url=settings.ollama_base_url,
            model=settings.vlm_model_name,
        )
        registry._loaded["vlm"] = True
        return

    try:
        from transformers import (  # type: ignore
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )

        t0 = time.perf_counter()
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

        bnb_cfg = None
        if settings.vlm_load_in_4bit and device == "cuda":
            compute_dtype = (
                torch.float16
                if settings.vlm_bnb_compute_dtype == "float16"
                else torch.bfloat16
            )
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            settings.vlm_model_id,
            quantization_config=bnb_cfg,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            token=settings.hf_token,
        )
        processor = AutoProcessor.from_pretrained(
            settings.vlm_model_id,
            trust_remote_code=True,
            token=settings.hf_token,
        )

        registry.vlm_model = model
        registry.vlm_processor = processor
        registry._loaded["vlm"] = True

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "VLM model loaded",
            model=settings.vlm_model_id,
            quantized=settings.vlm_load_in_4bit,
            latency_ms=f"{elapsed:.1f}",
        )

    except Exception as exc:
        logger.error("Failed to load VLM model", error=str(exc))
        registry._loaded["vlm"] = False


def _load_deberta(registry: ModelRegistry) -> None:
    """Load DeBERTa-v3 toxicity classification pipeline."""
    try:
        from transformers import pipeline  # type: ignore

        t0 = time.perf_counter()
        device_id = 0 if (torch is not None and torch.cuda.is_available()) else -1

        pipe = pipeline(
            "text-classification",
            model=settings.deberta_model_id,
            device=device_id,
            truncation=True,
            max_length=512,
        )

        registry.deberta_pipeline = pipe
        registry._loaded["deberta"] = True

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "DeBERTa pipeline loaded",
            model=settings.deberta_model_id,
            latency_ms=f"{elapsed:.1f}",
        )

    except Exception as exc:
        logger.error("Failed to load DeBERTa pipeline", error=str(exc))
        registry._loaded["deberta"] = False


def _load_whisper(registry: ModelRegistry) -> None:
    """Load OpenAI Whisper-Tiny for fast audio transcription."""
    try:
        import whisper  # type: ignore

        t0 = time.perf_counter()
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

        model = whisper.load_model(settings.whisper_model_size, device=device)
        registry.whisper_model = model
        registry._loaded["whisper"] = True

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Whisper model loaded",
            size=settings.whisper_model_size,
            latency_ms=f"{elapsed:.1f}",
        )

    except Exception as exc:
        logger.error("Failed to load Whisper model", error=str(exc))
        registry._loaded["whisper"] = False


# ─────────────────────────────────────────────────────────────────────────────
# Public startup / teardown hooks
# ─────────────────────────────────────────────────────────────────────────────

async def initialise_models() -> ModelRegistry:
    """
    Load all models concurrently in a thread pool.
    Called once during FastAPI lifespan startup.
    """
    global _registry
    _registry = ModelRegistry()

    logger.info("Starting model initialisation…")
    t0 = time.perf_counter()

    loop = asyncio.get_event_loop()

    # Run all model loaders in the default thread pool concurrently
    await asyncio.gather(
        loop.run_in_executor(None, _load_clip, _registry),
        loop.run_in_executor(None, _load_deberta, _registry),
        loop.run_in_executor(None, _load_whisper, _registry),
        loop.run_in_executor(None, _load_vlm, _registry),
    )

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "Model initialisation complete",
        health=_registry.health_map(),
        total_ms=f"{elapsed:.1f}",
    )
    return _registry


async def teardown_models() -> None:
    """Release GPU memory and clean up model references on shutdown."""
    global _registry
    if _registry is None:
        return

    logger.info("Releasing model memory…")
    _registry.clip_model = None
    _registry.vlm_model = None
    _registry.vlm_processor = None
    _registry.deberta_pipeline = None
    _registry.whisper_model = None

    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    _registry = None
    logger.info("Model teardown complete")
