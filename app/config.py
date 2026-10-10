"""
SafeSight AI — Application Configuration
=========================================
Central Pydantic-Settings model that reads from .env and environment variables.
All tuneable knobs live here so no magic strings are scattered in the codebase.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Single source of truth for all runtime configuration.
    Values are read from environment variables (and optionally a .env file).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── Model IDs ──────────────────────────────────────────────────────────────
    vlm_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    deberta_model_id: str = "martin-ha/toxic-comment-model"
    whisper_model_size: Literal["tiny", "base", "small"] = "tiny"

    # ── VLM Quantization ───────────────────────────────────────────────────────
    vlm_load_in_4bit: bool = True
    vlm_bnb_compute_dtype: Literal["float16", "bfloat16"] = "float16"

    # ── Tier-1 CLIP Gate ───────────────────────────────────────────────────────
    clip_model_name: str = (
        "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    )
    tier1_confidence_threshold: float = Field(0.82, ge=0.0, le=1.0)
    tier1_suspicious_threshold: float = Field(0.45, ge=0.0, le=1.0)

    # ── Latency Budgets (milliseconds) ─────────────────────────────────────────
    tier1_timeout_ms: int = 50
    tier2_timeout_ms: int = 700

    # ── Redis Cache ────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # ── Device ─────────────────────────────────────────────────────────────────
    device: Literal["cuda", "cpu", "mps"] = "cuda"

    # ── Ollama Runtime Integration ─────────────────────────────────────────────
    ollama_base_url: str | None = None
    vlm_model_name: str = "qwen2.5vl:7b"
    ollama_timeout_seconds: float = 90.0
    ollama_max_retries: int = 3

    # ── Media Forensics & Authenticity ─────────────────────────────────────────
    enable_forensics: bool = True
    forensics_ela_quality: int = 90
    forensics_ela_tamper_threshold: float = 180.0
    forensics_ela_synth_threshold: float = 60.0
    forensics_audio_phase_jump_threshold: float = 0.35
    forensics_video_jitter_threshold: float = 12.0
    forensics_text_low_perplexity_threshold: float = 22.0

    # ── HuggingFace ────────────────────────────────────────────────────────────
    hf_token: str | None = None

    # ── CORS ───────────────────────────────────────────────────────────────────
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v: str | list) -> list[str]:
        """Accept ALLOWED_ORIGINS as a JSON string or a Python list."""
        if isinstance(v, str):
            return json.loads(v)
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


# ── Bounding-Box Grounded Schema Re-exports ──────────────────────────────────
# Convenience aliases to support direct imports from app.config
from app.schemas import (
    AuthenticityReport,
    AuthenticityReportSchema,
    DetectedTamperedRegion,
    TamperedRegionCategory,
    ModerationResponse,
    ModerationResult,
)

__all__ = [
    "Settings",
    "get_settings",
    "DetectedTamperedRegion",
    "TamperedRegionCategory",
    "AuthenticityReport",
    "AuthenticityReportSchema",
    "ModerationResponse",
    "ModerationResult",
]

