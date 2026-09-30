"""
SafeSight AI — Text Processor
================================
Text normalization and preprocessing utilities for the moderation pipeline.

Cleans, normalizes, and prepares raw user-submitted text before it is
passed to the Tier-1 DeBERTa scorer and Tier-2 VLM prompt builder.
"""

from __future__ import annotations

import re
import unicodedata


# ─────────────────────────────────────────────────────────────────────────────
# Compiled regex patterns
# ─────────────────────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_EXTRA_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_PUNCT_RE = re.compile(r"([!?.,]){3,}")
_EMOJI_RE = re.compile(
    "[\U00010000-\U0010ffff]",
    flags=re.UNICODE,
)


def preprocess_text(
    text: str,
    max_length: int = 2048,
    preserve_urls: bool = False,
) -> str:
    """
    Normalize and clean raw text for moderation analysis.

    Operations (in order):
      1. Unicode NFKC normalization (handles homoglyph attacks like 'а' → 'a')
      2. Strip leading/trailing whitespace
      3. URL replacement (optionally preserve for phishing detection)
      4. @mention normalization
      5. Hashtag expansion (keep readable text)
      6. Repeated punctuation collapse (!!!!! → !)
      7. Whitespace normalization
      8. Truncation to max_length

    Args:
        text:          Raw input text string.
        max_length:    Maximum character length (truncates beyond this).
        preserve_urls: If True, keep URLs in place (useful for phishing checks).

    Returns:
        Cleaned, normalized text string.
    """
    if not text:
        return ""

    # 1. Unicode normalization — defeats homoglyph substitution attacks
    text = unicodedata.normalize("NFKC", text)

    # 2. Strip
    text = text.strip()

    # 3. URL handling
    if preserve_urls:
        text = _URL_RE.sub("[URL]", text)
    else:
        text = _URL_RE.sub(" ", text)

    # 4. Normalize mentions
    text = _MENTION_RE.sub("[USER]", text)

    # 5. Expand hashtags — "#KillAllX" → "KillAllX" (preserves harmful content)
    text = _HASHTAG_RE.sub(r"\1", text)

    # 6. Collapse repeated punctuation
    text = _REPEATED_PUNCT_RE.sub(r"\1\1", text)

    # 7. Collapse extra whitespace
    text = _EXTRA_WHITESPACE_RE.sub(" ", text).strip()

    # 8. Truncate
    if len(text) > max_length:
        text = text[:max_length]

    return text


def extract_language_hints(text: str) -> dict[str, str]:
    """
    Lightweight heuristic language detection for logging / routing.
    Returns dict with detected_script and estimated_language.
    Not used for moderation decisions — informational only.
    """
    # Detect dominant Unicode script by sampling characters
    sample = text[:200]
    scripts: dict[str, int] = {}

    for char in sample:
        try:
            script = unicodedata.name(char).split()[0]
            scripts[script] = scripts.get(script, 0) + 1
        except ValueError:
            pass

    dominant = max(scripts, key=scripts.get, default="LATIN")

    script_to_language = {
        "LATIN": "en/es/fr/de (Latin script)",
        "ARABIC": "ar",
        "DEVANAGARI": "hi/mr/ne",
        "CJK": "zh/ja/ko",
        "CYRILLIC": "ru/uk/bg",
        "HANGUL": "ko",
        "HIRAGANA": "ja",
        "KATAKANA": "ja",
    }

    return {
        "detected_script": dominant,
        "estimated_language": script_to_language.get(dominant, "unknown"),
    }
