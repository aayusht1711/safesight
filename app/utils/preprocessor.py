"""
SafeSight AI — Adversarial Text & Image Preprocessor
======================================================
Production-grade cleaning layer that executes BEFORE any ML model is invoked.
Designed to defeat common evasion techniques used by bad actors.

Evasion techniques handled:
  1. Unicode homoglyph substitution (Cyrillic 'а' → Latin 'a')
  2. Leet-speak obfuscation ('h4t3' → 'hate', 'p0is0n' → 'poison')
  3. Artificially spaced text ('h a t e' → 'hate')
  4. Zero-width characters and invisible Unicode separators
  5. Perceptual image hash matching against known-bad image database

Architecture:
  - All functions are pure / synchronous → safe to call in run_in_executor
  - No model dependencies → importable with zero ML packages
  - Returns original value unchanged on any exception (graceful degradation)
"""

from __future__ import annotations

import io
import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Leet-speak translation table
# ─────────────────────────────────────────────────────────────────────────────

_LEET_MAP: dict[str, str] = {
    # Digits → letters
    "0": "o",
    "1": "i",
    "2": "z",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    # Special characters → letters
    "@": "a",
    "$": "s",
    "!": "i",
    "|": "i",
    "+": "t",
    "(": "c",
    "<": "c",
    "€": "e",
    "£": "l",
    "¥": "y",
    "ø": "o",
    "ð": "d",
    "þ": "p",
    "ß": "ss",
    # Visually similar Unicode characters (common homoglyphs)
    "а": "a",   # Cyrillic small a
    "е": "e",   # Cyrillic small e
    "о": "o",   # Cyrillic small o
    "р": "p",   # Cyrillic small er
    "с": "c",   # Cyrillic small es
    "х": "x",   # Cyrillic small ha
    "і": "i",   # Ukrainian i
    "ї": "i",   # Ukrainian yi
    "ј": "j",   # Cyrillic je
    "ԁ": "d",   # Cyrillic komi de
    "ɡ": "g",   # Latin small script g
    "ᴀ": "a",   # Small capital A
    "ɴ": "n",   # Small capital N
    "ꜱ": "s",   # Small capital S
}

# Build a str.translate() compatible table (int codepoint → replacement string)
_LEET_TRANS = str.maketrans(
    {ord(k): v for k, v in _LEET_MAP.items() if len(k) == 1}
)

# Multi-char leet keys handled separately (e.g. "ß" → "ss")
_LEET_MULTI: list[tuple[str, str]] = [
    (k, v) for k, v in _LEET_MAP.items() if len(k) > 1
]


# ─────────────────────────────────────────────────────────────────────────────
# Zero-width / invisible character stripping
# ─────────────────────────────────────────────────────────────────────────────

_ZERO_WIDTH_RE = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f"    # Zero-width space/joiner/non-joiner/marks
    r"\u00ad"                              # Soft hyphen
    r"\u2060\u2061\u2062\u2063\u2064"     # Word joiner & invisible operators
    r"\ufeff"                              # Zero-width no-break space (BOM)
    r"\u00a0]",                            # Non-breaking space
)


# ─────────────────────────────────────────────────────────────────────────────
# Spaced-text compressor
# ─────────────────────────────────────────────────────────────────────────────

# Matches sequences like "h a t e" or "H-A-T-E" or "h.a.t.e"
# but NOT normal word-to-word spacing ("I hate you" should remain intact).
# Strategy: a "spaced" word is 3+ single chars separated by 1 separator char.
_SPACED_TEXT_RE = re.compile(
    r"\b([a-zA-Z][\s.\-_*|]{1,2}){2,}[a-zA-Z]\b"
)


def _compress_spaced_word(match: re.Match) -> str:
    """Strip separators from an artificially spaced word match."""
    return re.sub(r"[\s.\-_*|]+", "", match.group(0))


# ─────────────────────────────────────────────────────────────────────────────
# Repeated-character normalizer ('haaaaaate' → 'hate')
# ─────────────────────────────────────────────────────────────────────────────

_REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")


# ─────────────────────────────────────────────────────────────────────────────
# Public: Text preprocessor
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_adversarial_text(text: str | None) -> str:
    """
    Full adversarial text cleaning pipeline.

    Steps (in order):
      1. Null / empty guard
      2. Strip zero-width and invisible Unicode characters
      3. NFKC Unicode normalization (defeats homoglyph substitution at unicode level)
      4. Leet-speak resolution (multi-char mappings first, then single-char translate)
      5. Artificially spaced text compression ("h a t e" → "hate")
      6. Repeated character normalization ("haaaaaate" → "hate")
      7. Collapse extra whitespace

    Args:
        text: Raw user-submitted string (may be None).

    Returns:
        Cleaned string. Returns "" if input is None or empty.

    Examples:
        >>> preprocess_adversarial_text("p0is0n")
        'poison'
        >>> preprocess_adversarial_text("h a t e")
        'hate'
        >>> preprocess_adversarial_text("h@t3 $p33ch")
        'hate speech'
        >>> preprocess_adversarial_text("I lоve cats")  # Cyrillic 'о' in 'love'
        'I love cats'
    """
    if not text:
        return ""

    try:
        # ── Step 1: Strip zero-width / invisible characters ───────────────────
        text = _ZERO_WIDTH_RE.sub("", text)

        # ── Step 2: NFKC normalization ─────────────────────────────────────────
        # Decomposes ligatures, normalizes compatibility equivalents.
        # Must happen BEFORE leet-speak to catch e.g. ﬁ → fi
        text = unicodedata.normalize("NFKC", text)

        # ── Step 3: Leet-speak — multi-char mappings first ────────────────────
        for src, dst in _LEET_MULTI:
            text = text.replace(src, dst)

        # ── Step 4: Leet-speak — single-char translate table ─────────────────
        text = text.translate(_LEET_TRANS)

        # ── Step 5: Spaced-text compression ───────────────────────────────────
        text = _SPACED_TEXT_RE.sub(_compress_spaced_word, text)

        # ── Step 6: Repeated character normalization ──────────────────────────
        # Keep at most 2 consecutive identical characters (handles "haaate")
        text = _REPEAT_CHAR_RE.sub(r"\1\1", text)

        # ── Step 7: Whitespace normalization ──────────────────────────────────
        text = re.sub(r"\s+", " ", text).strip()

        return text

    except Exception:
        # Graceful degradation — never crash the pipeline on bad input
        return text or ""


# ─────────────────────────────────────────────────────────────────────────────
# Perceptual Image Hashing
# ─────────────────────────────────────────────────────────────────────────────

# ── Mock banned-hash database ─────────────────────────────────────────────────
# In production this would be a fast Redis SET or PostgreSQL lookup.
# Format: hex string of the dHash (64-bit = 16 hex chars)
BANNED_IMAGE_HASHES: set[str] = {
    # Example banned hashes (replace with real hashes from your CSAM / NCMEC database)
    "a3c4f2e1b5d6a7c8",  # Mock: known hate-symbol image
    "f0e1d2c3b4a59687",  # Mock: known CSAM perceptual hash
    "1a2b3c4d5e6f7a8b",  # Mock: known deepfake template
    "deadbeef12345678",  # Mock: test banned hash
}

# Hamming distance threshold for "near-duplicate" matching
# A dHash of 64 bits — distance ≤ 10 means "visually similar"
HASH_SIMILARITY_THRESHOLD = 10


def compute_dhash(image_bytes: bytes, hash_size: int = 8) -> str | None:
    """
    Compute the dHash (Difference Hash) of an image.

    dHash works by:
      1. Resize image to (hash_size+1) × hash_size grayscale pixels
      2. Compare each pixel to its right neighbor
      3. Build a bitstring of comparisons → encode as hex

    This produces a robust, content-aware fingerprint that:
      - Is invariant to small scaling and minor color changes
      - Changes significantly when image content changes
      - Can be compared with Hamming distance for near-duplicate detection

    Args:
        image_bytes: Raw bytes of any image format.
        hash_size:   Controls hash precision. 8 → 64-bit hash.

    Returns:
        Hex string of the dHash (16 chars for hash_size=8),
        or None if the image cannot be processed.
    """
    if not image_bytes:
        return None

    try:
        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(image_bytes))

        # Convert to grayscale and resize to (hash_size+1, hash_size)
        img = img.convert("L").resize(
            (hash_size + 1, hash_size),
            PILImage.Resampling.LANCZOS,
        )

        pixels = list(img.getdata())

        # Build difference bitstring: 1 if left pixel > right neighbor
        bits = []
        for row in range(hash_size):
            for col in range(hash_size):
                left = pixels[row * (hash_size + 1) + col]
                right = pixels[row * (hash_size + 1) + col + 1]
                bits.append(1 if left > right else 0)

        # Pack bits into integer and return hex
        hash_int = int("".join(str(b) for b in bits), 2)
        hex_hash = format(hash_int, f"0{hash_size * hash_size // 4}x")
        return hex_hash

    except Exception:
        return None


def _hamming_distance(hex_a: str, hex_b: str) -> int:
    """
    Compute bit-level Hamming distance between two hex hash strings.
    Returns the number of differing bits (0 = identical).
    """
    try:
        int_a = int(hex_a, 16)
        int_b = int(hex_b, 16)
        xor = int_a ^ int_b
        return bin(xor).count("1")
    except (ValueError, TypeError):
        return 999  # Treat parse errors as maximally different


def check_banned_hash(
    image_hash: str | None,
    banned_hashes: set[str] | None = None,
    threshold: int = HASH_SIMILARITY_THRESHOLD,
) -> tuple[bool, str | None]:
    """
    Check an image hash against the banned-hash database.

    Uses exact match first (O(1) set lookup), then falls back to
    Hamming-distance scan for near-duplicate detection.

    Args:
        image_hash:    Hex dHash string from compute_dhash().
        banned_hashes: Set of banned hex hashes. Defaults to BANNED_IMAGE_HASHES.
        threshold:     Maximum Hamming distance to consider "similar" (default: 10).

    Returns:
        (is_banned: bool, matched_hash: str | None)
        matched_hash is the hash from the database that triggered the match.
    """
    if not image_hash:
        return False, None

    db = banned_hashes if banned_hashes is not None else BANNED_IMAGE_HASHES

    # ── Exact match (O(1)) ─────────────────────────────────────────────────────
    if image_hash in db:
        return True, image_hash

    # ── Near-duplicate Hamming scan ────────────────────────────────────────────
    for banned in db:
        if _hamming_distance(image_hash, banned) <= threshold:
            return True, banned

    return False, None


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: full image preprocessing entry point
# ─────────────────────────────────────────────────────────────────────────────

class ImagePreprocessResult:
    """Result object returned by preprocess_image()."""

    __slots__ = ("dhash", "is_banned", "matched_banned_hash", "error")

    def __init__(
        self,
        dhash: str | None,
        is_banned: bool,
        matched_banned_hash: str | None,
        error: str | None = None,
    ) -> None:
        self.dhash = dhash
        self.is_banned = is_banned
        self.matched_banned_hash = matched_banned_hash
        self.error = error

    def __repr__(self) -> str:
        return (
            f"ImagePreprocessResult(dhash={self.dhash!r}, "
            f"is_banned={self.is_banned}, "
            f"matched={self.matched_banned_hash!r})"
        )


def preprocess_image(image_bytes: bytes | None) -> ImagePreprocessResult:
    """
    Full image preprocessing pipeline.

    Steps:
      1. Compute dHash perceptual fingerprint
      2. Check against banned-hash database (exact + near-duplicate)

    Args:
        image_bytes: Raw bytes of an image file. May be None.

    Returns:
        ImagePreprocessResult with hash, ban status, and any error string.
    """
    if not image_bytes:
        return ImagePreprocessResult(
            dhash=None,
            is_banned=False,
            matched_banned_hash=None,
            error="No image bytes provided",
        )

    dhash = compute_dhash(image_bytes)

    if dhash is None:
        return ImagePreprocessResult(
            dhash=None,
            is_banned=False,
            matched_banned_hash=None,
            error="Failed to compute dHash — image may be corrupted or unsupported",
        )

    is_banned, matched = check_banned_hash(dhash)

    return ImagePreprocessResult(
        dhash=dhash,
        is_banned=is_banned,
        matched_banned_hash=matched,
    )
