"""
SafeSight AI — Unit Tests for Preprocessor & Cache (v2)
=========================================================
Tests for the three new production components:
  - app/utils/preprocessor.py  (adversarial text + image hashing)
  - app/utils/cache.py          (InMemoryCache + key builder)
  - Integration: preprocessor → mock pipeline end-to-end

Run with:
    python -m pytest tests/test_unit_v2.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessor — Text
# ─────────────────────────────────────────────────────────────────────────────

class TestAdversarialTextPreprocessor:
    """Unit tests for preprocess_adversarial_text()."""

    def _clean(self, text: str) -> str:
        from app.utils.preprocessor import preprocess_adversarial_text
        return preprocess_adversarial_text(text)

    # ── Leet-speak ────────────────────────────────────────────────────────────

    def test_leet_zero_to_o(self):
        assert "poison" in self._clean("p0is0n")

    def test_leet_at_to_a(self):
        assert "hate" in self._clean("h@te")

    def test_leet_dollar_to_s(self):
        assert "self" in self._clean("$elf")

    def test_leet_one_to_i(self):
        assert "kill" in self._clean("k1ll")

    def test_leet_three_to_e(self):
        # 3 → e in leet table: h3te → hete (e replaces 3)
        result = self._clean("h3te")
        assert "hete" in result  # 3 maps to 'e', producing 'hete'
        assert "3" not in result  # digit '3' stripped

    def test_leet_compound(self):
        result = self._clean("h@t3 $p33ch")
        assert "hate" in result
        assert "speech" in result

    def test_leet_full_phrase(self):
        result = self._clean("dr1nk p0is0n and k1ll y0urs3lf")
        assert "drink" in result
        assert "poison" in result
        assert "kill" in result
        assert "yourself" in result

    # ── Spaced-text compression ────────────────────────────────────────────────

    def test_spaced_word_hate(self):
        result = self._clean("h a t e")
        assert "hate" in result

    def test_spaced_word_kill(self):
        result = self._clean("k i l l")
        assert "kill" in result

    def test_spaced_deepfake(self):
        result = self._clean("d e e p f a k e video")
        assert "deepfake" in result

    def test_normal_sentence_not_compressed(self):
        # "I love cats" — single chars preceded/followed by spaces should not collapse
        result = self._clean("I love cats and dogs")
        assert "love" in result
        assert "cats" in result

    # ── Unicode / homoglyphs ───────────────────────────────────────────────────

    def test_cyrillic_o_normalized(self):
        # Cyrillic 'о' (U+043E) in 'poison'
        result = self._clean("p\u043eis\u043en")
        assert "poison" in result

    def test_cyrillic_a_normalized(self):
        # Cyrillic 'а' (U+0430) in 'hate'
        result = self._clean("h\u0430te")
        assert "hate" in result

    def test_nfkc_normalizes_ligatures(self):
        # ﬁ (fi ligature U+FB01) should become "fi"
        result = self._clean("\ufb01lter")
        assert "filter" in result

    # ── Zero-width characters ─────────────────────────────────────────────────

    def test_zero_width_space_stripped(self):
        # Zero-width space between chars
        result = self._clean("h\u200bate")
        assert "\u200b" not in result

    def test_zero_width_joiner_stripped(self):
        result = self._clean("kill\u200d yourself")
        assert "\u200d" not in result

    # ── Repeated characters ────────────────────────────────────────────────────

    def test_repeated_chars_normalized(self):
        result = self._clean("haaaaaate")
        assert len(result) < len("haaaaaate")
        assert "ha" in result  # at most 2 repeated chars kept

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_none_input(self):
        from app.utils.preprocessor import preprocess_adversarial_text
        assert preprocess_adversarial_text(None) == ""

    def test_empty_string(self):
        assert self._clean("") == ""

    def test_clean_text_unchanged(self):
        text = "Hello, I love spending time with my family."
        result = self._clean(text)
        # Clean text should pass through essentially unchanged
        assert "Hello" in result
        assert "family" in result

    def test_whitespace_collapsed(self):
        result = self._clean("too   many    spaces")
        assert "  " not in result  # no double-spaces


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessor — Image hashing
# ─────────────────────────────────────────────────────────────────────────────

class TestImagePreprocessor:
    """Unit tests for compute_dhash, check_banned_hash, preprocess_image."""

    def _make_test_image_bytes(self, width: int = 100, height: int = 100, color=(128, 128, 128)) -> bytes:
        """Create minimal valid PNG bytes for testing."""
        try:
            import io
            from PIL import Image
            img = Image.new("RGB", (width, height), color=color)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            pytest.skip("Pillow not installed — skipping image hash tests")

    def test_dhash_returns_hex_string(self):
        from app.utils.preprocessor import compute_dhash
        img_bytes = self._make_test_image_bytes()
        h = compute_dhash(img_bytes)
        assert h is not None
        assert isinstance(h, str)
        assert len(h) == 16  # 64-bit hash = 16 hex chars

    def test_dhash_consistent(self):
        """Same image bytes should always produce the same hash."""
        from app.utils.preprocessor import compute_dhash
        img_bytes = self._make_test_image_bytes(color=(200, 100, 50))
        assert compute_dhash(img_bytes) == compute_dhash(img_bytes)

    def test_dhash_different_images(self):
        """Different images should produce different hashes."""
        from app.utils.preprocessor import compute_dhash
        import io
        from PIL import Image

        img1 = Image.new("RGB", (100, 100), color=(0, 0, 0))
        for x in range(50, 100):
            for y in range(100):
                img1.putpixel((x, y), (255, 255, 255))
        b1 = io.BytesIO()
        img1.save(b1, format="PNG")

        img2 = Image.new("RGB", (100, 100), color=(255, 255, 255))
        for x in range(50, 100):
            for y in range(100):
                img2.putpixel((x, y), (0, 0, 0))
        b2 = io.BytesIO()
        img2.save(b2, format="PNG")

        h1 = compute_dhash(b1.getvalue())
        h2 = compute_dhash(b2.getvalue())
        assert h1 != h2

    def test_dhash_none_input(self):
        from app.utils.preprocessor import compute_dhash
        assert compute_dhash(b"") is None
        assert compute_dhash(None) is None  # type: ignore

    def test_dhash_corrupted_bytes(self):
        from app.utils.preprocessor import compute_dhash
        result = compute_dhash(b"not_an_image_at_all_12345")
        assert result is None  # Should return None, not raise

    def test_check_banned_hash_exact_match(self):
        from app.utils.preprocessor import check_banned_hash
        banned = {"deadbeef12345678", "a1b2c3d4e5f6a7b8"}
        is_banned, matched = check_banned_hash("deadbeef12345678", banned_hashes=banned)
        assert is_banned is True
        assert matched == "deadbeef12345678"

    def test_check_banned_hash_no_match(self):
        from app.utils.preprocessor import check_banned_hash
        banned = {"deadbeef12345678"}
        is_banned, matched = check_banned_hash("1111111111111111", banned_hashes=banned, threshold=0)
        assert is_banned is False
        assert matched is None

    def test_check_banned_hash_none(self):
        from app.utils.preprocessor import check_banned_hash
        is_banned, matched = check_banned_hash(None)
        assert is_banned is False
        assert matched is None

    def test_preprocess_image_clean(self):
        from app.utils.preprocessor import preprocess_image
        img_bytes = self._make_test_image_bytes(color=(10, 20, 30))
        result = preprocess_image(img_bytes)
        assert result.dhash is not None
        # Fresh image shouldn't match the mock banned hashes (unless extremely unlikely collision)
        assert result.error is None

    def test_preprocess_image_none_input(self):
        from app.utils.preprocessor import preprocess_image
        result = preprocess_image(None)
        assert result.dhash is None
        assert result.is_banned is False
        assert result.error is not None

    def test_preprocess_image_corrupted(self):
        from app.utils.preprocessor import preprocess_image
        result = preprocess_image(b"garbage_bytes")
        assert result.dhash is None
        assert result.error is not None

    def test_hamming_distance(self):
        from app.utils.preprocessor import _hamming_distance
        # Identical hashes → distance 0
        assert _hamming_distance("ffffffffffffffff", "ffffffffffffffff") == 0
        # All bits flipped → distance 64
        assert _hamming_distance("ffffffffffffffff", "0000000000000000") == 64


# ─────────────────────────────────────────────────────────────────────────────
# Cache — InMemoryCache
# ─────────────────────────────────────────────────────────────────────────────

class TestInMemoryCache:
    """Unit tests for InMemoryCache."""

    def _make_cache(self):
        from app.utils.cache import InMemoryCache
        return InMemoryCache(max_entries=5)

    def _run(self, coro):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                return asyncio.run(coro)
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    def test_miss_on_empty(self):
        cache = self._make_cache()
        result = self._run(cache.get("nonexistent_key"))
        assert result is None

    def test_set_then_get(self):
        cache = self._make_cache()
        payload = {"decision": "APPROVED", "confidence_score": 0.99}
        self._run(cache.set("mykey", payload, ttl_seconds=60))
        result = self._run(cache.get("mykey"))
        assert result is not None
        assert result["decision"] == "APPROVED"

    def test_ttl_expiry(self):
        import time
        from app.utils.cache import InMemoryCache, _CacheEntry
        cache = InMemoryCache()
        # Manually insert an already-expired entry
        cache._store["expired_key"] = _CacheEntry({"decision": "FLAGGED"}, ttl=-1)
        result = self._run(cache.get("expired_key"))
        assert result is None  # Should be expired

    def test_lru_eviction(self):
        cache = self._make_cache()  # max_entries=5
        for i in range(6):
            self._run(cache.set(f"key_{i}", {"v": i}, ttl_seconds=3600))
        # key_0 should have been evicted (LRU = oldest)
        assert self._run(cache.get("key_0")) is None
        assert self._run(cache.get("key_5")) is not None

    def test_delete(self):
        cache = self._make_cache()
        self._run(cache.set("del_key", {"x": 1}, ttl_seconds=60))
        self._run(cache.delete("del_key"))
        assert self._run(cache.get("del_key")) is None

    def test_clear(self):
        cache = self._make_cache()
        self._run(cache.set("k1", {"v": 1}, ttl_seconds=60))
        self._run(cache.set("k2", {"v": 2}, ttl_seconds=60))
        self._run(cache.clear())
        assert self._run(cache.get("k1")) is None
        assert self._run(cache.get("k2")) is None

    def test_is_available(self):
        cache = self._make_cache()
        assert self._run(cache.is_available()) is True

    def test_hit_rate_stats(self):
        cache = self._make_cache()
        self._run(cache.set("s", {"v": 1}, ttl_seconds=60))
        self._run(cache.get("s"))      # hit
        self._run(cache.get("miss"))   # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate_pct"] == 50


# ─────────────────────────────────────────────────────────────────────────────
# Cache — Key builder
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheKeyBuilder:
    """Unit tests for build_cache_key()."""

    def test_image_dhash_key(self):
        from app.utils.cache import build_cache_key
        key = build_cache_key(image_dhash="abc123def456")
        assert "safesight:mod:v1" in key
        assert "abc123def456" in key

    def test_text_key(self):
        from app.utils.cache import build_cache_key
        key = build_cache_key(text="hello world")
        assert "safesight:mod:v1" in key
        assert "txt" in key

    def test_combo_key(self):
        from app.utils.cache import build_cache_key
        key = build_cache_key(image_dhash="abc", text="hello")
        assert "combo" in key

    def test_deterministic(self):
        from app.utils.cache import build_cache_key
        k1 = build_cache_key(text="same input")
        k2 = build_cache_key(text="same input")
        assert k1 == k2

    def test_different_inputs_different_keys(self):
        from app.utils.cache import build_cache_key
        k1 = build_cache_key(text="input A")
        k2 = build_cache_key(text="input B")
        assert k1 != k2

    def test_no_input_raises(self):
        from app.utils.cache import build_cache_key
        with pytest.raises(ValueError):
            build_cache_key()


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: Preprocessor → Mock Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessorPipelineIntegration:
    """Verify preprocessor correctly routes obfuscated text through the pipeline."""

    def _moderate(self, text: str):
        from app.utils.preprocessor import preprocess_adversarial_text
        from tests.mock_pipeline import mock_moderate
        cleaned = preprocess_adversarial_text(text)
        return mock_moderate(text=cleaned, simulate_latency=False)

    def test_leet_poison_flagged_as_self_harm(self):
        from app.schemas import ModerationDecision, ViolationCategory
        result = self._moderate("you should dr1nk p0is0n and k1ll y0urs3lf")
        assert result.decision == ModerationDecision.FLAGGED
        cats = {v.category for v in result.detected_violations}
        assert ViolationCategory.SELF_HARM in cats

    def test_leet_hate_speech_flagged(self):
        from app.schemas import ModerationDecision, ViolationCategory
        result = self._moderate("I h@t3 all [inferior race] people, d3@th t0 th3m")
        assert result.decision == ModerationDecision.FLAGGED
        cats = {v.category for v in result.detected_violations}
        assert ViolationCategory.HATE_SPEECH_AND_HARASSMENT in cats

    def test_spaced_deepfake_flagged(self):
        from app.schemas import ModerationDecision, ViolationCategory
        result = self._moderate("this is a d e e p f a k e synthetic face swap video")
        assert result.decision == ModerationDecision.FLAGGED
        cats = {v.category for v in result.detected_violations}
        assert ViolationCategory.DEEPFAKE in cats

    def test_layered_threat_flagged(self):
        from app.schemas import ModerationDecision, ViolationCategory
        result = self._moderate("1 kn0w wh3r3 y0u l1v3 and 1 w1ll k1ll y0u")
        assert result.decision == ModerationDecision.FLAGGED
        cats = {v.category for v in result.detected_violations}
        assert ViolationCategory.VIOLENT_THREATS in cats

    def test_clean_safe_text_approved(self):
        from app.schemas import ModerationDecision
        result = self._moderate("Good morning! Hope you have a wonderful day.")
        assert result.decision == ModerationDecision.APPROVED
        assert result.detected_violations == []

    def test_obfuscated_safe_text_still_approved(self):
        """Safe text with numbers shouldn't be falsely flagged."""
        from app.schemas import ModerationDecision
        # "1 love cats" → "i love cats" (leet) — still safe
        result = self._moderate("1 l0ve cats and d0gs. They are gr3at p3ts!")
        assert result.decision == ModerationDecision.APPROVED
