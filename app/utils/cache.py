"""
SafeSight AI — Async Caching Layer
=====================================
Provides a unified async caching interface that:
  - Uses Redis in production (non-blocking, distributed, 24h TTL)
  - Falls back to a thread-safe in-memory LRU dict in local dev
    (no Redis installation required to run tests)

Cache keys are content-addressed:
  - Image:  sha256 of image bytes  OR  dHash perceptual fingerprint
  - Text:   sha256 of cleaned text
  - Combo:  sha256(image_hash + ":" + text_hash)

The cache stores serialized ModerationResult JSON so the full pipeline
is bypassed entirely on a hit — zero model inference needed.

Integration:
  - Called at the START of orchestrator.run_moderation_pipeline()
  - Written at the END if the result is APPROVED or FLAGGED
    (HUMAN_AUDIT_REQUIRED results are NOT cached — they need fresh review)

Thread-safety:
  - InMemoryCache uses asyncio.Lock (safe in async context)
  - RedisCache is inherently async via aioredis
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

logger = logging.getLogger("safesight.cache")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TTL_SECONDS: int = 86_400          # 24 hours
MAX_INMEMORY_ENTRIES: int = 10_000         # LRU eviction after this
CACHE_KEY_PREFIX: str = "safesight:mod:v1"


# ─────────────────────────────────────────────────────────────────────────────
# Cache key builder
# ─────────────────────────────────────────────────────────────────────────────

def build_cache_key(
    image_dhash: str | None = None,
    image_bytes: bytes | None = None,
    text: str | None = None,
) -> str:
    """
    Build a deterministic cache key from available content identifiers.

    Priority:
      1. dHash (fast, content-aware fingerprint — preferred for images)
      2. SHA-256 of raw image bytes (exact match fallback)
      3. SHA-256 of cleaned text
      4. Combination of image + text identifiers

    Args:
        image_dhash:  Perceptual hash string from preprocessor.
        image_bytes:  Raw image bytes (used if no dhash available).
        text:         Cleaned text string.

    Returns:
        Namespaced cache key string.

    Examples:
        build_cache_key(image_dhash="a3c4f2e1b5d6a7c8")
        → "safesight:mod:v1:img:a3c4f2e1b5d6a7c8"

        build_cache_key(text="I love puppies")
        → "safesight:mod:v1:txt:sha256(I love puppies)"

        build_cache_key(image_dhash="abc123", text="hello world")
        → "safesight:mod:v1:combo:sha256(abc123:hello world)"
    """
    parts: list[str] = []

    if image_dhash:
        image_id = f"dhash:{image_dhash}"
    elif image_bytes:
        image_id = f"sha:{hashlib.sha256(image_bytes).hexdigest()[:32]}"
    else:
        image_id = None

    if text:
        text_id = f"sha:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:32]}"
    else:
        text_id = None

    if image_id and text_id:
        # Multimodal: hash the combination
        combo = f"{image_id}:{text_id}"
        key_body = f"combo:{hashlib.sha256(combo.encode()).hexdigest()[:32]}"
    elif image_id:
        key_body = f"img:{image_id}"
    elif text_id:
        key_body = f"txt:{text_id}"
    else:
        raise ValueError("At least one of image_dhash, image_bytes, or text must be provided")

    return f"{CACHE_KEY_PREFIX}:{key_body}"


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class ModerationCache(ABC):
    """Abstract async cache interface for moderation results."""

    @abstractmethod
    async def get(self, key: str) -> dict[str, Any] | None:
        """Return cached moderation result dict, or None on miss."""

    @abstractmethod
    async def set(
        self,
        key: str,
        value: dict[str, Any],
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        """Store a moderation result dict with the given TTL."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove a specific key (used for cache invalidation)."""

    @abstractmethod
    async def clear(self) -> None:
        """Flush all cached entries (use with care in production)."""

    @abstractmethod
    async def is_available(self) -> bool:
        """Return True if the cache backend is reachable."""


# ─────────────────────────────────────────────────────────────────────────────
# In-Memory LRU Cache (development / CI fallback)
# ─────────────────────────────────────────────────────────────────────────────

class _CacheEntry:
    """Internal holder for a cached value and its expiry timestamp."""
    __slots__ = ("value", "expires_at")

    def __init__(self, value: dict[str, Any], ttl: int) -> None:
        self.value = value
        self.expires_at = time.monotonic() + ttl


class InMemoryCache(ModerationCache):
    """
    Thread-safe async in-memory LRU cache with TTL expiry.

    Designed for local development and CI environments where Redis
    is not available. Uses asyncio.Lock for safe concurrent access.

    LRU eviction: oldest entries are removed when MAX_INMEMORY_ENTRIES
    is reached, preventing unbounded memory growth.
    """

    def __init__(self, max_entries: int = MAX_INMEMORY_ENTRIES) -> None:
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._max_entries = max_entries
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> dict[str, Any] | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            # TTL check
            if time.monotonic() > entry.expires_at:
                del self._store[key]
                self._misses += 1
                logger.debug("Cache TTL expired", extra={"key": key})
                return None

            # LRU: move to end (most recently used)
            self._store.move_to_end(key)
            self._hits += 1
            logger.debug("Cache hit", extra={"key": key})
            return entry.value

    async def set(
        self,
        key: str,
        value: dict[str, Any],
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        async with self._lock:
            # LRU eviction if at capacity
            while len(self._store) >= self._max_entries:
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug("LRU eviction", extra={"evicted_key": evicted_key})

            self._store[key] = _CacheEntry(value, ttl_seconds)
            self._store.move_to_end(key)
            logger.debug("Cache set", extra={"key": key, "ttl": ttl_seconds})

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
            logger.info("InMemoryCache cleared")

    async def is_available(self) -> bool:
        return True  # Always available

    @property
    def stats(self) -> dict[str, int]:
        """Return cache hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate_pct": round(100 * self._hits / total) if total else 0,
            "entries": len(self._store),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Redis Cache (production)
# ─────────────────────────────────────────────────────────────────────────────

class RedisCache(ModerationCache):
    """
    Async Redis-backed cache for production deployments.

    Uses aioredis for non-blocking I/O. Serializes ModerationResult
    dicts as JSON strings. Falls back gracefully on connection errors
    (cache miss semantics — the pipeline continues without caching).

    Usage:
        redis = await aioredis.from_url("redis://localhost:6379/0")
        cache = RedisCache(redis)
    """

    def __init__(self, redis_client: Any) -> None:
        """
        Args:
            redis_client: An aioredis client instance (already connected).
        """
        self._redis = redis_client

    async def get(self, key: str) -> dict[str, Any] | None:
        try:
            raw = await self._redis.get(key)
            if raw is None:
                logger.debug("Redis cache miss", extra={"key": key})
                return None
            logger.debug("Redis cache hit", extra={"key": key})
            return json.loads(raw)
        except Exception as exc:
            logger.warning(
                "Redis GET error — treating as cache miss",
                extra={"key": key, "error": str(exc)},
            )
            return None

    async def set(
        self,
        key: str,
        value: dict[str, Any],
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        try:
            serialized = json.dumps(value, default=str)
            await self._redis.setex(key, ttl_seconds, serialized)
            logger.debug("Redis cache set", extra={"key": key, "ttl": ttl_seconds})
        except Exception as exc:
            logger.warning(
                "Redis SET error — result not cached",
                extra={"key": key, "error": str(exc)},
            )

    async def delete(self, key: str) -> None:
        try:
            await self._redis.delete(key)
        except Exception as exc:
            logger.warning("Redis DELETE error", extra={"key": key, "error": str(exc)})

    async def clear(self) -> None:
        """Flush all SafeSight cache keys (uses SCAN — safe for large dbs)."""
        try:
            cursor = 0
            pattern = f"{CACHE_KEY_PREFIX}:*"
            deleted = 0
            while True:
                cursor, keys = await self._redis.scan(
                    cursor, match=pattern, count=100
                )
                if keys:
                    await self._redis.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            logger.info(f"RedisCache cleared {deleted} keys")
        except Exception as exc:
            logger.error("Redis CLEAR error", extra={"error": str(exc)})

    async def is_available(self) -> bool:
        try:
            await self._redis.ping()
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

async def create_cache(redis_url: str | None = None) -> ModerationCache:
    """
    Factory that returns the best available cache backend.

    If redis_url is provided and Redis is reachable → RedisCache.
    Otherwise → InMemoryCache (with a log warning).

    Args:
        redis_url: Redis connection URL e.g. "redis://localhost:6379/0"

    Returns:
        A ready-to-use ModerationCache instance.
    """
    if redis_url:
        try:
            import aioredis  # type: ignore

            client = await aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await client.ping()
            logger.info("Cache backend: Redis", extra={"url": redis_url})
            return RedisCache(client)
        except Exception as exc:
            logger.warning(
                "Redis unavailable — using InMemoryCache fallback",
                extra={"error": str(exc)},
            )

    logger.info("Cache backend: InMemoryCache (local dev mode)")
    return InMemoryCache()
