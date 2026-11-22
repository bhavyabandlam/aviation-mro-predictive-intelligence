"""Safe caching for expensive data loading and computations (SRS v1.1 NFR-08).

Provides a TTL-aware, argument-keyed cache decorator for expensive dashboard
operations (JSON/Parquet loading, report generation). Design goals:

* Cache keys derived from function arguments only — no positional reliance.
* TTL-based expiration so results do not go stale.
* File-mtime invalidation: if a source file changes on disk, the cache entry
  is invalidated even before TTL expires.
* Never caches sensitive data (PII, secrets, model weights).
* Explicit ``clear()`` and ``cache_info()`` for observability.
"""

from __future__ import annotations

import functools
import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

__all__ = ["ttl_cache", "CacheEntry", "clear_all_caches", "cache_stats"]

P = ParamSpec("P")
T = TypeVar("T")

# Default TTL in seconds (5 minutes).
DEFAULT_TTL_SECONDS = 300

# Maximum number of cached entries per decorated function.
DEFAULT_MAX_SIZE = 128

# Registry of all active cache wrappers for global clear/stats.
_ACTIVE_CACHES: list[Any] = []


class CacheEntry:
    """A single cached value with its expiry time and source mtime."""

    __slots__ = ("value", "expires_at", "source_mtime")

    def __init__(self, value: Any, ttl_seconds: float, source_mtime: float | None) -> None:
        self.value = value
        self.expires_at = time.time() + ttl_seconds
        self.source_mtime = source_mtime

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def source_changed(self, path: Path | str | None) -> bool:
        """Return True if the source file's mtime differs from when cached."""
        if self.source_mtime is None or path is None:
            return False
        try:
            current = Path(path).stat().st_mtime
        except OSError:
            return True  # File missing → treat as changed
        return current != self.source_mtime


class _TTLCacheWrapper[**P, T]:
    """Internal wrapper implementing TTL + mtime-based invalidation."""

    def __init__(self, func: Callable[P, T], ttl_seconds: float, max_size: int) -> None:
        self.func = func
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: dict[str, CacheEntry] = {}
        self._order: list[str] = []  # LRU order
        _ACTIVE_CACHES.append(self)
        functools.update_wrapper(self, func)

    def _make_key(self, args: P.args, kwargs: P.kwargs) -> str:  # type: ignore[valid-type]
        """Build a stable, argument-only cache key."""
        payload: dict[str, Any] = {
            "args": [str(a) for a in args],
            "kwargs": {k: str(v) for k, v in sorted(kwargs.items())},
        }
        serialized = repr(payload).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()[:32]

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.max_size and self._order:
            oldest = self._order.pop(0)
            self._cache.pop(oldest, None)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        key = self._make_key(args, kwargs)

        entry = self._cache.get(key)
        if entry is not None and not entry.is_expired():
            # Check source file mtimes for any Path arguments.
            changed = False
            for arg in args:
                if isinstance(arg, (str, Path)) and entry.source_changed(arg):
                    changed = True
                    break
            for v in kwargs.values():
                if isinstance(v, (str, Path)) and entry.source_changed(v):
                    changed = True
                    break
            if not changed:
                # Move to end of LRU order.
                if key in self._order:
                    self._order.remove(key)
                self._order.append(key)
                return entry.value  # type: ignore[no-any-return]
            # Expired or source changed: fall through to recompute.

        # Recompute.
        value = self.func(*args, **kwargs)

        # Capture source mtime from any Path/str argument that looks like a file path.
        source_mtime: float | None = None
        for arg in args:
            if isinstance(arg, (str, Path)):
                try:
                    p = Path(arg)
                    if p.exists():
                        source_mtime = p.stat().st_mtime
                        break
                except OSError:
                    pass

        entry = CacheEntry(value, self.ttl_seconds, source_mtime)
        self._cache[key] = entry
        self._order.append(key)
        self._evict_if_needed()
        return value

    def clear(self) -> None:
        """Clear all entries for this wrapped function."""
        self._cache.clear()
        self._order.clear()

    def info(self) -> dict[str, Any]:
        """Return cache statistics for observability."""
        return {
            "currsize": len(self._cache),
            "maxsize": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hits": getattr(self, "_hits", 0),
            "misses": getattr(self, "_misses", 0),
        }


def ttl_cache(
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    max_size: int = DEFAULT_MAX_SIZE,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator factory for TTL + mtime-aware caching.

    Args:
        ttl_seconds: Time-to-live in seconds for each cache entry.
        max_size: Maximum number of entries per decorated function.

    Returns:
        A decorator that wraps the function with safe caching.

    Example::

        @ttl_cache(ttl_seconds=60, max_size=64)
        def load_report(path: Path) -> dict:
            return json.loads(path.read_text())
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        return _TTLCacheWrapper(func, ttl_seconds=ttl_seconds, max_size=max_size)

    return decorator


def clear_all_caches() -> None:
    """Clear every active TTL cache."""
    for wrapper in _ACTIVE_CACHES:
        wrapper.clear()


def cache_stats() -> list[dict[str, Any]]:
    """Return statistics for all active caches."""
    return [w.info() for w in _ACTIVE_CACHES]
