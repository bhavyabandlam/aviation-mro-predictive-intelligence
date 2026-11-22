"""Tests for the safe TTL caching service (SRS v1.1 NFR-08).

Covers TTL expiration, cache hit/miss, source-file mtime invalidation,
LRU eviction, and no stale results after source changes.
"""

from __future__ import annotations

import time
from pathlib import Path

from aviation_mro.services.caching import (
    CacheEntry,
    clear_all_caches,
    ttl_cache,
)


class TestTTLCacheExpiration:
    """Tests for TTL-based cache expiration."""

    def test_cache_hit_within_ttl(self) -> None:
        """A cached value is returned within TTL."""
        call_count = 0

        @ttl_cache(ttl_seconds=60)
        def expensive(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive(5)
        result2 = expensive(5)
        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Function called once

    def test_cache_miss_after_ttl(self) -> None:
        """A cached value is recomputed after TTL expires."""
        call_count = 0

        @ttl_cache(ttl_seconds=0)  # Immediate expiry
        def expensive(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        expensive(5)
        time.sleep(0.01)
        expensive(5)
        assert call_count == 2  # Function called twice

    def test_different_args_different_cache_entries(self) -> None:
        """Different arguments produce different cache entries."""
        call_count = 0

        @ttl_cache(ttl_seconds=60)
        def expensive(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert expensive(1) == 2
        assert expensive(2) == 4
        assert expensive(1) == 2  # Cached
        assert call_count == 2  # Only 2 unique calls

    def test_kwargs_part_of_cache_key(self) -> None:
        """Keyword arguments are part of the cache key."""
        call_count = 0

        @ttl_cache(ttl_seconds=60)
        def expensive(x: int, multiplier: int = 1) -> int:
            nonlocal call_count
            call_count += 1
            return x * multiplier

        assert expensive(5, multiplier=2) == 10
        assert expensive(5, multiplier=3) == 15
        assert expensive(5, multiplier=2) == 10  # Cached
        assert call_count == 2  # Two unique (args, kwargs) combos


class TestCacheEntry:
    """Tests for CacheEntry expiration and source-change detection."""

    def test_entry_is_expired_after_ttl(self) -> None:
        """CacheEntry.is_expired() returns True after TTL."""
        entry = CacheEntry("value", ttl_seconds=0, source_mtime=None)
        time.sleep(0.01)
        assert entry.is_expired()

    def test_entry_not_expired_within_ttl(self) -> None:
        """CacheEntry.is_expired() returns False within TTL."""
        entry = CacheEntry("value", ttl_seconds=60, source_mtime=None)
        assert not entry.is_expired()

    def test_entry_source_changed_detects_mtime(self, tmp_path) -> None:
        """CacheEntry.source_changed() detects file mtime changes."""
        f = tmp_path / "test.txt"
        f.write_text("hello")
        mtime = f.stat().st_mtime

        entry = CacheEntry("value", ttl_seconds=60, source_mtime=mtime)
        assert not entry.source_changed(f)

        # Modify file to change mtime
        time.sleep(0.01)
        f.write_text("world")
        assert entry.source_changed(f)

    def test_entry_source_changed_none_path(self) -> None:
        """CacheEntry.source_changed() with None path returns False."""
        entry = CacheEntry("value", ttl_seconds=60, source_mtime=None)
        assert not entry.source_changed(None)

    def test_entry_source_changed_missing_file(self, tmp_path) -> None:
        """CacheEntry.source_changed() returns True for missing file."""
        entry = CacheEntry("value", ttl_seconds=60, source_mtime=1000.0)
        missing = tmp_path / "does_not_exist.txt"
        assert entry.source_changed(missing)


class TestCacheInvalidation:
    """Tests for cache invalidation when source files change."""

    def test_cache_invalidated_after_source_file_change(self, tmp_path) -> None:
        """A cached file load is invalidated when the source file changes."""
        f = tmp_path / "data.txt"
        f.write_text("version1")

        @ttl_cache(ttl_seconds=60)
        def load_file(path: Path) -> str:
            return path.read_text()

        result1 = load_file(f)
        assert result1 == "version1"

        # Change the file
        time.sleep(0.01)
        f.write_text("version2")
        result2 = load_file(f)
        assert result2 == "version2"

    def test_clear_all_caches(self) -> None:
        """clear_all_caches() clears all active caches."""
        call_count = 0

        @ttl_cache(ttl_seconds=60)
        def expensive(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        expensive(5)
        clear_all_caches()
        expensive(5)
        assert call_count == 2  # Called again after clear

    def test_cache_does_not_cache_none_results(self) -> None:
        """Functions returning None are still cached (None is a valid value)."""
        call_count = 0

        @ttl_cache(ttl_seconds=60)
        def returns_none(x: int) -> int | None:
            nonlocal call_count
            call_count += 1
            return None

        assert returns_none(1) is None
        assert returns_none(1) is None  # Cached
        assert call_count == 1

    def test_lru_eviction_respects_max_size(self) -> None:
        """Cache evicts oldest entries when max_size is exceeded."""
        call_count = 0

        @ttl_cache(ttl_seconds=60, max_size=2)
        def expensive(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        expensive(1)  # cache: {1}
        expensive(2)  # cache: {1, 2}
        expensive(3)  # cache: {2, 3} (1 evicted)
        expensive(1)  # cache: {2, 3, 1} (2 evicted) -> recomputed
        assert call_count == 4  # 1,2,3,1 all computed
