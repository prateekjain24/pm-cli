"""
Search cache implementation.

Provides multi-level caching for search results to minimize API calls
and improve response times.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import diskcache

from pmkit.llm.models import SearchResult
from pmkit.utils.logger import get_logger

logger = get_logger(__name__)


class SearchCache:
    """
    Multi-level cache for search results.
    
    Features:
    - L1: In-memory cache for immediate hits
    - L2: Disk-based cache for persistence
    - Content-based cache keys
    - TTL support
    - Cache statistics
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        memory_size_limit: int = 100,
        disk_size_limit: int = 500 * 1024 * 1024,  # 500 MB
        default_ttl: int = 86400,  # 24 hours
    ):
        """
        Initialize search cache.
        
        Args:
            cache_dir: Directory for disk cache
            memory_size_limit: Max items in memory cache
            disk_size_limit: Max size of disk cache in bytes
            default_ttl: Default time-to-live in seconds
        """
        self.cache_dir = cache_dir or Path.home() / ".pmkit" / "search_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # L1: In-memory cache (simple dict with TTL)
        self.memory_cache: Dict[str, Tuple[SearchResult, float]] = {}
        self.memory_size_limit = memory_size_limit
        
        # L2: Disk cache using diskcache
        self.disk_cache = diskcache.Cache(
            str(self.cache_dir),
            size_limit=disk_size_limit,
            eviction_policy='least-recently-used'
        )
        
        self.default_ttl = default_ttl
        
        # Statistics
        self.stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "total_requests": 0,
        }
        
        logger.info(f"Search cache initialized at {self.cache_dir}")
    
    def get(self, query: str, provider: str = "default") -> Optional[SearchResult]:
        """
        Get cached search result.
        
        Args:
            query: The search query
            provider: Provider name for namespace isolation
            
        Returns:
            Cached SearchResult or None if not found/expired
        """
        self.stats["total_requests"] += 1
        cache_key = self._compute_cache_key(query, provider)
        
        # Check L1 (memory)
        if cache_key in self.memory_cache:
            result, expiry = self.memory_cache[cache_key]
            if time.time() < expiry:
                self.stats["memory_hits"] += 1
                logger.debug(f"Cache hit (memory): {cache_key[:8]}...")
                # Mark as cached
                result.cached = True
                return result
            else:
                # Expired, remove from memory
                del self.memory_cache[cache_key]
        
        # Check L2 (disk)
        try:
            cached_data = self.disk_cache.get(cache_key)
            if cached_data:
                result_dict, expiry = cached_data
                if time.time() < expiry:
                    self.stats["disk_hits"] += 1
                    logger.debug(f"Cache hit (disk): {cache_key[:8]}...")
                    
                    # Reconstruct SearchResult
                    result = SearchResult(**result_dict)
                    result.cached = True
                    
                    # Promote to L1
                    self._add_to_memory(cache_key, result, expiry)
                    
                    return result
                else:
                    # Expired, remove from disk
                    del self.disk_cache[cache_key]
        except Exception as e:
            logger.warning(f"Error reading from disk cache: {e}")
        
        self.stats["misses"] += 1
        logger.debug(f"Cache miss: {cache_key[:8]}...")
        return None
    
    def set(
        self,
        query: str,
        result: SearchResult,
        provider: str = "default",
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache a search result.
        
        Args:
            query: The search query
            result: The search result to cache
            provider: Provider name for namespace isolation
            ttl: Time-to-live in seconds (uses default if None)
        """
        cache_key = self._compute_cache_key(query, provider)
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        
        # Add to L1 (memory)
        self._add_to_memory(cache_key, result, expiry)
        
        # Add to L2 (disk)
        try:
            # Convert to dict for serialization
            result_dict = {
                "content": result.content,
                "citations": [str(url) for url in result.citations],
                "query": result.query,
                "timestamp": result.timestamp.isoformat(),
                "cached": False,  # Will be marked as cached when retrieved
            }
            self.disk_cache.set(cache_key, (result_dict, expiry))
            logger.debug(f"Cached result: {cache_key[:8]}... (TTL: {ttl}s)")
        except Exception as e:
            logger.warning(f"Error writing to disk cache: {e}")
    
    def clear(self, provider: Optional[str] = None) -> None:
        """
        Clear cache.
        
        Args:
            provider: Clear only for specific provider, or all if None
        """
        if provider:
            # Clear specific provider
            # This is more complex as we'd need to track keys by provider
            logger.warning("Provider-specific cache clear not implemented")
        else:
            # Clear all
            self.memory_cache.clear()
            self.disk_cache.clear()
            logger.info("Search cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total = self.stats["total_requests"]
        if total > 0:
            hit_rate = (
                (self.stats["memory_hits"] + self.stats["disk_hits"]) / total
            ) * 100
        else:
            hit_rate = 0
        
        return {
            **self.stats,
            "hit_rate": f"{hit_rate:.1f}%",
            "memory_size": len(self.memory_cache),
            "memory_limit": self.memory_size_limit,
            "disk_size": self.disk_cache.size,
            "disk_limit": self.disk_cache.size_limit,
        }
    
    def _compute_cache_key(self, query: str, provider: str) -> str:
        """
        Compute cache key from query and provider.
        
        Args:
            query: The search query
            provider: Provider name
            
        Returns:
            SHA256 hash as cache key
        """
        # Normalize query for better cache hits
        normalized_query = " ".join(query.lower().split())
        
        # Include provider in key for namespace isolation
        key_data = f"{provider}:{normalized_query}"
        
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _add_to_memory(
        self,
        cache_key: str,
        result: SearchResult,
        expiry: float
    ) -> None:
        """
        Add result to memory cache with LRU eviction.
        
        Args:
            cache_key: The cache key
            result: The search result
            expiry: Expiry timestamp
        """
        # Evict oldest if at limit
        if len(self.memory_cache) >= self.memory_size_limit:
            # Find oldest entry
            oldest_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k][1]
            )
            del self.memory_cache[oldest_key]
            logger.debug(f"Evicted from memory cache: {oldest_key[:8]}...")
        
        self.memory_cache[cache_key] = (result, expiry)
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, 'disk_cache'):
                self.disk_cache.close()
        except Exception as e:
            logger.debug(f"Error closing disk cache: {e}")