"""
Simple search cache for reducing API calls.

Provides a basic in-memory cache with optional disk persistence
for search results. Simplified from multi-level to single-level caching.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from pmkit.llm.models import SearchResult
from pmkit.utils.logger import get_logger
from pmkit.utils.paths import get_pmkit_dir

logger = get_logger(__name__)


class SearchCache:
    """
    Simple cache for search results.
    
    Features:
    - In-memory LRU cache with size limit
    - Optional disk persistence
    - TTL-based expiration
    - Simple statistics tracking
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        memory_size_limit: int = 100,
        default_ttl: int = 3600,  # 1 hour
    ):
        """
        Initialize search cache.
        
        Args:
            cache_dir: Directory for disk cache (optional)
            memory_size_limit: Maximum items in memory cache
            default_ttl: Default time-to-live in seconds
        """
        self.memory_cache: OrderedDict[str, tuple[SearchResult, datetime]] = OrderedDict()
        self.memory_size_limit = memory_size_limit
        self.default_ttl = default_ttl
        
        # Optional disk cache directory
        self.cache_dir = cache_dir or (get_pmkit_dir() / ".cache" / "search")
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
        }
        
        logger.debug(f"SearchCache initialized (memory_limit={memory_size_limit})")
    
    def _compute_cache_key(self, query: str, provider: str) -> str:
        """
        Compute cache key from query and provider.
        
        Args:
            query: Search query
            provider: Provider name
            
        Returns:
            SHA256 hash as cache key
        """
        # Normalize query - collapse multiple spaces
        import re
        normalized = re.sub(r'\s+', ' ', query.lower().strip())
        key_string = f"{provider}:{normalized}"
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, query: str, provider: str) -> Optional[SearchResult]:
        """
        Get cached search result.
        
        Args:
            query: Search query
            provider: Provider name
            
        Returns:
            Cached result if found and not expired, None otherwise
        """
        cache_key = self._compute_cache_key(query, provider)
        
        # Check memory cache
        if cache_key in self.memory_cache:
            result, timestamp = self.memory_cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.default_ttl):
                # Move to end (LRU)
                self.memory_cache.move_to_end(cache_key)
                self.stats["hits"] += 1
                result.cached = True
                logger.debug(f"Cache hit for: {query[:50]}...")
                return result
            else:
                # Expired
                del self.memory_cache[cache_key]
        
        # Check disk cache if available
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    
                    # Check expiration
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    if datetime.now() - timestamp < timedelta(seconds=self.default_ttl):
                        result = SearchResult(**data['result'])
                        result.cached = True
                        
                        # Add to memory cache
                        self._add_to_memory(cache_key, result)
                        
                        self.stats["hits"] += 1
                        logger.debug(f"Disk cache hit for: {query[:50]}...")
                        return result
                    else:
                        # Expired - remove file
                        cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to load from disk cache: {e}")
        
        self.stats["misses"] += 1
        return None
    
    def set(
        self,
        query: str,
        result: SearchResult,
        provider: str,
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache a search result.
        
        Args:
            query: Search query
            result: Search result to cache
            provider: Provider name
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        cache_key = self._compute_cache_key(query, provider)
        ttl = ttl or self.default_ttl
        
        # Add to memory cache
        self._add_to_memory(cache_key, result)
        
        # Save to disk if available
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.json"
            try:
                data = {
                    'result': result.model_dump(),
                    'timestamp': datetime.now().isoformat(),
                    'ttl': ttl,
                    'provider': provider,
                    'query': query,
                }
                with open(cache_file, 'w') as f:
                    json.dump(data, f, default=str)
                logger.debug(f"Cached to disk: {query[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to save to disk cache: {e}")
    
    def _add_to_memory(self, cache_key: str, result: SearchResult) -> None:
        """
        Add result to memory cache with LRU eviction.
        
        Args:
            cache_key: Cache key
            result: Search result
        """
        # Evict oldest if at limit
        if len(self.memory_cache) >= self.memory_size_limit:
            self.memory_cache.popitem(last=False)  # Remove oldest
        
        self.memory_cache[cache_key] = (result, datetime.now())
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.memory_cache.clear()
        
        # Clear disk cache
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove cache file: {e}")
        
        logger.info("Search cache cleared")
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
        
        return {
            "memory_items": len(self.memory_cache),
            "memory_limit": self.memory_size_limit,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "total_requests": total,
            "hit_rate": f"{hit_rate:.1f}%",
        }