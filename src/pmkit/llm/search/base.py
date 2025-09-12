"""
Base classes for search providers.

Provides abstract interfaces and common functionality for all search providers
to ensure consistent behavior across different implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pmkit.llm.models import SearchResult


class SearchDepth(str, Enum):
    """Search depth/reasoning levels."""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class SearchOptions:
    """
    Common search options for all providers.
    
    Providers may support additional options through the extras field.
    """
    
    # Search depth/reasoning level
    depth: SearchDepth = SearchDepth.MEDIUM
    
    # Domain filtering
    allowed_domains: Optional[List[str]] = None
    blocked_domains: Optional[List[str]] = None
    
    # Result limits
    max_results: int = 10
    
    # Timeout in seconds
    timeout: float = 30.0
    
    # Whether to include citations
    include_citations: bool = True
    
    # Provider-specific options
    extras: Dict[str, Any] = field(default_factory=dict)


class BaseSearchProvider(ABC):
    """
    Abstract base class for all search providers.
    
    All search providers must inherit from this class and implement
    the search method. This ensures consistent interface across providers.
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize search provider.
        
        Args:
            api_key: API key for the provider
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs
    
    @abstractmethod
    async def search(
        self,
        query: str,
        options: Optional[SearchOptions] = None
    ) -> SearchResult:
        """
        Perform a web search.
        
        Args:
            query: The search query
            options: Search configuration options
            
        Returns:
            SearchResult containing content and citations
            
        Raises:
            SearchUnavailableError: If search service is unavailable
            SearchTimeoutError: If search times out
        """
        pass
    
    async def is_available(self) -> bool:
        """
        Check if the search provider is available.
        
        Returns:
            True if provider is available, False otherwise
        """
        try:
            # Try a minimal search to test availability
            await self.search("test", SearchOptions(
                depth=SearchDepth.MINIMAL,
                max_results=1,
                timeout=5.0
            ))
            return True
        except Exception:
            return False
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for better caching.
        
        Args:
            query: Raw search query
            
        Returns:
            Normalized query string
        """
        # Remove extra whitespace
        query = " ".join(query.split())
        # Convert to lowercase for cache key
        return query.lower().strip()


class SearchError(Exception):
    """Base exception for search-related errors."""
    pass


class SearchUnavailableError(SearchError):
    """Raised when search service is unavailable."""
    
    def __init__(self, provider: str, message: str = ""):
        super().__init__(f"Search unavailable for {provider}: {message}")
        self.provider = provider


class SearchTimeoutError(SearchError):
    """Raised when search operation times out."""
    
    def __init__(self, timeout: float):
        super().__init__(f"Search timed out after {timeout} seconds")
        self.timeout = timeout


class SearchProviderNotFoundError(SearchError):
    """Raised when requested search provider is not found."""
    
    def __init__(self, provider: str):
        super().__init__(f"Search provider '{provider}' not found")
        self.provider = provider