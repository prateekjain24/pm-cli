"""
GroundingAdapter - Unified interface for web search across LLM providers.

This module provides a provider-agnostic interface for web search capabilities,
making it easy to switch between different search providers or add new ones.

EXTENDING THIS MODULE:
======================
To add a new search provider, follow these steps:

1. Create a new provider class in src/pmkit/llm/search/
   Example: src/pmkit/llm/search/gemini_search.py
   
2. Inherit from BaseSearchProvider:
   ```python
   from pmkit.llm.search.base import BaseSearchProvider, SearchOptions
   from pmkit.llm.models import SearchResult
   
   class GeminiSearchProvider(BaseSearchProvider):
       async def search(self, query: str, options: Optional[SearchOptions] = None) -> SearchResult:
           # Implement Gemini-specific search logic
           # Use google.generativeai library
           # Parse results to SearchResult format
   ```
   
3. Register your provider in PROVIDER_REGISTRY below
   
4. Add provider-specific configuration if needed
   
5. Write tests in tests/test_search_providers.py

EXAMPLE IMPLEMENTATIONS:
========================

Gemini (Google Grounding):
--------------------------
    class GeminiSearchProvider(BaseSearchProvider):
        async def search(self, query: str, options: Optional[SearchOptions] = None) -> SearchResult:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            response = await model.generate_content_async(
                query,
                grounding_config={"enable_grounding": True}
            )
            
            # Parse grounding results
            content = response.text
            citations = [source.url for source in response.grounding_metadata.sources]
            
            return SearchResult(
                content=content,
                citations=citations,
                query=query
            )

Perplexity:
-----------
    class PerplexitySearchProvider(BaseSearchProvider):
        async def search(self, query: str, options: Optional[SearchOptions] = None) -> SearchResult:
            # Use Perplexity's dedicated search API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.perplexity.ai/search",
                    json={"query": query, "mode": "concise"},
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                data = response.json()
                
                return SearchResult(
                    content=data["answer"],
                    citations=data["sources"],
                    query=query
                )

Anthropic (Native Search):
--------------------------
    class AnthropicSearchProvider(BaseSearchProvider):
        async def search(self, query: str, options: Optional[SearchOptions] = None) -> SearchResult:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=self.api_key)
            
            response = await client.messages.create(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": query}],
                web_search=True  # Enable native web search
            )
            
            # Parse search results from response metadata
            content = response.content[0].text
            citations = response.search_results.urls if hasattr(response, 'search_results') else []
            
            return SearchResult(
                content=content,
                citations=citations,
                query=query
            )

Tavily Search API:
------------------
    class TavilySearchProvider(BaseSearchProvider):
        async def search(self, query: str, options: Optional[SearchOptions] = None) -> SearchResult:
            from tavily import AsyncTavily
            client = AsyncTavily(api_key=self.api_key)
            
            response = await client.search(
                query=query,
                max_results=options.max_results if options else 10
            )
            
            # Combine results into content
            content = "\\n\\n".join([
                f"{r['title']}\\n{r['snippet']}" 
                for r in response['results']
            ])
            citations = [r['url'] for r in response['results']]
            
            return SearchResult(
                content=content,
                citations=citations,
                query=query
            )
"""

from __future__ import annotations

import importlib
from typing import Dict, Optional, Type

from pmkit.config.models import LLMProviderConfig
from pmkit.exceptions import LLMError
from pmkit.llm.models import SearchResult
from pmkit.llm.search.base import (
    BaseSearchProvider,
    SearchOptions,
    SearchProviderNotFoundError,
    SearchUnavailableError,
)
from pmkit.llm.search.cache import SearchCache
from pmkit.utils.logger import get_logger

logger = get_logger(__name__)


class GroundingAdapter:
    """
    Unified interface for web search across LLM providers.
    
    This adapter provides a consistent interface for web search regardless
    of the underlying provider (OpenAI, Gemini, Anthropic, etc.).
    
    Features:
    - Provider abstraction
    - Automatic caching
    - Graceful degradation
    - Easy provider addition
    
    Usage:
        config = LLMProviderConfig(provider="openai", api_key="...")
        adapter = GroundingAdapter(config)
        result = await adapter.search("Latest AI developments")
    """
    
    # Provider registry - maps provider names to their implementation classes
    # Add new providers here after implementing them
    PROVIDER_REGISTRY: Dict[str, str] = {
        'openai': 'pmkit.llm.search.openai_search.OpenAISearchProvider',
        
        # Future providers - uncomment when implemented:
        # 'gemini': 'pmkit.llm.search.gemini_search.GeminiSearchProvider',
        # 'anthropic': 'pmkit.llm.search.anthropic_search.AnthropicSearchProvider',
        # 'perplexity': 'pmkit.llm.search.perplexity_search.PerplexitySearchProvider',
        # 'tavily': 'pmkit.llm.search.tavily_search.TavilySearchProvider',
        # 'serp': 'pmkit.llm.search.serp_search.SerpAPISearchProvider',
        # 'you': 'pmkit.llm.search.you_search.YouSearchProvider',
        # 'brave': 'pmkit.llm.search.brave_search.BraveSearchProvider',
    }
    
    def __init__(
        self,
        config: LLMProviderConfig,
        cache_enabled: bool = True,
        cache_ttl: int = 86400,  # 24 hours
    ):
        """
        Initialize the grounding adapter.
        
        Args:
            config: LLM provider configuration
            cache_enabled: Whether to enable caching
            cache_ttl: Cache time-to-live in seconds
        """
        self.config = config
        self.provider_name = config.provider
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        
        # Initialize cache if enabled
        self.cache = SearchCache(default_ttl=cache_ttl) if cache_enabled else None
        
        # Initialize the provider
        self.provider = self._create_provider()
        
        logger.info(
            f"GroundingAdapter initialized with {self.provider_name} provider"
            f" (cache: {cache_enabled})"
        )
    
    def _create_provider(self) -> BaseSearchProvider:
        """
        Create the search provider instance.
        
        Returns:
            Initialized search provider
            
        Raises:
            SearchProviderNotFoundError: If provider not found
        """
        if self.provider_name not in self.PROVIDER_REGISTRY:
            available = ", ".join(self.PROVIDER_REGISTRY.keys())
            raise SearchProviderNotFoundError(
                f"Provider '{self.provider_name}' not found. "
                f"Available providers: {available}"
            )
        
        # Dynamically import and instantiate the provider
        provider_path = self.PROVIDER_REGISTRY[self.provider_name]
        module_path, class_name = provider_path.rsplit('.', 1)
        
        try:
            module = importlib.import_module(module_path)
            provider_class: Type[BaseSearchProvider] = getattr(module, class_name)
            
            # Get API key from config
            api_key = None
            if self.config.api_key:
                api_key = self.config.api_key.get_secret_value()
            
            return provider_class(api_key=api_key, config=self.config)
            
        except ImportError as e:
            logger.error(f"Failed to import {provider_path}: {e}")
            raise SearchProviderNotFoundError(
                f"Could not load provider '{self.provider_name}': {e}"
            )
        except AttributeError as e:
            logger.error(f"Provider class not found in {module_path}: {e}")
            raise SearchProviderNotFoundError(
                f"Provider class '{class_name}' not found in module: {e}"
            )
    
    async def search(
        self,
        query: str,
        options: Optional[SearchOptions] = None,
        use_cache: bool = True,
    ) -> SearchResult:
        """
        Perform a web search using the configured provider.
        
        Args:
            query: The search query
            options: Search configuration options
            use_cache: Whether to use cache for this search
            
        Returns:
            SearchResult containing content and citations
            
        Raises:
            SearchUnavailableError: If search is unavailable
        """
        # Check cache first if enabled
        if self.cache and use_cache and self.cache_enabled:
            cached_result = self.cache.get(query, self.provider_name)
            if cached_result:
                logger.debug(f"Returning cached result for: {query[:50]}...")
                return cached_result
        
        try:
            # Perform the search
            logger.debug(f"Searching with {self.provider_name}: {query[:50]}...")
            result = await self.provider.search(query, options)
            
            # Cache the result if enabled
            if self.cache and use_cache and self.cache_enabled:
                self.cache.set(query, result, self.provider_name, self.cache_ttl)
            
            return result
            
        except SearchUnavailableError:
            # Try to return cached result even if expired
            if self.cache:
                logger.warning(
                    f"Search unavailable, checking for expired cache: {query[:50]}..."
                )
                # Try with extended time window
                cached_result = self.cache.get(query, self.provider_name)
                if cached_result:
                    logger.info("Returning expired cached result")
                    return cached_result
            
            # No cache available, return empty result (graceful degradation)
            logger.warning(f"Search unavailable and no cache for: {query[:50]}...")
            return SearchResult(
                content="",
                citations=[],
                query=query,
                cached=False,
            )
        
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise LLMError(
                f"Search failed: {str(e)}",
                provider=self.provider_name,
            )
    
    async def is_available(self) -> bool:
        """
        Check if the search provider is available.
        
        Returns:
            True if available, False otherwise
        """
        try:
            return await self.provider.is_available()
        except Exception as e:
            logger.warning(f"Provider availability check failed: {e}")
            return False
    
    def get_cache_stats(self) -> Optional[Dict]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics or None if cache disabled
        """
        if self.cache:
            return self.cache.get_stats()
        return None
    
    def clear_cache(self) -> None:
        """Clear the search cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Search cache cleared")
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """
        List available search providers.
        
        Returns:
            List of provider names
        """
        return list(cls.PROVIDER_REGISTRY.keys())
    
    @classmethod
    def register_provider(cls, name: str, provider_path: str) -> None:
        """
        Register a new search provider.
        
        This method allows dynamic registration of providers at runtime.
        
        Args:
            name: Provider name (e.g., 'custom')
            provider_path: Full module path to provider class
                          (e.g., 'myapp.search.CustomProvider')
        
        Example:
            GroundingAdapter.register_provider(
                'custom',
                'myapp.search.CustomSearchProvider'
            )
        """
        cls.PROVIDER_REGISTRY[name] = provider_path
        logger.info(f"Registered search provider: {name} -> {provider_path}")