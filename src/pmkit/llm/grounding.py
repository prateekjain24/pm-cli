"""
GroundingAdapter - Unified interface for web search across LLM providers.

This module provides a thin orchestration layer for web search capabilities,
routing requests to the appropriate native search provider implementation.

EXTENDING THIS MODULE:
======================
To add a new search provider:

1. Create a provider class in src/pmkit/llm/search/
2. Inherit from BaseSearchProvider
3. Register in PROVIDER_REGISTRY below
4. Write tests

The adapter now focuses on routing rather than complex abstraction,
as providers use their native search capabilities.
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
from pmkit.utils.logger import get_logger

logger = get_logger(__name__)


class GroundingAdapter:
    """
    Unified interface for web search across LLM providers.
    
    This thin orchestrator routes search requests to the appropriate
    provider's native search implementation.
    
    Features:
    - Simple provider routing
    - Graceful degradation
    - Easy provider addition
    
    Usage:
        config = LLMProviderConfig(provider="openai", api_key="...")
        adapter = GroundingAdapter(config)
        result = await adapter.search("Latest AI developments")
    """
    
    # Provider registry - maps provider names to their implementation classes
    PROVIDER_REGISTRY: Dict[str, str] = {
        'openai': 'pmkit.llm.search.openai_search.OpenAISearchProvider',
        'gemini': 'pmkit.llm.search.gemini_search.GeminiSearchProvider',
        
        # Future providers - uncomment when implemented:
        # 'anthropic': 'pmkit.llm.search.anthropic_search.AnthropicSearchProvider',
        # 'perplexity': 'pmkit.llm.search.perplexity_search.PerplexitySearchProvider',
    }
    
    def __init__(self, config: LLMProviderConfig):
        """
        Initialize the grounding adapter.
        
        Args:
            config: LLM provider configuration
        """
        self.config = config
        self.provider_name = config.provider
        
        # Initialize the provider
        self.provider = self._create_provider()
        
        logger.info(f"GroundingAdapter initialized with {self.provider_name} provider")
    
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
    ) -> SearchResult:
        """
        Perform a web search using the configured provider.
        
        Args:
            query: The search query
            options: Search configuration options
            
        Returns:
            SearchResult containing content and citations
        """
        try:
            # Perform the search using native provider
            logger.debug(f"Searching with {self.provider_name}: {query[:50]}...")
            result = await self.provider.search(query, options)
            return result
            
        except SearchUnavailableError:
            # Graceful degradation - return empty result
            logger.warning(f"Search unavailable for: {query[:50]}...")
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