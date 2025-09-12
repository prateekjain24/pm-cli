"""
Search providers package for PM-Kit.

This package contains modular search providers for different LLM services,
enabling web search capabilities across multiple platforms.
"""

from pmkit.llm.search.base import (
    BaseSearchProvider,
    SearchDepth,
    SearchError,
    SearchOptions,
    SearchProviderNotFoundError,
    SearchTimeoutError,
    SearchUnavailableError,
)
from pmkit.llm.search.cache import SearchCache
from pmkit.llm.search.gemini_search import GeminiSearchProvider
from pmkit.llm.search.openai_search import OpenAISearchProvider

__all__ = [
    # Base classes
    "BaseSearchProvider",
    "SearchOptions",
    "SearchDepth",
    # Providers
    "OpenAISearchProvider",
    "GeminiSearchProvider",
    # Cache
    "SearchCache",
    # Errors
    "SearchError",
    "SearchUnavailableError",
    "SearchTimeoutError",
    "SearchProviderNotFoundError",
]