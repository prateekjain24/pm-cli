"""
Tests for search providers and GroundingAdapter.

Tests the modular search system including caching, provider selection,
and graceful degradation.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from pmkit.config.models import LLMProviderConfig
from pmkit.llm.grounding import GroundingAdapter
from pmkit.llm.models import SearchResult
from pmkit.llm.search import (
    BaseSearchProvider,
    OpenAISearchProvider,
    SearchCache,
    SearchDepth,
    SearchOptions,
    SearchProviderNotFoundError,
    SearchTimeoutError,
    SearchUnavailableError,
)


@pytest.fixture
def mock_config():
    """Create a mock LLM provider config."""
    return LLMProviderConfig(
        provider="openai",
        api_key=SecretStr("test-api-key"),
        model="gpt-5",
        timeout=30,
        max_retries=3,
    )


@pytest.fixture
def mock_search_result():
    """Create a mock search result."""
    return SearchResult(
        content="Test search content about AI developments",
        citations=["https://example.com/ai", "https://example.org/tech"],
        query="AI developments",
        timestamp=datetime.now(),
        cached=False,
    )


@pytest.fixture
async def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "search_cache"
    cache_dir.mkdir(exist_ok=True)
    yield cache_dir
    # Cleanup is handled by tmp_path fixture


class TestSearchCache:
    """Test search cache functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, temp_cache_dir, mock_search_result):
        """Test setting and getting from cache."""
        cache = SearchCache(cache_dir=temp_cache_dir, default_ttl=60)
        
        # Set in cache
        cache.set("test query", mock_search_result, provider="openai")
        
        # Get from cache
        result = cache.get("test query", provider="openai")
        
        assert result is not None
        assert result.content == mock_search_result.content
        assert result.cached is True
        assert len(result.citations) == 2
    
    @pytest.mark.asyncio
    async def test_cache_expiry(self, temp_cache_dir, mock_search_result):
        """Test cache expiry."""
        cache = SearchCache(cache_dir=temp_cache_dir, default_ttl=1)  # 1 second TTL
        
        # Set in cache
        cache.set("test query", mock_search_result, provider="openai", ttl=1)
        
        # Should be in cache immediately
        assert cache.get("test query", provider="openai") is not None
        
        # Wait for expiry
        await asyncio.sleep(1.5)
        
        # Should be expired
        assert cache.get("test query", provider="openai") is None
    
    def test_cache_stats(self, temp_cache_dir, mock_search_result):
        """Test cache statistics."""
        cache = SearchCache(cache_dir=temp_cache_dir)
        
        # Initial stats
        stats = cache.get_stats()
        assert stats["total_requests"] == 0
        assert stats["memory_hits"] == 0
        assert stats["disk_hits"] == 0
        assert stats["misses"] == 0
        
        # Miss
        cache.get("nonexistent", provider="openai")
        
        # Set and hit
        cache.set("test", mock_search_result, provider="openai")
        cache.get("test", provider="openai")  # Memory hit
        
        stats = cache.get_stats()
        assert stats["total_requests"] == 2
        assert stats["memory_hits"] == 1
        assert stats["misses"] == 1
    
    def test_cache_key_normalization(self, temp_cache_dir):
        """Test that cache keys are normalized."""
        cache = SearchCache(cache_dir=temp_cache_dir)
        
        # These should produce the same cache key
        key1 = cache._compute_cache_key("  Test  Query  ", "openai")
        key2 = cache._compute_cache_key("test query", "openai")
        
        assert key1 == key2
    
    def test_memory_cache_eviction(self, temp_cache_dir, mock_search_result):
        """Test memory cache LRU eviction."""
        cache = SearchCache(cache_dir=temp_cache_dir, memory_size_limit=2)
        
        # Add 3 items to cache with limit of 2
        cache.set("query1", mock_search_result, provider="openai")
        cache.set("query2", mock_search_result, provider="openai")
        cache.set("query3", mock_search_result, provider="openai")
        
        # Memory cache should only have 2 items
        assert len(cache.memory_cache) == 2
        
        # First item should be evicted
        key1 = cache._compute_cache_key("query1", "openai")
        assert key1 not in cache.memory_cache


class TestBaseSearchProvider:
    """Test base search provider functionality."""
    
    def test_normalize_query(self):
        """Test query normalization."""
        class TestProvider(BaseSearchProvider):
            async def search(self, query: str, options=None):
                return SearchResult(content="", citations=[], query=query)
        
        provider = TestProvider()
        
        # Test normalization
        assert provider._normalize_query("  Test  Query  ") == "test query"
        assert provider._normalize_query("UPPERCASE") == "uppercase"
        assert provider._normalize_query("multiple   spaces") == "multiple spaces"


class TestOpenAISearchProvider:
    """Test OpenAI search provider."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_config):
        """Test provider initialization."""
        provider = OpenAISearchProvider(config=mock_config)
        
        assert provider.api_key == "test-api-key"
        assert provider.model == "gpt-5"
    
    @pytest.mark.asyncio
    async def test_search_success(self, mock_config, mock_search_result):
        """Test successful search."""
        provider = OpenAISearchProvider(config=mock_config)
        
        # Mock the OpenAI client response
        with patch.object(provider.client, 'responses', new_callable=MagicMock) as mock_responses:
            mock_response = MagicMock()
            mock_response.output_text = "Search results about AI"
            mock_response.web_search_call.action.sources = [
                MagicMock(url="https://example.com/ai")
            ]
            
            mock_responses.create = AsyncMock(return_value=mock_response)
            
            result = await provider.search("AI developments")
            
            assert result.content == "Search results about AI"
            assert len(result.citations) == 1
            assert result.query == "AI developments"
    
    @pytest.mark.asyncio
    async def test_search_fallback(self, mock_config):
        """Test fallback to chat completions."""
        provider = OpenAISearchProvider(config=mock_config)
        
        # Mock chat completions for fallback
        with patch.object(
            provider.client.chat.completions,
            'create',
            new_callable=AsyncMock
        ) as mock_create:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content="Fallback search results"))
            ]
            mock_create.return_value = mock_response
            
            # Simulate responses API not available
            with patch.object(provider.client, 'responses', None):
                result = await provider.search("test query")
                
                assert result.content == "Fallback search results"
                assert result.query == "test query"
    
    @pytest.mark.asyncio
    async def test_search_with_options(self, mock_config):
        """Test search with custom options."""
        provider = OpenAISearchProvider(config=mock_config)
        
        options = SearchOptions(
            depth=SearchDepth.HIGH,
            allowed_domains=["example.com"],
            blocked_domains=["spam.com"],
            max_results=5,
            timeout=10.0,
        )
        
        # Mock the search
        with patch.object(provider, '_fallback_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Test"))]
            )
            
            result = await provider.search("test", options)
            
            # Verify options were processed
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_search_timeout(self, mock_config):
        """Test search timeout handling."""
        provider = OpenAISearchProvider(config=mock_config)
        
        options = SearchOptions(timeout=0.1)  # Very short timeout
        
        async def slow_search(*args, **kwargs):
            await asyncio.sleep(1)  # Longer than timeout
            return MagicMock()
        
        with patch.object(provider, '_fallback_search', new=slow_search):
            with pytest.raises(SearchTimeoutError):
                await provider.search("test", options)


class TestGroundingAdapter:
    """Test GroundingAdapter functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_config):
        """Test adapter initialization."""
        adapter = GroundingAdapter(mock_config)
        
        assert adapter.provider_name == "openai"
        assert adapter.cache_enabled is True
        assert adapter.cache is not None
    
    @pytest.mark.asyncio
    async def test_search_with_caching(self, mock_config, mock_search_result):
        """Test search with caching enabled."""
        adapter = GroundingAdapter(mock_config, cache_enabled=True)
        
        # Mock the provider search
        with patch.object(
            adapter.provider,
            'search',
            new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = mock_search_result
            
            # First search - should call provider
            result1 = await adapter.search("test query")
            assert mock_search.call_count == 1
            assert result1.cached is False
            
            # Second search - should use cache
            result2 = await adapter.search("test query")
            assert mock_search.call_count == 1  # Not called again
            assert result2.cached is True
    
    @pytest.mark.asyncio
    async def test_search_without_caching(self, mock_config, mock_search_result):
        """Test search with caching disabled."""
        adapter = GroundingAdapter(mock_config, cache_enabled=False)
        
        assert adapter.cache is None
        
        with patch.object(
            adapter.provider,
            'search',
            new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = mock_search_result
            
            # Multiple searches should all call provider
            await adapter.search("test query")
            await adapter.search("test query")
            
            assert mock_search.call_count == 2
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, mock_config):
        """Test graceful degradation when search unavailable."""
        adapter = GroundingAdapter(mock_config)
        
        # Mock provider to raise SearchUnavailableError
        with patch.object(
            adapter.provider,
            'search',
            side_effect=SearchUnavailableError("openai", "Service down")
        ):
            result = await adapter.search("test query")
            
            # Should return empty result instead of raising
            assert result.content == ""
            assert result.citations == []
            assert result.query == "test query"
    
    def test_list_providers(self):
        """Test listing available providers."""
        providers = GroundingAdapter.list_providers()
        
        assert "openai" in providers
        # Future providers would be listed here when implemented
    
    def test_register_provider(self):
        """Test dynamic provider registration."""
        # Register a custom provider
        GroundingAdapter.register_provider(
            "custom",
            "myapp.search.CustomProvider"
        )
        
        assert "custom" in GroundingAdapter.PROVIDER_REGISTRY
        assert GroundingAdapter.PROVIDER_REGISTRY["custom"] == "myapp.search.CustomProvider"
        
        # Clean up
        del GroundingAdapter.PROVIDER_REGISTRY["custom"]
    
    def test_provider_not_found(self, mock_config):
        """Test error when provider not found."""
        config = LLMProviderConfig(
            provider="nonexistent",
            api_key=SecretStr("test"),
            model="test",
        )
        
        with pytest.raises(SearchProviderNotFoundError) as exc_info:
            GroundingAdapter(config)
        
        assert "nonexistent" in str(exc_info.value)
        assert "Available providers" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, mock_config, mock_search_result):
        """Test getting cache statistics."""
        adapter = GroundingAdapter(mock_config)
        
        # Perform a search
        with patch.object(
            adapter.provider,
            'search',
            new_callable=AsyncMock,
            return_value=mock_search_result
        ):
            await adapter.search("test")
        
        stats = adapter.get_cache_stats()
        assert stats is not None
        assert "total_requests" in stats
        assert "hit_rate" in stats
    
    def test_clear_cache(self, mock_config):
        """Test clearing the cache."""
        adapter = GroundingAdapter(mock_config)
        
        # Clear cache should not raise
        adapter.clear_cache()
        
        # With cache disabled
        adapter_no_cache = GroundingAdapter(mock_config, cache_enabled=False)
        adapter_no_cache.clear_cache()  # Should handle gracefully