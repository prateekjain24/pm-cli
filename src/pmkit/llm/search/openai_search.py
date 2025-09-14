"""
OpenAI search provider using native Responses API.

Uses OpenAI's integrated web search capabilities through the Responses API.
GPT-5 includes native web search support with automatic citation handling.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any, Optional

from openai import AsyncOpenAI, OpenAIError

from pmkit.config.models import LLMProviderConfig
from pmkit.llm.models import SearchResult
from pmkit.llm.search.base import (
    BaseSearchProvider,
    SearchOptions,
    SearchTimeoutError,
    SearchUnavailableError,
)
from pmkit.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAISearchProvider(BaseSearchProvider):
    """
    OpenAI search provider using native Responses API.

    Leverages OpenAI's integrated web search capabilities for real-time
    web search with automatic citation extraction. GPT-5 includes native
    web search support as part of the Responses API.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[LLMProviderConfig] = None,
        **kwargs
    ):
        """
        Initialize OpenAI search provider.
        
        Args:
            api_key: OpenAI API key
            config: Optional LLM provider configuration
            **kwargs: Additional configuration
        """
        # Get API key from various sources
        if not api_key:
            if config and config.api_key:
                api_key = config.api_key.get_secret_value()
            else:
                api_key = os.getenv("OPENAI_API_KEY")
        
        super().__init__(api_key=api_key, **kwargs)
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Default model for search - GPT-5 supports web search via Responses API
        # Note: GPT-5 uses automatic tool selection (cannot force with tool_choice)
        self.model = "gpt-5"  # Default to GPT-5 which includes web search
        if config and config.model:
            self.model = config.model
        
        logger.info(f"OpenAI search provider initialized with model: {self.model}")
    
    async def search(
        self,
        query: str,
        options: Optional[SearchOptions] = None
    ) -> SearchResult:
        """
        Perform web search using OpenAI's native Responses API.

        GPT-5 includes integrated web search via openai.tools.webSearch.
        The model uses automatic tool selection to determine when to search.
        Note: Unlike GPT-4, GPT-5 cannot be forced to use web search via tool_choice.

        Args:
            query: The search query
            options: Search configuration options (mostly ignored - using native behavior)

        Returns:
            SearchResult containing content and citations

        Raises:
            SearchUnavailableError: If search service is unavailable
            SearchTimeoutError: If search times out
        """
        options = options or SearchOptions()
        
        try:
            # Use native Responses API with web search tool
            # Note: GPT-5 Responses API doesn't support temperature parameter
            response = await self.client.responses.create(
                model=self.model,
                tools=[{"type": "web_search"}],
                input=query,
                max_output_tokens=options.extras.get("max_tokens", 1500) if options.extras else 1500,
            )
            
            # Wait for completion if needed
            if hasattr(response, 'status'):
                while response.status == "in_progress":
                    await asyncio.sleep(0.5)
                    response = await self.client.responses.retrieve(response.id)
            
            # Parse the response
            return self._parse_response(response, query)
            
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise SearchUnavailableError("openai", str(e))
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI search: {e}")
            raise SearchUnavailableError("openai", f"Unexpected error: {e}")
    
    def _parse_response(self, response: Any, query: str) -> SearchResult:
        """
        Parse the Responses API response into a SearchResult.
        
        Args:
            response: API response object
            query: Original search query
            
        Returns:
            Parsed SearchResult
        """
        try:
            # Extract content from response
            content = ""
            citations = []
            
            # Responses API format
            if hasattr(response, 'output_text'):
                content = response.output_text or ""
            elif hasattr(response, 'output'):
                content = response.output or ""
            
            # Extract citations if available in response metadata
            # The Responses API includes citations automatically
            # but the exact format may vary
            if hasattr(response, 'annotations'):
                for annotation in response.annotations:
                    if hasattr(annotation, 'url'):
                        citations.append(annotation.url)
            
            # Fallback: extract URLs from content if no explicit citations
            if not citations and content:
                import re
                url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
                found_urls = re.findall(url_pattern, content)
                citations = list(set(found_urls))[:10]  # Dedupe and limit
            
            return SearchResult(
                content=content,
                citations=citations,
                query=query,
                timestamp=datetime.now(),
                cached=False,
            )
            
        except Exception as e:
            logger.error(f"Error parsing OpenAI search response: {e}")
            # Return minimal result on parse error
            return SearchResult(
                content="Search completed but failed to parse results.",
                citations=[],
                query=query,
                timestamp=datetime.now(),
                cached=False,
            )
    
    async def is_available(self) -> bool:
        """
        Check if OpenAI search is available.
        
        Returns:
            True if available, False otherwise
        """
        try:
            # Quick check with the Responses API
            response = await self.client.models.retrieve(self.model)
            return True
        except Exception as e:
            logger.warning(f"OpenAI search availability check failed: {e}")
            return False