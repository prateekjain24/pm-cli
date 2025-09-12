"""
OpenAI search provider implementation.

Uses GPT-5's web search capabilities through the Responses API.
Supports domain filtering, reasoning levels, and citation extraction.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, OpenAIError

from pmkit.config.models import LLMProviderConfig
from pmkit.llm.models import SearchResult
from pmkit.llm.search.base import (
    BaseSearchProvider,
    SearchDepth,
    SearchOptions,
    SearchTimeoutError,
    SearchUnavailableError,
)
from pmkit.utils.async_utils import timeout
from pmkit.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAISearchProvider(BaseSearchProvider):
    """
    OpenAI search provider using GPT-5's web search tools.
    
    This provider uses the new Responses API introduced with GPT-5
    that includes built-in web search capabilities.
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
        
        # Default model for search
        self.model = "gpt-5"  # GPT-5 supports web search
        if config and config.model:
            self.model = config.model
        
        logger.info(f"OpenAI search provider initialized with model: {self.model}")
    
    async def search(
        self,
        query: str,
        options: Optional[SearchOptions] = None
    ) -> SearchResult:
        """
        Perform web search using OpenAI's GPT-5.
        
        Args:
            query: The search query
            options: Search configuration options
            
        Returns:
            SearchResult containing content and citations
            
        Raises:
            SearchUnavailableError: If search service is unavailable
            SearchTimeoutError: If search times out
        """
        options = options or SearchOptions()
        
        try:
            # Build the web search tool configuration
            tools_config = self._build_tools_config(options)
            
            # Map search depth to reasoning effort
            reasoning_effort = self._map_depth_to_reasoning(options.depth)
            
            # Build the request
            request_params = {
                "model": self.model,
                "tools": [tools_config],
                "tool_choice": "auto",  # Let model decide when to search
                "input": query,
            }
            
            # Add reasoning configuration for GPT-5
            if reasoning_effort:
                request_params["reasoning"] = {"effort": reasoning_effort}
            
            # Include sources if requested
            if options.include_citations:
                request_params["include"] = ["web_search_call.action.sources"]
            
            # Perform the search with timeout
            @timeout(options.timeout)
            async def _search_with_timeout():
                # Use the new Responses API for GPT-5
                # Note: This is based on the expected API structure
                # The actual implementation may vary based on SDK updates
                try:
                    # Try new Responses API first (GPT-5)
                    if hasattr(self.client, 'responses'):
                        return await self.client.responses.create(**request_params)
                    else:
                        # Fallback to chat completions with tools
                        # This maintains compatibility with GPT-4
                        return await self._fallback_search(query, tools_config, options)
                except AttributeError:
                    # If responses API not available, use fallback
                    return await self._fallback_search(query, tools_config, options)
            
            response = await _search_with_timeout()
            
            # Parse the response
            return self._parse_response(response, query)
            
        except SearchTimeoutError:
            raise
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise SearchUnavailableError("openai", str(e))
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI search: {e}")
            raise SearchUnavailableError("openai", f"Unexpected error: {e}")
    
    def _build_tools_config(self, options: SearchOptions) -> Dict[str, Any]:
        """
        Build the web search tool configuration.
        
        Args:
            options: Search options
            
        Returns:
            Tool configuration dictionary
        """
        config = {"type": "web_search"}
        
        # Add domain filtering if specified
        filters = {}
        if options.allowed_domains:
            filters["allowed_domains"] = options.allowed_domains
        if options.blocked_domains:
            filters["blocked_domains"] = options.blocked_domains
        
        if filters:
            config["filters"] = filters
        
        return config
    
    def _map_depth_to_reasoning(self, depth: SearchDepth) -> str:
        """
        Map search depth to OpenAI reasoning effort.
        
        Args:
            depth: Search depth level
            
        Returns:
            Reasoning effort string for OpenAI API
        """
        mapping = {
            SearchDepth.MINIMAL: "minimal",
            SearchDepth.LOW: "low",
            SearchDepth.MEDIUM: "medium",
            SearchDepth.HIGH: "high",
        }
        return mapping.get(depth, "medium")
    
    async def _fallback_search(
        self,
        query: str,
        tools_config: Dict[str, Any],
        options: SearchOptions
    ) -> Any:
        """
        Fallback search using chat completions with function calling.
        
        This is used when the Responses API is not available.
        
        Args:
            query: Search query
            tools_config: Tools configuration
            options: Search options
            
        Returns:
            API response
        """
        # For now, use chat completions with a search-focused prompt
        # This will be updated when OpenAI releases official web search tools
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a search assistant. Provide comprehensive, "
                    "factual information based on the query. "
                    "Include relevant details and cite sources when possible."
                )
            },
            {"role": "user", "content": query}
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,  # Lower temperature for factual search
            max_tokens=options.extras.get("max_tokens", 1000),
        )
        
        return response
    
    def _parse_response(self, response: Any, query: str) -> SearchResult:
        """
        Parse the API response into a SearchResult.
        
        Args:
            response: API response object
            query: Original search query
            
        Returns:
            Parsed SearchResult
        """
        try:
            # Parse based on response type
            if hasattr(response, 'output_text'):
                # New Responses API format
                content = response.output_text
                
                # Extract citations from sources if available
                citations = []
                if hasattr(response, 'web_search_call'):
                    sources = getattr(response.web_search_call.action, 'sources', [])
                    citations = [source.url for source in sources if hasattr(source, 'url')]
                
            elif hasattr(response, 'choices'):
                # Chat completions format (fallback)
                content = response.choices[0].message.content
                
                # For fallback, we don't have real citations
                # This would be enhanced when proper web search tools are available
                citations = []
                
                # Try to extract URLs from the content if present
                import re
                url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
                found_urls = re.findall(url_pattern, content)
                citations = found_urls[:10]  # Limit to 10 citations
            
            else:
                # Unknown response format
                content = str(response)
                citations = []
            
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
            # Try a minimal API call to check availability
            response = await self.client.models.list()
            return True
        except Exception as e:
            logger.warning(f"OpenAI search availability check failed: {e}")
            return False