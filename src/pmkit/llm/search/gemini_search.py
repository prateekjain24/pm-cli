"""
Gemini search provider using native grounding.

Uses Google's native grounding feature through the google-genai SDK
for simplified, high-performance web search with automatic citation handling.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    Tool,
)

from pmkit.config.models import LLMProviderConfig
from pmkit.llm.models import SearchResult
from pmkit.llm.search.base import (
    BaseSearchProvider,
    SearchOptions,
    SearchUnavailableError,
)
from pmkit.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiSearchProvider(BaseSearchProvider):
    """
    Gemini search provider using native Google Search grounding.
    
    Leverages Google's grounding feature for real-time
    web search with automatic citation extraction.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[LLMProviderConfig] = None,
        **kwargs
    ):
        """
        Initialize Gemini search provider.
        
        Args:
            api_key: Google API key
            config: Optional LLM provider configuration
            **kwargs: Additional configuration
        """
        # Get API key from various sources
        if not api_key:
            if config and config.api_key:
                api_key = config.api_key.get_secret_value()
            else:
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        super().__init__(api_key=api_key, **kwargs)
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        # Default model for search
        self.model = "gemini-2.0-flash-latest"  
        if config and config.model:
            self.model = config.model
        
        logger.info(f"Gemini search provider initialized with model: {self.model}")
    
    async def search(
        self,
        query: str,
        options: Optional[SearchOptions] = None
    ) -> SearchResult:
        """
        Perform web search using Gemini's native grounding.
        
        Args:
            query: The search query
            options: Search configuration options (mostly ignored - using native behavior)
            
        Returns:
            SearchResult containing content and citations
            
        Raises:
            SearchUnavailableError: If search service is unavailable
        """
        options = options or SearchOptions()
        
        try:
            # Use native Google Search grounding
            response = await self.client.models.generate_content_async(
                model=self.model,
                contents=query,
                config=GenerateContentConfig(
                    tools=[
                        Tool(google_search=GoogleSearch())
                    ],
                    temperature=0.3,  # Lower temperature for factual search
                    max_output_tokens=options.extras.get("max_tokens", 1500) if options.extras else 1500,
                )
            )
            
            # Parse the response
            return self._parse_response(response, query)
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise SearchUnavailableError("gemini", str(e))
    
    def _parse_response(self, response, query: str) -> SearchResult:
        """
        Parse the Gemini response into a SearchResult.
        
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
            
            # Get the text content
            if hasattr(response, 'text'):
                content = response.text or ""
            elif hasattr(response, 'candidates') and response.candidates:
                # Access first candidate's content
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    content = " ".join([part.text for part in parts if hasattr(part, 'text')])
            
            # Extract citations from grounding metadata
            if hasattr(response, 'grounding_metadata'):
                metadata = response.grounding_metadata
                
                # Check for web search queries performed
                if hasattr(metadata, 'web_search_queries'):
                    logger.debug(f"Search queries performed: {metadata.web_search_queries}")
                
                # Extract grounding chunks (sources)
                if hasattr(metadata, 'grounding_chunks'):
                    for chunk in metadata.grounding_chunks:
                        if hasattr(chunk, 'web') and hasattr(chunk.web, 'uri'):
                            citations.append(chunk.web.uri)
                
                # Alternative: check grounding_supports
                if not citations and hasattr(metadata, 'grounding_supports'):
                    for support in metadata.grounding_supports:
                        if hasattr(support, 'grounding_chunk_indices'):
                            # Reference to chunks, already processed above
                            pass
            
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
            logger.error(f"Error parsing Gemini search response: {e}")
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
        Check if Gemini search is available.
        
        Returns:
            True if available, False otherwise
        """
        try:
            # Quick check by listing models
            models = await self.client.models.list_async()
            return True
        except Exception as e:
            logger.warning(f"Gemini search availability check failed: {e}")
            return False