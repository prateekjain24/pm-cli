"""
OpenAI client wrapper for PM-Kit.

Provides async OpenAI API access with connection pooling, rate limiting,
retry logic, and graceful error handling.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Union

import httpx
from openai import AsyncOpenAI, AuthenticationError, RateLimitError, APITimeoutError
from openai.types.chat import ChatCompletion
from pydantic import SecretStr

from pmkit.config.models import LLMProviderConfig
from pmkit.exceptions import LLMError
from pmkit.llm.models import (
    ChatMessage,
    Choice,
    CompletionResponse,
    MessageRole,
    SearchResult,
    ToolCall,
    Usage,
)
from pmkit.utils.async_utils import retry_with_backoff, timeout
from pmkit.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIClient:
    """
    Async OpenAI client with enterprise features.
    
    Features:
    - Connection pooling with httpx
    - Rate limiting with semaphore
    - Automatic retries with exponential backoff
    - API key validation
    - Graceful error handling
    - Structured logging
    """
    
    def __init__(
        self,
        config: LLMProviderConfig,
        max_concurrent_requests: int = 10,
        validate_on_init: bool = True,
    ):
        """
        Initialize OpenAI client.
        
        Args:
            config: LLM provider configuration
            max_concurrent_requests: Maximum concurrent API requests
            validate_on_init: Whether to validate API key on initialization
        """
        self.config = config
        self.max_concurrent_requests = max_concurrent_requests
        
        # Get API key from config or environment
        api_key = self._get_api_key()
        
        # Create custom httpx client with connection pooling
        self.http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=25,
                max_connections=100,
                keepalive_expiry=30,
            ),
            timeout=httpx.Timeout(
                timeout=float(config.timeout),
                connect=5.0,
            ),
        )
        
        # Initialize OpenAI client with custom HTTP client
        self.client = AsyncOpenAI(
            api_key=api_key,
            http_client=self.http_client,
            max_retries=0,  # We handle retries ourselves
        )
        
        # Rate limiting semaphore
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Track if client is validated
        self._validated = False
        self.validate_on_init = validate_on_init
        
        logger.info(
            f"OpenAI client initialized",
            extra={
                "model": config.model,
                "max_concurrent": max_concurrent_requests,
                "timeout": config.timeout,
            }
        )
    
    def _get_api_key(self) -> str:
        """
        Get API key from config or environment.
        
        Returns:
            API key string
            
        Raises:
            LLMError: If no API key is found
        """
        # Check config first
        if self.config.api_key:
            return self.config.api_key.get_secret_value()
        
        # Check environment
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key
        
        raise LLMError(
            "OpenAI API key not found",
            provider="openai",
            suggestion="Set OPENAI_API_KEY in environment or .env file"
        )
    
    async def validate_api_key(self) -> bool:
        """
        Validate API key with a minimal API call.
        
        Returns:
            True if API key is valid
            
        Raises:
            LLMError: If API key is invalid
        """
        try:
            # Make a minimal API call to validate the key
            response = await self.client.models.list()
            self._validated = True
            logger.info("OpenAI API key validated successfully")
            return True
            
        except AuthenticationError as e:
            raise LLMError(
                f"Invalid OpenAI API key: {str(e)}",
                provider="openai",
                suggestion="Check your OPENAI_API_KEY is correct"
            )
        except Exception as e:
            raise LLMError(
                f"Failed to validate API key: {str(e)}",
                provider="openai"
            )
    
    async def __aenter__(self) -> OpenAIClient:
        """Async context manager entry."""
        if self.validate_on_init and not self._validated:
            await self.validate_api_key()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - cleanup resources."""
        await self.close()
    
    @retry_with_backoff(
        max_attempts=3,
        retry_on=(RateLimitError, APITimeoutError, httpx.TimeoutException)
    )
    @timeout(30.0)
    async def chat_completion(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Create a chat completion with rate limiting and retries.
        
        Args:
            messages: List of messages (ChatMessage objects or dicts)
            model: Model to use (defaults to config model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            tools: List of tool/function definitions
            tool_choice: How to handle tool selection
            **kwargs: Additional OpenAI API parameters
            
        Returns:
            CompletionResponse object
            
        Raises:
            LLMError: If API call fails after retries
        """
        # Convert ChatMessage objects to dicts
        message_dicts = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                message_dicts.append(msg.to_dict())
            else:
                message_dicts.append(msg)
        
        # Use configured model if not specified
        if model is None:
            model = self.config.model
        
        # Rate limiting
        async with self.semaphore:
            try:
                logger.debug(
                    f"Calling OpenAI chat completion",
                    extra={
                        "model": model,
                        "messages": len(message_dicts),
                        "temperature": temperature,
                        "has_tools": bool(tools),
                    }
                )
                
                # Build API call parameters
                params = {
                    "model": model,
                    "messages": message_dicts,
                    "temperature": temperature,
                    **kwargs,
                }
                
                if max_tokens:
                    params["max_tokens"] = max_tokens
                if tools:
                    params["tools"] = tools
                if tool_choice:
                    params["tool_choice"] = tool_choice
                
                # Make API call
                response = await self.client.chat.completions.create(**params)
                
                # Convert to our response model
                return self._parse_completion_response(response)
                
            except RateLimitError as e:
                logger.warning(f"Rate limit hit: {e}")
                raise  # Let retry decorator handle it
                
            except APITimeoutError as e:
                logger.warning(f"API timeout: {e}")
                raise  # Let retry decorator handle it
                
            except AuthenticationError as e:
                raise LLMError(
                    f"Authentication failed: {str(e)}",
                    provider="openai",
                    suggestion="Check your API key is valid"
                )
                
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                raise LLMError(
                    f"OpenAI API call failed: {str(e)}",
                    provider="openai"
                )
    
    def _parse_completion_response(self, response: ChatCompletion) -> CompletionResponse:
        """
        Parse OpenAI response into our CompletionResponse model.
        
        Args:
            response: Raw OpenAI ChatCompletion response
            
        Returns:
            Parsed CompletionResponse
        """
        choices = []
        for choice in response.choices:
            # Parse message
            message = ChatMessage(
                role=MessageRole(choice.message.role),
                content=choice.message.content or "",
            )
            
            # Parse tool calls if present
            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        type=tc.type,
                        function={
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    )
                    for tc in choice.message.tool_calls
                ]
            
            choices.append(
                Choice(
                    index=choice.index,
                    message=message,
                    finish_reason=choice.finish_reason,
                    tool_calls=tool_calls,
                )
            )
        
        # Parse usage
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
        
        return CompletionResponse(
            id=response.id,
            model=response.model,
            created=response.created,
            choices=choices,
            usage=usage,
        )
    
    async def search(
        self,
        query: str,
        model: Optional[str] = None,
        use_cache: bool = True,
        search_options: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """
        Perform web search using OpenAI's GPT-5 web search capabilities.
        
        This method now delegates to the modular search provider system
        for better extensibility and caching.
        
        Args:
            query: Search query
            model: Model to use for search (defaults to config model)
            use_cache: Whether to use cached results
            search_options: Additional search options (depth, domains, etc.)
            
        Returns:
            SearchResult object
        """
        # Use GroundingAdapter for modular search
        from pmkit.llm.grounding import GroundingAdapter
        from pmkit.llm.search.base import SearchOptions, SearchDepth
        
        # Create search options
        options = None
        if search_options:
            options = SearchOptions(
                depth=SearchDepth(search_options.get("depth", "medium")),
                allowed_domains=search_options.get("allowed_domains"),
                blocked_domains=search_options.get("blocked_domains"),
                max_results=search_options.get("max_results", 10),
                timeout=search_options.get("timeout", 30.0),
                include_citations=search_options.get("include_citations", True),
            )
        
        # Create adapter with current config
        adapter = GroundingAdapter(
            config=self.config,
            cache_enabled=use_cache,
        )
        
        try:
            # Perform search
            result = await adapter.search(query, options, use_cache=use_cache)
            
            logger.debug(
                f"Search completed for: {query[:50]}... "
                f"(cached: {result.cached}, citations: {len(result.citations)})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            # Return empty result as fallback
            return SearchResult(
                content="",
                citations=[],
                query=query,
                cached=False,
            )
    
    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        if hasattr(self, 'http_client'):
            await self.http_client.aclose()
            logger.debug("OpenAI HTTP client closed")
    
    def __del__(self):
        """Cleanup on deletion."""
        # Try to close the client if not already closed
        try:
            if hasattr(self, 'http_client') and not self.http_client.is_closed:
                # Create event loop if needed for cleanup
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.close())
                    loop.close()
                else:
                    asyncio.create_task(self.close())
        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")