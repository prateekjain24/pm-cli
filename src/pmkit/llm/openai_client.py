"""
OpenAI client wrapper for PM-Kit.

Provides async OpenAI API access with connection pooling, rate limiting,
retry logic, and graceful error handling.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
import tiktoken
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import SecretStr

from pmkit.config.models import LLMProviderConfig
from pmkit.exceptions import LLMError
from pmkit.llm.models import (
    ChatMessage,
    Choice,
    CompletionResponse,
    MessageRole,
    SearchResult,
    StreamingChunk,
    TokenEstimate,
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
    
    @retry_with_backoff(
        max_attempts=2,  # Fewer retries for validation
        retry_on=(APITimeoutError, APIConnectionError, httpx.TimeoutException)
    )
    @timeout(10.0)  # Quick timeout for validation
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
            # Don't retry on auth errors - key is invalid
            raise LLMError(
                f"Invalid OpenAI API key: {str(e)}",
                provider="openai",
                suggestion="Check your OPENAI_API_KEY is correct"
            )
        except (APITimeoutError, APIConnectionError) as e:
            # These will be retried by decorator
            logger.warning(f"Network issue during API key validation: {e}")
            raise
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
        retry_on=(RateLimitError, APITimeoutError, APIConnectionError, httpx.TimeoutException)
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
                
            except APIConnectionError as e:
                logger.warning(f"Connection error: {e}")
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
    
    @retry_with_backoff(
        max_attempts=3,
        retry_on=(RateLimitError, APITimeoutError, APIConnectionError, httpx.TimeoutException)
    )
    @timeout(60.0)  # Longer timeout for streaming
    async def chat_completion_stream(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        stream_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        Create a streaming chat completion with real-time token updates.
        
        Args:
            messages: List of messages (ChatMessage objects or dicts)
            model: Model to use (defaults to config model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            tools: List of tool/function definitions
            tool_choice: How to handle tool selection
            stream_options: Options for streaming (e.g., {"include_usage": True})
            **kwargs: Additional OpenAI API parameters
            
        Yields:
            StreamingChunk objects with incremental content
            
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
        
        # Default stream options to include usage
        if stream_options is None:
            stream_options = {"include_usage": True}
        
        # Rate limiting
        async with self.semaphore:
            try:
                logger.debug(
                    f"Starting streaming chat completion",
                    extra={
                        "model": model,
                        "messages": len(message_dicts),
                        "temperature": temperature,
                        "has_tools": bool(tools),
                        "stream_options": stream_options,
                    }
                )
                
                # Build API call parameters
                params = {
                    "model": model,
                    "messages": message_dicts,
                    "temperature": temperature,
                    "stream": True,
                    "stream_options": stream_options,
                    **kwargs,
                }
                
                if max_tokens:
                    params["max_tokens"] = max_tokens
                if tools:
                    params["tools"] = tools
                if tool_choice:
                    params["tool_choice"] = tool_choice
                
                # Make streaming API call
                stream = await self.client.chat.completions.create(**params)
                
                # Process stream chunks
                async for chunk in stream:
                    yield self._parse_stream_chunk(chunk)
                    
            except RateLimitError as e:
                logger.warning(f"Rate limit hit during streaming: {e}")
                raise  # Let retry decorator handle it
                
            except APITimeoutError as e:
                logger.warning(f"API timeout during streaming: {e}")
                raise  # Let retry decorator handle it
                
            except APIConnectionError as e:
                logger.warning(f"Connection error during streaming: {e}")
                raise  # Let retry decorator handle it
                
            except AuthenticationError as e:
                raise LLMError(
                    f"Authentication failed: {str(e)}",
                    provider="openai",
                    suggestion="Check your API key is valid"
                )
                
            except Exception as e:
                logger.error(f"OpenAI streaming error: {e}")
                raise LLMError(
                    f"OpenAI streaming failed: {str(e)}",
                    provider="openai"
                )
    
    def _parse_stream_chunk(self, chunk: ChatCompletionChunk) -> StreamingChunk:
        """
        Parse a streaming chunk into our StreamingChunk model.
        
        Args:
            chunk: Raw OpenAI streaming chunk
            
        Returns:
            Parsed StreamingChunk
        """
        # Extract content from delta
        content = None
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
        
        # Extract finish reason
        finish_reason = None
        if chunk.choices and chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
        
        # Extract tool calls from delta
        tool_calls = None
        if chunk.choices and chunk.choices[0].delta.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id or "",
                    type=tc.type or "function",
                    function={
                        "name": tc.function.name if tc.function else "",
                        "arguments": tc.function.arguments if tc.function else "",
                    }
                )
                for tc in chunk.choices[0].delta.tool_calls
            ]
        
        # Extract usage if present (only in final chunk with stream_options)
        usage = None
        if hasattr(chunk, 'usage') and chunk.usage:
            usage = Usage(
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens,
                total_tokens=chunk.usage.total_tokens,
            )
        
        return StreamingChunk(
            id=chunk.id,
            model=chunk.model,
            created=chunk.created,
            content=content,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            usage=usage,
        )
    
    def count_tokens(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
            model: Model name (defaults to config model)
            
        Returns:
            Number of tokens
        """
        if model is None:
            model = self.config.model
        
        try:
            # Try to get encoding for specific model
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fall back to o200k_base for GPT-5 models
            if "gpt-5" in model.lower():
                encoding = tiktoken.get_encoding("o200k_base")
            else:
                # Default to cl100k_base for other models
                encoding = tiktoken.get_encoding("cl100k_base")
        
        tokens = encoding.encode(text)
        return len(tokens)
    
    def count_messages_tokens(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        model: Optional[str] = None,
    ) -> int:
        """
        Count tokens in a list of messages.
        
        Accounts for message structure overhead.
        
        Args:
            messages: List of messages
            model: Model name (defaults to config model)
            
        Returns:
            Total token count including overhead
        """
        if model is None:
            model = self.config.model
        
        # Convert messages to text
        total_tokens = 0
        
        for msg in messages:
            if isinstance(msg, ChatMessage):
                # Count role tokens (usually 1-2)
                total_tokens += self.count_tokens(msg.role.value, model)
                # Count content tokens
                total_tokens += self.count_tokens(msg.content, model)
                # Add message structure overhead (typically 3-4 tokens)
                total_tokens += 4
            else:
                # Handle dict format
                total_tokens += self.count_tokens(msg.get("role", ""), model)
                total_tokens += self.count_tokens(msg.get("content", ""), model)
                total_tokens += 4
        
        # Add conversation structure overhead
        total_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>
        
        return total_tokens
    
    def estimate_tokens_before_call(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        model: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
    ) -> TokenEstimate:
        """
        Estimate tokens and cost before making an API call.
        
        Args:
            messages: List of messages
            model: Model name (defaults to config model)
            max_completion_tokens: Expected max completion tokens
            
        Returns:
            TokenEstimate with cost and context window check
        """
        if model is None:
            model = self.config.model
        
        # Count input tokens
        input_tokens = self.count_messages_tokens(messages, model)
        
        # Estimate output tokens if not provided
        if max_completion_tokens is None:
            # Rough estimate: 20% of input tokens or 500, whichever is larger
            max_completion_tokens = max(int(input_tokens * 0.2), 500)
        
        total_tokens = input_tokens + max_completion_tokens
        
        # Get model info for pricing and context window
        model_info = self._get_model_info(model)
        
        # Calculate estimated cost
        input_cost = (input_tokens / 1_000_000) * model_info["input_price"]
        output_cost = (max_completion_tokens / 1_000_000) * model_info["output_price"]
        estimated_cost = input_cost + output_cost
        
        # Check if fits in context window
        fits_context = total_tokens <= model_info["context_window"]
        
        return TokenEstimate(
            tokens=total_tokens,
            estimated_cost=estimated_cost,
            fits_context=fits_context,
            context_window=model_info["context_window"],
        )
    
    def _get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get model information including pricing and context window.
        
        Args:
            model: Model name
            
        Returns:
            Dict with model info
        """
        # GPT-5 models (as of August 2025)
        if "gpt-5-nano" in model:
            return {
                "input_price": 0.05,  # $0.05 per 1M tokens
                "output_price": 0.40,  # $0.40 per 1M tokens
                "context_window": 272000,
            }
        elif "gpt-5-mini" in model:
            return {
                "input_price": 0.25,  # $0.25 per 1M tokens
                "output_price": 2.00,  # $2.00 per 1M tokens
                "context_window": 272000,
            }
        elif "gpt-5" in model:
            return {
                "input_price": 1.25,  # $1.25 per 1M tokens
                "output_price": 10.00,  # $10.00 per 1M tokens
                "context_window": 272000,
            }
        else:
            # Default to GPT-4 pricing for unknown models
            return {
                "input_price": 10.00,  # $10 per 1M tokens
                "output_price": 30.00,  # $30 per 1M tokens
                "context_window": 128000,
            }
    
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
        search_options: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """
        Perform web search using OpenAI's native web search capabilities.
        
        Uses the simplified GroundingAdapter for provider routing.
        
        Args:
            query: Search query
            model: Model to use for search (defaults to config model)
            search_options: Additional search options (depth, domains, etc.)
            
        Returns:
            SearchResult object
        """
        # Use GroundingAdapter for provider routing
        from pmkit.llm.grounding import GroundingAdapter
        from pmkit.llm.search.base import SearchOptions, SearchDepth
        
        # Create search options if provided
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
        adapter = GroundingAdapter(config=self.config)
        
        try:
            # Perform search
            result = await adapter.search(query, options)
            
            logger.debug(
                f"Search completed for: {query[:50]}... "
                f"(citations: {len(result.citations)})"
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