"""
Tests for OpenAI client wrapper.

Tests async client initialization, API key validation, connection pooling,
rate limiting, retry logic, and error handling.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import AuthenticationError, RateLimitError, APITimeoutError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as OpenAIChoice
from openai.types.completion_usage import CompletionUsage
from pydantic import SecretStr

from pmkit.config.models import LLMProviderConfig
from pmkit.exceptions import LLMError
from pmkit.llm import (
    ChatMessage,
    CompletionResponse,
    MessageRole,
    OpenAIClient,
    SearchResult,
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
def mock_config_no_key():
    """Create a mock config without API key."""
    return LLMProviderConfig(
        provider="openai",
        model="gpt-5",
        timeout=30,
        max_retries=3,
    )


@pytest.fixture
async def mock_openai_response():
    """Create a mock OpenAI ChatCompletion response."""
    return ChatCompletion(
        id="test-id",
        object="chat.completion",  # Add required object field
        model="gpt-5",
        created=1234567890,
        choices=[
            OpenAIChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Test response"
                ),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        ),
    )


class TestOpenAIClient:
    """Test OpenAI client functionality."""
    
    def test_init_with_api_key(self, mock_config):
        """Test client initialization with API key in config."""
        client = OpenAIClient(mock_config, validate_on_init=False)
        
        assert client.config == mock_config
        assert client.max_concurrent_requests == 10
        assert isinstance(client.semaphore, asyncio.Semaphore)
        assert isinstance(client.http_client, httpx.AsyncClient)
    
    def test_init_with_env_api_key(self, mock_config_no_key, monkeypatch):
        """Test client initialization with API key from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-api-key")
        
        client = OpenAIClient(mock_config_no_key, validate_on_init=False)
        assert client.client.api_key == "env-api-key"
    
    def test_init_no_api_key_raises(self, mock_config_no_key):
        """Test that initialization without API key raises error."""
        with pytest.raises(LLMError) as exc_info:
            OpenAIClient(mock_config_no_key, validate_on_init=False)
        
        assert "API key not found" in str(exc_info.value)
        assert "OPENAI_API_KEY" in str(exc_info.value.suggestion)
    
    @pytest.mark.asyncio
    async def test_validate_api_key_success(self, mock_config):
        """Test successful API key validation."""
        client = OpenAIClient(mock_config, validate_on_init=False)
        
        # Mock the models.list() call
        with patch.object(client.client.models, 'list', new_callable=AsyncMock) as mock_list:
            mock_list.return_value = MagicMock()  # Simulate successful response
            
            result = await client.validate_api_key()
            assert result is True
            assert client._validated is True
            mock_list.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validate_api_key_invalid(self, mock_config):
        """Test API key validation with invalid key."""
        client = OpenAIClient(mock_config, validate_on_init=False)
        
        with patch.object(client.client.models, 'list', new_callable=AsyncMock) as mock_list:
            mock_list.side_effect = AuthenticationError(
                "Invalid API key",
                response=MagicMock(status_code=401),
                body={"error": {"message": "Invalid API key"}}
            )
            
            with pytest.raises(LLMError) as exc_info:
                await client.validate_api_key()
            
            assert "Invalid OpenAI API key" in str(exc_info.value)
            assert "Check your OPENAI_API_KEY" in str(exc_info.value.suggestion)
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_config):
        """Test async context manager functionality."""
        with patch('pmkit.llm.openai_client.AsyncOpenAI'):
            client = OpenAIClient(mock_config, validate_on_init=False)
            
            # Mock validate_api_key
            with patch.object(client, 'validate_api_key', new_callable=AsyncMock) as mock_validate:
                mock_validate.return_value = True
                client.validate_on_init = True
                
                async with client as ctx_client:
                    assert ctx_client == client
                    mock_validate.assert_called_once()
                
                # Verify close was called
                assert client.http_client.is_closed
    
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, mock_config, mock_openai_response):
        """Test successful chat completion."""
        client = OpenAIClient(mock_config, validate_on_init=False)
        
        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello"),
        ]
        
        with patch.object(
            client.client.chat.completions,
            'create',
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_openai_response
            
            response = await client.chat_completion(messages)
            
            assert isinstance(response, CompletionResponse)
            assert response.content == "Test response"
            assert response.usage.total_tokens == 30
            # GPT-5 pricing: (10/1000*0.00125 + 20/1000*0.01) = 0.0000125 + 0.0002 = 0.0002125
            assert response.usage.estimated_cost == 0.000213  # Rounded to 6 decimal places
            
            # Verify API call
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args.kwargs["model"] == "gpt-5"
            assert call_args.kwargs["temperature"] == 0.7
            assert len(call_args.kwargs["messages"]) == 1
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_tools(self, mock_config):
        """Test chat completion with tools/functions."""
        client = OpenAIClient(mock_config, validate_on_init=False)
        
        messages = [ChatMessage(role=MessageRole.USER, content="What's the weather?")]
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}}
            }
        }]
        
        # Create response with tool calls using MagicMock to avoid validation issues
        response_with_tools = MagicMock()
        response_with_tools.id = "test-id"
        response_with_tools.model = "gpt-5"
        response_with_tools.created = 1234567890
        
        # Create mock choice with tool calls
        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.finish_reason = "tool_calls"
        
        # Create mock message with tool calls
        mock_message = MagicMock()
        mock_message.role = "assistant"
        mock_message.content = ""
        
        # Create mock tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = "{}"
        
        mock_message.tool_calls = [mock_tool_call]
        mock_choice.message = mock_message
        
        response_with_tools.choices = [mock_choice]
        
        # Create mock usage
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        response_with_tools.usage = mock_usage
        
        with patch.object(
            client.client.chat.completions,
            'create',
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = response_with_tools
            
            response = await client.chat_completion(messages, tools=tools)
            
            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].function["name"] == "get_weather"
    
    @pytest.mark.asyncio
    async def test_chat_completion_rate_limit_retry(self, mock_config):
        """Test that rate limit errors trigger retry."""
        client = OpenAIClient(mock_config, validate_on_init=False)
        
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        
        # Mock to fail once with rate limit, then succeed
        call_count = 0
        async def mock_create_with_rate_limit(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError(
                    "Rate limit exceeded",
                    response=MagicMock(status_code=429),
                    body={"error": {"message": "Rate limit"}}
                )
            return MagicMock(
                id="test-id",
                model="gpt-5",
                created=1234567890,
                choices=[MagicMock(
                    index=0,
                    message=MagicMock(
                        role="assistant",
                        content="Success after retry",
                        tool_calls=None
                    ),
                    finish_reason="stop",
                )],
                usage=MagicMock(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                ),
            )
        
        with patch.object(
            client.client.chat.completions,
            'create',
            new=mock_create_with_rate_limit
        ):
            response = await client.chat_completion(messages)
            assert response.content == "Success after retry"
            assert call_count == 2  # First call failed, second succeeded
    
    @pytest.mark.asyncio
    async def test_chat_completion_timeout(self, mock_config):
        """Test that timeout errors are handled."""
        client = OpenAIClient(mock_config, validate_on_init=False)
        
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        
        # Mock the create method to raise APITimeoutError
        with patch.object(
            client.client.chat.completions,
            'create',
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = APITimeoutError(request=MagicMock())
            
            # The retry decorator will retry 3 times then raise the error
            with pytest.raises(APITimeoutError):
                await client.chat_completion(messages)
    
    @pytest.mark.asyncio
    async def test_chat_completion_auth_error(self, mock_config):
        """Test that authentication errors are properly raised."""
        client = OpenAIClient(mock_config, validate_on_init=False)
        
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        
        with patch.object(
            client.client.chat.completions,
            'create',
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = AuthenticationError(
                "Invalid API key",
                response=MagicMock(status_code=401),
                body={"error": {"message": "Invalid API key"}}
            )
            
            with pytest.raises(LLMError) as exc_info:
                await client.chat_completion(messages)
            
            assert "Authentication failed" in str(exc_info.value)
            assert "Check your API key" in str(exc_info.value.suggestion)
    
    @pytest.mark.asyncio
    async def test_rate_limiting_semaphore(self, mock_config, mock_openai_response):
        """Test that rate limiting semaphore works."""
        client = OpenAIClient(mock_config, max_concurrent_requests=2)
        
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        
        call_times = []
        async def mock_create_with_timing(**kwargs):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)  # Simulate API delay
            return mock_openai_response
        
        with patch.object(
            client.client.chat.completions,
            'create',
            new=mock_create_with_timing
        ):
            # Start 4 concurrent requests with max_concurrent=2
            tasks = [
                client.chat_completion(messages)
                for _ in range(4)
            ]
            
            await asyncio.gather(*tasks)
            
            # Check that requests were rate limited (not all started at once)
            assert len(call_times) == 4
            # First 2 should start immediately, next 2 should wait
            # Allow some tolerance for timing
            assert call_times[2] > call_times[0] + 0.05
            assert call_times[3] > call_times[1] + 0.05
    
    @pytest.mark.asyncio
    async def test_search_placeholder(self, mock_config):
        """Test that search returns placeholder result."""
        client = OpenAIClient(mock_config, validate_on_init=False)
        
        result = await client.search("test query")
        
        assert isinstance(result, SearchResult)
        assert result.query == "test query"
        assert result.content == ""
        assert result.citations == []
        assert result.cached is False
    
    @pytest.mark.asyncio
    async def test_close(self, mock_config):
        """Test client cleanup."""
        client = OpenAIClient(mock_config, validate_on_init=False)
        
        # Verify client is open
        assert not client.http_client.is_closed
        
        # Close the client
        await client.close()
        
        # Verify client is closed
        assert client.http_client.is_closed
    
    def test_message_to_dict_conversion(self):
        """Test ChatMessage to dict conversion."""
        message = ChatMessage(
            role=MessageRole.USER,
            content="Test message",
            name="user1"
        )
        
        message_dict = message.to_dict()
        
        assert message_dict == {
            "role": "user",
            "content": "Test message",
            "name": "user1"
        }
    
    def test_usage_cost_estimation(self):
        """Test token usage cost estimation."""
        from pmkit.llm.models import Usage
        
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=2000,
            total_tokens=3000
        )
        
        # GPT-5 pricing: (1000/1000 * 0.00125) + (2000/1000 * 0.01) = 0.00125 + 0.02 = 0.02125
        assert usage.estimated_cost == 0.02125