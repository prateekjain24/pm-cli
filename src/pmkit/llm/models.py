"""
Data models for LLM interactions.

Provides type-safe models for OpenAI API responses, search results,
and chat messages using Pydantic for validation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class MessageRole(str, Enum):
    """Valid roles for chat messages."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"  # Deprecated but kept for compatibility


class ChatMessage(BaseModel):
    """
    Represents a message in a chat conversation.
    
    Used for both input to and output from the LLM.
    """
    
    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    
    role: MessageRole
    content: str
    name: Optional[str] = None  # For tool/function messages
    tool_call_id: Optional[str] = None  # For tool responses
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI API format."""
        data = {"role": self.role.value, "content": self.content}
        if self.name:
            data["name"] = self.name
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        return data


class ToolCall(BaseModel):
    """Represents a tool/function call from the model."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    id: str
    type: str = "function"
    function: Dict[str, Any]  # Contains name and arguments


class Choice(BaseModel):
    """Represents a single choice in a completion response."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class Usage(BaseModel):
    """Token usage information from API response."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    @property
    def estimated_cost(self) -> float:
        """
        Estimate cost based on token usage.
        
        Uses GPT-5 pricing as default (as of August 2025):
        - Input: $0.00125 per 1K tokens ($1.25 per 1M)
        - Output: $0.01 per 1K tokens ($10 per 1M)
        """
        input_cost = (self.prompt_tokens / 1000) * 0.00125
        output_cost = (self.completion_tokens / 1000) * 0.01
        return round(input_cost + output_cost, 6)  # More precision for lower costs


class CompletionResponse(BaseModel):
    """
    Represents a chat completion response from OpenAI API.
    
    Simplified model focusing on the essential fields.
    """
    
    model_config = ConfigDict(validate_assignment=True)
    
    id: str
    model: str
    created: datetime
    choices: List[Choice]
    usage: Usage
    
    @property
    def content(self) -> str:
        """Get the content of the first choice."""
        if self.choices:
            return self.choices[0].message.content
        return ""
    
    @property
    def tool_calls(self) -> Optional[List[ToolCall]]:
        """Get tool calls from the first choice."""
        if self.choices and self.choices[0].tool_calls:
            return self.choices[0].tool_calls
        return None


class SearchResult(BaseModel):
    """
    Represents a web search result.
    
    Used for grounding LLM responses with web search data.
    """
    
    model_config = ConfigDict(validate_assignment=True)
    
    content: str = Field(description="The main content/summary from search")
    citations: List[HttpUrl] = Field(
        default_factory=list,
        description="URLs of sources cited"
    )
    query: str = Field(description="The original search query")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the search was performed"
    )
    cached: bool = Field(default=False, description="Whether this was from cache")
    
    def has_content(self) -> bool:
        """Check if search returned meaningful content."""
        return bool(self.content and self.content.strip())


class ModelInfo(BaseModel):
    """Information about an available model."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    id: str
    name: str
    context_window: int
    max_output_tokens: Optional[int] = None
    supports_tools: bool = True
    supports_vision: bool = False
    pricing: Optional[Dict[str, float]] = None  # per 1K tokens


# Common model configurations (as of August 2025)
# GPT-5 models launched August 7, 2025
GPT5 = ModelInfo(
    id="gpt-5",
    name="GPT-5",
    context_window=272000,  # 272K input tokens
    max_output_tokens=128000,  # 128K output tokens
    supports_tools=True,
    supports_vision=True,
    pricing={
        "input": 0.00125,  # $1.25 per 1M tokens
        "output": 0.01,  # $10 per 1M tokens
        "cached_input": 0.000125  # $0.125 per 1M tokens (90% discount)
    }
)

GPT5_MINI = ModelInfo(
    id="gpt-5-mini",
    name="GPT-5 Mini",
    context_window=272000,
    max_output_tokens=128000,
    supports_tools=True,
    supports_vision=True,
    pricing={
        "input": 0.00025,  # $0.25 per 1M tokens
        "output": 0.002,  # $2 per 1M tokens
        "cached_input": 0.000025  # $0.025 per 1M tokens (90% discount)
    }
)

GPT5_NANO = ModelInfo(
    id="gpt-5-nano",
    name="GPT-5 Nano",
    context_window=272000,
    max_output_tokens=128000,
    supports_tools=True,
    supports_vision=False,  # Basic model, no vision
    pricing={
        "input": 0.00005,  # $0.05 per 1M tokens
        "output": 0.0004,  # $0.40 per 1M tokens
        "cached_input": 0.000005  # $0.005 per 1M tokens (90% discount)
    }
)

# Thinking variants for deeper reasoning
GPT5_THINKING = ModelInfo(
    id="gpt-5-thinking",
    name="GPT-5 Thinking",
    context_window=272000,
    max_output_tokens=128000,
    supports_tools=True,
    supports_vision=True,
    pricing={
        "input": 0.00125,  # Same as GPT-5
        "output": 0.01,  # Same as GPT-5 but uses more tokens
        "cached_input": 0.000125
    }
)

# Legacy models (deprecated but kept for compatibility)
GPT4_TURBO = ModelInfo(
    id="gpt-4-turbo-preview",
    name="GPT-4 Turbo (Legacy)",
    context_window=128000,
    max_output_tokens=4096,
    supports_tools=True,
    supports_vision=True,
    pricing={"input": 0.01, "output": 0.03}
)