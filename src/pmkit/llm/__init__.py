"""
LLM package for PM-Kit.

Provides unified interfaces for LLM providers with async support,
rate limiting, and graceful error handling.
"""

from pmkit.llm.models import (
    ChatMessage,
    Choice,
    CompletionResponse,
    MessageRole,
    ModelInfo,
    SearchResult,
    ToolCall,
    Usage,
    GPT5,
    GPT5_MINI,
    GPT5_NANO,
    GPT5_THINKING,
    GPT4_TURBO,  # Legacy, kept for compatibility
)
from pmkit.llm.openai_client import OpenAIClient

__all__ = [
    # Client
    "OpenAIClient",
    # Models
    "ChatMessage",
    "Choice",
    "CompletionResponse",
    "MessageRole",
    "ModelInfo",
    "SearchResult",
    "ToolCall",
    "Usage",
    # Model configs (GPT-5)
    "GPT5",
    "GPT5_MINI",
    "GPT5_NANO",
    "GPT5_THINKING",
    # Legacy models
    "GPT4_TURBO",
]