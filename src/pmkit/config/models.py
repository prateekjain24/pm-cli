"""
Pydantic models for PM-Kit configuration.

Provides strongly-typed configuration models with validation, secure secret handling,
and support for multiple provider configurations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr, field_validator


class LLMProviderConfig(BaseModel):
    """Configuration for LLM providers with secure API key handling."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
    )
    
    provider: Literal["openai", "anthropic", "gemini", "ollama"] = "openai"
    api_key: Optional[SecretStr] = None
    model: str = Field(default="", description="Model name, defaults based on provider")
    base_url: Optional[HttpUrl] = None  # For Ollama or custom endpoints
    timeout: int = Field(30, ge=1, le=300, description="Request timeout in seconds")
    max_retries: int = Field(3, ge=1, le=10, description="Maximum retry attempts")
    
    @field_validator('model')
    @classmethod
    def set_default_model(cls, v: str, info: Any) -> str:
        """Set default model based on provider if not specified."""
        if v:  # If model is explicitly set, use it
            return v
            
        # Get provider from the same validation context
        provider = info.data.get('provider', 'openai')
        
        # Default models per provider (latest as of 2025)
        defaults = {
            'openai': 'gpt-4-turbo-preview',
            'anthropic': 'claude-3-5-sonnet-20241022',
            'gemini': 'gemini-2.0-flash-exp',
            'ollama': 'llama3.2:latest',
        }
        
        return defaults.get(provider, 'gpt-4-turbo-preview')
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: Optional[SecretStr], info: Any) -> Optional[SecretStr]:
        """Validate API key and attempt to load from environment if not provided."""
        if v is not None:
            return v
        
        # Try to get from environment based on provider
        provider = info.data.get('provider', 'openai')
        env_key_map = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'gemini': 'GOOGLE_API_KEY',
            'ollama': None,  # Ollama typically doesn't need API key for local
        }
        
        env_key = env_key_map.get(provider)
        if env_key:
            env_value = os.getenv(env_key)
            if env_value:
                return SecretStr(env_value)
        
        # For ollama, API key is optional
        if provider == 'ollama':
            return None
            
        # For other providers, warn but don't fail (might be set later)
        return None


class CacheConfig(BaseModel):
    """Configuration for caching system."""
    
    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    
    enabled: bool = True
    directory: Path = Field(default=Path(".pmkit/.cache"), description="Cache directory path")
    ttl_seconds: int = Field(
        86400, 
        ge=0, 
        description="Time-to-live for cache entries in seconds (86400 = 24 hours)"
    )
    max_size_mb: int = Field(
        100, 
        ge=1, 
        le=1000, 
        description="Maximum cache size in megabytes"
    )
    
    @field_validator('directory')
    @classmethod
    def validate_directory(cls, v: Path) -> Path:
        """Ensure directory path is absolute."""
        if not v.is_absolute():
            return Path.cwd() / v
        return v


class IntegrationConfig(BaseModel):
    """Configuration for external service integrations."""
    
    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    
    # Confluence
    confluence_url: Optional[HttpUrl] = None
    confluence_username: Optional[str] = None
    confluence_api_token: Optional[SecretStr] = None
    confluence_space_key: Optional[str] = None
    
    # Jira
    jira_url: Optional[HttpUrl] = None
    jira_project_key: Optional[str] = None
    jira_api_token: Optional[SecretStr] = None
    
    # GitHub
    github_token: Optional[SecretStr] = None
    github_owner: Optional[str] = None
    github_repo: Optional[str] = None
    
    @field_validator('confluence_api_token', 'jira_api_token', 'github_token')
    @classmethod
    def load_token_from_env(cls, v: Optional[SecretStr], info: Any) -> Optional[SecretStr]:
        """Load tokens from environment if not provided."""
        if v is not None:
            return v
        
        field_name = info.field_name
        env_map = {
            'confluence_api_token': 'CONFLUENCE_API_TOKEN',
            'jira_api_token': 'JIRA_API_TOKEN',
            'github_token': 'GITHUB_TOKEN',
        }
        
        env_key = env_map.get(field_name)
        if env_key:
            env_value = os.getenv(env_key)
            if env_value:
                return SecretStr(env_value)
        
        return None


class ApplicationConfig(BaseModel):
    """General application settings."""
    
    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", 
        description="Logging level"
    )
    pretty_json: bool = Field(True, description="Pretty print JSON output")
    enable_telemetry: bool = Field(False, description="Enable anonymous usage telemetry")
    no_color: bool = Field(False, description="Disable colored output")
    
    @field_validator('debug')
    @classmethod
    def check_debug_env(cls, v: bool) -> bool:
        """Check PMKIT_DEBUG environment variable."""
        env_debug = os.getenv('PMKIT_DEBUG')
        if env_debug:
            return env_debug.lower() in ('1', 'true', 'yes', 'on')
        return v
    
    @field_validator('log_level')
    @classmethod
    def check_log_level_env(cls, v: str) -> str:
        """Check PMKIT_LOG_LEVEL environment variable."""
        env_level = os.getenv('PMKIT_LOG_LEVEL')
        if env_level and env_level.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            return env_level.upper()
        return v
    
    @field_validator('no_color')
    @classmethod
    def check_no_color_env(cls, v: bool) -> bool:
        """Check NO_COLOR environment variable."""
        # Respect standard NO_COLOR environment variable
        if os.getenv('NO_COLOR'):
            return True
        return v


class ContextConfig(BaseModel):
    """Configuration for context management."""
    
    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    
    auto_enrich: bool = Field(
        True, 
        description="Automatically enrich context with external data"
    )
    validation_mode: Literal["strict", "relaxed", "none"] = Field(
        "strict",
        description="Context validation mode"
    )
    auto_backup: bool = Field(
        True,
        description="Automatically backup context changes"
    )
    max_history_items: int = Field(
        10,
        ge=1,
        le=100,
        description="Maximum number of context history items to keep"
    )


class Config(BaseModel):
    """
    Main configuration model for PM-Kit.
    
    Combines all configuration sections with proper validation and defaults.
    Supports loading from multiple sources with environment variable overrides.
    """
    
    model_config = ConfigDict(
        validate_assignment=True, 
        extra="forbid",
        arbitrary_types_allowed=True,
    )
    
    # Project information
    project_name: Optional[str] = Field(
        None, 
        description="Name of the current project"
    )
    project_root: Path = Field(
        default_factory=Path.cwd,
        description="Root directory of the project"
    )
    
    # Configuration sections
    llm: LLMProviderConfig = Field(
        default_factory=LLMProviderConfig,
        description="LLM provider configuration"
    )
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Cache configuration"
    )
    integrations: IntegrationConfig = Field(
        default_factory=IntegrationConfig,
        description="External integrations configuration"
    )
    app: ApplicationConfig = Field(
        default_factory=ApplicationConfig,
        description="Application settings"
    )
    context: ContextConfig = Field(
        default_factory=ContextConfig,
        description="Context management configuration"
    )
    
    @field_validator('project_root')
    @classmethod
    def validate_project_root(cls, v: Path) -> Path:
        """Ensure project root is absolute."""
        if not v.is_absolute():
            return v.resolve()
        return v
    
    def model_dump_safe(self) -> Dict[str, Any]:
        """
        Dump model to dict with secrets masked for safe display.
        
        Returns:
            Dictionary with SecretStr values replaced with "[REDACTED]"
        """
        data = self.model_dump()
        
        def mask_secrets(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: mask_secrets(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [mask_secrets(item) for item in obj]
            elif hasattr(obj, 'get_secret_value'):  # SecretStr
                return "[REDACTED]"
            else:
                return obj
        
        return mask_secrets(data)
    
    def __repr__(self) -> str:
        """String representation with masked secrets."""
        safe_data = self.model_dump_safe()
        return f"Config({safe_data})"


__all__ = [
    "Config",
    "LLMProviderConfig", 
    "CacheConfig",
    "IntegrationConfig",
    "ApplicationConfig",
    "ContextConfig",
]