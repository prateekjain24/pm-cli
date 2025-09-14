"""
Configuration loader for PM-Kit.

Implements hierarchical configuration loading with proper precedence:
Environment Variables > .pmrc.yaml > ~/.pmkit/config.yaml > defaults

Provides beautiful error handling and validation feedback.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError
from rich.panel import Panel
from rich.text import Text

from pmkit.utils.console import console
from .models import Config


class ConfigLoader:
    """
    Loads configuration from multiple sources with proper precedence.
    
    Loading order (highest to lowest precedence):
    1. Environment variables (PMKIT_*, OPENAI_API_KEY, etc.)
    2. .env file in current directory  
    3. .pmrc.yaml in current directory (project config)
    4. ~/.pmkit/config.yaml (user global config)
    5. Built-in defaults
    """
    
    def __init__(self) -> None:
        """Initialize the configuration loader."""
        self.config_paths = [
            Path.home() / ".pmkit" / "config.yaml",  # User global
            Path.cwd() / ".pmrc.yaml",  # Project config
        ]
        
    def load(self) -> Config:
        """
        Load configuration from all sources with proper precedence.
        
        Returns:
            Validated Config instance
            
        Raises:
            ValidationError: If configuration is invalid
            FileNotFoundError: If required config files are missing
        """
        # 1. Start with empty config dict
        config_dict: Dict[str, Any] = {}
        
        # 2. Load from global config if exists
        global_config_path = Path.home() / ".pmkit" / "config.yaml"
        global_config = self._load_yaml(global_config_path)
        if global_config:
            config_dict = self._deep_merge(config_dict, global_config)
            console.print(
                f"[dim]📁 Loaded global config from {global_config_path}[/dim]"
            )
        
        # 3. Load from project config if exists
        project_config_path = Path.cwd() / ".pmrc.yaml"
        project_config = self._load_yaml(project_config_path)
        if project_config:
            config_dict = self._deep_merge(config_dict, project_config)
            console.print(
                f"[dim]📁 Loaded project config from {project_config_path}[/dim]"
            )
        
        # 4. Load from .env file
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            console.print(f"[dim]🔐 Loaded environment from {env_path}[/dim]")
        
        # 5. Override with environment variables
        config_dict = self._apply_env_overrides(config_dict)
        
        # 6. Create and validate Config object
        try:
            config = Config(**config_dict)
            console.print("[success]✅ Configuration loaded successfully[/success]")
            return config
            
        except ValidationError as e:
            self._handle_validation_error(e, config_dict)
            raise
    
    def _load_yaml(self, path: Path) -> Optional[Dict[str, Any]]:
        """
        Load YAML config file safely.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Parsed YAML data or None if file doesn't exist
            
        Raises:
            yaml.YAMLError: If YAML is malformed
        """
        if not path.exists():
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f) or {}
                return content if isinstance(content, dict) else {}
                
        except yaml.YAMLError as e:
            console.error(f"Invalid YAML in {path}: {e}")
            raise
        except OSError as e:
            console.error(f"Failed to read {path}: {e}")
            raise
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries with override taking precedence.
        
        Args:
            base: Base dictionary
            override: Override dictionary (takes precedence)
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if (
                key in result 
                and isinstance(result[key], dict) 
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Args:
            config_dict: Base configuration dictionary
            
        Returns:
            Configuration with environment overrides applied
        """
        # Map environment variables to config paths
        env_mapping = {
            # Application settings
            "PMKIT_DEBUG": ("app", "debug"),
            "PMKIT_LOG_LEVEL": ("app", "log_level"),
            "PMKIT_PRETTY_JSON": ("app", "pretty_json"),
            "PMKIT_ENABLE_TELEMETRY": ("app", "enable_telemetry"),
            "NO_COLOR": ("app", "no_color"),
            
            # LLM settings
            "DEFAULT_LLM_PROVIDER": ("llm", "provider"),
            "LLM_MODEL": ("llm", "model"),
            "LLM_TIMEOUT": ("llm", "timeout"),
            "LLM_MAX_RETRIES": ("llm", "max_retries"),
            "LLM_BASE_URL": ("llm", "base_url"),

            # API Keys are handled directly by LLMProviderConfig model
            # based on the provider setting, so we don't map them here
            # to avoid overwriting each other
            
            # Cache settings
            "PMKIT_CACHE_ENABLED": ("cache", "enabled"),
            "PMKIT_CACHE_DIR": ("cache", "directory"),
            "PMKIT_CACHE_TTL": ("cache", "ttl_seconds"),
            "PMKIT_CACHE_MAX_SIZE": ("cache", "max_size_mb"),
            
            # Context settings
            "PMKIT_AUTO_ENRICH": ("context", "auto_enrich"),
            "PMKIT_VALIDATION_MODE": ("context", "validation_mode"),
            "PMKIT_AUTO_BACKUP": ("context", "auto_backup"),
            
            # Integration settings
            "CONFLUENCE_URL": ("integrations", "confluence_url"),
            "CONFLUENCE_USERNAME": ("integrations", "confluence_username"),
            "CONFLUENCE_API_TOKEN": ("integrations", "confluence_api_token"),
            "CONFLUENCE_SPACE_KEY": ("integrations", "confluence_space_key"),
            
            "JIRA_URL": ("integrations", "jira_url"),
            "JIRA_PROJECT_KEY": ("integrations", "jira_project_key"),
            "JIRA_API_TOKEN": ("integrations", "jira_api_token"),
            
            "GITHUB_TOKEN": ("integrations", "github_token"),
            "GITHUB_OWNER": ("integrations", "github_owner"),
            "GITHUB_REPO": ("integrations", "github_repo"),
            
            # Project settings
            "PMKIT_PROJECT_NAME": ("project_name",),
            "PMKIT_PROJECT_ROOT": ("project_root",),
        }
        
        for env_var, config_path in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Type conversion for specific fields
                converted_value = self._convert_env_value(env_var, value)
                self._set_nested(config_dict, config_path, converted_value)
        
        return config_dict
    
    def _convert_env_value(self, env_var: str, value: str) -> Any:
        """
        Convert environment variable string to appropriate type.
        
        Args:
            env_var: Environment variable name
            value: String value from environment
            
        Returns:
            Converted value
        """
        # Boolean conversions
        bool_vars = {
            "PMKIT_DEBUG", "PMKIT_PRETTY_JSON", "PMKIT_ENABLE_TELEMETRY",
            "NO_COLOR", "PMKIT_CACHE_ENABLED", "PMKIT_AUTO_ENRICH", 
            "PMKIT_AUTO_BACKUP"
        }
        if env_var in bool_vars:
            return value.lower() in ('1', 'true', 'yes', 'on')
        
        # Integer conversions
        int_vars = {
            "LLM_TIMEOUT", "LLM_MAX_RETRIES", "PMKIT_CACHE_TTL", 
            "PMKIT_CACHE_MAX_SIZE"
        }
        if env_var in int_vars:
            try:
                return int(value)
            except ValueError:
                console.warning(f"Invalid integer value for {env_var}: {value}")
                return value
        
        # Path conversions
        path_vars = {"PMKIT_CACHE_DIR", "PMKIT_PROJECT_ROOT"}
        if env_var in path_vars:
            return Path(value)
        
        return value
    
    def _set_nested(
        self,
        config_dict: Dict[str, Any],
        path: tuple[str, ...],
        value: Any
    ) -> None:
        """
        Set a nested dictionary value using a path tuple.

        Args:
            config_dict: Dictionary to modify
            path: Tuple of keys representing the nested path
            value: Value to set
        """
        current = config_dict

        # Navigate to the parent of the target key
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                # If the value exists but is not a dict (could be None or other type),
                # replace it with a dict to allow nested assignment
                current[key] = {}
            current = current[key]

        # Set the final value
        current[path[-1]] = value
    
    def _handle_validation_error(
        self, 
        error: ValidationError, 
        config_dict: Dict[str, Any]
    ) -> None:
        """
        Display beautiful validation errors with helpful suggestions.
        
        Args:
            error: Pydantic validation error
            config_dict: Configuration that failed validation
        """
        console.print()
        console.print(
            Panel(
                self._format_validation_errors(error),
                title="[error]❌ Configuration Validation Error[/error]",
                title_align="left",
                border_style="error.text",
                padding=(1, 2),
            )
        )
        
        # Show example config if major sections are missing
        if self._is_major_config_missing(error):
            self._show_config_example()
    
    def _format_validation_errors(self, error: ValidationError) -> Text:
        """
        Format validation errors into rich text with suggestions.
        
        Args:
            error: Pydantic validation error
            
        Returns:
            Formatted rich text
        """
        text = Text()
        
        for i, err in enumerate(error.errors()):
            if i > 0:
                text.append("\n")
            
            # Format field path
            field_path = " → ".join(str(loc) for loc in err['loc'])
            text.append(f"Field: ", style="dim")
            text.append(field_path, style="warning.text")
            text.append("\n")
            
            # Format error message
            text.append(f"Error: ", style="dim")
            text.append(err['msg'], style="error.text")
            text.append("\n")
            
            # Add helpful suggestions
            suggestion = self._get_field_suggestion(field_path, err)
            if suggestion:
                text.append(f"💡 Tip: ", style="info.text")
                text.append(suggestion, style="dim")
                text.append("\n")
        
        return text
    
    def _get_field_suggestion(self, field_path: str, error: Dict[str, Any]) -> str:
        """
        Get helpful suggestion for a validation error.
        
        Args:
            field_path: Dot-separated field path
            error: Error dictionary from Pydantic
            
        Returns:
            Suggestion string or empty string
        """
        field_lower = field_path.lower()
        
        if "api_key" in field_lower:
            # API keys are loaded directly from environment based on provider
            return "Set the appropriate API key environment variable (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY) based on your configured provider"
        
        if "url" in field_lower:
            return "Ensure the URL is valid and includes the protocol (https://)"
        
        if "directory" in field_lower or "path" in field_lower:
            return "Ensure the path exists and is writable"
        
        if "timeout" in field_lower:
            return "Timeout must be between 1 and 300 seconds"
        
        if "retries" in field_lower:
            return "Max retries must be between 1 and 10"
        
        return ""
    
    def _is_major_config_missing(self, error: ValidationError) -> bool:
        """
        Check if major configuration sections are missing.
        
        Args:
            error: Validation error
            
        Returns:
            True if major config is missing
        """
        major_fields = {'llm', 'cache', 'app', 'context'}
        error_fields = {err['loc'][0] for err in error.errors() if err['loc']}
        return len(major_fields.intersection(error_fields)) > 1
    
    def _show_config_example(self) -> None:
        """Show an example configuration file."""
        example = """# Example .pmrc.yaml configuration
project_name: "My Product"

llm:
  provider: openai
  model: gpt-4-turbo-preview
  timeout: 30

cache:
  enabled: true
  directory: .pmkit/.cache

app:
  debug: false
  log_level: INFO

context:
  auto_enrich: true
  validation_mode: strict
"""
        
        console.print()
        console.print(
            Panel(
                example,
                title="[info]💡 Example Configuration[/info]",
                title_align="left",
                border_style="info.text",
                padding=(1, 2),
            )
        )
        
        console.print(
            "\n[dim]Run [bold]pm config init[/bold] to create a starter configuration file.[/dim]"
        )


__all__ = ["ConfigLoader"]