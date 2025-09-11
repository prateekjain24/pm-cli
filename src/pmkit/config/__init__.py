"""
PM-Kit Configuration Management.

Provides thread-safe singleton access to configuration with hierarchical loading
from multiple sources. Supports beautiful error handling and project initialization.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from .loader import ConfigLoader
from .models import Config


class ConfigManager:
    """
    Thread-safe singleton configuration manager.
    
    Provides global access to configuration while ensuring thread safety
    and proper initialization. Supports configuration reloading and 
    project-specific overrides.
    """
    
    _instance: Optional[ConfigManager] = None
    _lock = threading.Lock()
    _config: Optional[Config] = None
    _config_lock = threading.RLock()
    
    def __new__(cls) -> ConfigManager:
        """Thread-safe singleton implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_config(self, reload: bool = False) -> Config:
        """
        Load configuration with thread-safe singleton pattern.
        
        Args:
            reload: Force reload even if config is already loaded
            
        Returns:
            Validated Config instance
            
        Raises:
            ValidationError: If configuration is invalid
        """
        with self._config_lock:
            if self._config is None or reload:
                loader = ConfigLoader()
                self._config = loader.load()
            
            return self._config
    
    def get_config(self) -> Config:
        """
        Get current configuration.
        
        Returns:
            Current Config instance
            
        Raises:
            RuntimeError: If configuration hasn't been loaded yet
        """
        with self._config_lock:
            if self._config is None:
                raise RuntimeError(
                    "Configuration not loaded. Call load_config() first or use get_config_safe()."
                )
            return self._config
    
    def get_config_safe(self) -> Config:
        """
        Get configuration, loading it if necessary.
        
        Returns:
            Config instance (loads automatically if needed)
        """
        with self._config_lock:
            if self._config is None:
                return self.load_config()
            return self._config
    
    def is_loaded(self) -> bool:
        """
        Check if configuration is loaded.
        
        Returns:
            True if configuration is loaded
        """
        with self._config_lock:
            return self._config is not None
    
    def clear(self) -> None:
        """Clear loaded configuration (useful for testing)."""
        with self._config_lock:
            self._config = None


# Global instance
_manager = ConfigManager()


def load_config(reload: bool = False) -> Config:
    """
    Load configuration (singleton pattern).
    
    Args:
        reload: Force reload configuration even if already loaded
        
    Returns:
        Validated Config instance
        
    Raises:
        ValidationError: If configuration is invalid
    """
    return _manager.load_config(reload=reload)


def get_config() -> Config:
    """
    Get current configuration.
    
    Returns:
        Current Config instance
        
    Raises:
        RuntimeError: If configuration not loaded yet
    """
    return _manager.get_config()


def get_config_safe() -> Config:
    """
    Get configuration, loading it automatically if needed.
    
    Returns:
        Config instance
    """
    return _manager.get_config_safe()


def is_config_loaded() -> bool:
    """
    Check if configuration is loaded.
    
    Returns:
        True if configuration is loaded
    """
    return _manager.is_loaded()


def clear_config() -> None:
    """Clear loaded configuration (useful for testing)."""
    _manager.clear()


def init_project_config(path: Optional[Path] = None) -> Path:
    """
    Initialize a new .pmrc.yaml configuration file.
    
    Creates a comprehensive configuration template with sensible defaults
    and helpful comments for customization.
    
    Args:
        path: Path for the config file (defaults to .pmrc.yaml in current directory)
        
    Returns:
        Path to the created configuration file
        
    Raises:
        OSError: If file cannot be created
        FileExistsError: If config file already exists (unless force=True)
    """
    config_path = path or Path.cwd() / ".pmrc.yaml"
    
    if config_path.exists():
        raise FileExistsError(
            f"Configuration file already exists: {config_path}\\n"
            "Use --force to overwrite or choose a different path."
        )
    
    template = '''# PM-Kit Configuration
# This file configures PM-Kit for your project
# Learn more: https://github.com/yourusername/pmkit

# Project Information
project_name: "My Product"
# project_root: /path/to/project  # Defaults to current directory

# LLM Provider Settings
llm:
  provider: openai  # Options: openai, anthropic, gemini, ollama
  # model: gpt-4-turbo-preview  # Auto-detected based on provider
  timeout: 30  # Request timeout in seconds
  max_retries: 3  # Number of retry attempts
  # base_url: http://localhost:11434  # For Ollama or custom endpoints

# Cache Settings
cache:
  enabled: true
  directory: .pmkit/.cache  # Cache directory path
  ttl_seconds: 86400  # 24 hours cache TTL
  max_size_mb: 100  # Maximum cache size

# Context Management
context:
  auto_enrich: true  # Automatically enrich context with external data
  validation_mode: strict  # Options: strict, relaxed, none
  auto_backup: true  # Backup context changes
  max_history_items: 10  # Number of history items to keep

# Application Settings  
app:
  debug: false  # Enable debug mode
  log_level: INFO  # Options: DEBUG, INFO, WARNING, ERROR
  pretty_json: true  # Pretty print JSON output
  enable_telemetry: false  # Anonymous usage telemetry
  no_color: false  # Disable colored output

# External Integrations (Optional)
# Uncomment and configure as needed
integrations:
  # Confluence
  # confluence_url: https://your-domain.atlassian.net
  # confluence_username: your-email@company.com
  # confluence_space_key: PROD
  
  # Jira
  # jira_url: https://your-domain.atlassian.net
  # jira_project_key: PROJ
  
  # GitHub
  # github_owner: your-org
  # github_repo: your-repo

# API Keys and Secrets
# Set these as environment variables for security:
# - OPENAI_API_KEY=your_openai_key
# - ANTHROPIC_API_KEY=your_anthropic_key
# - GOOGLE_API_KEY=your_google_key
# - CONFLUENCE_API_TOKEN=your_confluence_token
# - JIRA_API_TOKEN=your_jira_token
# - GITHUB_TOKEN=your_github_token

# You can also create a .env file in this directory with:
# OPENAI_API_KEY=your_key_here
# PMKIT_DEBUG=false
# PMKIT_LOG_LEVEL=INFO
'''
    
    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the configuration file
    config_path.write_text(template, encoding='utf-8')
    
    return config_path


def init_global_config() -> Path:
    """
    Initialize global user configuration at ~/.pmkit/config.yaml.
    
    Returns:
        Path to the created global configuration file
    """
    global_config_dir = Path.home() / ".pmkit"
    global_config_path = global_config_dir / "config.yaml"
    
    # Create directory if it doesn't exist
    global_config_dir.mkdir(exist_ok=True)
    
    if global_config_path.exists():
        raise FileExistsError(
            f"Global configuration already exists: {global_config_path}\\n"
            "Edit the file directly or remove it first."
        )
    
    template = '''# PM-Kit Global Configuration
# This configuration applies to all projects unless overridden
# Location: ~/.pmkit/config.yaml

# Default LLM Provider
llm:
  provider: openai
  timeout: 30
  max_retries: 3

# Global Cache Settings
cache:
  enabled: true
  ttl_seconds: 86400  # 24 hours

# Default Application Settings
app:
  debug: false
  log_level: INFO
  pretty_json: true
  enable_telemetry: false

# Default Context Settings
context:
  auto_enrich: true
  validation_mode: strict
  auto_backup: true

# Your preferred integrations (used as defaults for new projects)
integrations:
  # confluence_url: https://your-company.atlassian.net
  # jira_url: https://your-company.atlassian.net
  # github_owner: your-username
'''
    
    global_config_path.write_text(template, encoding='utf-8')
    return global_config_path


# Export main classes and functions
__all__ = [
    "Config",
    "ConfigManager",
    "load_config",
    "get_config", 
    "get_config_safe",
    "is_config_loaded",
    "clear_config",
    "init_project_config",
    "init_global_config",
]