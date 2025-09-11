# Changelog

All notable changes to PM-Kit are documented in this file.

## [0.1.0] - 2025-01-11

### Added

#### PMKIT-001: Initialize project structure with pyproject.toml
- Created src layout with proper Python package organization
- Configured modern Python packaging with PEP 517/518
- Added comprehensive dependencies including LLM SDKs
- Setup development tools (black, ruff, mypy)

#### PMKIT-002: Setup beautiful CLI foundation with Typer
- Implemented Rich-based CLI with custom theme and colors
- Created command structure (init, new, status, run, publish, sync)
- Added debug mode support via PMKIT_DEBUG
- Integrated beautiful help text with emojis and formatting

#### PMKIT-003: Create configuration management module
- Built hierarchical config loading (env > .pmrc.yaml > ~/.pmkit/config.yaml)
- Implemented thread-safe singleton ConfigManager
- Added CLI commands for config management (show/init/validate/reset)
- Integrated SecretStr for secure API key handling

#### PMKIT-004: Setup logging infrastructure
- Implemented structured logging with loguru
- Added Rich console output with beautiful formatting
- Created JSON file logging with rotation (10MB, 7 days)
- Built request tracing with context injection and performance monitoring

#### PMKIT-005: Create base exception hierarchy
- Defined PMKitError base class with Rich formatting
- Added specific exceptions (ConfigError, ContextError, LLMError, ValidationError)
- Implemented beautiful error panels with helpful suggestions
- Integrated with console for user-friendly error display

#### PMKIT-006: Setup async utilities module
- Created async/sync bridge with run_async() helper
- Implemented retry decorator with tenacity for API calls
- Added timeout decorator for both sync/async functions
- Built ensure_async() converter for API flexibility

#### PMKIT-007: Implement path utilities
- Created project root detection (.pmkit or .git)
- Added thread-safe directory management
- Implemented atomic file writes with backup
- Built path security validation against traversal attacks

#### PMKIT-008: Setup testing infrastructure
- Configured pytest with comprehensive fixtures
- Created 179 tests covering all modules
- Added mock factories for testing
- Achieved 100% test pass rate