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

## [0.2.0] - 2025-01-12

### Added

#### PMKIT-009: Create OpenAI client wrapper
- Implemented async OpenAI client with GPT-5 support
- Added connection pooling with httpx (25 keepalive, 100 max connections)
- Integrated rate limiting with semaphore (10 concurrent requests)
- Built retry logic with exponential backoff
- Added API key validation on initialization
- Implemented proper error handling with custom exceptions
- Created comprehensive test suite with 16 passing tests

#### PMKIT-010: Implement OpenAI web search method
- Created modular search system with GroundingAdapter pattern
- Built multi-level caching (L1 memory, L2 disk) with SHA256 keys
- Implemented BaseSearchProvider for extensibility
- Added OpenAI search provider with GPT-5 Responses API support
- Created comprehensive extension guide with examples for future providers
- Added domain filtering and reasoning level configuration
- Implemented graceful degradation when search unavailable
- Built SearchCache with TTL support and LRU eviction

### Updated

#### GPT-5 Model Support
- Updated all model constants to GPT-5 variants (standard, mini, nano, thinking)
- Changed default model from GPT-4 to GPT-5
- Updated pricing: $1.25/1M input, $10/1M output tokens
- Added support for 272K input tokens and 128K output tokens
- Implemented 90% cache discount for recently used tokens
- Updated cost estimation with higher precision (6 decimal places)