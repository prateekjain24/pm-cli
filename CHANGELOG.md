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

#### PMKIT-010: Implement modular web search with native provider support
- Created simplified GroundingAdapter as thin orchestrator for provider routing
- Implemented OpenAISearchProvider using native Responses API with web_search_preview tool
- Added GeminiSearchProvider using native Google Search grounding
- Built simplified SearchCache with single-level caching and LRU eviction
- Removed complex multi-level caching in favor of provider-native optimizations
- Added comprehensive extension guide for future providers (Anthropic, Perplexity)
- Implemented graceful degradation when search unavailable
- Created test suite covering all search functionality (16 passing tests)

### Updated

#### GPT-5 Model Support
- Updated all model constants to GPT-5 variants (standard, mini, nano, thinking)
- Changed default model from GPT-4 to GPT-5
- Updated pricing: $1.25/1M input, $10/1M output tokens
- Added support for 272K input tokens and 128K output tokens
- Implemented 90% cache discount for recently used tokens
- Updated cost estimation with higher precision (6 decimal places)

### Refactored

#### Native Search Integration
- Refactored search providers to use native API capabilities instead of custom implementations
- OpenAI: Now uses Responses API with web_search_preview tool
- Gemini: Now uses google-genai SDK with GoogleSearch() tool
- Simplified caching strategy to rely on provider-native optimizations
- Reduced abstraction layers for better performance and maintainability

## [0.3.0] - 2025-01-12

### Added

#### PMKIT-011: Add OpenAI chat completion wrapper with streaming and token counting
- Implemented streaming chat completions with real-time token updates via `chat_completion_stream()`
- Added native token usage tracking with `stream_options={"include_usage": true}` support
- Integrated tiktoken library for accurate token counting with GPT-5 models
- Added `count_tokens()` method using o200k_base encoding for GPT-5
- Implemented `count_messages_tokens()` with message structure overhead accounting
- Created `estimate_tokens_before_call()` for pre-flight cost estimation
- Added StreamingChunk model for incremental streaming responses
- Added TokenEstimate model with cost formatting and context window validation
- Implemented proper streaming error handling and retry logic
- Added comprehensive tests for streaming, token counting, and cost estimation
- All 218 tests passing with new functionality integrated

#### PMKIT-012: Create in-memory cache for searches
- Implemented SearchCache with TTL support (24 hours default)
- Added SHA256-based cache key generation for deterministic caching
- Created LRU eviction when cache size exceeds limit (100 items default)
- Added comprehensive cache hit/miss metrics tracking
- Built memory-efficient dual-layer caching (in-memory + disk)
- 5 comprehensive cache tests passing covering all cache scenarios

#### PMKIT-013: Add enhanced retry logic for OpenAI calls
- Added APIConnectionError to retry exceptions for better network resilience
- Implemented retry logic on validate_api_key with 2 attempts and 10s timeout
- Added 60-second timeout to streaming chat completions for long-running streams
- Enhanced error handling to properly catch and retry on network issues
- Added comprehensive tests for APIConnectionError retry behavior
- Verified AuthenticationError does NOT trigger retry (as expected)
- All retry attempts are properly logged with tenacity
- 26 OpenAI client tests passing with complete retry coverage

#### PMKIT-014: Write OpenAI integration tests
- Created comprehensive test suite with 26 tests for OpenAI client
- Tests cover search functionality with mocks
- Verified retry behavior with exponential backoff (rate limit, connection, timeout)
- Added cache hit/miss scenario tests (5 cache-specific tests)
- Validated error handling for authentication, timeout, and rate limit errors

#### PMKIT-015: Define context Pydantic models
- Created lean, PM-focused context models (CompanyContext, ProductContext, MarketContext, TeamContext, OKRContext)
- Implemented smart defaults and B2B/B2C differentiation logic
- Added validation rules for required fields without overengineering
- Used Pydantic V2 best practices (ConfigDict, field_validator, model_dump)
- Created comprehensive test suite with 22 tests covering all models
- Models focus on essential fields that directly improve PRD quality or save PM time

#### PMKIT-016: Complete team and OKR models with confidence scoring
- Added confidence scoring (0-100%) to KeyResult model for tracking achievement probability
- Implemented average_confidence property on Objective to aggregate KR confidence
- Added at_risk_key_results property to OKRContext to identify KRs with <50% confidence
- Created tests for confidence validation and at-risk identification
- All 23 context model tests passing with confidence scoring features
- Kept implementation minimal - just one new field and two helper properties

#### PMKIT-017: Implement context versioning with SHA256
- Created ContextVersion class with content-based SHA256 hashing
- Version computed as SHA256(company.yaml + product.yaml + market.yaml + team.yaml + okrs.yaml)
- Implemented minimal MVP: compute_hash, has_changed, get_current methods
- Added load_stored and save_current for version persistence
- Files processed in deterministic order for consistent hashing
- Created 10 comprehensive tests covering all versioning scenarios
- No overengineering: no history tracking, no comparison operators, just ~50 lines of code
- Enables cache invalidation when context changes, avoiding unnecessary PRD regeneration
- Implemented proper mocking for OpenAI API responses and streaming
- All tests passing with 100% coverage of integration scenarios
- Total project test count: 221 tests

#### PMKIT-018: Build ContextManager for persistence
- Implemented ContextManager class for YAML-based context persistence
- Added save_context() method to save complete context to individual YAML files
- Added load_context() method to load and optionally validate context from disk
- Implemented atomic write operations using temp file + rename pattern to prevent corruption
- Added automatic backup creation (.backup files) before overwriting existing files
- Created individual save methods for each context component (save_company, save_product, etc.)
- Added context_exists() method to check if valid context is present
- Implemented get_context_summary() to report file existence and version status
- Integrated with ContextVersion for automatic version hash updates on save
- Created comprehensive test suite with 15 tests covering all persistence scenarios
- No over-engineering: simple YAML files, no database, straightforward file operations

#### PMKIT-019: Add context validation layer
- Created ContextValidator class with comprehensive validation rules
- Validates required fields presence (company and product are mandatory)
- Checks data consistency (e.g., team size matches sum of roles)
- Provides detailed validation errors with field path and actionable messages
- Differentiates between errors (blocking) and warnings (informational)
- Implemented auto-repair capability for minor issues (missing default metrics, team size mismatch)
- Added B2B/B2C specific validations (e.g., B2B companies should have sales teams)
- Validates OKR confidence levels and identifies at-risk key results
- Integrated validation into ContextManager with optional enable/disable
- Added auto_repair parameter to save_context() for automatic fixes
- Created comprehensive test suite with 12 tests covering all validation scenarios
- Total project test count: 248 tests (all passing)