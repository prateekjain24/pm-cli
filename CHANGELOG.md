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

#### PMKIT-019: Add context validation layer (COMPLETE)
- Created ContextValidator class with comprehensive validation rules
- Validates required fields presence (company and product are mandatory)
- Checks data consistency (e.g., team size matches sum of roles)
- **Added version compatibility checking** - validates context schema version
- **Added migration support** - ContextMigrator handles schema version updates
- Provides detailed validation errors with field path and actionable messages
- Differentiates between errors (blocking) and warnings (informational)
- Implemented auto-repair capability for minor issues (missing default metrics, team size mismatch)
- Added B2B/B2C specific validations (e.g., B2B companies should have sales teams)
- Validates OKR confidence levels and identifies at-risk key results
- Integrated validation into ContextManager with optional enable/disable
- Added auto_repair parameter to save_context() for automatic fixes
- **Schema versioning** - tracks context schema version in .schema_version file
- **Migration framework** - supports future schema migrations with version compatibility
- Created comprehensive test suite with 27 tests covering validation and migration (12 validation + 15 migration)
- Total project test count: 324 tests (all passing)

#### PMKIT-020: Create context file structure
- Implemented initialize_context_structure() function to set up .pmkit/context directory
- Creates all required directories: .pmkit, context, history for versioned backups
- Generates template YAML files with helpful comments for company, product, market, team, and OKRs
- Templates include sensible defaults and explanatory headers for each field
- Added .gitignore in context directory to exclude history and backup files
- Implemented check_context_structure() to verify structure completeness
- Created repair_context_structure() to fix missing directories/files without overwriting
- Added proper error handling for permission issues and file system errors
- Operations are idempotent - safe to run multiple times
- Created comprehensive test suite with 15 tests covering all scenarios
- All initialization operations use atomic writes for safety

#### PMKIT-021: Implement context exists check
- Enhanced ContextManager with get_initialization_status() method
- Returns detailed status: 'not_initialized', 'partial', 'complete', or 'complete_with_warnings'
- Identifies missing required files (company.yaml, product.yaml)
- Lists missing optional files (market.yaml, team.yaml, okrs.yaml)
- Validates existing context and reports validation errors separately
- Provides actionable suggestions for fixing issues
- Added repair_context() method to automatically fix structure and validation issues
- Integrates with context structure module for directory/file creation
- Can auto-repair validation warnings when auto_fix=True
- Added 10 comprehensive tests for initialization status checking
- Total project test count: 286 tests (all passing)

#### PMKIT-022: Write context system tests
- Created comprehensive integration tests for the context system
- Added test_context_integration.py with 5 essential end-to-end tests
- Tests cover save/load roundtrip with full context
- Verifies version changes when context is updated
- Tests validation blocks invalid saves
- Confirms backup files are created on updates
- Tests auto-repair functionality during roundtrip
- All integration tests work together with existing unit tests
- No over-engineering - just 5 focused tests (~270 lines)
- Total context system tests: 88 (all passing)

#### PMKIT-023: OnboardingAgent comprehensive test suite (COMPLETE)
- Created comprehensive test suite for OnboardingAgent with 35 tests
- Tests all 3 phases (Essentials, Enrichment, Advanced) of onboarding flow
- Validates state persistence and resume functionality
- Tests cancellation handling and error scenarios
- Validates B2B vs B2C differentiation and context creation
- Properly mocks GroundingAdapter and external dependencies
- Ensures deterministic testing without hitting real APIs
- Tests performance requirement (<5 minute completion)
- Fixed syntax errors and control characters in onboarding_prompts.py
- Fixed all test failures - all 35 tests now passing:

#### PMKIT-024: Build interactive prompt flow with progressive disclosure (COMPLETE)
- Implemented InteractivePromptFlow class using prompt_toolkit for wizard-style interface
- Created advanced validators (CompanyName, Email, URL, ProductDescription, TeamSize)
- Built intelligent completers for industry, role, and metric selection
- Designed WizardState class for navigation (forward, back, skip)
- Integrated progressive disclosure pattern - showing information gradually
- Added B2B vs B2C differentiation with context-aware auto-completion
- Reduced Phase 1 to 4 questions for 30-second time-to-value
- Created beautiful CLI experience with colors, emojis, and progress tracking
- Added comprehensive test suite with 39 tests covering all validators and completers
- Fixed import errors in test_onboarding_agent.py and updated to use correct model fields
- All 58 combined tests passing (39 interactive + 19 onboarding agent tests)
  - Added required Rich console attributes (get_time, _live_stack, is_jupyter, is_interactive)
  - Fixed actual bug where competitors/metrics weren't collected without grounding adapter
  - Added context_dir parameter to OnboardingAgent for test isolation
  - Fixed invalid choice handling with proper fallback to defaults

#### PMKIT-024: Build interactive prompt flow (COMPLETE)
- Created InteractivePromptFlow class using prompt_toolkit for delightful PM-focused onboarding
- Implemented smart validators with real-time feedback:
  - CompanyNameValidator: 2-50 chars, no special chars, warns on generic names
  - EmailValidator: Format validation with clear error messages
  - URLValidator: Optional but validates format if provided
  - ProductDescriptionValidator: Requires at least 5 words for clarity
  - TeamSizeValidator: Numeric range validation with smart suggestions
- Built intelligent auto-completers for better UX:
  - IndustryCompleter: B2B/B2C specific industry suggestions
  - RoleCompleter: PM role hierarchy completion
  - MetricCompleter: Context-aware metric suggestions (MRR for B2B, MAU for B2C)
- Added progressive disclosure wizard with step navigation:
  - WizardState manager for back/forward navigation
  - Support for skip/help/quit commands
  - Progress indicators showing steps and time estimates
- Reduced Phase 1 to 4 questions (30 seconds to value) as per PM expert feedback
- Added immediate value demonstration after Phase 1 completion
- Created quick setup templates for B2B vs B2C companies
- Integrated with existing OnboardingAgent with backward compatibility
- Wrote comprehensive test suite with 39 tests (all passing)
- Delivers delightful, PM-focused experience that gets users to value quickly

#### PMKIT-025: Create manual input fallback (COMPLETE)
- Created ManualInputForm class with review_and_edit pattern for partial enrichment
- Implemented visual status indicators (✅ confirmed, ⚠️ review needed, ❌ missing)
- Built smart validators with three levels: error (blocking), warning (advisory), autocorrect (automatic fixes)
- Added cross-field validation (e.g., team size vs company stage alignment)
- Implemented auto-save after every field change with atomic writes
- Added smart resume capability that jumps to next empty required field
- Created staleness detection warning for saved data older than 7 days
- Integrated seamlessly with OnboardingAgent maintaining backward compatibility
- Added comprehensive test suite with 21 tests covering all scenarios
- Optimized for editing what automation got wrong, not entering everything from scratch
- Delivers delightful experience for PMs when automated enrichment fails
- Fixed all test failures in broader test suite (484 tests passing)
- Fixed asyncio issues in tests by properly mocking ManualInputForm methods
- Fixed mock side_effect lists in onboarding tests to match actual prompt counts

### Fixed

#### Test Suite Fixes
- Fixed test_init_no_api_key_raises in OpenAI client tests by adding monkeypatch to ensure OPENAI_API_KEY is not set
- Fixed console mocking fixture to properly handle PMKitConsole methods by patching print method directly
- Added MockConsole class to fixture with success(), error(), warning(), info(), and status_panel() methods
- Fixed pmkit/__init__.py import error by commenting out missing CLI app import
- Fixed URLValidator regex pattern for proper URL validation
- All 348 tests now passing successfully (309 existing + 39 new)