# PHASE1 Implementation - 1 Story Point Ticket Breakdown

Based on PHASE1.md and the PRD, here's the complete breakdown into 1 story point tickets (2-4 hours each):

## Epic 1: Project Foundation (8 tickets)

### PMKIT-001: Initialize project structure with pyproject.toml
- Create project skeleton with src layout
- Setup pyproject.toml with modern Python packaging
- Configure black, ruff, mypy settings
- Add .gitignore, .env.example

### PMKIT-002: Setup CLI foundation with Typer
- Initialize Typer app with Rich support
- Create main entry point (pm command)
- Add --version and --help flags
- Setup error handling and debug mode

### PMKIT-003: Create configuration management module
- Implement Config class using Pydantic v2
- Load from .pmrc.yaml and environment
- Add validation for required fields
- Support provider selection (openai/anthropic/gemini)

### PMKIT-004: Setup logging infrastructure
- Configure structured logging with loguru
- Add console and file handlers
- Implement log levels (DEBUG/INFO/ERROR)
- Add context injection for tracing

### PMKIT-005: Create base exception hierarchy
- Define PMKitError base class
- Add specific exceptions (ConfigError, ContextError, LLMError)
- Implement error formatting with Rich
- Add user-friendly error messages

### PMKIT-006: Setup async utilities module
- Create async/sync bridge helpers
- Implement timeout decorators
- Add retry logic with tenacity
- Create async context managers

### PMKIT-007: Implement path utilities
- Create project root detection
- Add .pmkit directory management
- Implement safe file operations
- Add path validation helpers

### PMKIT-008: Setup testing infrastructure
- Configure pytest with asyncio support
- Create test fixtures for config/context
- Add mock factories for LLM responses
- Setup coverage reporting

## Epic 2: OpenAI Integration (6 tickets)

### PMKIT-009: Create OpenAI client wrapper
- Implement async OpenAI client initialization
- Add API key validation
- Setup connection pooling with httpx
- Handle rate limiting gracefully

### PMKIT-010: Implement OpenAI web search method
- Create search function with tools API
- Parse search results to SearchResult model
- Extract citations from response
- Handle search unavailable scenarios

### PMKIT-011: Add OpenAI chat completion wrapper
- Implement streaming and non-streaming modes
- Add token counting with tiktoken
- Support system/user/assistant messages
- Handle model selection (gpt-5)

### PMKIT-012: Create in-memory cache for searches
- Implement simple dict-based cache
- Add TTL support (24 hours default)
- Create cache key generation
- Add cache hit/miss metrics

### PMKIT-013: Add retry logic for OpenAI calls
- Implement exponential backoff (3 attempts)
- Handle specific OpenAI exceptions
- Add timeout handling (30s default)
- Log retry attempts

### PMKIT-014: Write OpenAI integration tests
- Test search functionality with mocks
- Verify retry behavior
- Test cache hit/miss scenarios
- Validate error handling

## Epic 3: Context System (8 tickets)

### PMKIT-015: Define context Pydantic models
- Create CompanyContext model
- Create ProductContext model
- Create MarketContext model
- Add validation rules

### PMKIT-016: Define team and OKR models
- Create TeamContext model
- Create OKRContext model
- Add nested models (Objective, KeyResult)
- Implement confidence scoring

### PMKIT-017: Implement context versioning with SHA256
- Create ContextVersion class
- Implement content-based hashing
- Add version comparison methods
- Create version history tracking

### PMKIT-018: Build ContextManager for persistence
- Implement save_context to YAML
- Implement load_context from disk
- Add atomic write operations
- Create backup before updates

### PMKIT-019: Add context validation layer
- Validate required fields presence
- Check data consistency
- Verify version compatibility
- Add migration support

### PMKIT-020: Create context file structure
- Setup .pmkit/context directory
- Initialize YAML templates
- Create history subdirectory
- Add .gitignore patterns

### PMKIT-021: Implement context exists check
- Check for initialization status
- Validate context completeness
- Report missing sections
- Add repair functionality

### PMKIT-022: Write context system tests
- Test save/load roundtrip
- Verify version changes
- Test validation rules
- Check backup creation

## Epic 4: Onboarding Flow (7 tickets)

### PMKIT-023: Create OnboardingAgent class
- Initialize with config and grounding
- Setup state management
- Add progress tracking
- Implement cancellation support

### PMKIT-024: Build interactive prompt flow
- Implement company name collection
- Add progressive disclosure UI
- Create confirmation steps
- Add input validation

### PMKIT-025: Create manual input fallback
- Build forms for manual entry
- Add field-by-field validation
- Support editing enriched data
- Save partial progress

### PMKIT-026: Implement OKR collection UI
- Create objective input flow
- Add key result collection
- Support multiple objectives
- Calculate confidence scores

### PMKIT-027: Add onboarding enrichment (OpenAI)
- Build search query templates
- Parse company information
- Extract competitors list
- Identify industry/model

### PMKIT-028: Create onboarding completion
- Validate all required fields
- Save context to disk
- Display success summary
- Create initialization marker

### PMKIT-029: Implement 'pm init' command
- Wire up OnboardingAgent
- Handle existing context
- Add --force flag
- Show progress indicators

## Epic 5: PRD Generation Core (10 tickets)

### PMKIT-030: Create PRDAgent base class
- Load context on init
- Setup phase orchestration
- Add progress tracking
- Implement cancellation

### PMKIT-031: Define PRD phase interfaces
- Create PhaseInput model
- Create PhaseOutput model
- Define BasePhase abstract class
- Add phase metadata

### PMKIT-032: Implement ProblemPhase class
- Create B2B problem template
- Create B2C problem template
- Add context injection
- Generate problem statement

### PMKIT-033: Implement SolutionPhase class
- Create solution templates
- Reference problem phase output
- Add market research integration
- Generate solution approach

### PMKIT-034: Implement RequirementsPhase class
- Create requirements template
- Parse user stories
- Add acceptance criteria
- Generate technical requirements

### PMKIT-035: Implement PrototypePhase class
- Create prototype prompt templates
- Reference previous phases
- Add UI/UX considerations
- Generate prototype descriptions

### PMKIT-036: Implement FinalPRDPhase class
- Combine all phase outputs
- Create final PRD structure
- Add executive summary
- Generate table of contents

### PMKIT-037: Build phase orchestration
- Execute phases sequentially
- Pass context between phases
- Handle phase failures
- Track execution time

### PMKIT-038: Add B2B vs B2C differentiation
- Detect model from context
- Select appropriate templates
- Adjust focus areas
- Customize metrics

### PMKIT-039: Create PRD output models
- Define PRD dataclass
- Add phase results storage
- Include metadata
- Support serialization

## Epic 6: Caching System (5 tickets)

### PMKIT-040: Create cache key generation
- Implement deterministic hashing
- Include context version
- Add phase dependencies
- Support cache invalidation

### PMKIT-041: Build in-memory L1 cache
- Implement LRU cache
- Set size limits
- Add TTL support
- Track hit rates

### PMKIT-042: Implement disk-based L2 cache
- Create DiskCache class
- Use file-based storage
- Add compression support
- Implement cleanup

### PMKIT-043: Create multi-level cache
- Implement PRDCache class
- Add L1/L2 coordination
- Handle promotion/demotion
- Add cache statistics

### PMKIT-044: Write cache tests
- Test key generation
- Verify TTL behavior
- Test size limits
- Check invalidation

## Epic 7: CLI Integration (6 tickets)

### PMKIT-045: Implement 'pm new prd' command
- Parse title argument
- Create PRD directory
- Initialize manifest.yaml
- Show creation summary

### PMKIT-046: Implement 'pm run' command
- Load PRD manifest
- Execute phases
- Save outputs
- Display progress

### PMKIT-047: Implement 'pm status' command
- Check context existence
- Display context summary
- Show OKR progress
- List recent PRDs

### PMKIT-048: Add CLI output formatting
- Use Rich for tables
- Add progress bars
- Format errors nicely
- Support --json flag

### PMKIT-049: Create file output structure
- Save phase markdown files
- Update manifest.yaml
- Create .cache directory
- Generate final PRD

### PMKIT-050: Add CLI error handling
- Catch and format exceptions
- Provide helpful messages
- Add --debug flag
- Log to file

## Epic 8: Testing & Documentation (5 tickets)

### PMKIT-051: Write integration tests
- Test full onboarding flow
- Test PRD generation
- Verify file outputs
- Check CLI commands

### PMKIT-052: Add B2B/B2C differentiation tests
- Verify template selection
- Check output differences
- Test context influence
- Validate metrics

### PMKIT-053: Create performance tests
- Measure generation time
- Check memory usage
- Test cache effectiveness
- Verify async performance

### PMKIT-054: Write user documentation
- Create README.md
- Add CLI help text
- Document context schema
- Provide examples

### PMKIT-055: Setup CI/CD pipeline
- Configure GitHub Actions
- Run tests on PR
- Check code quality
- Automate releases

## Summary

- **Total Tickets**: 55
- **Total Story Points**: 55 (1 point each)
- **Estimated Time**: 110-220 hours
- **Team Velocity**: ~20 points/week (1 developer)
- **Timeline**: ~3 weeks for MVP

### Priority Order for MVP

1. **Week 1**: Epics 1-3 (Foundation, OpenAI, Context) - 22 tickets
2. **Week 2**: Epics 4-5 (Onboarding, PRD Core) - 17 tickets  
3. **Week 3**: Epics 6-8 (Caching, CLI, Testing) - 16 tickets

### Key Dependencies

- Epic 2 (OpenAI) depends on Epic 1 (Foundation)
- Epic 4 (Onboarding) depends on Epics 2-3 (OpenAI, Context)
- Epic 5 (PRD) depends on Epics 3-4 (Context, Onboarding)
- Epic 7 (CLI) depends on Epics 4-5 (Onboarding, PRD)

### Technology Stack

- **Language**: Python 3.11+
- **CLI**: Typer with Rich
- **Data Models**: Pydantic v2
- **Async**: asyncio, httpx
- **LLM**: OpenAI SDK (latest)
- **Testing**: pytest, pytest-asyncio
- **Packaging**: pyproject.toml (PEP 517/518)
- **Code Quality**: black, ruff, mypy
- **Logging**: loguru
- **Caching**: diskcache
- **YAML**: PyYAML