# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CLI Design Guidelines

PM-Kit prioritizes **beautiful, delightful CLI experiences**. When implementing CLI features:
- Use Rich library for colors, tables, progress bars, and panels
- Provide immediate visual feedback for all actions
- Include emojis for better visual hierarchy (🚀 ✅ ❌ ⚠️ 📝 etc.)
- Show progress with animations for long operations
- Format errors as helpful suggestions, not stack traces
- Use consistent color palette (see DESIGN.md)

## Project Overview

PM-CLI (pmkit) is a context-aware PM assistant that treats PRDs, roadmaps, OKRs, and release notes as code. It uses a two-layer architecture:
- **Context Layer**: Persistent company/product/market/team data
- **Task Layer**: Stateless agents for PRD generation, roadmaps, personas, etc.

## Development Commands

### Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev,test]"
```

### Running Tests
```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_context_version.py

# Run specific test
pytest tests/test_context_version.py::test_version_changes_with_content

# Run with verbose output
pytest -v

# Run async tests only
pytest -k "async"
```

### Code Quality
```bash
# Format code
black pmkit tests

# Lint code
ruff check pmkit tests

# Type checking
mypy pmkit

# Run all checks (format, lint, type)
black pmkit tests && ruff check pmkit tests && mypy pmkit
```

### CLI Development
```bash
# Install in development mode
pip install -e .

# Run CLI directly
pm init          # Initialize context
pm new prd      # Create new PRD
pm status       # Check context status

# Debug mode with rich traceback
PMKIT_DEBUG=1 pm new prd
```

## Architecture & Key Design Patterns

### Context Versioning Strategy
The system uses **content-based SHA256 hashing** for context versioning, NOT timestamps:
- Version = `SHA256(company.yaml + product.yaml + market.yaml + team.yaml + okrs.yaml)`
- Cache invalidation happens automatically when content changes
- See `pmkit/context/version.py` for implementation

### Async/Sync Pattern Rules
**Critical**: Follow these patterns consistently:
- **Async**: All LLM API calls, web search, network requests
- **Sync**: File I/O (YAML files are small), context hashing, cache lookups
- **Bridge**: CLI commands use `asyncio.run()` to call async operations

Example:
```python
# CLI layer (sync)
def new(artifact: str):
    result = asyncio.run(agent.generate_prd(title))  # Bridge to async
    
# Agent layer (async)
async def generate_prd(self, title: str):
    result = await self.llm.generate()  # Async LLM call
```

### Grounding/Search Abstraction
The system abstracts web search across providers. **MVP focuses on OpenAI only**, but the architecture supports:
- OpenAI: Native web search tools
- Anthropic: Native web search
- Gemini: Grounding feature
- Ollama: Fallback to external APIs

Implementation in `pmkit/llm/grounding.py`

### B2B vs B2C Context Differentiation
The system adapts based on context:
- **B2B**: Focus on ROI, enterprise features, integrations, compliance
- **B2C**: Focus on user engagement, retention, viral loops, mobile experience

This affects PRD templates, personas, and roadmap priorities.

### Caching Strategy
Multi-level caching to minimize API costs:
1. **L1**: In-memory cache (fast, limited size)
2. **L2**: Disk cache with SHA256 keys
3. **L3**: Context-aware TTL (company data: 7 days, market data: 1 day)

## Phase 1 Implementation Focus

### Week 1: Context Foundation
1. `pm init` command with company enrichment
2. Context schema (company, product, market, team, OKRs)
3. Content-based versioning
4. Manual fallback when search unavailable

### Week 2: PRD Generation
1. 5-phase PRD pipeline (problem → solution → requirements → prototype → final)
2. Context-aware templates
3. Deterministic caching
4. B2B/B2C differentiation

## Testing Requirements

### Essential Tests for MVP
```python
# Context versioning
test_version_changes_with_content()
test_version_stable_without_changes()

# B2B vs B2C
test_b2b_uses_correct_template()
test_b2c_uses_correct_template()

# Cache invalidation
test_cache_invalidates_on_context_change()
```

## LLM Integration Notes

### API Key Management
Keys are loaded from environment variables:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`

Never commit keys. Use `.env` file locally.

### Error Handling Pattern
All LLM calls must have:
1. Retry logic (3 attempts with exponential backoff)
2. Timeout handling (30s default)
3. Graceful degradation (return None or cached result)
4. Proper logging

## File Structure Conventions

### Context Files
```
.pmkit/
  context/
    company.yaml      # Company profile
    product.yaml      # Product details
    market.yaml       # Market intelligence
    team.yaml         # Team structure
    okrs.yaml         # Current OKRs
    history/          # Versioned backups
```

### PRD Output Structure
```
product/
  prds/<slug>/
    01_problem.md
    02_solution.md
    03_requirements.md
    04_prototype_prompts.md
    05_final_prd.md
    manifest.yaml
    .cache/
```

## Common Issues & Solutions

### Import Errors
If you see import errors, ensure:
1. Package is installed in editable mode: `pip install -e .`
2. Virtual environment is activated
3. Python version is ≥3.11

### Async/Await Issues
- Never mix sync file operations with async LLM calls in the same function
- Always use `asyncio.run()` to bridge from CLI to async functions
- Mock async functions properly in tests using `pytest-asyncio`

### Context Version Mismatch
If cache isn't invalidating properly:
1. Check `ContextVersion.compute_version()` is hashing all context files
2. Verify file paths are sorted for deterministic ordering
3. Clear `.pmkit/.cache/` manually if needed
- never mention claude or anthropic or claude code in commit messages
- do not use co-authored in commit messages
- always think if you can use subagents for the job. it helps you save context space