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

## CRITICAL: Directory Structure

### ⚠️ Source Code Location
**THE SOURCE CODE LIVES IN `/src/pmkit/`, NOT `/pmkit/`**

This is the #1 mistake to avoid. The project uses a src-layout:
```
pm-cli/
├── src/
│   └── pmkit/          # ← ALL SOURCE CODE GOES HERE
│       ├── __init__.py
│       ├── agents/
│       ├── cli/
│       ├── config/
│       ├── context/
│       ├── llm/
│       └── ...
├── tests/              # Test files
├── pyproject.toml      # Points to src/ directory
└── ...
```

### Before Starting Any Work:
1. **Check where existing code lives**: `ls -la src/pmkit/`
2. **Verify pyproject.toml**: Should have `where = ["src"]`
3. **Test imports immediately**: Run a simple test after creating new files
4. **Never create `/pmkit/` directory**: This will cause duplicate, broken implementation

### Real Example of This Mistake:
When implementing PMKIT-023 (OnboardingAgent), the entire implementation was accidentally created in `/pmkit/` instead of `/src/pmkit/`, causing:
- All imports failed with `ModuleNotFoundError`
- Tests couldn't find the new modules
- Duplicate, incompatible directory structure
- Hours of debugging that could have been avoided

### How to Fix If This Happens:
1. Move all new code from `/pmkit/` to `/src/pmkit/`
2. Delete the duplicate `/pmkit/` directory
3. Reinstall package: `pip install -e .`
4. Run tests to verify imports work

## Common Issues & Solutions

### Import Errors
If you see import errors, ensure:
1. **Check directory structure first**: Code must be in `/src/pmkit/`, not `/pmkit/`
2. Package is installed in editable mode: `pip install -e .`
3. Virtual environment is activated
4. Python version is ≥3.11
5. Run a test immediately after creating new modules

### Async/Await Issues
- Never mix sync file operations with async LLM calls in the same function
- Always use `asyncio.run()` to bridge from CLI to async functions
- Mock async functions properly in tests using `pytest-asyncio`

### Context Version Mismatch
If cache isn't invalidating properly:
1. Check `ContextVersion.compute_version()` is hashing all context files
2. Verify file paths are sorted for deterministic ordering
3. Clear `.pmkit/.cache/` manually if needed

### Testing Best Practices
- **ALWAYS run tests after implementation** - don't assume they pass
- Use `pytest tests/test_specific_file.py -v` to see actual output
- If tests fail with import errors, check directory structure first
- Mock external API calls properly to avoid hitting real services
- never mention claude or anthropic or claude code in commit messages
- do not use co-authored in commit messages
- always think if you can use subagents for the job. it helps you save context space
- you have access to MCP such as sequential thinking & perplexity Search
- over engineering is an enemy of value delivery
- always update the CHANGELOG.md after finishing work. Commit after testing.
- never cheat on tests and make them pass by rewriting the tests or writing the test in a way it always passes.
- latest Openai model is GPT-5 that was released in August 2025
- overall implementation plan is in IMP_Claud.md . But over the cousersew of implemntation we may have made some minor changes

## OKR Management Flow

### Progressive Disclosure Philosophy
OKRs are **completely optional** during initial setup to maintain the 90-second time-to-value target. PMs can add OKRs when they're ready, not when the tool demands it.

### Multiple Entry Points for OKRs

1. **Dedicated OKR Commands** (Primary Method)
```bash
pm okrs add        # Launch the OKR wizard anytime
pm okrs edit       # Modify existing OKRs
pm okrs status     # View progress and confidence scores
pm okrs archive    # Archive completed OKRs
```

2. **Contextual Prompts**
- When creating PRDs without OKRs: Shows tip about alignment benefits
- During status checks: Displays OKR section with add prompt if empty
- Never blocks workflow, just gentle suggestions

3. **Natural Timing**
- Week 1: Get context set up, generate first PRD
- Week 2: After seeing value, optionally add OKRs for alignment
- Month 2: Review and update OKRs based on learnings

### Why This Works for PMs
- No cognitive overload during setup
- OKRs defined AFTER understanding market context (better quality)
- Flexible for teams that don't use OKRs
- Just-in-time approach reduces abandonment

## 90-Second Init Flow

### Phase Structure
1. **0-30 seconds**: 2 questions only (company, product)
2. **30-60 seconds**: Live enrichment with progress updates
3. **60-90 seconds**: Display concrete value and insights
4. **Post-90 seconds**: Optional advanced setup (team, OKRs, etc.)

### PM Archetype Templates
Quick-start templates for common PM scenarios:
- **B2B SaaS** (Slack-like): Enterprise, seats-based
- **Developer Tool** (Stripe-like): API-first, usage-based
- **Consumer App** (Spotify-like): Engagement, freemium
- **Marketplace** (Airbnb-like): Two-sided dynamics
- **PLG B2B** (Figma-like): Bottom-up adoption

### Value Metrics That Matter
Instead of generic "time saved", show:
- "Generated 12 user personas from real market data"
- "Discovered 5 competitor features you haven't tracked"
- "Identified 3 unserved market segments"
- "Ready to generate your first PRD in 30 seconds"