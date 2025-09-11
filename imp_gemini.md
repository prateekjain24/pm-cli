# pm-kit: Detailed Implementation Plan

This document provides a comprehensive, step-by-step implementation plan for building the `pm-kit` CLI tool. It synthesizes the vision from the product documents (`prd_pm_cli.md`, `pm-kit_phasing_story.md`) with the technical blueprint provided in `pmkit_final_wired_stack_and_repl.md`.

The plan is divided into phases, starting with a foundational setup and progressively adding the core product management features and integrations.

## 1. Overview & Core Architecture

The `pm-kit` will be a Python application built using the Typer framework for the CLI. It will feature a dual interface:
1.  **Direct Commands:** Standard CLI subcommands like `pmkit new prd`, `pmkit run`, `pmkit publish`.
2.  **REPL Shell:** An interactive shell (`pmkit sh`) with slash commands (`/prd`, `/review`) for a more conversational workflow.

The core logic will be encapsulated in **Agents** (e.g., `PRDAgent`, `ReviewAgent`) and **Services** (e.g., `ConfluencePublisher`, `JiraSynchronizer`). Both CLI interfaces will act as thin wrappers around this shared core logic, ensuring consistency.

LLM interaction will be managed by a multi-provider backend supporting OpenAI, Anthropic, Gemini, and Ollama, with API keys handled securely via environment variables.

## 2. Project Setup & Dependencies

This phase prepares the development environment.

**1. Directory Structure:**
Create the following initial project structure:
```
pmkit/
  __init__.py
  cli.py
  config/
    __init__.py
    loader.py
  context/
    __init__.py
    store.py
  llm/
    __init__.py
    backends.py
    normalize.py
  slash/
    __init__.py
    parser.py
    registry.py
    handlers.py
  agents/
    __init__.py
    prd_agent.py
    review_agent.py
  utils/
    __init__.py
    hashing.py
    files.py
    update_check.py
  ide/
    __init__.py
    ide_server.py
    vscode/
      package.json
      src/extension.ts
.gitignore
pyproject.toml
README.md
```

**2. Virtual Environment:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**3. Dependencies:**
Install the required libraries using the `pyproject.toml` file below. The versions have been updated to the latest stable releases as of the time of this plan's creation.

```bash
pip install -e .
```

**(See Appendix for the final `pyproject.toml` content)**

**4. Configuration:**
Create the global configuration directory and file as specified in the technical docs:
```bash
mkdir -p ~/.pmkit
touch ~/.pmkit/config.yaml
```
Populate `~/.pmkit/config.yaml` with initial defaults, and instruct the user to add their API keys via environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`).

## 3. Phase 0: Core Scaffolding & REPL

This phase implements the application's skeleton and the interactive REPL.

**1. Implement Config Loader (`pmkit/config/loader.py`):**
   - Create the `load_config` function to read `~/.pmkit/config.yaml`, merge it with default values, and override with any environment variables. Use the code provided in `pmkit_final_wired_stack_and_repl.md`.

**2. Implement Utilities (`pmkit/utils/`):**
   - Implement the helper functions for hashing, file I/O, and PyPI update checking. Use the code provided in the technical document.

**3. Implement CLI Entrypoint (`pmkit/cli.py`):**
   - Use Typer to create the main `app`.
   - Implement the `pmkit sh` command to launch the REPL.
   - Use `prompt_toolkit` for the interactive session, including the command completer and the bottom toolbar.
   - The toolbar should display dynamic context like the current working directory, git branch, and selected LLM, as detailed in the spec.
   - Implement the initial banner and the update-check toast.
   - Use the code from `pmkit_final_wired_stack_and_repl.md` as the basis for this file.

## 4. Phase 1: LLM Integration

This phase builds the engine that powers the tool's intelligent features.

**1. Implement Context Store (`pmkit/context/store.py`):**
   - Create the `JsonContextStore` class to manage conversational history for different agents. This class will handle loading, saving, and appending messages to a JSON file within the project's `.pmkit` directory.

**2. Implement LLM Backends (`pmkit/llm/backends.py`):**
   - Define a base `LLMBackend` abstract class or interface.
   - Create concrete implementations for OpenAI, Anthropic, Gemini, and Ollama.
   - **Crucially, for the Gemini backend, use the `google-genai` package instead of the deprecated `google-generativeai` package.**
   - Implement the `resolve_backend` factory function that instantiates the correct backend based on the loaded configuration.
   - Implement the `provider_tools_for_grounding` function to enable native web search capabilities for supported providers (OpenAI, Anthropic, Gemini).

## 5. Phase 2: Agent & Command Layer

This phase connects the user interface (REPL) to the LLM backends.

**1. Implement Slash Command Parser (`pmkit/slash/parser.py`):**
   - Create the `parse` function and `SlashCmd` dataclass to parse the REPL input (e.g., `/prd "Title" --flag @target`) into a structured command object.

**2. Implement Command Registry (`pmkit/slash/registry.py`):**
   - Implement the `@slash` decorator and the `dispatch` function to register and execute handlers for different slash commands.

**3. Implement Initial Agents (`pmkit/agents/`):**
   - Create `PRDAgent` and `ReviewAgent`. Each agent will be initialized with an LLM backend and a context store.
   - They will have a `run` method that prepares a prompt, calls the LLM, and processes the result.

**4. Implement Slash Handlers (`pmkit/slash/handlers.py`):**
   - Create the initial handlers for `/help`, `/provider`, `/model`, `/ground`, `/prd`, and `/review`.
   - These handlers will parse the command, instantiate the appropriate agent (e.g., `PRDAgent` for `/prd`), call its `run` method, and print the output to the console.

## 6. Phase 3: Product-as-Code Features (MVP)

This phase implements the core PM workflow features from the PRD.

**1. Unify CLI and REPL Commands:**
   - For each feature (e.g., `new`, `run`), create a core function in a new module (e.g., `pmkit/core/scaffolding.py`, `pmkit/core/orchestration.py`).
   - Create a Typer command (e.g., `pmkit new prd`) that calls the core function.
   - Create a slash handler (e.g., `/new prd`) that also calls the same core function.

**2. Implement `pmkit new`:**
   - Create the scaffolding logic to generate the directory structure for a new PRD (`/product/prds/<slug>/`) and the initial phase files (`01_problem.md`, etc.) based on the information architecture in the PRD.

**3. Implement `pmkit run`:**
   - Develop an orchestrator that processes the PRD phases in a directed acyclic graph (DAG).
   - Use the `hashing` utility to hash the content of input files for each phase.
   - The orchestrator should only re-run a phase if its inputs (prior phase outputs, templates) have changed, storing content hashes in the `.cache/` directory.

**4. Implement `pmkit status`:**
   - Create a "gates" module that contains linting rules (e.g., checking for "TBD", ambiguous terms, missing metrics).
   - The `pmkit status` command will run these gates locally on the PRD files and report success or failure. This will be integrated into a CI/CD pipeline later.

## 7. Phase 4: Integrations & Advanced Features

This phase connects the local tool to external services.

**1. Implement `pmkit publish`:**
   - Create a `pmkit/publishers/confluence.py` module.
   - Implement a `ConfluencePublisher` class that uses `httpx` to make calls to the Confluence REST API.
   - The publisher will be responsible for creating and updating pages idempotently, managing page trees, and handling Confluence's storage format.

**2. Implement `pmkit sync issues`:**
   - Create a `pmkit/sync/jira.py` module.
   - Implement a `JiraSynchronizer` class that parses user stories from `03_requirements.md`.
   - It will use `httpx` to interact with the Jira Cloud REST API to create or update issues.
   - It must store a mapping of story IDs to Jira issue keys in `manifest.yaml` to ensure idempotency.

**3. Implement `pmkit release draft`:**
   - Create a `pmkit/release/generator.py` module.
   - Use Python's `subprocess` module to call `git` commands to find all merged PRs and commits since the last git tag.
   - Pass this information to an LLM agent to categorize and summarize the changes into a draft release notes document.

## 8. Testing & Verification Strategy

To ensure quality and stability, a robust testing strategy is essential.

**1. Framework:**
   - Use `pytest` as the testing framework. Add it to the development dependencies.

**2. Unit Tests:**
   - Write unit tests for all utility functions, the slash command parser, and individual methods within agents and publishers.
   - Mock all external network calls (to LLMs, Jira, Confluence) using `pytest-mock`.

**3. Integration Tests:**
   - Write integration tests for the Typer CLI commands. Use Typer's `CliRunner` to invoke commands and assert on the output and file system changes.
   - These tests can run against local LLM providers like Ollama to verify the end-to-end flow without relying on paid APIs.

## Appendix: Final `pyproject.toml`

```toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pmkit"
version = "0.1.0"
description = "pm-kit: CLI-first PM docs with LLM superpowers"
authors = [{name = "You"}]
dependencies = [
  "typer>=0.17.4",
  "rich>=14.1.0",
  "prompt_toolkit>=3.0.52",
  "pyyaml>=6.0.2",
  "httpx>=0.28.1",
  "openai>=1.107.1",
  "anthropic>=0.66.0",
  "google-genai>=1.35.0",
  "ollama>=0.5.3",
  "websockets>=15.0.1"
]

[project.scripts]
pmkit = "pmkit.cli:app"

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-mock"
]
```
