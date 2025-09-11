# 🚀 PM-Kit

A context-aware, CLI-first assistant for product managers that treats your documentation like code.

PM-Kit helps you create, manage, and publish PRDs, roadmaps, and other product artifacts directly from your terminal, using the power of LLMs with deep context about your company, product, and market.

## ✨ Key Features

- **Intelligent Onboarding**: Run `pmkit init` to automatically build a rich context profile of your company, product, and market using AI-powered web research.
- **Context-Aware Generation**: All generated documents are tailored to your specific business (B2B/B2C), competitors, and OKRs.
- **Native LLM Integrations**: Leverages native search and grounding features from OpenAI, Gemini, and Anthropic for high-quality, cited results.
- **CLI-First Workflow**: A powerful and fast command-line interface, including an interactive REPL shell, designed for keyboard-driven workflows.
- **Docs-as-Code**: Treat your product documentation like code. Store it in Git, review it via pull requests, and integrate it with your existing engineering workflows.
- **Extensible & Flexible**: Designed to be provider-agnostic with graceful fallbacks and a clear architecture for adding new commands and artifact types.

## ⚙️ Installation

You can install PM-Kit via pip:

```bash
pip install pmkit
```

## 🚀 Quick Start

Getting started with PM-Kit takes less than 3 minutes.

1.  **Install the package:**
    ```bash
    pip install pmkit
    ```

2.  **Set up your environment:**
    Copy the `.env.example` file to `.env` and add your API key for at least one LLM provider (OpenAI is recommended for the MVP).

    ```bash
    cp .env.example .env
    # Now edit .env to add your API key
    ```

3.  **Initialize your PM Context:**
    Run the interactive onboarding command. This will ask for your company name and use AI to build a detailed context profile for you to verify.

    ```bash
    pmkit init
    ```

4.  **Generate your first PRD:**
    Once the context is initialized, you can create your first context-aware document.

    ```bash
    pmkit new prd "Unified Search Experience for Mobile App"
    ```
    This will generate a new PRD in the `product/prds/` directory, with each section tailored to your company's specific context.

5.  **Use the Interactive Shell (Optional):**
    For a more conversational workflow, you can use the REPL shell.

    ```bash
    pmkit sh
    ```
    Inside the shell, you can use slash commands:
    ```
    > /new prd "My New Feature"
    > /status
    ```

## 📚 Commands

| Command | Alias | Description |
| :--- | :--- | :--- |
| `pmkit init` | | Starts the interactive onboarding to build your PM context. |
| `pmkit sh` | `shell`, `repl` | Starts the interactive REPL shell for slash commands. |
| `pmkit run "<command>"` | | Runs a single slash command non-interactively. |
| `pmkit new <type> <name>` | | Creates a new product artifact (e.g., `prd`, `persona`). |
| `pmkit status` | | Checks and displays the current context status. |
| `pmkit publish [target]` | | Publishes artifacts to configured destinations (e.g., Confluence). |
| `pmkit sync [resource]` | | Syncs artifacts with external services (e.g., Jira issues). |
| `pmkit release [action]` | | Generates or publishes release notes. |
| `pmkit version` | | Displays the installed version of PM-Kit. |

## 🛠️ Development

Interested in contributing? Here’s how to set up your development environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/prateekjain24/pmkit.git
    cd pmkit
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install all dependencies:**
    This command installs the core package in editable mode along with all dev, test, and docs dependencies.
    ```bash
    pip install -e ".[all]"
    ```

4.  **Set up pre-commit hooks:**
    This will run linters and formatters automatically before each commit to ensure code quality.
    ```bash
    pre-commit install
    ```

5.  **Run the test suite:**
    To make sure everything is working correctly, run the full test suite.
    ```bash
    pytest
    ```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
