"""
PM-Kit status and health check command.

Provides overview of PM-Kit context, project status, and system health.
"""

from __future__ import annotations

import os
from pathlib import Path

from pmkit.utils.console import console


def check_status() -> None:
    """
    Check PM-Kit context and project status.
    
    Provides a comprehensive overview of the current PM-Kit setup,
    context status, and project health indicators.
    """
    current_dir = Path.cwd()
    pmkit_dir = current_dir / ".pmkit"
    
    console.command_help_panel(
        command="pm status",
        description="Check PM-Kit configuration, context, and project health",
        examples=[
            "pm status                  # Show full status overview",
            "pm status --format json    # Output in JSON format (coming soon)",
            "pm status --verbose        # Show detailed diagnostics (coming soon)",
        ]
    )
    
    # Check if PM-Kit is initialized
    if not pmkit_dir.exists():
        console.status_panel(
            title="PM-Kit Not Initialized",
            content=f"No PM-Kit configuration found in {current_dir}\n\n"
                    "To get started, run:\n"
                    "[bold primary]pm init[/bold primary]\n\n"
                    "This will set up:\n"
                    "• Company and product context\n"
                    "• Team structure and OKRs\n" 
                    "• LLM provider configuration\n"
                    "• Git hooks for quality gates",
            status="warning",
            emoji="⚠️"
        )
        return
    
    # Coming soon - full status implementation
    console.status_panel(
        title="Status Overview",
        content="Comprehensive status checking is coming soon! 📊\n\n"
                "This will show:\n"
                "• Context health and versioning\n"
                "• LLM provider connectivity\n"
                "• Git repository status\n"
                "• PRD quality gate status\n"
                "• Cache usage and performance\n"
                "• External integrations status\n"
                "• Recent activity summary",
        status="info",
        emoji="📊"
    )
    
    # Show current directory info
    console.print(f"\n[dim]Current Directory:[/dim] {current_dir}")
    console.print(f"[dim]PM-Kit Config:[/dim] {pmkit_dir} {'✅' if pmkit_dir.exists() else '❌'}")
    
    # Show environment status
    console.print("\n[dim]Environment Status:[/dim]")
    debug_mode = os.getenv("PMKIT_DEBUG", "false").lower() == "true"
    console.print(f"[muted]• Debug Mode: {'Enabled' if debug_mode else 'Disabled'}[/muted]")
    
    # Check for API keys (without exposing them)
    api_keys = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY", 
        "Google": "GEMINI_API_KEY",
        "Ollama": "OLLAMA_HOST",
    }
    
    console.print(f"[dim]LLM Provider Keys:[/dim]")
    for provider, env_var in api_keys.items():
        has_key = bool(os.getenv(env_var))
        status = "✅ Configured" if has_key else "❌ Missing"
        console.print(f"[muted]• {provider}: {status}[/muted]")
    
    # Show git status if available
    git_dir = current_dir / ".git"
    if git_dir.exists():
        console.print(f"[dim]Git Repository:[/dim] ✅ Detected")
    else:
        console.print(f"[dim]Git Repository:[/dim] ❌ Not detected")
        console.print("[muted]  Consider initializing git for version control[/muted]")


def check_context_health() -> dict[str, bool]:
    """
    Check the health of PM-Kit context files.
    
    Returns:
        Dict mapping context file names to health status
    """
    # Placeholder for context health checking
    return {
        "company.yaml": False,
        "product.yaml": False,
        "market.yaml": False,
        "team.yaml": False,
        "okrs.yaml": False,
    }


def check_integration_status() -> dict[str, str]:
    """
    Check status of external integrations.
    
    Returns:
        Dict mapping integration names to status
    """
    # Placeholder for integration checking
    return {
        "jira": "not_configured",
        "github": "not_configured", 
        "confluence": "not_configured",
        "notion": "not_configured",
        "slack": "not_configured",
    }


if __name__ == "__main__":
    check_status()