"""
PM-Kit initialization command.

Handles setting up PM-Kit configuration and context in a new project.
"""

from __future__ import annotations

import os
from pathlib import Path

from pmkit.utils.console import console


def init_pmkit(force: bool = False) -> None:
    """
    Initialize PM-Kit configuration in the current directory.
    
    Args:
        force: Whether to overwrite existing configuration
    """
    current_dir = Path.cwd()
    pmkit_dir = current_dir / ".pmkit"
    
    console.command_help_panel(
        command="pm init",
        description="Initialize PM-Kit configuration in your project directory",
        examples=[
            "pm init                    # Initialize in current directory",
            "pm init --force            # Overwrite existing configuration",
        ]
    )
    
    # Check if already initialized
    if pmkit_dir.exists() and not force:
        console.warning(
            f"PM-Kit already initialized in {current_dir}\n"
            "Use --force to overwrite existing configuration"
        )
        return
    
    # Coming soon message
    console.status_panel(
        title="Coming Soon",
        content="PM-Kit initialization is coming soon! 🚧\n\n"
                "This will set up:\n"
                "• Context configuration (.pmkit/context/)\n"
                "• Company and product profiles\n"
                "• Team structure and OKRs\n"
                "• Market intelligence setup\n"
                "• LLM provider configuration\n"
                "• Git hooks for quality gates\n\n"
                f"Target directory: {current_dir}",
        status="info",
        emoji="🏗️"
    )
    
    # Show what would be created
    console.print("\n[dim]Configuration structure that will be created:[/dim]")
    console.print("[muted].pmkit/[/muted]")
    console.print("[muted]├── config.yaml          # Main configuration[/muted]")
    console.print("[muted]├── context/            # Context files[/muted]")
    console.print("[muted]│   ├── company.yaml    # Company profile[/muted]")
    console.print("[muted]│   ├── product.yaml    # Product details[/muted]")
    console.print("[muted]│   ├── market.yaml     # Market intelligence[/muted]")
    console.print("[muted]│   ├── team.yaml       # Team structure[/muted]")
    console.print("[muted]│   └── okrs.yaml       # Current OKRs[/muted]")
    console.print("[muted]├── templates/          # Custom templates[/muted]")
    console.print("[muted]└── cache/              # LLM response cache[/muted]")


if __name__ == "__main__":
    init_pmkit()