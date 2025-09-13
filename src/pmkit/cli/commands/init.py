"""
PM-Kit initialization command.

Handles setting up PM-Kit configuration and context in a new project
using the interactive OnboardingAgent for a delightful setup experience.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pmkit.agents.onboarding import run_onboarding
from pmkit.config.loader import ConfigLoader
from pmkit.config.models import Config
from pmkit.context.structure import initialize_context_structure
from pmkit.exceptions import ConfigError, ContextError
from pmkit.utils.console import console
from pmkit.utils.logger import get_logger

logger = get_logger(__name__)


def init_pmkit(
    force: bool = False,
    resume: bool = True,
    skip_enrichment: bool = False,
    non_interactive: bool = False,
) -> bool:
    """
    Initialize PM-Kit configuration in the current directory.

    Uses the interactive OnboardingAgent for a beautiful setup experience
    with progressive disclosure, web enrichment, and state persistence.

    Args:
        force: Whether to overwrite existing configuration
        resume: Whether to resume from saved onboarding state
        skip_enrichment: Whether to skip web enrichment phase
        non_interactive: Whether to use non-interactive mode (minimal setup)

    Returns:
        True if initialization successful, False otherwise
    """
    current_dir = Path.cwd()
    pmkit_dir = current_dir / ".pmkit"
    context_dir = pmkit_dir / "context"

    # Check if already initialized
    if pmkit_dir.exists() and not force and not resume:
        # Check if there's a saved onboarding state
        state_file = context_dir / "onboarding_state.yaml"
        if state_file.exists():
            console.warning(
                f"PM-Kit initialization in progress in {current_dir}\n"
                "Use --resume to continue where you left off, or --force to start over"
            )
        else:
            console.warning(
                f"PM-Kit already initialized in {current_dir}\n"
                "Use --force to overwrite existing configuration"
            )
        return False

    try:
        # Ensure directory structure exists
        logger.info(f"Ensuring context structure in {pmkit_dir}")
        initialize_context_structure(current_dir)

        # Load or create configuration
        try:
            config_loader = ConfigLoader(pmkit_dir / "config.yaml")
            config = config_loader.load()
        except Exception as e:
            logger.warning(f"Could not load config, using defaults: {e}")
            config = Config(project_root=current_dir)

        # Non-interactive mode (for CI/CD or automation)
        if non_interactive:
            return _init_non_interactive(context_dir)

        # Run interactive onboarding
        console.info("Starting interactive PM-Kit setup...\n")

        success, context = run_onboarding(
            config=config,
            context_dir=context_dir,
            resume=resume and not force,
            skip_enrichment=skip_enrichment,
        )

        if success:
            console.success("\nPM-Kit initialization complete! 🎉")
            return True
        else:
            console.error("\nPM-Kit initialization failed or was cancelled")
            return False

    except ConfigError as e:
        e.display()
        return False
    except ContextError as e:
        e.display()
        return False
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        console.error(f"Failed to initialize PM-Kit: {e}")
        return False


def _init_non_interactive(context_dir: Path) -> bool:
    """
    Initialize with minimal non-interactive setup.

    Creates template files with placeholder values for CI/CD environments.

    Args:
        context_dir: Directory for context files

    Returns:
        True if successful
    """
    from pmkit.context.manager import ContextManager
    from pmkit.context.models import (
        CompanyContext,
        Context,
        ProductContext,
    )

    console.info("Non-interactive initialization - creating template files")

    try:
        # Create minimal context with environment variables or defaults
        company = CompanyContext(
            name=os.getenv("PMKIT_COMPANY_NAME", "MyCompany"),
            type=os.getenv("PMKIT_COMPANY_TYPE", "b2b"),
            stage=os.getenv("PMKIT_COMPANY_STAGE", "seed"),
        )

        product = ProductContext(
            name=os.getenv("PMKIT_PRODUCT_NAME", "MyProduct"),
            description=os.getenv(
                "PMKIT_PRODUCT_DESCRIPTION",
                "Product description to be updated"
            ),
            stage=os.getenv("PMKIT_PRODUCT_STAGE", "mvp"),
        )

        context = Context(
            company=company,
            product=product,
        )

        # Save context
        manager = ContextManager(context_dir, validate=False)
        success, errors = manager.save_context(context)

        if success:
            console.success("Created template context files")
            console.info(
                "Edit the files in .pmkit/context/ to customize your context"
            )
            return True
        else:
            console.error("Failed to create context files")
            for error in errors:
                console.print(f"  - {error.field}: {error.message}", style="dim")
            return False

    except Exception as e:
        logger.error(f"Non-interactive init failed: {e}")
        console.error(f"Failed to create template files: {e}")
        return False


def init_pmkit_legacy(force: bool = False) -> None:
    """
    Legacy initialization for backward compatibility.

    Shows the coming soon message from the original implementation.

    Args:
        force: Whether to overwrite existing configuration
    """
    current_dir = Path.cwd()
    pmkit_dir = current_dir / ".pmkit"

    console.command_help_panel(
        command="pm init",
        description="Initialize PM-Kit configuration in your project directory",
        examples=[
            "pm init                    # Interactive setup",
            "pm init --resume           # Resume interrupted setup",
            "pm init --force            # Start over (overwrites existing)",
            "pm init --skip-enrichment  # Skip web enrichment phase",
            "pm init --non-interactive  # Create template files only",
        ]
    )

    # Show what will be created
    console.print("\n[dim]Configuration structure to be created:[/dim]")
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
    # For testing, run interactive init
    init_pmkit()