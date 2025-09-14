"""
PM-Kit initialization command.

Handles setting up PM-Kit configuration and context in a new project
using the interactive OnboardingAgent for a delightful setup experience.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict
import yaml

from pmkit.agents.onboarding import run_onboarding
from pmkit.agents.templates import (
    PMArchetype,
    get_template_by_name,
    apply_template,
    format_template_choices,
)
from pmkit.agents.value_display import (
    calculate_value_metrics,
    display_value_metrics,
    display_90_second_progress,
)
from pmkit.config.loader import ConfigLoader
from pmkit.config.models import Config
from pmkit.context.structure import initialize_context_structure
from pmkit.exceptions import ConfigError, ContextError
from pmkit.utils.console import console
from pmkit.utils.logger import get_logger
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
import time

logger = get_logger(__name__)


def init_pmkit(
    force: bool = False,
    resume: bool = True,
    skip_enrichment: bool = False,
    non_interactive: bool = False,
    template: Optional[str] = None,
    quick: bool = False,
) -> bool:
    """
    Initialize PM-Kit configuration in the current directory.

    Uses the interactive OnboardingAgent for a beautiful setup experience
    with progressive disclosure, web enrichment, and state persistence.
    Follows the 90-second rule for fast time-to-value.

    Args:
        force: Whether to overwrite existing configuration
        resume: Whether to resume from saved onboarding state
        skip_enrichment: Whether to skip web enrichment phase
        non_interactive: Whether to use non-interactive mode (minimal setup)
        template: PM archetype template to use (e.g., 'b2b_saas', 'developer_tool')
        quick: Quick mode - minimal questions, maximum enrichment

    Returns:
        True if initialization successful, False otherwise
    """
    current_dir = Path.cwd()
    pmkit_dir = current_dir / ".pmkit"
    context_dir = pmkit_dir / "context"

    # Start timing for 90-second goal
    start_time = time.time()

    # Check if already initialized (check context dir, not just pmkit dir)
    # The .pmkit dir might exist just for logs
    if context_dir.exists() and not force and not resume:
        # Check if there's a saved onboarding state
        state_file = context_dir / "onboarding_state.yaml"
        if state_file.exists():
            # Smart resume detection
            import yaml
            with open(state_file) as f:
                state = yaml.safe_load(f)
            progress = _calculate_progress(state)
            console.print(
                f"\n[yellow]Welcome back![/] Setup is {progress}% complete.\n"
                f"Continue where you left off? Use --resume\n"
                f"Start over? Use --force"
            )
        else:
            # Already initialized - show status and next actions
            console.print(
                f"\n[green]✅ PM-Kit already initialized![/]\n\n"
                f"Next actions:\n"
                f"  • Generate your first PRD: [bold]pm new prd 'Your Feature'[/]\n"
                f"  • Add OKRs: [bold]pm okrs add[/]\n"
                f"  • Check status: [bold]pm status[/]\n\n"
                f"To reinitialize, use --force"
            )
        return False

    try:
        # Ensure directory structure exists
        logger.info(f"Ensuring context structure in {pmkit_dir}")
        initialize_context_structure(current_dir)

        # Load or create configuration
        try:
            config_loader = ConfigLoader()
            config = config_loader.load()
        except Exception as e:
            logger.warning(f"Could not load config, using defaults: {e}")
            config = Config(project_root=current_dir)

        # Template mode - use PM archetype for instant setup
        if template:
            return _init_with_template(template, context_dir, config)

        # Quick mode - minimal questions, maximum enrichment
        if quick:
            console.info("[bold cyan]Quick mode:[/] 1 question + maximum enrichment")
            skip_enrichment = False  # Force enrichment in quick mode

        # Non-interactive mode (for CI/CD or automation)
        if non_interactive:
            return _init_non_interactive(context_dir)

        # Run interactive onboarding with 90-second tracking
        console.print(
            Panel(
                "[bold cyan]🚀 Welcome to PM-Kit![/]\n\n"
                "Let's set up your PM workspace in [bold]90 seconds![/]",
                title="PM-Kit Setup",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        # Show progress tracking
        display_90_second_progress(
            console,
            phase="Starting",
            elapsed=int(time.time() - start_time),
            details="Initializing context structure...",
        )

        success, context = run_onboarding(
            config=config,
            context_dir=context_dir,
            resume=resume and not force,
            skip_enrichment=skip_enrichment,
        )

        elapsed = int(time.time() - start_time)

        if success and context:
            # Calculate and display value metrics
            metrics = calculate_value_metrics(context, elapsed_seconds=elapsed)
            display_value_metrics(metrics, console)

            # Show next actions
            console.print(
                "\n[bold]Ready to go! Next actions:[/]\n"
                f"  → [cyan]pm new prd 'Your First Feature'[/] - Generate your first PRD\n"
                f"  → [cyan]pm okrs add[/] - Define quarterly objectives\n"
                f"  → [cyan]pm status[/] - Check context completeness\n"
            )

            if elapsed <= 90:
                console.success(f"\n🏆 Setup complete in {elapsed} seconds! (Goal: 90s)")
            else:
                console.success(f"\n✅ Setup complete in {elapsed} seconds")

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


def _init_with_template(
    template_name: str,
    context_dir: Path,
    config: Config,
) -> bool:
    """
    Initialize with a PM archetype template for instant value.

    Args:
        template_name: Name of the template (e.g., 'b2b_saas')
        context_dir: Directory for context files
        config: Configuration object

    Returns:
        True if successful
    """
    from pmkit.context.manager import ContextManager
    from pmkit.context.models import Context

    template = get_template_by_name(template_name)
    if not template:
        console.error(f"Unknown template: {template_name}")
        console.print("\nAvailable templates:")
        for choice in format_template_choices():
            console.print(f"  • {choice}")
        return False

    console.print(
        Panel(
            f"[bold cyan]Using {template.name} template[/]\n"
            f"{template.description}\n"
            f"Examples: {', '.join(template.example_companies[:3])}",
            title="Template Mode",
            border_style="cyan",
        )
    )

    # Get company and product names
    company_name = Prompt.ask("[bold]Company name[/]", console=console)
    product_name = Prompt.ask("[bold]Product name[/]", console=console)

    # Apply template
    context_data = apply_template(
        PMArchetype(template_name),
        company_name=company_name,
        product_name=product_name,
    )

    # Create context from template
    context = Context(
        company=context_data['company'],
        product=context_data['product'],
        market=context_data['market'],
        team=context_data['team'],
        okrs=context_data['okrs'],
    )

    # Save context
    manager = ContextManager(context_dir)
    success, errors = manager.save_context(context)

    if success:
        # Display template benefits
        console.print(
            Panel(
                f"✅ Created {template.name} workspace with:\n\n"
                f"• {len(template.personas)} pre-configured personas\n"
                f"• {len(template.primary_metrics)} key metrics to track\n"
                f"• {len(template.sample_objectives)} sample OKRs\n"
                f"• {len(template.prd_focus_areas)} PRD focus areas\n\n"
                f"[bold green]Ready to generate your first PRD![/]",
                title="✨ Template Applied",
                border_style="green",
            )
        )
        return True
    else:
        console.error("Failed to apply template")
        for error in errors:
            console.print(f"  - {error.field}: {error.message}", style="dim")
        return False


def _calculate_progress(state: dict) -> int:
    """
    Calculate onboarding progress percentage.

    Args:
        state: Onboarding state dictionary

    Returns:
        Progress percentage (0-100)
    """
    total_fields = 10  # Approximate total fields
    completed = 0

    # Check Phase 1 (essentials)
    if state.get('company_name'):
        completed += 1
    if state.get('product_name'):
        completed += 1

    # Check Phase 2 (enrichment)
    if state.get('phase2_complete'):
        completed += 4

    # Check Phase 3 (advanced)
    if state.get('team_size'):
        completed += 2
    if state.get('okrs'):
        completed += 2

    return int((completed / total_fields) * 100)


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