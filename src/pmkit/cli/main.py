"""
PM-Kit CLI - Beautiful command-line interface.

Main entry point for the PM-Kit CLI following the design philosophy
of delightful, beautiful command-line experiences.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Optional

import typer
from rich.text import Text
from rich.panel import Panel

from pmkit import __version__
from pmkit.utils.console import console
from pmkit.utils.logger import setup_logging, get_logger
from pmkit.config import get_config_safe


# Create main Typer app with beautiful styling
app = typer.Typer(
    name="pmkit",
    help="🚀 PM-Kit: CLI-first PM docs with LLM superpowers",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
)

# Create subcommand for 'new' to handle PRDs, personas, etc.
new_app = typer.Typer(
    name="new",
    help="📝 Create new PM resources (PRDs, personas, OKRs)",
    no_args_is_help=True,
)
app.add_typer(new_app, name="new")

# Add configuration management subcommand
from pmkit.cli.commands.config import app as config_app
app.add_typer(config_app, name="config")

# Add OKR management subcommand
from pmkit.cli.commands.okrs import app as okrs_app
app.add_typer(okrs_app, name="okrs")


def initialize_logging() -> None:
    """Initialize logging system early in the application lifecycle."""
    try:
        # Try to get configuration for logging setup
        config = get_config_safe()
        setup_logging(config)
    except Exception:
        # Fall back to environment-based setup if config is not available
        setup_logging(None)


def handle_debug_mode() -> None:
    """Configure debug mode based on environment variable."""
    if os.getenv("PMKIT_DEBUG"):
        # Enable debug mode - more verbose output and tracebacks
        console.info("Debug mode enabled", emoji=True)
        # Re-initialize logging with debug level if config is available
        try:
            config = get_config_safe() 
            if config:
                config.app.debug = True
                setup_logging(config)
        except Exception:
            pass


def global_exception_handler(exc_type: type, exc_value: Exception, exc_tb: Any) -> None:
    """
    Global exception handler with beautiful error formatting and logging.
    
    Args:
        exc_type: Exception type
        exc_value: Exception instance
        exc_tb: Exception traceback
    """
    # Get logger for exception handling
    logger = get_logger(__name__)
    
    if isinstance(exc_value, KeyboardInterrupt):
        logger.info("Application interrupted by user")
        console.print("\n👋 Goodbye!")
        sys.exit(0)
    
    # Check if this is a Typer exit (normal CLI behavior)
    if hasattr(exc_value, 'exit_code'):
        sys.exit(getattr(exc_value, 'exit_code', 1))
    
    # Format the error beautifully
    error_title = f"Unexpected Error: {exc_type.__name__}"
    error_message = str(exc_value) if str(exc_value) else "An unexpected error occurred"
    
    # Log the exception with full context
    logger.exception(
        f"Uncaught exception: {error_title}",
        extra={
            'error_type': exc_type.__name__,
            'error_message': error_message,
        }
    )
    
    console.error(f"{error_title}: {error_message}")
    
    # Show traceback in debug mode
    if os.getenv("PMKIT_DEBUG"):
        console.print("\n[dim]Full traceback (debug mode):[/dim]")
        traceback.print_exception(exc_type, exc_value, exc_tb)
    else:
        console.info("Run with PMKIT_DEBUG=1 for full traceback", emoji=False)
    
    sys.exit(1)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", help="Show version and exit"
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug mode"
    ),
) -> None:
    """
    PM-Kit: CLI-first PM documentation system.
    
    Make PRDs, roadmaps, OKRs, personas, and release notes live in Git,
    flow through review gates like code, and publish to Confluence/Notion.
    """
    # Initialize logging early
    initialize_logging()
    logger = get_logger(__name__)
    
    logger.debug("PM-Kit CLI started", extra={
        'version': __version__,
        'debug': debug,
        'command': ctx.invoked_subcommand,
        'args': sys.argv[1:],
    })
    
    if version:
        logger.debug("Version requested")
        console.print(f"PM-Kit version [bold primary]{__version__}[/bold primary]")
        raise typer.Exit()
    
    if debug:
        os.environ["PMKIT_DEBUG"] = "1"
        logger.debug("Debug mode enabled via CLI flag")
    
    handle_debug_mode()
    
    # If no subcommand was invoked and no options, show help
    if ctx.invoked_subcommand is None and not version:
        logger.debug("No subcommand provided, showing help")
        console.print(ctx.get_help(), color_system="auto")


@app.command("init")
def init_command(
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing configuration"
    ),
    resume: bool = typer.Option(
        False, "--resume", "-r", help="Resume interrupted setup"
    ),
    template: Optional[str] = typer.Option(
        None, "--template", "-t",
        help="Use PM archetype template (b2b_saas, developer_tool, consumer_app, marketplace, plg_b2b)"
    ),
    quick: bool = typer.Option(
        False, "--quick", "-q", help="Quick mode: 1 question + maximum enrichment"
    ),
    skip_enrichment: bool = typer.Option(
        False, "--skip-enrichment", help="Skip web enrichment phase"
    ),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", help="Create template files only (for CI/CD)"
    ),
) -> None:
    """🏗️ Initialize PM-Kit in current directory (90-second setup!)."""
    logger = get_logger(__name__)
    logger.info("Initializing PM-Kit", extra={
        'force': force,
        'resume': resume,
        'template': template,
        'quick': quick,
    })

    try:
        from pmkit.cli.commands.init import init_pmkit
        success = init_pmkit(
            force=force,
            resume=resume,
            template=template,
            quick=quick,
            skip_enrichment=skip_enrichment,
            non_interactive=non_interactive,
        )
        if success:
            logger.success("PM-Kit initialization completed")
        else:
            logger.info("PM-Kit initialization skipped or incomplete")
    except Exception as e:
        logger.error(f"PM-Kit initialization failed: {e}")
        raise


@app.command("status")
def status_command() -> None:
    """📊 Check PM-Kit context and project status."""
    logger = get_logger(__name__)
    logger.info("Checking PM-Kit status")
    
    try:
        from pmkit.cli.commands.status import check_status
        check_status()
        logger.debug("Status check completed")
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise


@new_app.command("prd")
def new_prd_command(
    title: str = typer.Argument(
        ..., 
        help="Title for the new PRD",
        metavar="TITLE"
    ),
    template: str = typer.Option(
        "default", 
        "--template", 
        "-t", 
        help="PRD template to use"
    ),
) -> None:
    """📋 Create a new Product Requirements Document (PRD)."""
    logger = get_logger(__name__)
    logger.info("Creating new PRD", extra={
        'title': title,
        'template': template
    })
    
    try:
        from pmkit.cli.commands.new import create_prd
        create_prd(title=title, template=template)
        logger.success("PRD creation completed", extra={'title': title})
    except Exception as e:
        logger.error(f"PRD creation failed: {e}", extra={'title': title})
        raise


@app.command("run")
def run_command(
    workflow: str = typer.Argument(
        ...,
        help="Workflow to execute",
        metavar="WORKFLOW"
    ),
) -> None:
    """⚡ Run PM workflow or automation."""
    # Placeholder for workflow execution
    console.status_panel(
        title="Run Workflow",
        content=f"Workflow execution for '{workflow}' is coming soon!\n\nThis will handle automated PM workflows like:\n• PRD generation pipelines\n• Roadmap updates\n• OKR tracking\n• Release planning",
        status="info",
        emoji="⚡"
    )


@app.command("publish")
def publish_command(
    target: Optional[str] = typer.Argument(
        None,
        help="Specific artifact to publish",
        metavar="TARGET"
    ),
    platform: str = typer.Option(
        "confluence",
        "--platform",
        "-p",
        help="Publishing platform (confluence, notion, github)"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview without publishing"
    ),
) -> None:
    """🚀 Publish PRDs and docs to external platforms."""
    # Placeholder for publishing
    action = "Preview" if dry_run else "Publish"
    target_info = f" '{target}'" if target else " all artifacts"
    
    console.status_panel(
        title=f"{action} to {platform.title()}",
        content=f"Publishing{target_info} to {platform} is coming soon!\n\nSupported platforms:\n• Confluence\n• Notion\n• GitHub Wiki\n• Custom webhooks",
        status="info",
        emoji="🚀"
    )


@app.command("sync")
def sync_command(
    resource: str = typer.Argument(
        "all",
        help="Resource to sync (issues, status, metrics)"
    ),
) -> None:
    """🔄 Sync with external tools (Jira, GitHub, etc.)."""
    # Placeholder for sync functionality
    console.status_panel(
        title="Sync External Resources",
        content=f"Syncing '{resource}' is coming soon!\n\nSupported integrations:\n• Jira issues\n• GitHub issues/PRs\n• Linear tickets\n• Slack channels\n• Analytics platforms",
        status="info",
        emoji="🔄"
    )


def setup_exception_handling() -> None:
    """Install global exception handler."""
    sys.excepthook = global_exception_handler


def cli_main() -> None:
    """Main CLI entry point with exception handling and logging."""
    # Initialize logging first thing
    initialize_logging()
    logger = get_logger(__name__)
    
    logger.debug("Starting PM-Kit CLI", extra={
        'python_version': sys.version,
        'argv': sys.argv,
    })
    
    setup_exception_handling()
    
    try:
        app()
    except Exception as e:
        logger.critical(f"Critical error in CLI main: {e}")
        raise
    finally:
        logger.debug("PM-Kit CLI session ended")


if __name__ == "__main__":
    cli_main()