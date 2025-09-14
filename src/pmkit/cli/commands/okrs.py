"""
OKR management commands for PM-Kit.

Provides commands for adding, editing, and managing OKRs after initialization.
Following the progressive disclosure philosophy, OKRs are optional during init
but easily accessible when PMs are ready.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from pmkit.agents.okr_wizard import OKRWizard
from pmkit.utils.async_utils import run_async
from pmkit.context.manager import ContextManager
from pmkit.context.models import OKRContext
from pmkit.exceptions import ContextError
from pmkit.utils.console import console
from pmkit.utils.logger import get_logger


logger = get_logger(__name__)

# Create OKR subcommand app
app = typer.Typer(
    name="okrs",
    help="📊 Manage Objectives and Key Results",
    no_args_is_help=True,
)


@app.command("add")
def add_okrs(
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", "-i/-n",
        help="Use interactive wizard (default) or manual entry"
    ),
) -> None:
    """✨ Add OKRs using the delightful wizard experience."""
    logger.info("Adding OKRs", extra={'interactive': interactive})

    try:
        context_dir = Path.cwd() / ".pmkit" / "context"
        if not context_dir.exists():
            console.error("PM-Kit not initialized. Run 'pm init' first.")
            return

        manager = ContextManager(context_dir)

        if interactive:
            # Use the OKRWizard for delightful experience
            console.print(
                Panel(
                    "[bold cyan]🎯 OKR Setup Wizard[/]\n\n"
                    "Let's define your quarterly objectives and key results.\n"
                    "This takes about 5-10 mins and helps align your PRDs.",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )

            wizard = OKRWizard(console=console)
            okrs = run_async(wizard.collect_okrs())

            if okrs and okrs.objectives:
                # Save OKRs
                context = manager.load_context()
                if not context:
                    context = manager.load_context(validate=False)

                context.okrs = okrs
                success, errors = manager.save_context(context)

                if success:
                    console.success("\n✅ OKRs saved successfully!")
                    _display_okr_summary(okrs)
                else:
                    console.error("Failed to save OKRs")
                    for error in errors:
                        console.print(f"  - {error.field}: {error.message}", style="dim")
            else:
                console.info("OKR setup cancelled")
        else:
            # Manual entry mode
            console.info("Manual OKR entry not yet implemented. Use interactive mode.")

    except ContextError as e:
        e.display()
    except Exception as e:
        logger.error(f"Failed to add OKRs: {e}")
        console.error(f"Failed to add OKRs: {e}")


@app.command("status")
def okr_status() -> None:
    """📈 View OKR progress and confidence scores."""
    logger.info("Checking OKR status")

    try:
        context_dir = Path.cwd() / ".pmkit" / "context"
        if not context_dir.exists():
            console.error("PM-Kit not initialized. Run 'pm init' first.")
            return

        manager = ContextManager(context_dir)
        context = manager.load_context(validate=False)

        if not context or not context.okrs or not context.okrs.objectives:
            console.print(
                Panel(
                    "[yellow]No OKRs defined yet[/]\n\n"
                    "Define your quarterly objectives to:\n"
                    "  • Align PRDs with strategic goals\n"
                    "  • Track progress systematically\n"
                    "  • Communicate priorities clearly\n\n"
                    "Run [bold]pm okrs add[/] to get started!",
                    title="📊 OKR Status",
                    border_style="yellow",
                )
            )
            return

        # Display OKR status
        _display_okr_status(context.okrs)

    except Exception as e:
        logger.error(f"Failed to check OKR status: {e}")
        console.error(f"Failed to check OKR status: {e}")


@app.command("edit")
def edit_okrs() -> None:
    """✏️ Edit existing OKRs."""
    logger.info("Editing OKRs")

    try:
        context_dir = Path.cwd() / ".pmkit" / "context"
        if not context_dir.exists():
            console.error("PM-Kit not initialized. Run 'pm init' first.")
            return

        manager = ContextManager(context_dir)
        context = manager.load_context(validate=False)

        if not context or not context.okrs or not context.okrs.objectives:
            console.info("No OKRs to edit. Run 'pm okrs add' first.")
            return

        # Re-run wizard with existing OKRs as starting point
        console.print(
            Panel(
                "[bold cyan]📝 Edit OKRs[/]\n\n"
                "Modifying existing objectives and key results.",
                border_style="cyan",
            )
        )

        wizard = OKRWizard(console=console, existing_okrs=context.okrs)
        updated_okrs = run_async(wizard.collect_okrs())

        if updated_okrs:
            context.okrs = updated_okrs
            success, errors = manager.save_context(context)

            if success:
                console.success("\n✅ OKRs updated successfully!")
                _display_okr_summary(updated_okrs)
            else:
                console.error("Failed to update OKRs")
        else:
            console.info("Edit cancelled")

    except Exception as e:
        logger.error(f"Failed to edit OKRs: {e}")
        console.error(f"Failed to edit OKRs: {e}")


@app.command("archive")
def archive_okrs() -> None:
    """📦 Archive completed OKRs."""
    logger.info("Archiving OKRs")

    try:
        context_dir = Path.cwd() / ".pmkit" / "context"
        if not context_dir.exists():
            console.error("PM-Kit not initialized. Run 'pm init' first.")
            return

        # Archive logic would go here
        console.info("OKR archiving will be available in a future update.")

    except Exception as e:
        logger.error(f"Failed to archive OKRs: {e}")
        console.error(f"Failed to archive OKRs: {e}")


def _display_okr_summary(okrs: OKRContext) -> None:
    """Display a summary of OKRs."""
    if not okrs or not okrs.objectives:
        return

    console.print("\n[bold]Your OKRs:[/]\n")

    for i, obj in enumerate(okrs.objectives, 1):
        # Objective with confidence indicator
        confidence = obj.average_confidence
        if confidence >= 70:
            conf_emoji = "🟢"
            conf_style = "green"
        elif confidence >= 50:
            conf_emoji = "🟡"
            conf_style = "yellow"
        else:
            conf_emoji = "🔴"
            conf_style = "red"

        console.print(f"{i}. [bold]{obj.text}[/] {conf_emoji} [{conf_style}]{confidence}% confidence[/]")

        # Key results
        for j, kr in enumerate(obj.key_results, 1):
            kr_conf = kr.confidence
            if kr_conf >= 70:
                kr_emoji = "✅"
            elif kr_conf >= 50:
                kr_emoji = "⚠️"
            else:
                kr_emoji = "❌"

            console.print(f"   {i}.{j} {kr_emoji} {kr.text}")
            if kr.target_value:
                console.print(f"       Target: {kr.target_value}", style="dim")

        console.print()


def _display_okr_status(okrs: OKRContext) -> None:
    """Display detailed OKR status with progress."""
    table = Table(
        title=f"📊 OKR Status - {okrs.quarter} {okrs.year}" if okrs.quarter else "📊 OKR Status",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Objective", style="white", width=40)
    table.add_column("Key Results", style="dim", width=30)
    table.add_column("Confidence", justify="center", width=15)
    table.add_column("Status", justify="center", width=10)

    for obj in okrs.objectives:
        # Build KR summary
        kr_summary = []
        for kr in obj.key_results[:2]:  # Show first 2 KRs
            kr_text = kr.text[:25] + "..." if len(kr.text) > 25 else kr.text
            kr_summary.append(kr_text)
        if len(obj.key_results) > 2:
            kr_summary.append(f"... +{len(obj.key_results) - 2} more")

        # Confidence styling
        conf = obj.average_confidence
        if conf >= 70:
            conf_text = Text(f"{conf}%", style="green")
            status = "✅ On Track"
        elif conf >= 50:
            conf_text = Text(f"{conf}%", style="yellow")
            status = "⚠️ At Risk"
        else:
            conf_text = Text(f"{conf}%", style="red")
            status = "❌ Needs Help"

        table.add_row(
            obj.text[:40] + "..." if len(obj.text) > 40 else obj.text,
            "\n".join(kr_summary),
            conf_text,
            status,
        )

    console.print(table)

    # Show at-risk items
    at_risk = okrs.at_risk_key_results
    if at_risk:
        console.print("\n[yellow]⚠️ At-Risk Key Results:[/]")
        for kr in at_risk[:3]:  # Show top 3
            console.print(f"  • {kr.text} ({kr.confidence}% confidence)", style="yellow")


# Export the app for use in main CLI
__all__ = ["app"]