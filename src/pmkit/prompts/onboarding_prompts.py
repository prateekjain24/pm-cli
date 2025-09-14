"""
Onboarding-specific prompts and UI text for PM-Kit.

This module contains all the prompts, messages, and UI text used
during the onboarding flow to ensure consistency and easy updates.
"""

from __future__ import annotations

from typing import Dict, List


class OnboardingPrompts:
    """
    UI text and prompts specifically for the onboarding experience.

    Separated from PromptLibrary to keep user-facing text distinct
    from LLM prompts.
    """

    # ============================================================
    # WELCOME & INTRO
    # ============================================================

    WELCOME_MESSAGE = """
[bold cyan]Welcome to PM-Kit![/bold cyan]

Let's set up your product context in under [bold]5 minutes[/bold].
We'll use this to generate amazing PRDs tailored to your product.

[dim]You can always update these settings later with 'pm context edit'[/dim]
"""

    PHASE_INTRO = {
        1: "[bold]Phase 1: Essentials[/bold] (30 seconds)\n[dim]Just the basics to get you started[/dim]",
        2: "[bold]Phase 2: Enrichment[/bold] (2 minutes)\n[dim]We'll search for additional context[/dim]",
        3: "[bold]Phase 3: Advanced[/bold] (2 minutes)\n[dim]Optional details for better PRDs[/dim]",
    }

    # ============================================================
    # PHASE 1: ESSENTIALS
    # ============================================================

    COMPANY_NAME_PROMPT = "What's your company name?"
    COMPANY_NAME_HELP = "[dim]We'll use this to search for additional context[/dim]"

    COMPANY_TYPE_PROMPT = "What type of business model?"
    COMPANY_TYPE_CHOICES = ["B2B", "B2C", "B2B2C (marketplace/platform)"]
    COMPANY_TYPE_HELP = "[dim]This helps us tailor PRD templates and metrics[/dim]"

    PRODUCT_NAME_PROMPT = "What's your product called?"
    PRODUCT_NAME_HELP = "[dim]The name of your product or service[/dim]"

    PRODUCT_DESC_PROMPT = "Describe your product in one sentence"
    PRODUCT_DESC_HELP = "[dim]Example: 'AI-powered code review tool for engineering teams'[/dim]"

    YOUR_ROLE_PROMPT = "What's your role?"
    YOUR_ROLE_CHOICES = ["Product Manager", "Senior PM", "Director of Product", "CPO", "Founder", "Other"]

    ESSENTIALS_COMPLETE = """
[green]Great! You can start creating PRDs now.[/green]

Would you like to add more context for better PRDs?
This takes ~2 more minutes but improves quality significantly.
"""

    # ============================================================
    # PHASE 2: ENRICHMENT
    # ============================================================

    ENRICHMENT_SEARCHING = "Searching for information about {company_name}..."
    ENRICHMENT_FOUND = """
[green]Found information about {company_name}![/green]

Here's what we discovered:
"""

    ENRICHMENT_NOT_FOUND = """
[yellow]Couldn't auto-enrich company data[/yellow]
No worries! Let's fill in a few details manually:
"""

    ENRICHMENT_CONFIRM = """
Do these details look correct?

[dim]You can edit any field before proceeding[/dim]
"""

    COMPETITOR_PROMPT = "Who are your main competitors? (comma-separated)"
    COMPETITOR_HELP = "[dim]We found: {found_competitors}. Add more or press Enter to accept[/dim]"

    METRIC_PROMPT = "What's your north star metric?"
    METRIC_DEFAULT_B2B = "[dim]Suggested: MRR (Monthly Recurring Revenue)[/dim]"
    METRIC_DEFAULT_B2C = "[dim]Suggested: MAU (Monthly Active Users)[/dim]"

    # ============================================================
    # PHASE 3: ADVANCED (OPTIONAL)
    # ============================================================

    ADVANCED_INTRO = """
[bold]Almost done! These optional details help generate better PRDs:[/bold]

[dim]You can skip any of these or add them later[/dim]
"""

    TEAM_SIZE_PROMPT = "How big is your team? (number or press Enter to skip)"
    TEAM_COMPOSITION_PROMPT = "Team composition? (e.g., '5 engineers, 2 designers, 1 PM')"

    OKR_INTRO = """
[bold]Current OKRs (Optional)[/bold]
PRDs aligned with OKRs are more likely to get approved!
"""

    OKR_SKIP = "Press Enter to skip OKRs for now"
    OKR_QUICK = "What's your main goal this quarter? (we'll convert to an objective)"
    OKR_ADD_MORE = "Add another objective? (y/N)"

    OBJECTIVE_PROMPT = "Describe the objective in one sentence"
    KEY_RESULT_PROMPT = "Key Result {number}: What's the measurable outcome?"
    KEY_RESULT_COUNT = "How many key results for this objective? (1-5, default: 3)"
    CONFIDENCE_PROMPT = "Confidence in achieving this? (0-100%, default: 70%)"

    MARKET_DIFFERENTIATOR_PROMPT = "What's your unique value proposition? (one-liner)"
    MARKET_DIFFERENTIATOR_HELP = "[dim]What makes you different from competitors?[/dim]"

    # ============================================================
    # COMPLETION & SUMMARY
    # ============================================================

    SETUP_COMPLETE = """
[bold green]Setup Complete![/bold green]

Your context has been saved to [cyan].pmkit/context/[/cyan]

You can now:
• Create PRDs: [cyan]pm new prd "Your PRD Title"[/cyan]
• View context: [cyan]pm status[/cyan]
• Edit context: [cyan]pm context edit[/cyan]

[dim]Tip: Your first PRD will take ~30 seconds to generate[/dim]
"""

    CONTEXT_SUMMARY_TEMPLATE = """
[bold]Context Summary[/bold]

[bold cyan]Company[/bold cyan]
• Name: {company_name}
• Type: {company_type}
• Stage: {company_stage}

[bold cyan]Product[/bold cyan]
• Name: {product_name}
• Description: {product_description}
• North Star: {metric}

[bold cyan]Market[/bold cyan]
• Competitors: {competitors}
• Differentiator: {differentiator}

[bold cyan]Team[/bold cyan]
• Size: {team_size}
• Composition: {team_composition}

[bold cyan]OKRs[/bold cyan]
{okr_summary}
"""

    # ============================================================
    # ERROR MESSAGES
    # ============================================================

    ERROR_MESSAGES = {
        'network_error': "[red]Network error. Let's continue with manual input.[/red]",
        'invalid_input': "[red]Invalid input. Please try again.[/red]",
        'save_failed': "[red]Failed to save context. Check permissions for .pmkit/[/red]",
        'context_exists': "[yellow]Context already exists. Use --force to overwrite.[/yellow]",
    }

    # ============================================================
    # PROGRESS INDICATORS
    # ============================================================

    STEP_INDICATOR = "Step {current} of {total} {emoji}"

    STEP_EMOJIS = {
        'company': '',
        'product': '',
        'market': '',
        'team': '',
        'okrs': '',
        'complete': '',
    }

    # ============================================================
    # HELPER METHODS
    # ============================================================

    @classmethod
    def get_step_indicator(cls, current: int, total: int, step_type: str) -> str:
        """
        Get a formatted step indicator with emoji.

        Args:
            current: Current step number
            total: Total number of steps
            step_type: Type of step for emoji selection

        Returns:
            Formatted step indicator string
        """
        emoji = cls.STEP_EMOJIS.get(step_type, '')
        return cls.STEP_INDICATOR.format(
            current=current,
            total=total,
            emoji=emoji
        )

    @classmethod
    def format_okr_summary(cls, okrs: List[Dict]) -> str:
        """
        Format OKRs for display in the summary.

        Args:
            okrs: List of objective dictionaries

        Returns:
            Formatted OKR summary string
        """
        if not okrs:
            return "[dim]No OKRs set (you can add them later)[/dim]"

        summary_lines = []
        for i, objective in enumerate(okrs, 1):
            summary_lines.append(f"• Objective {i}: {objective.get('title', 'Unnamed')}")
            for kr in objective.get('key_results', []):
                confidence = kr.get('confidence', 70)
                color = 'green' if confidence >= 70 else 'yellow' if confidence >= 50 else 'red'
                summary_lines.append(
                    f"  - {kr.get('description', 'Unnamed KR')} [{color}]{confidence}%[/{color}]"
                )

        return '\n'.join(summary_lines)

    @classmethod
    def format_progress_bar(cls, current: int, total: int, width: int = 30) -> str:
        """
        Format a simple progress bar.

        Args:
            current: Current step
            total: Total steps
            width: Width of the progress bar

        Returns:
            Formatted progress bar string
        """
        percentage = int((current / total) * 100) if total > 0 else 0
        filled = int((current / total) * width) if total > 0 else 0
        bar = "█" * filled + "░" * (width - filled)

        return f"[cyan]{bar}[/cyan] {percentage}%"

    @classmethod
    def get_phase_name(cls, phase: int) -> str:
        """
        Get the name of a phase.

        Args:
            phase: Phase number (1-3)

        Returns:
            Phase name string
        """
        phase_names = {
            1: "Essentials",
            2: "Enrichment",
            3: "Advanced",
        }
        return phase_names.get(phase, f"Phase {phase}")

    @classmethod
    def format_list_input(cls, items: str) -> List[str]:
        """
        Parse comma-separated input into a list.

        Args:
            items: Comma-separated string

        Returns:
            List of trimmed items
        """
        if not items:
            return []

        return [item.strip() for item in items.split(',') if item.strip()]