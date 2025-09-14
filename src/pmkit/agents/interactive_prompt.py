"""
Interactive prompt flow with prompt_toolkit for delightful onboarding experience.

This module provides a wizard-style interface with progressive disclosure,
validation, auto-completion, and beautiful feedback for PM-Kit onboarding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, WordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import confirm, message_dialog
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import ValidationError, Validator
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from pmkit.utils.console import console as pmkit_console
from pmkit.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# ENUMS & CONSTANTS
# ============================================================

class StepAction(Enum):
    """Actions available during wizard navigation."""
    NEXT = "next"
    BACK = "back"
    SKIP = "skip"
    CANCEL = "cancel"
    HELP = "help"


# Quick setup templates for B2B vs B2C
B2B_QUICK_SETUP = {
    'metrics': ['MRR', 'ARR', 'Logo Retention', 'NPS', 'CAC Payback'],
    'personas': ['IT Admin', 'End User', 'Executive Buyer', 'Champion'],
    'features': ['API', 'SSO', 'Audit Logs', 'SLAs', 'Integrations'],
    'industries': ['SaaS', 'Enterprise Software', 'Infrastructure', 'DevTools'],
}

B2C_QUICK_SETUP = {
    'metrics': ['MAU', 'DAU', 'Retention Rate', 'Viral Coefficient', 'LTV'],
    'personas': ['Power User', 'Casual User', 'Churned User', 'New User'],
    'features': ['Mobile App', 'Social Sharing', 'Gamification', 'Notifications'],
    'industries': ['Consumer Apps', 'Gaming', 'Social Media', 'E-commerce'],
}


# ============================================================
# VALIDATORS
# ============================================================

class CompanyNameValidator(Validator):
    """Validator for company names with helpful feedback."""

    GENERIC_NAMES = ['test', 'demo', 'asdf', 'company', 'my company', 'startup']
    FORBIDDEN_CHARS = r'[<>:"/\\|?*]'

    def validate(self, document: Document) -> None:
        text = document.text.strip()

        # Check length
        if len(text) < 2:
            raise ValidationError(
                message='Company name must be at least 2 characters',
                cursor_position=len(text)
            )

        if len(text) > 50:
            raise ValidationError(
                message='Company name must be less than 50 characters',
                cursor_position=50
            )

        # Check for forbidden characters
        if re.search(self.FORBIDDEN_CHARS, text):
            raise ValidationError(
                message='Company name cannot contain special characters: < > : " / \\ | ? *',
                cursor_position=len(text)
            )

        # Warn about generic names (but don't block)
        if text.lower() in self.GENERIC_NAMES:
            # This is a warning, not an error - we'll handle this separately
            pass


class EmailValidator(Validator):
    """Email validator with clear error messages."""

    EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    def validate(self, document: Document) -> None:
        text = document.text.strip()

        if not text:  # Email is optional
            return

        if '@' not in text:
            raise ValidationError(
                message='Email must contain @ symbol',
                cursor_position=len(text)
            )

        if not re.match(self.EMAIL_PATTERN, text):
            raise ValidationError(
                message='Please enter a valid email (e.g., pm@company.com)',
                cursor_position=len(text)
            )


class URLValidator(Validator):
    """URL validator for optional website field."""

    URL_PATTERN = r'^https?://[^\s]+$'  # Simplified pattern - just check basic format

    def validate(self, document: Document) -> None:
        text = document.text.strip()

        if not text:  # URL is optional
            return

        if not text.startswith(('http://', 'https://')):
            raise ValidationError(
                message='URL must start with http:// or https://',
                cursor_position=0
            )

        if not re.match(self.URL_PATTERN, text):
            raise ValidationError(
                message='Please enter a valid URL (e.g., https://company.com)',
                cursor_position=len(text)
            )


class ProductDescriptionValidator(Validator):
    """Validator for product descriptions."""

    def validate(self, document: Document) -> None:
        text = document.text.strip()
        words = text.split()

        if len(words) < 5:
            raise ValidationError(
                message='Please describe your product in at least 5 words (e.g., "AI code review tool for teams")',
                cursor_position=len(text)
            )

        # Check for WHO it's for
        has_target = any(word in text.lower() for word in [
            'for', 'helps', 'enables', 'allows', 'lets'
        ])

        if not has_target:
            # This is a warning we'll handle separately
            pass


class TeamSizeValidator(Validator):
    """Validator for team size input."""

    def validate(self, document: Document) -> None:
        text = document.text.strip()

        if not text:  # Team size is optional
            return

        if not text.isdigit():
            raise ValidationError(
                message='Team size must be a number',
                cursor_position=len(text)
            )

        size = int(text)
        if size < 1:
            raise ValidationError(
                message='Team size must be at least 1',
                cursor_position=len(text)
            )

        if size > 10000:
            raise ValidationError(
                message='Team size seems too large. Please verify.',
                cursor_position=len(text)
            )


# ============================================================
# COMPLETERS
# ============================================================

class IndustryCompleter(Completer):
    """Auto-completer for industry selection."""

    B2B_INDUSTRIES = [
        'SaaS', 'Enterprise Software', 'Infrastructure', 'DevTools',
        'Security', 'Data Analytics', 'API Platform', 'Cloud Services',
        'HR Tech', 'FinTech B2B', 'MarTech', 'LegalTech'
    ]

    B2C_INDUSTRIES = [
        'Consumer Apps', 'Gaming', 'Social Media', 'E-commerce',
        'EdTech', 'HealthTech', 'FinTech B2C', 'Travel', 'Food Delivery',
        'Entertainment', 'Fitness', 'Dating'
    ]

    def __init__(self, company_type: str = 'b2b'):
        self.company_type = company_type

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        word = document.get_word_before_cursor()
        industries = self.B2B_INDUSTRIES if self.company_type == 'b2b' else self.B2C_INDUSTRIES

        for industry in industries:
            if industry.lower().startswith(word.lower()):
                yield Completion(
                    industry,
                    start_position=-len(word),
                    display=industry,
                    display_meta='(Common industry)'
                )


class RoleCompleter(Completer):
    """Auto-completer for PM roles."""

    ROLES = [
        'Product Manager',
        'Senior Product Manager',
        'Lead Product Manager',
        'Principal Product Manager',
        'Group Product Manager',
        'Director of Product',
        'VP of Product',
        'CPO',
        'Founder',
        'Product Owner',
        'Technical Product Manager',
    ]

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        word = document.get_word_before_cursor()

        for role in self.ROLES:
            if role.lower().startswith(word.lower()):
                yield Completion(
                    role,
                    start_position=-len(word),
                    display=role,
                )


class MetricCompleter(Completer):
    """Auto-completer for metrics based on company type."""

    def __init__(self, company_type: str = 'b2b'):
        self.company_type = company_type
        self.metrics = B2B_QUICK_SETUP['metrics'] if company_type == 'b2b' else B2C_QUICK_SETUP['metrics']

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        word = document.get_word_before_cursor()

        for metric in self.metrics:
            if metric.lower().startswith(word.lower()):
                yield Completion(
                    metric,
                    start_position=-len(word),
                    display=metric,
                    display_meta='(Recommended metric)'
                )


# ============================================================
# WIZARD STATE MANAGEMENT
# ============================================================

@dataclass
class WizardStep:
    """Represents a single step in the wizard."""
    name: str
    prompt: str
    validator: Optional[Validator] = None
    completer: Optional[Completer] = None
    default: str = ""
    required: bool = True
    help_text: str = ""
    phase: int = 1  # 1=Essentials, 2=Enrichment, 3=Advanced


class WizardState:
    """Manages wizard state and navigation."""

    def __init__(self):
        self.steps: List[WizardStep] = []
        self.current_step: int = 0
        self.data: Dict[str, Any] = {}
        self.history: List[int] = []  # For back navigation

    def add_step(self, step: WizardStep) -> None:
        """Add a step to the wizard."""
        self.steps.append(step)

    def can_go_back(self) -> bool:
        """Check if we can navigate back."""
        return len(self.history) > 0

    def go_back(self) -> Optional[WizardStep]:
        """Navigate to previous step."""
        if self.can_go_back():
            self.current_step = self.history.pop()
            return self.steps[self.current_step]
        return None

    def next_step(self) -> Optional[WizardStep]:
        """Move to next step."""
        if self.current_step < len(self.steps) - 1:
            self.history.append(self.current_step)
            self.current_step += 1
            return self.steps[self.current_step]
        return None

    def get_current_step(self) -> Optional[WizardStep]:
        """Get current step."""
        if 0 <= self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None

    def save_step_data(self, key: str, value: Any) -> None:
        """Save data for current step."""
        self.data[key] = value

    def get_progress(self) -> Tuple[int, int]:
        """Get progress as (current, total)."""
        return (self.current_step + 1, len(self.steps))


# ============================================================
# INTERACTIVE PROMPT FLOW
# ============================================================

class InteractivePromptFlow:
    """
    Main class for interactive prompt flow with prompt_toolkit.

    Provides wizard-style interface with validation, completion,
    and progressive disclosure for delightful onboarding.
    """

    def __init__(self, console: Optional[Console] = None):
        """Initialize the interactive prompt flow."""
        self.console = console or pmkit_console.console
        self.session = PromptSession()
        self.wizard_state = WizardState()

        # Custom style for prompt_toolkit
        self.style = Style.from_dict({
            'prompt': '#00a6fb bold',  # Primary blue
            'success': '#52c41a',      # Success green
            'warning': '#faad14',      # Warning orange
            'error': '#ff4d4f',        # Error red
            'info': '#1890ff',         # Info blue
            'muted': '#8c8c8c',        # Muted gray
        })

        # Setup key bindings for navigation
        self._setup_key_bindings()

    def _setup_key_bindings(self) -> None:
        """Setup custom key bindings for navigation."""
        self.kb = KeyBindings()

        # Add key bindings for special commands
        # Note: These would be handled in the prompt processing
        pass

    def prompt_with_validation(
        self,
        prompt_text: str,
        validator: Optional[Validator] = None,
        completer: Optional[Completer] = None,
        default: str = "",
        help_text: str = "",
        multiline: bool = False,
    ) -> Optional[str]:
        """
        Prompt for input with validation and completion.

        Args:
            prompt_text: The prompt to display
            validator: Optional validator for input
            completer: Optional completer for suggestions
            default: Default value
            help_text: Help text to display
            multiline: Whether to allow multiline input

        Returns:
            User input or None if cancelled
        """
        # Display help text if provided
        if help_text:
            self.console.print(f"[dim]{help_text}[/dim]")

        # Format the prompt
        formatted_prompt = HTML(f'<prompt>{prompt_text}</prompt> ')

        try:
            result = self.session.prompt(
                formatted_prompt,
                validator=validator,
                completer=completer,
                default=default,
                multiline=multiline,
                style=self.style,
                complete_while_typing=True,
                enable_history_search=True,
            )

            # Check for navigation commands
            if result.lower() in ['back', ':back', ':b']:
                return ':back'
            elif result.lower() in ['skip', ':skip', ':s']:
                return ':skip'
            elif result.lower() in ['help', ':help', ':h', '?']:
                return ':help'
            elif result.lower() in ['quit', 'exit', ':quit', ':q']:
                return ':quit'

            return result.strip()

        except KeyboardInterrupt:
            return ':quit'
        except EOFError:
            return ':quit'

    def select_with_completion(
        self,
        prompt_text: str,
        choices: List[str],
        default: Optional[str] = None,
        help_text: str = "",
    ) -> Optional[str]:
        """
        Select from choices with auto-completion.

        Args:
            prompt_text: The prompt to display
            choices: List of valid choices
            default: Default choice
            help_text: Help text to display

        Returns:
            Selected choice or None if cancelled
        """
        # Display choices
        self.console.print(f"\n[bold]{prompt_text}[/bold]")
        if help_text:
            self.console.print(f"[dim]{help_text}[/dim]")

        for i, choice in enumerate(choices, 1):
            self.console.print(f"  {i}. {choice}")

        # Create completer for choices
        completer = WordCompleter(choices, ignore_case=True, sentence=True)

        # Prompt for selection
        result = self.prompt_with_validation(
            "Enter choice (number or text)",
            completer=completer,
            default=default or "",
        )

        if result and result.startswith(':'):
            return result  # Navigation command

        # Handle numeric input
        if result and result.isdigit():
            idx = int(result) - 1
            if 0 <= idx < len(choices):
                return choices[idx]

        # Handle text input
        if result in choices:
            return result

        # Try case-insensitive match
        for choice in choices:
            if choice.lower() == result.lower():
                return choice

        return None

    def show_progress(self, current: int, total: int, phase_name: str) -> None:
        """
        Display progress indicator.

        Args:
            current: Current step number
            total: Total number of steps
            phase_name: Name of current phase
        """
        percentage = int((current / total) * 100) if total > 0 else 0
        time_estimate = self._estimate_time_remaining(current, total)

        # Create progress text
        progress_text = Text()
        progress_text.append(f"Step {current} of {total}", style="bold cyan")
        progress_text.append(" • ", style="dim")
        progress_text.append(phase_name, style="bold")
        progress_text.append(" • ", style="dim")
        progress_text.append(f"{percentage}% complete", style="green")

        if time_estimate:
            progress_text.append(" • ", style="dim")
            progress_text.append(time_estimate, style="yellow")

        # Display in a panel
        panel = Panel(
            progress_text,
            border_style="cyan",
            padding=(0, 1),
        )
        self.console.print(panel)

    def _estimate_time_remaining(self, current: int, total: int) -> str:
        """Estimate time remaining based on progress."""
        if current <= 4:  # Phase 1
            return "15 seconds to your first PRD"
        elif current <= 8:  # Phase 2
            return "5-10 mins remaining"
        else:  # Phase 3
            return "Almost done!"

    def preview_and_confirm(self, data: Dict[str, Any]) -> bool:
        """
        Show preview of collected data and confirm.

        Args:
            data: Collected data to preview

        Returns:
            True if confirmed, False otherwise
        """
        # Create a summary table
        table = Table(title="Review Your Information", show_header=False)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")

        # Add essential fields
        essential_fields = [
            ('Company', data.get('company_name', 'Not set')),
            ('Type', data.get('company_type', 'Not set')),
            ('Product', data.get('product_name', 'Not set')),
            ('Description', data.get('product_description', 'Not set')),
        ]

        for field, value in essential_fields:
            if value and value != 'Not set':
                table.add_row(field, str(value))

        # Add optional fields if present
        optional_fields = [
            ('Industry', data.get('industry')),
            ('Competitors', ', '.join(data.get('competitors', [])) if data.get('competitors') else None),
            ('Metric', data.get('north_star_metric')),
            ('Team Size', data.get('team_size')),
        ]

        for field, value in optional_fields:
            if value:
                table.add_row(field, str(value))

        # Display the table
        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")

        # Confirm
        return confirm("Save this configuration and continue?")

    def show_value_proposition(self, company_type: str) -> None:
        """
        Show immediate value after Phase 1.

        Args:
            company_type: The company type (b2b/b2c)
        """
        self.console.print("\n")
        panel_content = Text()
        panel_content.append("🎉 Great! You can now create PRDs!\n\n", style="bold green")
        panel_content.append("Try this command:\n", style="bold")
        panel_content.append("pm new prd 'Your First PRD Title' --quick\n\n", style="cyan")
        panel_content.append("This will generate a PRD in ~15 seconds using smart defaults.\n\n", style="dim")

        if company_type == 'b2b':
            panel_content.append("Popular B2B PRD templates:\n", style="bold")
            panel_content.append("• API Integration PRD\n", style="cyan")
            panel_content.append("• Enterprise Security PRD\n", style="cyan")
            panel_content.append("• Admin Dashboard PRD\n", style="cyan")
        else:
            panel_content.append("Popular B2C PRD templates:\n", style="bold")
            panel_content.append("• Mobile App Redesign PRD\n", style="cyan")
            panel_content.append("• User Onboarding PRD\n", style="cyan")
            panel_content.append("• Social Features PRD\n", style="cyan")

        panel = Panel(
            panel_content,
            title="✨ You're Ready to Go!",
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(panel)

    def multi_step_wizard(
        self,
        steps: List[WizardStep],
        allow_skip: bool = True,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a multi-step wizard with navigation.

        Args:
            steps: List of wizard steps
            allow_skip: Whether to allow skipping optional steps
            show_progress: Whether to show progress indicator

        Returns:
            Dictionary of collected data
        """
        # Initialize wizard state
        self.wizard_state = WizardState()
        for step in steps:
            self.wizard_state.add_step(step)

        while True:
            step = self.wizard_state.get_current_step()
            if not step:
                break

            # Show progress
            if show_progress:
                current, total = self.wizard_state.get_progress()
                phase_name = self._get_phase_name(step.phase)
                self.show_progress(current, total, phase_name)

            # Get input for current step
            result = self.prompt_with_validation(
                prompt_text=step.prompt,
                validator=step.validator,
                completer=step.completer,
                default=step.default,
                help_text=step.help_text,
            )

            # Handle navigation
            if result == ':back':
                if self.wizard_state.can_go_back():
                    self.wizard_state.go_back()
                    self.console.print("[yellow]Going back to previous step...[/yellow]")
                else:
                    self.console.print("[red]Cannot go back from first step[/red]")
                continue

            elif result == ':skip':
                if allow_skip and not step.required:
                    self.wizard_state.save_step_data(step.name, None)
                    self.wizard_state.next_step()
                    self.console.print("[yellow]Skipping optional step...[/yellow]")
                else:
                    self.console.print("[red]This step is required and cannot be skipped[/red]")
                continue

            elif result == ':help':
                self._show_step_help(step)
                continue

            elif result == ':quit':
                if confirm("Are you sure you want to quit? Your progress will be saved."):
                    return self.wizard_state.data
                continue

            # Validate and save data
            if result or not step.required:
                self.wizard_state.save_step_data(step.name, result)

                # Check if this is end of Phase 1
                if step.phase == 1 and self._is_end_of_phase(step, 1):
                    self.show_value_proposition(
                        self.wizard_state.data.get('company_type', 'b2b')
                    )
                    if not confirm("Would you like to add more context for better PRDs?"):
                        break

                # Move to next step
                if not self.wizard_state.next_step():
                    break  # No more steps

        return self.wizard_state.data

    def _get_phase_name(self, phase: int) -> str:
        """Get the name of a phase."""
        phase_names = {
            1: "Essentials",
            2: "Enrichment",
            3: "Advanced",
        }
        return phase_names.get(phase, f"Phase {phase}")

    def _is_end_of_phase(self, step: WizardStep, phase: int) -> bool:
        """Check if this is the last step of a phase."""
        current_idx = self.wizard_state.current_step
        if current_idx >= len(self.wizard_state.steps) - 1:
            return True

        next_step = self.wizard_state.steps[current_idx + 1]
        return next_step.phase > phase

    def _show_step_help(self, step: WizardStep) -> None:
        """Show help for current step."""
        help_panel = Panel(
            f"{step.help_text}\n\n[dim]Navigation commands:[/dim]\n"
            "• Type 'back' to go to previous step\n"
            "• Type 'skip' to skip optional fields\n"
            "• Type 'help' to see this message\n"
            "• Press Ctrl+C to save and quit",
            title=f"ℹ️ Help: {step.name}",
            border_style="blue",
        )
        self.console.print(help_panel)


# ============================================================
# QUICK SETUP HELPERS
# ============================================================

def create_quick_setup_wizard(company_type: str) -> List[WizardStep]:
    """
    Create a quick setup wizard based on company type.

    Args:
        company_type: 'b2b' or 'b2c'

    Returns:
        List of wizard steps for quick setup
    """
    setup = B2B_QUICK_SETUP if company_type == 'b2b' else B2C_QUICK_SETUP

    steps = [
        # Phase 1: Essentials (4 questions only)
        WizardStep(
            name="company_name",
            prompt="What's your company name?",
            validator=CompanyNameValidator(),
            required=True,
            help_text="We'll use this to search for additional context",
            phase=1,
        ),
        WizardStep(
            name="company_type",
            prompt="Business model?",
            completer=WordCompleter(['B2B', 'B2C', 'B2B2C'], ignore_case=True),
            default=company_type.upper(),
            required=True,
            help_text="This helps us tailor PRD templates and metrics",
            phase=1,
        ),
        WizardStep(
            name="product_name",
            prompt="What's your product called?",
            required=True,
            help_text="The name of your product or service",
            phase=1,
        ),
        WizardStep(
            name="product_description",
            prompt="Describe your product in one sentence",
            validator=ProductDescriptionValidator(),
            required=True,
            help_text="Example: 'AI-powered code review tool for engineering teams'",
            phase=1,
        ),

        # Phase 2: Enrichment (optional but recommended)
        WizardStep(
            name="industry",
            prompt="What industry are you in?",
            completer=IndustryCompleter(company_type),
            required=False,
            help_text="This helps us suggest relevant templates and competitors",
            phase=2,
        ),
        WizardStep(
            name="competitors",
            prompt="Who are your main competitors? (comma-separated, or press Enter to skip)",
            required=False,
            help_text="We'll use these for competitive analysis in PRDs",
            phase=2,
        ),
        WizardStep(
            name="north_star_metric",
            prompt="What's your north star metric?",
            completer=MetricCompleter(company_type),
            default=setup['metrics'][0],
            required=False,
            help_text=f"Suggested: {setup['metrics'][0]} for {company_type.upper()} companies",
            phase=2,
        ),

        # Phase 3: Advanced (completely optional)
        WizardStep(
            name="team_size",
            prompt="How big is your team? (number or press Enter to skip)",
            validator=TeamSizeValidator(),
            required=False,
            help_text="This helps us scale PRD complexity appropriately",
            phase=3,
        ),
    ]

    return steps