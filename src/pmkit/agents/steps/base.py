"""
Base class for onboarding steps.

Provides common functionality for all onboarding steps including
validation, state management, and UI helpers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from pmkit.prompts.onboarding_prompts import OnboardingPrompts
from pmkit.utils.console import console as pmkit_console


class BaseStep(ABC):
    """
    Abstract base class for onboarding steps.

    Each step in the onboarding process inherits from this class
    and implements the run method.
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        state: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the step.

        Args:
            console: Rich console instance (uses pmkit_console if not provided)
            state: Shared state dictionary for passing data between steps
        """
        self.console = console or pmkit_console.console
        self.state = state or {}
        self.prompts = OnboardingPrompts

    @abstractmethod
    async def run(self) -> Dict[str, Any]:
        """
        Execute the step and return results.

        Returns:
            Dictionary containing step results to be merged into state

        Raises:
            Exception: If step fails or is cancelled
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name of the step.

        Returns:
            Step name for display purposes
        """
        pass

    @property
    @abstractmethod
    def step_type(self) -> str:
        """
        Type of step for emoji selection.

        Returns:
            Step type (e.g., 'company', 'product', 'team')
        """
        pass

    def show_step_header(self, current: int, total: int) -> None:
        """
        Display the step header with progress indicator.

        Args:
            current: Current step number
            total: Total number of steps
        """
        indicator = self.prompts.get_step_indicator(current, total, self.step_type)
        self.console.print(f"\n{indicator}")
        self.console.print(f"[bold]{self.name}[/bold]\n")

    def prompt_with_default(
        self,
        prompt_text: str,
        default: Optional[str] = None,
        help_text: Optional[str] = None,
        password: bool = False,
    ) -> str:
        """
        Prompt for input with optional default and help text.

        Args:
            prompt_text: The prompt to display
            default: Default value if user presses Enter
            help_text: Additional help text to show
            password: Whether to hide input (for sensitive data)

        Returns:
            User input or default value
        """
        if help_text:
            self.console.print(help_text)

        return Prompt.ask(
            prompt_text,
            default=default,
            password=password,
            console=self.console,
        )

    def prompt_choice(
        self,
        prompt_text: str,
        choices: list[str],
        default: Optional[str] = None,
        help_text: Optional[str] = None,
    ) -> str:
        """
        Prompt for a choice from a list.

        Args:
            prompt_text: The prompt to display
            choices: List of valid choices
            default: Default choice if user presses Enter
            help_text: Additional help text to show

        Returns:
            Selected choice
        """
        if help_text:
            self.console.print(help_text)

        # Display choices
        for i, choice in enumerate(choices, 1):
            self.console.print(f"  {i}. {choice}")

        while True:
            response = Prompt.ask(
                prompt_text,
                default=default or str(choices.index(default) + 1) if default in choices else "1",
                console=self.console,
            )

            # Allow selection by number
            if response.isdigit():
                idx = int(response) - 1
                if 0 <= idx < len(choices):
                    return choices[idx]

            # Allow selection by name
            if response in choices:
                return response

            # Check if it's a partial match
            matches = [c for c in choices if c.lower().startswith(response.lower())]
            if len(matches) == 1:
                return matches[0]

            self.console.print("[red]Invalid choice. Please try again.[/red]")

    def confirm(
        self,
        prompt_text: str,
        default: bool = False,
    ) -> bool:
        """
        Prompt for yes/no confirmation.

        Args:
            prompt_text: The confirmation prompt
            default: Default value if user presses Enter

        Returns:
            True if confirmed, False otherwise
        """
        return Confirm.ask(
            prompt_text,
            default=default,
            console=self.console,
        )

    def show_panel(
        self,
        content: str,
        title: Optional[str] = None,
        style: str = "info",
    ) -> None:
        """
        Display content in a styled panel.

        Args:
            content: Panel content
            title: Optional panel title
            style: Panel style (success, warning, error, info)
        """
        border_styles = {
            'success': 'green',
            'warning': 'yellow',
            'error': 'red',
            'info': 'cyan',
        }

        panel = Panel(
            content,
            title=title,
            title_align="left",
            border_style=border_styles.get(style, 'cyan'),
            padding=(1, 2),
        )
        self.console.print(panel)

    def show_error(self, message: str) -> None:
        """
        Display an error message.

        Args:
            message: Error message to display
        """
        self.console.print(f"[red]L {message}[/red]")

    def show_warning(self, message: str) -> None:
        """
        Display a warning message.

        Args:
            message: Warning message to display
        """
        self.console.print(f"[yellow]� {message}[/yellow]")

    def show_success(self, message: str) -> None:
        """
        Display a success message.

        Args:
            message: Success message to display
        """
        self.console.print(f"[green] {message}[/green]")

    def show_info(self, message: str) -> None:
        """
        Display an info message.

        Args:
            message: Info message to display
        """
        self.console.print(f"[cyan]9 {message}[/cyan]")

    def validate_required(self, value: Any, field_name: str) -> bool:
        """
        Validate that a required field has a value.

        Args:
            value: The value to validate
            field_name: Name of the field for error messages

        Returns:
            True if valid, False otherwise
        """
        if not value or (isinstance(value, str) and not value.strip()):
            self.show_error(f"{field_name} is required")
            return False
        return True

    def save_to_state(self, data: Dict[str, Any]) -> None:
        """
        Save data to the shared state.

        Args:
            data: Dictionary of data to merge into state
        """
        self.state.update(data)