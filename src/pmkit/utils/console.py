"""
Rich console singleton and utilities for PM-Kit CLI.

Provides a centralized console instance with custom theme and
helper methods for consistent styling across all commands.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.text import Text

from pmkit.cli.theme import pmkit_theme


class PMKitConsole:
    """
    Singleton console instance with PM-Kit theming and helper methods.
    
    Provides consistent styling and utilities for all CLI commands.
    """
    
    _instance: Optional[PMKitConsole] = None
    
    def __new__(cls) -> PMKitConsole:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize the Rich console with PM-Kit theme."""
        # Respect NO_COLOR environment variable
        force_terminal = None
        if os.getenv("PMKIT_DEBUG"):
            force_terminal = True
            
        self._console = Console(
            theme=pmkit_theme,
            force_terminal=force_terminal,
            width=None,  # Auto-detect terminal width
        )
    
    @property
    def console(self) -> Console:
        """Access to the underlying Rich console."""
        return self._console
    
    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print with themed console."""
        self._console.print(*args, **kwargs)
    
    def success(self, message: str, emoji: bool = True) -> None:
        """Print a success message with consistent styling."""
        prefix = "✅ " if emoji else ""
        self._console.print(f"{prefix}{message}", style="success.text")
    
    def error(self, message: str, emoji: bool = True) -> None:
        """Print an error message with consistent styling."""
        prefix = "❌ " if emoji else ""
        self._console.print(f"{prefix}{message}", style="error.text")
    
    def warning(self, message: str, emoji: bool = True) -> None:
        """Print a warning message with consistent styling."""
        prefix = "⚠️  " if emoji else ""
        self._console.print(f"{prefix}{message}", style="warning.text")
    
    def info(self, message: str, emoji: bool = True) -> None:
        """Print an info message with consistent styling."""
        prefix = "ℹ️  " if emoji else ""
        self._console.print(f"{prefix}{message}", style="info.text")
    
    def status_panel(
        self,
        title: str,
        content: str,
        status: str = "info",
        emoji: str = "",
    ) -> None:
        """
        Display a status panel with consistent styling.
        
        Args:
            title: Panel title
            content: Panel content
            status: Status type (success, warning, error, info)
            emoji: Optional emoji for the title
        """
        title_text = f"{emoji} {title}" if emoji else title
        
        panel = Panel(
            content,
            title=title_text,
            title_align="left",
            border_style=f"{status}.text" if status != "info" else "panel.border",
            padding=(1, 2),
        )
        self._console.print(panel)
    
    def create_progress(self, description: str = "Working...") -> Progress:
        """
        Create a themed progress bar.
        
        Args:
            description: Description text for the progress bar
            
        Returns:
            Configured Progress instance
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self._console,
        )
    
    def command_help_panel(
        self,
        command: str,
        description: str,
        examples: list[str],
    ) -> None:
        """
        Display a beautiful help panel for commands.
        
        Args:
            command: Command name
            description: Command description  
            examples: List of usage examples
        """
        content = Text()
        content.append(description + "\n\n", style="dim")
        
        if examples:
            content.append("Examples:\n", style="help.option")
            for example in examples:
                content.append(f"  {example}\n", style="help.example")
        
        panel = Panel(
            content,
            title=f"🚀 {command}",
            title_align="left",
            border_style="panel.border",
            padding=(1, 2),
        )
        self._console.print(panel)
    
    def clear(self) -> None:
        """Clear the console screen."""
        self._console.clear()


# Global console instance
console = PMKitConsole()

__all__ = ["console", "PMKitConsole"]