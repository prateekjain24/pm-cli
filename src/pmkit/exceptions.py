"""
PM-Kit Exception Hierarchy.

Simple, beautiful error handling with Rich formatting for MVP.
"""

from typing import Optional, Any
from rich.panel import Panel
from rich.text import Text

from pmkit.utils.console import console


class PMKitError(Exception):
    """
    Base exception for all PM-Kit errors.
    
    Provides beautiful error formatting with Rich panels.
    """
    
    def __init__(
        self, 
        message: str, 
        suggestion: Optional[str] = None,
        context: Optional[dict[str, Any]] = None
    ):
        """
        Initialize PM-Kit exception.
        
        Args:
            message: The error message
            suggestion: Optional helpful suggestion for fixing the error
            context: Optional context data for debugging
        """
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion
        self.context = context or {}
    
    def display(self) -> None:
        """Display the error beautifully in the console."""
        # Create error text with proper styling
        error_text = Text(self.message, style="bold red")
        
        # Add suggestion if provided
        if self.suggestion:
            error_text.append("\n\n💡 ", style="yellow")
            error_text.append(self.suggestion, style="italic yellow")
        
        # Create and print the error panel
        panel = Panel(
            error_text,
            title="❌ Error",
            title_align="left",
            border_style="red",
            padding=(1, 2)
        )
        console.print(panel)


class ConfigError(PMKitError):
    """Raised when there's a configuration issue."""
    
    def __init__(self, message: str, suggestion: Optional[str] = None):
        if not suggestion and "API" in message.upper():
            suggestion = "Set your API key in environment variables or .env file"
        super().__init__(message, suggestion)


class ContextError(PMKitError):
    """Raised when there's an issue with the context layer."""
    
    def __init__(self, message: str, suggestion: Optional[str] = None):
        if not suggestion and "not found" in message.lower():
            suggestion = "Run 'pm init' to initialize your project context"
        super().__init__(message, suggestion)


class LLMError(PMKitError):
    """Raised when LLM API calls fail."""
    
    def __init__(
        self, 
        message: str, 
        provider: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        if not suggestion:
            if "rate" in message.lower():
                suggestion = "Please wait a moment and try again"
            elif "timeout" in message.lower():
                suggestion = "Check your internet connection or try again"
            elif "key" in message.lower() or "auth" in message.lower():
                suggestion = f"Check your {provider or 'API'} key is valid"
        
        context = {"provider": provider} if provider else {}
        super().__init__(message, suggestion, context)


class ValidationError(PMKitError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        suggestion = f"Check the format of '{field}'" if field else "Check your input format"
        context = {"field": field} if field else {}
        super().__init__(message, suggestion, context)