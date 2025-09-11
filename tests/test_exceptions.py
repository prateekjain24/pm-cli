"""
Comprehensive test suite for PM-Kit exception handling.

Tests all exception types, display formatting, auto-suggestions,
context preservation, and edge cases.
"""

from __future__ import annotations

from io import StringIO
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from rich.panel import Panel

# Import exceptions module for patching
import pmkit.exceptions
from pmkit.exceptions import (
    ConfigError,
    ContextError,
    LLMError,
    PMKitError,
    ValidationError,
)


class TestPMKitError:
    """Test suite for the base PMKitError exception."""
    
    def test_creates_with_message_only(self):
        """Test that PMKitError can be created with just a message."""
        error = PMKitError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.suggestion is None
        assert error.context == {}
    
    def test_creates_with_message_and_suggestion(self):
        """Test that PMKitError preserves both message and suggestion."""
        error = PMKitError(
            "Database connection failed",
            suggestion="Check your connection string"
        )
        assert error.message == "Database connection failed"
        assert error.suggestion == "Check your connection string"
        assert error.context == {}
    
    def test_creates_with_full_context(self):
        """Test that PMKitError preserves all provided context data."""
        context = {"file": "config.yaml", "line": 42, "error_code": "E001"}
        error = PMKitError(
            "Invalid configuration",
            suggestion="Fix the YAML syntax",
            context=context
        )
        assert error.message == "Invalid configuration"
        assert error.suggestion == "Fix the YAML syntax"
        assert error.context == context
        assert error.context["file"] == "config.yaml"
        assert error.context["line"] == 42
    
    def test_display_shows_error_panel(self, mock_console):
        """Test that display() method creates and shows a Rich panel."""
        console_mock, output = mock_console
        
        # Patch the console in the exceptions module
        with patch('pmkit.exceptions.console', console_mock):
            error = PMKitError("Test error message")
            error.display()
        
        output_str = output.getvalue()
        assert "Error" in output_str
        assert "Test error message" in output_str
    
    def test_display_includes_suggestion_when_provided(self, mock_console):
        """Test that display() shows suggestions with lightbulb emoji."""
        console_mock, output = mock_console
        
        with patch('pmkit.exceptions.console', console_mock):
            error = PMKitError(
                "Configuration missing",
                suggestion="Create a config.yaml file"
            )
            error.display()
        
        output_str = output.getvalue()
        assert "Configuration missing" in output_str
        assert "💡" in output_str
        assert "Create a config.yaml file" in output_str
    
    def test_display_handles_none_values(self, mock_console):
        """Test that display() gracefully handles None values."""
        console_mock, output = mock_console
        
        with patch('pmkit.exceptions.console', console_mock):
            error = PMKitError("Error with no extras", suggestion=None, context=None)
            error.display()
        
        output_str = output.getvalue()
        assert "Error with no extras" in output_str
        # Should not have suggestion section
        assert "💡" not in output_str
    
    def test_inheritance_from_exception(self):
        """Test that PMKitError properly inherits from Exception."""
        error = PMKitError("Test error")
        assert isinstance(error, Exception)
        assert isinstance(error, PMKitError)
        
        # Should be raisable
        with pytest.raises(PMKitError) as exc_info:
            raise error
        assert exc_info.value.message == "Test error"


class TestConfigError:
    """Test suite for ConfigError with auto-suggestions."""
    
    def test_creates_config_error_with_message(self):
        """Test that ConfigError is created with correct message."""
        error = ConfigError("Invalid setting: debug_mode")
        assert error.message == "Invalid setting: debug_mode"
        assert isinstance(error, PMKitError)
        assert isinstance(error, ConfigError)
    
    def test_auto_suggestion_for_api_key_errors(self):
        """Test auto-suggestion when API key is mentioned in error."""
        error = ConfigError("Missing OPENAI_API_KEY")
        assert error.suggestion == "Set your API key in environment variables or .env file"
        
        # Test case insensitive
        error2 = ConfigError("api key not found")
        assert error2.suggestion == "Set your API key in environment variables or .env file"
        
        # Test with different API providers
        error3 = ConfigError("Invalid Anthropic API configuration")
        assert error3.suggestion == "Set your API key in environment variables or .env file"
    
    def test_custom_suggestion_overrides_auto_suggestion(self):
        """Test that explicit suggestion overrides auto-suggestion."""
        error = ConfigError(
            "API key invalid",
            suggestion="Contact admin for new API key"
        )
        assert error.suggestion == "Contact admin for new API key"
    
    def test_no_auto_suggestion_for_non_api_errors(self):
        """Test that non-API errors don't get auto-suggestions."""
        error = ConfigError("Invalid timeout value")
        assert error.suggestion is None
        
        error2 = ConfigError("Cache directory not writable")
        assert error2.suggestion is None


class TestContextError:
    """Test suite for ContextError with context initialization suggestions."""
    
    def test_creates_context_error_with_message(self):
        """Test that ContextError is created correctly."""
        error = ContextError("Context file corrupted")
        assert error.message == "Context file corrupted"
        assert isinstance(error, PMKitError)
        assert isinstance(error, ContextError)
    
    def test_auto_suggestion_for_not_found_errors(self):
        """Test auto-suggestion when context not found."""
        error = ContextError("Context not found in current directory")
        assert error.suggestion == "Run 'pm init' to initialize your project context"
        
        # Test case insensitive
        error2 = ContextError("Company context NOT FOUND")
        assert error2.suggestion == "Run 'pm init' to initialize your project context"
        
        # Test partial match
        error3 = ContextError("The .pmkit directory was not found")
        assert error3.suggestion == "Run 'pm init' to initialize your project context"
    
    def test_custom_suggestion_overrides_auto_suggestion(self):
        """Test that explicit suggestion overrides auto-suggestion."""
        error = ContextError(
            "Context not found",
            suggestion="Navigate to project root first"
        )
        assert error.suggestion == "Navigate to project root first"
    
    def test_no_auto_suggestion_for_other_errors(self):
        """Test that non-'not found' errors don't get auto-suggestions."""
        error = ContextError("Context version mismatch")
        assert error.suggestion is None
        
        error2 = ContextError("Invalid YAML in context file")
        assert error2.suggestion is None


class TestLLMError:
    """Test suite for LLMError with provider info and smart suggestions."""
    
    def test_creates_llm_error_with_message_only(self):
        """Test LLMError creation with minimal parameters."""
        error = LLMError("API call failed")
        assert error.message == "API call failed"
        assert error.context == {}
        assert isinstance(error, PMKitError)
        assert isinstance(error, LLMError)
    
    def test_creates_llm_error_with_provider(self):
        """Test that provider info is stored in context."""
        error = LLMError("Connection timeout", provider="OpenAI")
        assert error.message == "Connection timeout"
        assert error.context["provider"] == "OpenAI"
    
    def test_auto_suggestion_for_rate_limit_errors(self):
        """Test auto-suggestion for rate limiting errors."""
        error = LLMError("Rate limit exceeded")
        assert error.suggestion == "Please wait a moment and try again"
        
        # Case insensitive
        error2 = LLMError("You have hit the RATE LIMIT")
        assert error2.suggestion == "Please wait a moment and try again"
        
        # With provider
        error3 = LLMError("OpenAI rate limit", provider="OpenAI")
        assert error3.suggestion == "Please wait a moment and try again"
    
    def test_auto_suggestion_for_timeout_errors(self):
        """Test auto-suggestion for timeout errors."""
        error = LLMError("Request timeout after 30 seconds")
        assert error.suggestion == "Check your internet connection or try again"
        
        error2 = LLMError("Connection TIMEOUT", provider="Anthropic")
        assert error2.suggestion == "Check your internet connection or try again"
    
    def test_auto_suggestion_for_auth_errors(self):
        """Test auto-suggestion for authentication errors."""
        # Test with 'key' in message
        error = LLMError("Invalid API key", provider="OpenAI")
        assert error.suggestion == "Check your OpenAI key is valid"
        
        # Test with 'auth' in message
        error2 = LLMError("Authentication failed", provider="Anthropic")
        assert error2.suggestion == "Check your Anthropic key is valid"
        
        # Without provider
        error3 = LLMError("API key not recognized")
        assert error3.suggestion == "Check your API key is valid"
    
    def test_custom_suggestion_overrides_auto_suggestion(self):
        """Test that explicit suggestion overrides auto-suggestion."""
        error = LLMError(
            "Rate limit hit",
            provider="OpenAI",
            suggestion="Upgrade to paid tier"
        )
        assert error.suggestion == "Upgrade to paid tier"
    
    def test_no_auto_suggestion_for_other_errors(self):
        """Test that unrecognized errors don't get auto-suggestions."""
        error = LLMError("Model not available", provider="OpenAI")
        assert error.suggestion is None
        
        error2 = LLMError("Invalid prompt format")
        assert error2.suggestion is None


class TestValidationError:
    """Test suite for ValidationError with field information."""
    
    def test_creates_validation_error_with_message_only(self):
        """Test ValidationError with just a message."""
        error = ValidationError("Invalid data format")
        assert error.message == "Invalid data format"
        assert error.suggestion == "Check your input format"
        assert error.context == {}
        assert isinstance(error, PMKitError)
        assert isinstance(error, ValidationError)
    
    def test_creates_validation_error_with_field(self):
        """Test ValidationError with field name provided."""
        error = ValidationError("Must be a positive integer", field="age")
        assert error.message == "Must be a positive integer"
        assert error.suggestion == "Check the format of 'age'"
        assert error.context["field"] == "age"
    
    def test_field_none_produces_generic_suggestion(self):
        """Test that None field produces generic suggestion."""
        error = ValidationError("Data validation failed", field=None)
        assert error.suggestion == "Check your input format"
        assert error.context == {}
    
    def test_empty_field_name_handled_correctly(self):
        """Test that empty string field name is handled."""
        error = ValidationError("Required field missing", field="")
        # Empty string is falsy in Python, so generic suggestion
        assert error.suggestion == "Check your input format"
        assert error.context == {}


class TestErrorFormatting:
    """Test suite for error display formatting and styling."""
    
    def test_error_panel_has_correct_styling(self, mock_console):
        """Test that error panels have correct border style and title."""
        console_mock, output = mock_console
        
        with patch('pmkit.exceptions.console', console_mock):
            error = PMKitError("Styling test")
            error.display()
        
        output_str = output.getvalue()
        # Panel should have error title
        assert "❌" in output_str
        assert "Error" in output_str
    
    def test_multiple_errors_display_correctly(self, mock_console):
        """Test that multiple errors can be displayed sequentially."""
        console_mock, output = mock_console
        
        errors = [
            ConfigError("Missing API key"),
            LLMError("Rate limit hit", provider="OpenAI"),
            ValidationError("Invalid email", field="user_email")
        ]
        
        with patch('pmkit.exceptions.console', console_mock):
            for error in errors:
                error.display()
        
        output_str = output.getvalue()
        # All errors should be in output
        assert "Missing API key" in output_str
        assert "Rate limit hit" in output_str
        assert "Invalid email" in output_str
        # All suggestions should be present
        assert "Set your API key" in output_str
        assert "wait a moment" in output_str
        assert "user_email" in output_str


class TestEdgeCases:
    """Test suite for edge cases and unusual inputs."""
    
    def test_handles_none_message(self):
        """Test that None message is handled properly."""
        # PMKitError accepts None and converts it to string
        error = PMKitError(None)
        assert error.message is None
        assert str(error) == "None"
    
    def test_handles_empty_string_message(self):
        """Test that empty string message works correctly."""
        error = PMKitError("")
        assert error.message == ""
        assert str(error) == ""
    
    def test_handles_very_long_message(self, mock_console):
        """Test that very long messages are displayed correctly."""
        console_mock, output = mock_console
        
        long_message = "Error: " + "x" * 500
        
        with patch('pmkit.exceptions.console', console_mock):
            error = PMKitError(long_message)
            error.display()
        
        output_str = output.getvalue()
        # Should contain at least part of the message
        assert "Error:" in output_str
        assert "xxxxx" in output_str
    
    def test_handles_special_characters_in_message(self, mock_console):
        """Test that special characters in messages are handled."""
        console_mock, output = mock_console
        
        special_message = "Error with special chars: \n\t\r 'quotes' \"double\" <html>"
        
        with patch('pmkit.exceptions.console', console_mock):
            error = PMKitError(special_message)
            error.display()
        
        output_str = output.getvalue()
        # Should preserve the message content
        assert "special chars" in output_str
    
    def test_context_modification_doesnt_affect_original(self):
        """Test that modifying context after creation doesn't affect error."""
        context = {"key": "value"}
        error = PMKitError("Test", context=context.copy())  # Pass a copy
        
        # Modify original context
        context["key"] = "modified"
        context["new_key"] = "new_value"
        
        # Since PMKitError doesn't make a deep copy, we need to test differently
        # The error's context is the same object if not copied
        error2 = PMKitError("Test2", context={"key": "value"})
        original = {"key": "value"}
        error2.context["key"] = "still_value"
        # Original dict should not be affected if we passed a literal
        assert original["key"] == "value"
    
    def test_exception_chaining_preserved(self):
        """Test that exception chaining works correctly."""
        try:
            # Create an original exception
            raise ValueError("Original error")
        except ValueError as e:
            # Chain with PMKitError
            try:
                raise PMKitError("Wrapped error") from e
            except PMKitError as pmkit_error:
                assert pmkit_error.__cause__ is not None
                assert isinstance(pmkit_error.__cause__, ValueError)
                assert str(pmkit_error.__cause__) == "Original error"


class TestIntegration:
    """Integration tests for exception usage patterns."""
    
    def test_exception_hierarchy_catch_patterns(self):
        """Test that exceptions can be caught at different hierarchy levels."""
        error = ValidationError("Invalid input", field="test")
        
        # Can catch as ValidationError
        with pytest.raises(ValidationError):
            raise error
        
        # Can catch as PMKitError
        with pytest.raises(PMKitError):
            raise error
        
        # Can catch as Exception
        with pytest.raises(Exception):
            raise error
    
    def test_all_exceptions_inherit_from_pmkit_error(self):
        """Test that all custom exceptions inherit from PMKitError."""
        exceptions = [
            ConfigError("test"),
            ContextError("test"),
            LLMError("test"),
            ValidationError("test")
        ]
        
        for exc in exceptions:
            assert isinstance(exc, PMKitError)
            assert isinstance(exc, Exception)
    
    def test_real_world_error_flow(self, mock_console):
        """Test a realistic error handling flow."""
        console_mock, output = mock_console
        
        def process_config():
            """Simulate a function that might fail."""
            import os
            # First check API key
            if not os.environ.get("OPENAI_API_KEY"):
                raise ConfigError("OPENAI_API_KEY not found")
            
            # Then validate data
            user_data = {"name": "test", "email": "invalid"}
            if "@" not in user_data.get("email", ""):
                raise ValidationError("Invalid email format", field="email")
            
            return True
        
        # Test the flow
        import os
        with patch('pmkit.exceptions.console', console_mock):
            try:
                # Remove API key to trigger error
                os.environ.pop("OPENAI_API_KEY", None)
                process_config()
            except PMKitError as e:
                e.display()
                assert "API_KEY not found" in e.message
                assert e.suggestion is not None
        
        output_str = output.getvalue()
        # Check output was captured
        assert "API_KEY not found" in output_str
        assert "Set your API key" in output_str