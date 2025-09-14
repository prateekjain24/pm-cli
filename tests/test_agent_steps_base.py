"""Tests for base step class."""

import asyncio
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from pmkit.agents.steps.base import BaseStep


class ConcreteStep(BaseStep):
    """Concrete implementation of BaseStep for testing."""

    @property
    def name(self) -> str:
        return "Test Step"

    @property
    def step_type(self) -> str:
        return "test"

    async def run(self) -> Dict[str, Any]:
        """Execute the test step."""
        return {"test_result": "success"}


class TestBaseStep:
    """Test the BaseStep abstract class."""

    def test_step_initialization(self):
        """Test step initialization."""
        step = ConcreteStep()
        assert step.state == {}
        assert step.console is not None
        assert step.prompts is not None

    def test_step_with_state(self):
        """Test step initialization with existing state."""
        initial_state = {"key": "value", "count": 42}
        step = ConcreteStep(state=initial_state)
        assert step.state == initial_state

    def test_show_step_header(self):
        """Test showing step header."""
        step = ConcreteStep()
        with patch.object(step.console, 'print') as mock_print:
            step.show_step_header(2, 5)
            # Should print indicator and step name
            assert mock_print.call_count >= 2

    def test_prompt_with_default(self):
        """Test prompting with default value."""
        step = ConcreteStep()
        with patch('pmkit.agents.steps.base.Prompt.ask', return_value="user_input"):
            result = step.prompt_with_default("Enter value:", default="default")
            assert result == "user_input"

    def test_prompt_with_default_and_help(self):
        """Test prompting with default and help text."""
        step = ConcreteStep()
        with patch('pmkit.agents.steps.base.Prompt.ask', return_value="test"):
            with patch.object(step.console, 'print') as mock_print:
                result = step.prompt_with_default(
                    "Enter value:",
                    default="default",
                    help_text="This is help text"
                )
                assert result == "test"
                # Help text should be printed
                mock_print.assert_called_once_with("This is help text")

    def test_prompt_choice_by_number(self):
        """Test choice prompt with number selection."""
        step = ConcreteStep()
        choices = ["Option A", "Option B", "Option C"]

        with patch('pmkit.agents.steps.base.Prompt.ask', return_value="2"):
            with patch.object(step.console, 'print'):
                result = step.prompt_choice("Choose:", choices)
                assert result == "Option B"

    def test_prompt_choice_by_name(self):
        """Test choice prompt with name selection."""
        step = ConcreteStep()
        choices = ["Option A", "Option B", "Option C"]

        with patch('pmkit.agents.steps.base.Prompt.ask', return_value="Option C"):
            with patch.object(step.console, 'print'):
                result = step.prompt_choice("Choose:", choices)
                assert result == "Option C"

    def test_prompt_choice_partial_match(self):
        """Test choice prompt with partial matching."""
        step = ConcreteStep()
        choices = ["Production", "Development", "Testing"]

        with patch('pmkit.agents.steps.base.Prompt.ask', return_value="prod"):
            with patch.object(step.console, 'print'):
                result = step.prompt_choice("Choose:", choices)
                assert result == "Production"

    def test_prompt_choice_invalid_then_valid(self):
        """Test choice prompt with invalid then valid input."""
        step = ConcreteStep()
        choices = ["Option A", "Option B"]

        # First return invalid, then valid
        with patch('pmkit.agents.steps.base.Prompt.ask', side_effect=["invalid", "1"]):
            with patch.object(step.console, 'print') as mock_print:
                result = step.prompt_choice("Choose:", choices)
                assert result == "Option A"
                # Should print error message for invalid choice
                error_calls = [call for call in mock_print.call_args_list
                              if "Invalid choice" in str(call)]
                assert len(error_calls) > 0

    def test_confirm(self):
        """Test confirmation prompt."""
        step = ConcreteStep()
        with patch('pmkit.agents.steps.base.Confirm.ask', return_value=True):
            result = step.confirm("Are you sure?")
            assert result is True

    def test_show_panel(self):
        """Test showing panel."""
        step = ConcreteStep()
        with patch.object(step.console, 'print') as mock_print:
            step.show_panel("Panel content", title="Test Panel", style="success")
            mock_print.assert_called_once()
            # Check that a Panel object was printed
            call_args = mock_print.call_args[0][0]
            assert hasattr(call_args, 'title')  # Panel has title attribute

    def test_show_error(self):
        """Test showing error message."""
        step = ConcreteStep()
        with patch.object(step.console, 'print') as mock_print:
            step.show_error("Error message")
            mock_print.assert_called_once()
            assert "Error message" in mock_print.call_args[0][0]
            assert "red" in mock_print.call_args[0][0]

    def test_show_warning(self):
        """Test showing warning message."""
        step = ConcreteStep()
        with patch.object(step.console, 'print') as mock_print:
            step.show_warning("Warning message")
            mock_print.assert_called_once()
            assert "Warning message" in mock_print.call_args[0][0]
            assert "yellow" in mock_print.call_args[0][0]

    def test_show_success(self):
        """Test showing success message."""
        step = ConcreteStep()
        with patch.object(step.console, 'print') as mock_print:
            step.show_success("Success message")
            mock_print.assert_called_once()
            assert "Success message" in mock_print.call_args[0][0]
            assert "green" in mock_print.call_args[0][0]

    def test_show_info(self):
        """Test showing info message."""
        step = ConcreteStep()
        with patch.object(step.console, 'print') as mock_print:
            step.show_info("Info message")
            mock_print.assert_called_once()
            assert "Info message" in mock_print.call_args[0][0]
            assert "cyan" in mock_print.call_args[0][0]

    def test_validate_required_valid(self):
        """Test validating required field with valid value."""
        step = ConcreteStep()
        assert step.validate_required("valid_value", "Field") is True
        assert step.validate_required(123, "Number") is True
        assert step.validate_required(["list"], "List") is True

    def test_validate_required_invalid(self):
        """Test validating required field with invalid value."""
        step = ConcreteStep()
        with patch.object(step, 'show_error') as mock_error:
            assert step.validate_required("", "Field") is False
            mock_error.assert_called_once()

        with patch.object(step, 'show_error') as mock_error:
            assert step.validate_required("   ", "Field") is False
            mock_error.assert_called_once()

        with patch.object(step, 'show_error') as mock_error:
            assert step.validate_required(None, "Field") is False
            mock_error.assert_called_once()

    def test_save_to_state(self):
        """Test saving data to state."""
        step = ConcreteStep()
        step.state = {"existing": "data"}

        step.save_to_state({"new": "value", "count": 42})

        assert step.state == {"existing": "data", "new": "value", "count": 42}

    @pytest.mark.asyncio
    async def test_run_method(self):
        """Test the run method implementation."""
        step = ConcreteStep()
        result = await step.run()
        assert result == {"test_result": "success"}

    def test_abstract_properties(self):
        """Test that abstract properties are implemented."""
        step = ConcreteStep()
        assert step.name == "Test Step"
        assert step.step_type == "test"