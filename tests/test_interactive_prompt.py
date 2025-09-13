"""
Tests for the interactive prompt flow module.

This module tests the InteractivePromptFlow class and its components
including validators, completers, and wizard functionality.
"""

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.validation import ValidationError
from unittest.mock import Mock, patch, MagicMock

from pmkit.agents.interactive_prompt import (
    CompanyNameValidator,
    EmailValidator,
    URLValidator,
    ProductDescriptionValidator,
    TeamSizeValidator,
    IndustryCompleter,
    RoleCompleter,
    MetricCompleter,
    WizardStep,
    WizardState,
    InteractivePromptFlow,
    create_quick_setup_wizard,
    B2B_QUICK_SETUP,
    B2C_QUICK_SETUP,
)


# ============================================================
# VALIDATOR TESTS
# ============================================================

class TestCompanyNameValidator:
    """Test the CompanyNameValidator class."""

    def test_valid_company_name(self):
        """Test that valid company names pass validation."""
        validator = CompanyNameValidator()

        valid_names = [
            "Acme Corp",
            "OpenAI",
            "Google",
            "My Awesome Startup",
            "123 Industries",
        ]

        for name in valid_names:
            doc = Document(name)
            # Should not raise
            validator.validate(doc)

    def test_company_name_too_short(self):
        """Test that company names less than 2 chars are rejected."""
        validator = CompanyNameValidator()
        doc = Document("A")

        with pytest.raises(ValidationError) as exc_info:
            validator.validate(doc)
        assert "at least 2 characters" in str(exc_info.value.message)

    def test_company_name_too_long(self):
        """Test that company names over 50 chars are rejected."""
        validator = CompanyNameValidator()
        doc = Document("A" * 51)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate(doc)
        assert "less than 50 characters" in str(exc_info.value.message)

    def test_company_name_forbidden_chars(self):
        """Test that forbidden characters are rejected."""
        validator = CompanyNameValidator()
        forbidden_names = [
            "Company<Name>",
            "Company:Name",
            "Company/Name",
            "Company\\Name",
            "Company|Name",
            "Company?Name",
            "Company*Name",
        ]

        for name in forbidden_names:
            doc = Document(name)
            with pytest.raises(ValidationError) as exc_info:
                validator.validate(doc)
            assert "cannot contain special characters" in str(exc_info.value.message)


class TestEmailValidator:
    """Test the EmailValidator class."""

    def test_valid_emails(self):
        """Test that valid emails pass validation."""
        validator = EmailValidator()

        valid_emails = [
            "user@example.com",
            "john.doe@company.co.uk",
            "pm+test@startup.io",
            "123@456.com",
        ]

        for email in valid_emails:
            doc = Document(email)
            validator.validate(doc)

    def test_empty_email_allowed(self):
        """Test that empty email is allowed (optional field)."""
        validator = EmailValidator()
        doc = Document("")
        validator.validate(doc)  # Should not raise

    def test_email_missing_at_symbol(self):
        """Test that emails without @ are rejected."""
        validator = EmailValidator()
        doc = Document("user.example.com")

        with pytest.raises(ValidationError) as exc_info:
            validator.validate(doc)
        assert "must contain @ symbol" in str(exc_info.value.message)

    def test_invalid_email_format(self):
        """Test that invalid email formats are rejected."""
        validator = EmailValidator()
        invalid_emails = [
            "@example.com",
            "user@",
            "user@.com",
            "user@example",
            "user @example.com",
        ]

        for email in invalid_emails:
            doc = Document(email)
            with pytest.raises(ValidationError):
                validator.validate(doc)


class TestURLValidator:
    """Test the URLValidator class."""

    def test_valid_urls(self):
        """Test that valid URLs pass validation."""
        validator = URLValidator()

        valid_urls = [
            "http://example.com",
            "https://www.example.com",
            "https://sub.domain.co.uk",
            "http://localhost:3000",
        ]

        for url in valid_urls:
            doc = Document(url)
            validator.validate(doc)

    def test_empty_url_allowed(self):
        """Test that empty URL is allowed (optional field)."""
        validator = URLValidator()
        doc = Document("")
        validator.validate(doc)  # Should not raise

    def test_url_missing_protocol(self):
        """Test that URLs without protocol are rejected."""
        validator = URLValidator()
        doc = Document("www.example.com")

        with pytest.raises(ValidationError) as exc_info:
            validator.validate(doc)
        assert "must start with http://" in str(exc_info.value.message)


class TestProductDescriptionValidator:
    """Test the ProductDescriptionValidator class."""

    def test_valid_descriptions(self):
        """Test that valid descriptions pass validation."""
        validator = ProductDescriptionValidator()

        valid_descriptions = [
            "AI-powered code review tool for engineering teams",
            "Mobile app for tracking personal fitness goals",
            "B2B SaaS platform that enables real-time collaboration",
            "E-commerce marketplace connecting local artisans with buyers",
            "Project management software designed specifically for remote teams",
        ]

        for desc in valid_descriptions:
            doc = Document(desc)
            validator.validate(doc)

    def test_description_too_short(self):
        """Test that descriptions with less than 5 words are rejected."""
        validator = ProductDescriptionValidator()
        doc = Document("Code review tool")  # Only 3 words

        with pytest.raises(ValidationError) as exc_info:
            validator.validate(doc)
        assert "at least 5 words" in str(exc_info.value.message)


class TestTeamSizeValidator:
    """Test the TeamSizeValidator class."""

    def test_valid_team_sizes(self):
        """Test that valid team sizes pass validation."""
        validator = TeamSizeValidator()

        valid_sizes = ["1", "10", "50", "500", "5000"]

        for size in valid_sizes:
            doc = Document(size)
            validator.validate(doc)

    def test_empty_team_size_allowed(self):
        """Test that empty team size is allowed (optional field)."""
        validator = TeamSizeValidator()
        doc = Document("")
        validator.validate(doc)  # Should not raise

    def test_non_numeric_team_size(self):
        """Test that non-numeric team sizes are rejected."""
        validator = TeamSizeValidator()
        doc = Document("twenty")

        with pytest.raises(ValidationError) as exc_info:
            validator.validate(doc)
        assert "must be a number" in str(exc_info.value.message)

    def test_team_size_too_small(self):
        """Test that team size less than 1 is rejected."""
        validator = TeamSizeValidator()
        doc = Document("0")

        with pytest.raises(ValidationError) as exc_info:
            validator.validate(doc)
        assert "at least 1" in str(exc_info.value.message)


# ============================================================
# COMPLETER TESTS
# ============================================================

class TestIndustryCompleter:
    """Test the IndustryCompleter class."""

    def test_b2b_industry_completion(self):
        """Test B2B industry completions."""
        completer = IndustryCompleter(company_type='b2b')
        doc = Document("Saa")  # Partial match for "SaaS"

        completions = list(completer.get_completions(doc, None))

        assert len(completions) == 1
        assert completions[0].text == "SaaS"

    def test_b2c_industry_completion(self):
        """Test B2C industry completions."""
        completer = IndustryCompleter(company_type='b2c')
        doc = Document("Gam")  # Partial match for "Gaming"

        completions = list(completer.get_completions(doc, None))

        assert len(completions) == 1
        assert completions[0].text == "Gaming"

    def test_case_insensitive_completion(self):
        """Test that completion is case-insensitive."""
        completer = IndustryCompleter(company_type='b2b')
        doc = Document("saas")  # Lowercase

        completions = list(completer.get_completions(doc, None))

        assert len(completions) == 1
        assert completions[0].text == "SaaS"


class TestRoleCompleter:
    """Test the RoleCompleter class."""

    def test_role_completion(self):
        """Test role completions."""
        completer = RoleCompleter()
        doc = Document("Senior")  # Partial match

        completions = list(completer.get_completions(doc, None))

        assert len(completions) == 1
        assert completions[0].text == "Senior Product Manager"

    def test_multiple_role_completions(self):
        """Test multiple matching role completions."""
        completer = RoleCompleter()
        doc = Document("Product")  # Matches multiple roles

        completions = list(completer.get_completions(doc, None))

        assert len(completions) > 1
        role_texts = [c.text for c in completions]
        assert "Product Manager" in role_texts
        assert "Product Owner" in role_texts


class TestMetricCompleter:
    """Test the MetricCompleter class."""

    def test_b2b_metric_completion(self):
        """Test B2B metric completions."""
        completer = MetricCompleter(company_type='b2b')
        doc = Document("MR")  # Partial match for "MRR"

        completions = list(completer.get_completions(doc, None))

        assert len(completions) == 1
        assert completions[0].text == "MRR"

    def test_b2c_metric_completion(self):
        """Test B2C metric completions."""
        completer = MetricCompleter(company_type='b2c')
        doc = Document("MA")  # Partial match for "MAU"

        completions = list(completer.get_completions(doc, None))

        assert len(completions) == 1
        assert completions[0].text == "MAU"


# ============================================================
# WIZARD STATE TESTS
# ============================================================

class TestWizardState:
    """Test the WizardState class."""

    def test_add_step(self):
        """Test adding steps to the wizard."""
        state = WizardState()
        step = WizardStep(name="test", prompt="Test prompt")

        state.add_step(step)

        assert len(state.steps) == 1
        assert state.steps[0] == step

    def test_navigation_forward(self):
        """Test forward navigation through wizard."""
        state = WizardState()
        step1 = WizardStep(name="step1", prompt="Step 1")
        step2 = WizardStep(name="step2", prompt="Step 2")

        state.add_step(step1)
        state.add_step(step2)

        assert state.get_current_step() == step1

        next_step = state.next_step()
        assert next_step == step2
        assert state.get_current_step() == step2

    def test_navigation_backward(self):
        """Test backward navigation through wizard."""
        state = WizardState()
        step1 = WizardStep(name="step1", prompt="Step 1")
        step2 = WizardStep(name="step2", prompt="Step 2")

        state.add_step(step1)
        state.add_step(step2)

        state.next_step()  # Move to step2
        assert state.can_go_back()

        prev_step = state.go_back()
        assert prev_step == step1
        assert state.get_current_step() == step1

    def test_save_step_data(self):
        """Test saving data for steps."""
        state = WizardState()

        state.save_step_data("company_name", "Acme Corp")
        state.save_step_data("company_type", "b2b")

        assert state.data["company_name"] == "Acme Corp"
        assert state.data["company_type"] == "b2b"

    def test_get_progress(self):
        """Test getting wizard progress."""
        state = WizardState()
        state.add_step(WizardStep(name="step1", prompt="Step 1"))
        state.add_step(WizardStep(name="step2", prompt="Step 2"))
        state.add_step(WizardStep(name="step3", prompt="Step 3"))

        current, total = state.get_progress()
        assert current == 1
        assert total == 3

        state.next_step()
        current, total = state.get_progress()
        assert current == 2
        assert total == 3


# ============================================================
# INTERACTIVE PROMPT FLOW TESTS
# ============================================================

class TestInteractivePromptFlow:
    """Test the InteractivePromptFlow class."""

    @patch('pmkit.agents.interactive_prompt.PromptSession')
    def test_prompt_with_validation(self, mock_session_class):
        """Test prompting with validation."""
        mock_session = Mock()
        mock_session.prompt.return_value = "Acme Corp"
        mock_session_class.return_value = mock_session

        flow = InteractivePromptFlow()
        validator = CompanyNameValidator()

        result = flow.prompt_with_validation(
            prompt_text="Company name?",
            validator=validator,
        )

        assert result == "Acme Corp"
        mock_session.prompt.assert_called_once()

    @patch('pmkit.agents.interactive_prompt.PromptSession')
    def test_navigation_commands(self, mock_session_class):
        """Test that navigation commands are recognized."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        flow = InteractivePromptFlow()

        # Test each navigation command
        commands = ['back', 'skip', 'help', 'quit']
        expected = [':back', ':skip', ':help', ':quit']

        for cmd, expected_result in zip(commands, expected):
            mock_session.prompt.return_value = cmd
            result = flow.prompt_with_validation("Test?")
            assert result == expected_result

    def test_show_progress(self):
        """Test progress display."""
        console = Mock()
        flow = InteractivePromptFlow(console=console)

        flow.show_progress(2, 5, "Enrichment")

        # Check that console.print was called
        console.print.assert_called()

    @patch('pmkit.agents.interactive_prompt.confirm')
    def test_preview_and_confirm(self, mock_confirm):
        """Test preview and confirmation."""
        mock_confirm.return_value = True

        console = Mock()
        flow = InteractivePromptFlow(console=console)

        data = {
            'company_name': 'Acme Corp',
            'company_type': 'b2b',
            'product_name': 'AcmeProduct',
            'product_description': 'Amazing product for businesses',
        }

        result = flow.preview_and_confirm(data)

        assert result is True
        mock_confirm.assert_called_once()
        console.print.assert_called()

    def test_show_value_proposition(self):
        """Test showing value proposition after Phase 1."""
        console = Mock()
        flow = InteractivePromptFlow(console=console)

        flow.show_value_proposition('b2b')

        # Check that console.print was called
        console.print.assert_called()


# ============================================================
# QUICK SETUP TESTS
# ============================================================

class TestQuickSetup:
    """Test quick setup functionality."""

    def test_create_b2b_quick_setup(self):
        """Test creating B2B quick setup wizard."""
        steps = create_quick_setup_wizard('b2b')

        assert len(steps) == 8  # 4 essential + 3 enrichment + 1 advanced

        # Check Phase 1 steps
        assert steps[0].name == "company_name"
        assert steps[1].name == "company_type"
        assert steps[2].name == "product_name"
        assert steps[3].name == "product_description"

        # Check that B2B defaults are used
        assert steps[6].default == "MRR"  # Default metric for B2B

    def test_create_b2c_quick_setup(self):
        """Test creating B2C quick setup wizard."""
        steps = create_quick_setup_wizard('b2c')

        assert len(steps) == 8

        # Check that B2C defaults are used
        assert steps[6].default == "MAU"  # Default metric for B2C

    def test_phase_assignment(self):
        """Test that steps are assigned to correct phases."""
        steps = create_quick_setup_wizard('b2b')

        # Phase 1: First 4 steps
        for i in range(4):
            assert steps[i].phase == 1

        # Phase 2: Next 3 steps
        for i in range(4, 7):
            assert steps[i].phase == 2

        # Phase 3: Last step
        assert steps[7].phase == 3


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestIntegration:
    """Integration tests for the interactive prompt flow."""

    @patch('pmkit.agents.interactive_prompt.PromptSession')
    @patch('pmkit.agents.interactive_prompt.confirm')
    def test_complete_wizard_flow(self, mock_confirm, mock_session_class):
        """Test a complete wizard flow."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_confirm.return_value = True

        # Simulate user inputs
        inputs = [
            "Acme Corp",      # Company name
            "B2B",            # Company type
            "AcmeProduct",    # Product name
            "Amazing B2B product for enterprise teams",  # Description
            "",               # Skip industry
            "",               # Skip competitors
            "MRR",            # Metric
            "50",             # Team size
        ]
        mock_session.prompt.side_effect = inputs

        console = Mock()
        flow = InteractivePromptFlow(console=console)

        steps = create_quick_setup_wizard('b2b')
        result = flow.multi_step_wizard(steps, allow_skip=True)

        assert result['company_name'] == "Acme Corp"
        assert result['company_type'] == "B2B"
        assert result['product_name'] == "AcmeProduct"
        assert result['north_star_metric'] == "MRR"
        assert result['team_size'] == "50"

    def test_constants_defined(self):
        """Test that quick setup constants are properly defined."""
        assert 'metrics' in B2B_QUICK_SETUP
        assert 'personas' in B2B_QUICK_SETUP
        assert 'features' in B2B_QUICK_SETUP
        assert 'industries' in B2B_QUICK_SETUP

        assert 'metrics' in B2C_QUICK_SETUP
        assert 'personas' in B2C_QUICK_SETUP
        assert 'features' in B2C_QUICK_SETUP
        assert 'industries' in B2C_QUICK_SETUP

        # Check that they have different values
        assert B2B_QUICK_SETUP['metrics'][0] != B2C_QUICK_SETUP['metrics'][0]